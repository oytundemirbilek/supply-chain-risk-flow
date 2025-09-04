"""Module for inference and testing scripts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from modelname.dataset import BBGSupplyChainNetwork
from modelname.autoencoder import SurfConvAutoencoder


class BaseInferer:
    """Inference loop for a trained model. Run the testing scheme."""

    def __init__(
        self,
        model: Module | None = None,
        model_path: str | None = None,
        model_params: dict[str, Any] | None = None,
        out_path: str | None = None,
        random_seed: int = 0,
        device: str | None = None,
    ) -> None:
        """
        Initialize the inference (or testing) setup.

        Parameters
        ----------
        dataset: string
            Which dataset should be used for inference.
        model: torch Module, optional
            The model needs to be tested or inferred. If None, model_path
            and model_params should be specified to load a model.
        model_path: string, optional
            Path to the expected model.
        model_params: dictionary, optional
            Parameters that was specified before the training of the model.
        out_path: string, optional
            If you want to save the predictions, specify a path.
        metric_name: string
            Metric to evaluate the test performance of the model.
        """
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.model_path = model_path
        self.out_path = out_path
        self.random_seed = random_seed
        self.model_params = model_params
        self.model: Module
        if model is None:
            if model_params is None:
                raise ValueError("Specify a model or model params and its path.")
            if model_path is None:
                raise ValueError("Specify a model or model params and its path.")
            self.model = self.load_model_from_file(
                model_path, model_params, self.device
            )
        else:
            self.model = model

    def absolute_percentage_errors(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the normalized error between prediction and target.

        Parameters
        ----------
        prediction: torch Tensor
            Model prediction.
        target: torch Tensor
            Ground truth target.

        Returns
        -------
        error: torch Tensor
            Normalized error.
        """
        return torch.abs((prediction - target) / target)

    @torch.no_grad()
    def run(self, network: BBGSupplyChainNetwork | None = None) -> Tensor:
        """
        Run inference loop whether for testing purposes or in-production.

        Parameters
        ----------
        mode: string
            Either 'test' or 'infer'. Whether to use all dataset samples or just the testing split.
            This can be handy when testing a pretrained model on your private dataset. Set 'infer'
            if you want to use your model in production.

        Returns
        -------
        test_losses: list of floats
            Test loss for each sample. Or any metric you will define. Calculates only if test_split_only is True.
        """
        self.model.eval()
        if network is None:
            network = BBGSupplyChainNetwork(material="cocoa", device=self.device)
        graph_data = network.get_pytorch_graph()
        if graph_data.x is None:
            raise ValueError("Node features are not defined in the graph data.")
        out = self.model(graph_data)

        self.model.train()
        return self.absolute_percentage_errors(out, graph_data.x)

    @staticmethod
    def load_model_from_file(
        model_path: str, model_params: dict[str, Any], device: str | None = None
    ) -> Module:
        """
        Load a pretrained model from file.

        Parameters
        ----------
        model_path: string
            Path to the file which is model saved.
        model_params: dictionary
            Parameters of the model is needed to initialize.

        Returns
        -------
        model: pytorch Module
            Pretrained model ready for inference, or continue training.
        """
        model = SurfConvAutoencoder(**model_params).to(device)
        if not model_path.endswith(".pth"):
            model_path += ".pth"
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device) if device is not None else None,
            )
        )
        return model


if __name__ == "__main__":
    network = BBGSupplyChainNetwork(material="cocoa")
    inferer = BaseInferer(
        model_path=os.path.join("models", "default_model_name", "fold0.pth"),
        model_params={
            "node_feat_dim": network.num_node_features,
            "hidden_dim": 128,
            "latent_dim": 32,
        },
        out_path=None,
    )
    test_losses = inferer.run(network)

    disruptions = network.create_disruptions(("registered_in_country", "ID"), "equals")

    network.apply_disruptions(
        # {"BARN SW Equity": 0.5, "NESN SW Equity": 0.2, "WMT US Equity": 0.0},
        disruptions,
        ["operating_margin", "profit_margin"],
    )
    print(disruptions)

    disrupted_losses = inferer.run(network)

    original_df = network.nodes_df.copy()
    disrupted_df = network.nodes_df.copy()

    original_df["loss"] = test_losses[:, 0].cpu().numpy()
    disrupted_df["loss"] = disrupted_losses[:, 0].cpu().numpy()

    import pandas as pd

    original_df["state"] = "original_state"
    disrupted_df["state"] = "disrupted_state (geological)"
    combined_df = pd.concat(
        [
            original_df,
            disrupted_df,
        ]
    ).reset_index(drop=True)

    from modelname.plotting import Boxplot

    boxplot = Boxplot(combined_df)
    boxplot.plot(
        metric="loss",
        group_by="state",
        split_by="registered_in_country",
        reference_by="Ticker",
    )
