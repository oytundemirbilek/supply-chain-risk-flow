"""Module for training script."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import MSELoss
from torch.optim import AdamW, Optimizer
from torch_geometric.data import Data as PygData
import torch.nn.functional as F
from sklearn.metrics import r2_score

from modelname.dataset import BBGSupplyChainNetwork
from modelname.autoencoder import SurfNNConvCrossModalityAutoencoder

if torch.cuda.is_available():
    # These two options should be seed to ensure reproducible (If you are using cudnn backend)
    # https://pytorch.org/docs/stable/notes/randomness.html
    from torch.backends import cudnn

    cudnn.deterministic = True
    cudnn.benchmark = False

FILE_PATH = os.path.dirname(__file__)


def compute_loss(x_true, x_recon, edge_true, edge_recon, lambda_edge=0.1):
    L_attr = F.mse_loss(x_recon, x_true)  # node features
    L_edge = F.mse_loss(edge_recon, edge_true)  # edge weights (continuous)
    return L_attr + lambda_edge * L_edge


class BaseTrainer:
    """Wrapper around training function to save all the training parameters."""

    def __init__(  # noqa: PLR0913
        self,
        # Data related:
        dataset: str,
        # Training related:
        n_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.001,
        batch_size: int = 1,
        validation_period: int = 5,
        patience: int | None = None,
        # Model related:
        n_folds: int = 5,
        layer_sizes: tuple[int, ...] = (8, 16),
        loss_weight: float = 1.0,
        loss_name: str = "mse_loss",
        model_name: str = "default_model_name",
        random_seed: int = 0,
        device: str | None = None,
    ) -> None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.random_seed = random_seed
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_period = validation_period
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.layer_sizes = layer_sizes
        self.model_name = model_name
        self.model_save_path = os.path.join(FILE_PATH, "..", "models", model_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.model_params_save_path = os.path.join(
            FILE_PATH, "..", "models", model_name + "_params.json"
        )
        with open(self.model_params_save_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4)

        if loss_name == "mse_loss":
            self.loss_fn = MSELoss()
        else:
            raise NotImplementedError("Specified loss function is not defined.")

        self.val_loss_per_epoch: list[float] = []

    def __repr__(self) -> str:
        """Return string representation of the Trainer as training parameters."""
        return str(self.__dict__)

    def train_step(self, model: Module, data: PygData, optimizer: Optimizer):
        model.train()
        optimizer.zero_grad()

        nodes_out, edges_out = model.forward(data)
        loss_main = compute_loss(data.x, nodes_out, data.edge_attr, edges_out)
        loss_main.backward()
        optimizer.step()

        return loss_main.item()

    @torch.no_grad()
    def validate_step(self, model: Module, data: PygData) -> dict[str, float]:
        """Run validation loop."""
        model.eval()
        nodes_out, edges_out = model.forward(data)

        if not isinstance(data.x, Tensor):
            raise ValueError("Data x should be a tensor.")

        if not isinstance(data.edge_attr, Tensor):
            raise ValueError("Data edge attributes should be a tensor.")

        # print(edges_out)
        # print(data.edge_attr)

        loss_main = compute_loss(data.x, nodes_out, data.edge_attr, edges_out)
        r2_nodes = r2_score(data.x.cpu().numpy(), nodes_out.cpu().numpy())
        r2_edges = r2_score(data.edge_attr.cpu().numpy(), edges_out.cpu().numpy())

        # print(f"Original nodes: {data.x}")
        # print(f"Reconstructed nodes: {nodes_out}")

        loss_dict = {
            "val_loss": loss_main.item(),
            "val_r2_nodes": r2_nodes,
            "val_r2_edges": r2_edges,
        }

        model.train()
        return loss_dict

    def train(self, current_fold: int = 0) -> Module:
        """Train model."""
        network = BBGSupplyChainNetwork(material=self.dataset, device=self.device)
        graph_data = network.get_pytorch_graph()
        model = SurfNNConvCrossModalityAutoencoder(
            graph_data.num_node_features, graph_data.num_edge_features, 128, 32
        ).to(self.device)
        optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_loss = 99999999999999999999999.0
        for epoch in range(self.n_epochs):
            train_loss = self.train_step(model, graph_data, optimizer)

            if (epoch + 1) % self.validation_period == 0:
                val_loss_dict = self.validate_step(model, graph_data)
                msg = " | ".join(
                    [f"{key}: {value:.4f}" for key, value in val_loss_dict.items()]
                )
                print(
                    f"Epoch: {epoch + 1}/{self.n_epochs}",
                    f" | Tr.Loss: {train_loss:.4f}",
                    msg,
                )
                val_loss = val_loss_dict["val_loss"]
                self.val_loss_per_epoch.append(val_loss)

                if val_loss < best_loss:
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.model_save_path, f"fold{current_fold}.pth"),
                    )
                    best_loss = val_loss

        return model


if __name__ == "__main__":
    trainer = BaseTrainer(
        dataset="cocoa",
        n_epochs=1000,
        learning_rate=0.001,
        validation_period=20,
        model_name="cross_modality_model_rel_size",
        # device="cpu",
    )
    trainer.train()
