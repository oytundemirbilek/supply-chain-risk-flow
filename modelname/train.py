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
from sklearn.metrics import r2_score

from modelname.dataset import BBGSupplyChainNetwork
from modelname.autoencoder import SurfConvAutoencoder

if torch.cuda.is_available():
    # These two options should be seed to ensure reproducible (If you are using cudnn backend)
    # https://pytorch.org/docs/stable/notes/randomness.html
    from torch.backends import cudnn

    cudnn.deterministic = True
    cudnn.benchmark = False

FILE_PATH = os.path.dirname(__file__)


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

        out = model.forward(data).squeeze()
        loss_main = self.loss_fn(out, data.x)
        loss_main.backward()
        optimizer.step()

        return loss_main.item()

    @torch.no_grad()
    def validate_step(self, model: Module, data: PygData) -> tuple[float, float]:
        """Run validation loop."""
        model.eval()
        out = model.forward(data).squeeze()

        if not isinstance(data.x, Tensor):
            raise ValueError("Data x should be a tensor.")

        loss_main = self.loss_fn(out, data.x)
        r2 = r2_score(data.x.cpu().numpy(), out.cpu().numpy())

        model.train()
        return loss_main.item(), r2

    def train(self, current_fold: int = 0) -> Module:
        """Train model."""
        network = BBGSupplyChainNetwork(material="cocoa", device=self.device)
        graph_data = network.get_pytorch_graph()
        model = SurfConvAutoencoder(graph_data.num_node_features, 128, 32).to(
            self.device
        )
        optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_loss = 99999999999999999999999.0
        for epoch in range(self.n_epochs):
            train_loss = self.train_step(model, graph_data, optimizer)

            if (epoch + 1) % self.validation_period == 0:
                val_loss, val_r2 = self.validate_step(model, graph_data)
                print(
                    f"Epoch: {epoch + 1}/{self.n_epochs}",
                    f" | Tr.Loss: {train_loss}",
                    f" | Val.Loss: {val_loss}",
                    f" | Val.R2: {val_r2}",
                )
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
        dataset="mock_dataset",
        n_epochs=100,
        learning_rate=0.001,
        validation_period=5,
        # device="cpu",
    )
    trainer.train()
