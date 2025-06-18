"""Module to define neural network, which is our solution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList, ReLU, Sequential

# from modelname.gconv import EdgeGAT, EdgeSAGE
from torch_geometric.nn import GATv2Conv, NNConv, RGCNConv

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data as PygData


class SurfnnconvNet(Module):
    """A node classifier/regression model that predicts ESG score of a company."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_mlp1 = Sequential(
            Linear(edge_feat_dim, node_feat_dim * hidden_dim), ReLU()
        )
        self.edge_mlp2 = Sequential(
            Linear(edge_feat_dim, node_feat_dim * hidden_dim), ReLU()
        )
        self.conv1 = NNConv(node_feat_dim, hidden_dim, self.edge_mlp1)
        # Captures Tier 2
        self.conv2 = NNConv(hidden_dim, hidden_dim // 2, self.edge_mlp2)
        self.regressor = Linear(hidden_dim // 2, out_dim)

    def forward(self, graph_data: PygData) -> Tensor:
        x, edge_index, edge_attr = (
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
        )
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)  # Tier 2 influence
        return self.regressor(x)


class SurfgateNet(Module):
    """A node classifier/regression model that predicts ESG score of a company."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = ModuleList()
        self.dropout = dropout

        # First GATv2 layer with edge feature incorporation
        self.conv1 = GATv2Conv(
            in_channels=node_feat_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_feat_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        # Second GATv2 layer
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_feat_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
        )

        # Final linear layer for regression
        self.regressor = Linear(hidden_dim * heads, out_dim)

        # Edge feature transformation
        self.edge_encoder = Linear(edge_feat_dim, edge_feat_dim)

    def forward(self, graph_data: PygData) -> Tensor:
        x, edge_index, edge_attr = (
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
        )

        # Transform edge features (for both layers)
        # edge_attr = self.edge_encoder(edge_attr)

        # First GATv2 layer with edge features
        x = F.relu(self.conv1(x, edge_index, edge_attr))

        # Second GATv2 layer with edge features
        x = self.conv2(x, edge_index, edge_attr)

        # Final regression output
        return self.regressor(x)


class SurfrelNet(Module):
    """A node classifier/regression model that predicts ESG score of a company."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Tier 1/Tier 2
        self.conv1 = RGCNConv(node_feat_dim, hidden_dim, num_relations=edge_feat_dim)
        # Tier 2 influence
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=edge_feat_dim)
        self.regressor = Linear(hidden_dim, out_dim)

    def forward(self, graph_data: PygData) -> Tensor:
        x, edge_index, edge_type = (
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_type,  # edge_type should be a single binary value per edge.
        )
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return self.regressor(x)
