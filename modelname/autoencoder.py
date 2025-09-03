"""Module to define graph autoencoder network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.nn as nn

# from modelname.gconv import EdgeGAT, EdgeSAGE
from torch_geometric.nn import SAGEConv

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data as PygData


class SurfSageEncoder(nn.Module):
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
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.regressor = nn.Linear(out_dim, out_dim)
        self.dropout = dropout

    def forward(self, graph_data: PygData) -> Tensor:
        x, edge_index = graph_data.x, graph_data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return self.regressor(x)


class SurfSageDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),  # output score for edge
        )

    def forward(self, latent_space: Tensor, edge_index: Tensor) -> Tensor:
        # z: node embeddings [num_nodes, out_channels]
        # edge_index: [2, num_edges]
        src, dst = edge_index
        h_src = latent_space[src]
        h_dst = latent_space[dst]
        h_pair = torch.cat([h_src, h_dst], dim=1)  # concatenate embeddings
        logits = self.mlp(h_pair).squeeze(-1)  # shape: [num_edges]
        return torch.sigmoid(logits)  # probability between 0 and 1


class SurfSageAutoencoder(nn.Module):
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
        self.encoder = SurfSageEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, num_layers, out_dim, dropout
        )
        self.decoder = SurfSageDecoder(out_dim, hidden_dim, dropout)

    def forward(self, graph_data: PygData) -> Tensor:
        x, edge_index, edge_attr = (
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
        )
        latent_space = self.encoder(graph_data)
        edge_logits = self.decoder(latent_space, edge_index)
        return edge_logits


if __name__ == "__main__":
    # Quick test
    from torch_geometric.data import Data as PygData

    x = torch.randn((4, 16))  # 4 nodes, 16 features each
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 0, 3, 2, 2, 3]])  # 6 edges
    edge_attr = torch.randn((6, 4))  # 6 edges, 4 features each
    graph_data = PygData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    ae = SurfSageAutoencoder(16, 4, 32, 2, 1)
    out = ae(graph_data)
    print(out)  # Should print edge probabilities
