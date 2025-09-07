"""Module to define graph autoencoder network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import NNConv
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_mean

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data as PygData


class SurfCrossModalityDecoder(nn.Module):
    """Decoder that reconstructs both node features and edges, depending on each other."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.attr_decoder = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self, z: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Edge reconstruction
        src, dst = edge_index
        edge_input = torch.cat([z[src], z[dst]], dim=-1)
        edge_recon = self.edge_decoder(edge_input)
        msg = edge_attr

        # Aggregate edge signals into node-level context
        struct_ctx = scatter_mean(torch.cat([msg, z[src]], dim=1), dst, dim=0)
        # struct_ctx = scatter_mean(edge_recon, dst, dim=0, dim_size=z.size(0))

        # Node reconstruction uses both embeddings + structure context
        x_recon = self.attr_decoder(struct_ctx)

        return x_recon, edge_recon


class SurfNNConvEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        # Input linear layer to project node features
        self.lin_in = nn.Linear(node_feat_dim, hidden_dim)

        # Edge network generates weights for NNConv
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(edge_feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                ),
                nn.Sequential(
                    nn.Linear(edge_feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                ),
                nn.Sequential(
                    nn.Linear(edge_feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                ),
            ]
        )
        self.convs = nn.ModuleList(
            [
                NNConv(
                    hidden_dim,
                    hidden_dim,
                    nn=self.edge_mlps[0],
                    aggr="mean",
                    root_weight=False,
                ),
                NNConv(
                    hidden_dim,
                    hidden_dim,
                    nn=self.edge_mlps[1],
                    aggr="mean",
                    root_weight=False,
                ),
                NNConv(
                    hidden_dim,
                    hidden_dim,
                    nn=self.edge_mlps[2],
                    aggr="mean",
                    root_weight=False,
                ),
            ]
        )

        # Output projection to desired embedding size
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph_data: PygData) -> Tensor:
        x, edge_index, edge_attr = (
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
        )
        if x is None:
            raise ValueError("graph.x cannot be None.")
        if edge_index is None:
            raise ValueError("Edge index cannot be None.")

        x0 = x.clone()
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # Initial node projection
        x = F.relu(self.lin_in(x))

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.lin_out(x)
        return x0[:, 0].unsqueeze(-1) * z  # [num_nodes, out_dim]


class SurfNNConvCrossModalityAutoencoder(nn.Module):
    """A graph autoencoder that reconstructs both node features and edges, depending on each other."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SurfNNConvEncoder(
            node_feat_dim, edge_feat_dim, hidden_dim, latent_dim, dropout
        )
        self.cross_decoder = SurfCrossModalityDecoder(
            latent_dim, hidden_dim, node_feat_dim
        )

    def forward(self, graph_data: PygData) -> tuple[Tensor, Tensor]:
        x, edge_index, edge_attr = (
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
        )
        latent_space = self.encoder(graph_data)
        node_recon, edge_recon = self.cross_decoder(latent_space, edge_index, edge_attr)
        return node_recon, edge_recon


if __name__ == "__main__":
    # Quick test
    from torch_geometric.data import Data as PygData

    x = torch.randn((4, 16))  # 4 nodes, 16 features each
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 0, 3, 2, 2, 3]])  # 6 edges
    edge_attr = torch.randn((6))  # 6 edges, 1 feature
    graph_data = PygData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    edge_attr = torch.randn((6, 4))  # 6 edges, 4 features each
    multigraph_data = PygData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    cross_modal_nnconv = SurfNNConvCrossModalityAutoencoder(16, 4, 32, 1)
    nodes_out, edges_out = cross_modal_nnconv(multigraph_data)
    print(nodes_out.shape)  # Should print node probabilities
    print(edges_out.shape)  # Should print edge probabilities
