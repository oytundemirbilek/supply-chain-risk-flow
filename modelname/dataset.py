"""Utility module for supply chain graph construction functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.model_selection import KFold
from torch_geometric.data import Data as PygData
from torch_geometric.utils import to_networkx

if TYPE_CHECKING:
    from networkx import Graph


ROOT_PATH = os.path.dirname(__file__)


class SupplyChainNetworkGraph(PygData):
    """
    Base class for common functionalities of all supply chain datasets.

    Examples
    --------
    >>> sc_path = os.path.join(ROOT_PATH, "datasets", "my_supply_chain_data.csv")
    >>> company_path = os.path.join(ROOT_PATH, "datasets", "my_company_data.csv")
    >>> esg_path = os.path.join(ROOT_PATH, "datasets", "my_esg_data.csv")
    >>> gbuilder = GraphBuilder(sc_path, company_path, esg_path)
    >>> pyg_graph = gbuilder.get_pytorch_graph()
    >>> labels = gbuilder.get_regression_labels()
    >>> tickers = gbuilder.get_ticker_ids()
    """

    def __init__(
        self,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        edge_feature_names: list[str] | None = None,
        node_feature_names: list[str] | None = None,
        label_encoding: dict[str, int] | None = None,
        label_name: str = "esg_score",
        label_type: Literal["classification", "regression"] = "regression",
        node_identifier: str = "id",
        graph_split: Literal["inductive", "transductive"] = "transductive",
        mode: str = "inference",
        device: str | torch.device | None = None,
        random_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.edges_df = edges_df
        self.nodes_df = nodes_df

        self.edge_feature_names = edge_feature_names
        if edge_feature_names is None:
            self.edge_feature_names = ["relation_size", "revenue_percentage"]

        self.node_feature_names = node_feature_names
        if node_feature_names is None:
            self.node_feature_names = ["pe_ratio_today"]

        self.node_identifier = node_identifier
        self.label_name = label_name
        self.label_type = label_type
        self.label_encoding = label_encoding

        self.graph_split = graph_split
        self.num_nodes: int = 0
        self.mode = mode
        self.label_name = label_name
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.random_seed = random_seed

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.infer_mask = None

    # def select_nodes(self, node_mask: Tensor) -> PygData:
    #     """
    #     Extract a subgraph from a PyTorch Geometric Data object using a node mask.

    #     Parameters
    #     ----------
    #     node_mask: Tensor
    #         Boolean tensor of shape [num_nodes] indicating which nodes to keep

    #     Returns
    #     -------
    #     PygData:
    #         A new Data object representing the subgraph
    #     """
    #     # Verify the node_mask is valid
    #     if node_mask.dtype != torch.bool:
    #         raise ValueError("node_mask must be a boolean tensor")
    #     if len(node_mask) != self.num_nodes:
    #         raise ValueError(
    #             f"node_mask length {len(node_mask)} doesn't match number of nodes {self.num_nodes}"
    #         )

    #     # Get the indices of the nodes to keep
    #     node_indices = torch.where(node_mask)[0]

    #     # Filter edges where both nodes are in the subgraph
    #     edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
    #     subgraph_edge_index = self.edge_index[:, edge_mask]

    #     # Remap node indices in edge_index to new numbering
    #     node_mapping = torch.zeros(self.num_nodes, dtype=torch.long)
    #     node_mapping[node_mask] = torch.arange(node_mask.sum())
    #     subgraph_edge_index = node_mapping[subgraph_edge_index]

    #     # Create new Data object with subgraph attributes
    #     subgraph_data = PygData()

    #     # Copy node features if they exist
    #     if self.x is not None:
    #         subgraph_data.x = self.x[node_mask]

    #     # Add the remapped edge indices
    #     subgraph_data.edge_index = subgraph_edge_index

    #     # Copy other attributes if they exist (and are node-level)
    #     for key, value in self:
    #         if key in ["x", "edge_index"]:
    #             continue
    #         if isinstance(value, torch.Tensor) and value.size(0) == self.num_nodes:
    #             subgraph_data[key] = value[node_mask]
    #         elif isinstance(value, torch.Tensor) and value.size(
    #             0
    #         ) == self.edge_index.size(1):
    #             # Handle edge attributes if they exist
    #             subgraph_data[key] = value[edge_mask]
    #         else:
    #             # Copy global attributes
    #             subgraph_data[key] = value

    #     return subgraph_data

    # def get_split_graph(self, mode: str) -> PygData:
    #     """Assign indices to the train-validation-test-inference splits."""
    #     if (
    #         self.train_mask is None
    #         or self.val_mask is None
    #         or self.test_mask is None
    #         or self.infer_mask is None
    #     ):
    #         self.transductive_split()

    #     if mode == "train":
    #         return self[self.train_mask]
    #     elif mode == "validation":
    #         return self[self.val_mask]
    #     elif mode == "test":
    #         return self[self.test_mask]
    #     elif mode == "inference":
    #         return self[self.infer_mask]
    #     else:
    #         raise ValueError(
    #             "mode should be 'train', 'validation', 'test' or 'inference'"
    #         )

    def transductive_split(self) -> None:
        """_summary_"""
        # Mask 20% of nodes with ESG scores for validation/test
        # split_transform = RandomNodeSplit(split="random", num_val=0.1, num_test=0.1)
        # self = split_transform(self)  # data.train_mask, data.val_mask, data.test_mask
        if not isinstance(self.y, Tensor):
            raise ValueError("Labels should be a tensor.")

        known_labels_mask = ~torch.isnan(self.y).squeeze()  # [num_nodes]
        known_indices = known_labels_mask.nonzero().squeeze()
        num_known = len(known_indices)

        # Random split of labeled nodes (e.g., 60/20/20)
        shuffled_indices = torch.randperm(num_known)
        train_idx = shuffled_indices[: int(0.6 * num_known)]
        val_idx = shuffled_indices[int(0.6 * num_known) : int(0.8 * num_known)]
        test_idx = shuffled_indices[int(0.8 * num_known) :]

        # Initialize full masks (False by default)
        self.train_mask = torch.zeros(self.y.shape, dtype=torch.bool)  # [num_nodes]
        self.val_mask = torch.zeros(self.y.shape, dtype=torch.bool)
        self.test_mask = torch.zeros(self.y.shape, dtype=torch.bool)

        # Assign splits only to labeled nodes
        self.train_mask[known_indices[train_idx]] = True
        self.val_mask[known_indices[val_idx]] = True
        self.test_mask[known_indices[test_idx]] = True
        self.infer_mask = torch.isnan(self.y)

    # def inductive_split(self) -> Tensor:
    #     # Inductive split (isolate nodes and edges)
    #     train_mask = torch.rand(data.num_nodes) < 0.7
    #     val_mask = ~train_mask & (torch.rand(data.num_nodes) < 0.15)
    #     test_mask = ~train_mask & ~val_mask

    #     # For edges: keep only training-train + train-val/test edges
    #     edge_train_mask = (train_mask[data.edge_index[0]]) & (
    #         train_mask[data.edge_index[1]]
    #     )
    #     edge_val_mask = (train_mask[data.edge_index[0]]) & (
    #         val_mask[data.edge_index[1]]
    #     )

    # def hybrid_split(self) -> Tensor:
    #     # Transductive mask (semi-supervised)
    #     data.y = ESG_scores  # Known scores, NaN for missing
    #     data.train_mask = ~torch.isnan(data.y)  # Start with all known labels
    #     data.train_mask[random_subset] = False  # Mask some for val/test

    #     # Inductive mask (isolated nodes)
    #     train_nodes = torch.randperm(data.num_nodes)[: int(0.7 * data.num_nodes)]
    #     data.train_mask = torch.zeros(data.num_nodes, dtype=bool)
    #     data.train_mask[train_nodes] = True

    def get_ticker_ids(self) -> pd.Series:
        """
        Get ticker ids from supply chain companies table.

        Returns
        -------
        pandas Series
            Pandas column that ticker IDs are located.
        """
        return self.nodes_df[self.node_identifier]

    @staticmethod
    def replace_string_nans(df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        return df.replace("#N/A Field Not Applicable", np.nan).replace(
            "#N/A N/A", np.nan
        )

    @staticmethod
    def cast_as_float(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        columns : list[str]
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        df[columns] = df[columns].astype(float)
        return df

    def get_isin_ids(self) -> pd.Series:
        """
        Get ISIN ids from supply chain companies table.

        Returns
        -------
        pandas Series
            Pandas column that ISIN IDs are located.
        """
        return self.nodes_df["id_isin"]

    def get_regression_labels(self) -> np.ndarray:
        """
        Get regression labels from ESG/SFDR output table.

        Returns
        -------
        numpy ndarray
            Labels in a numpy array.
        """
        return self.nodes_df[self.label_name].astype(float).to_numpy()

    def get_classification_labels(self) -> np.ndarray:
        """
        Get classification labels from ESG/SFDR output table.

        Returns
        -------
        numpy ndarray
            Labels encoded as integers in a numpy array.
        """
        if self.label_encoding is None:
            raise ValueError("Label encoding has to be provided for classification.")
        return self.nodes_df[self.label_name].map(self.label_encoding).to_numpy()

    def company_name_lookup(self, ticker: str) -> str:
        """_summary_

        Parameters
        ----------
        ticker : str
            _description_

        Returns
        -------
        str
            _description_
        """
        if ticker in self.nodes_df[self.node_identifier].to_numpy():
            return self.nodes_df[self.nodes_df[self.node_identifier] == ticker][
                "name"
            ].item()
        return "Not Found"

    def company_index_lookup(self, ticker: str) -> int | None:
        """_summary_

        Parameters
        ----------
        ticker : str
            _description_

        Returns
        -------
        int | None
            _description_
        """
        if ticker in self.nodes_df[self.node_identifier].to_numpy():
            return self.nodes_df[
                self.nodes_df[self.node_identifier] == ticker
            ].index.item()
        return None

    def create_pyg_nodes(self) -> Tensor:
        """"""
        nodes = self.nodes_df[self.node_feature_names].to_numpy()
        return torch.from_numpy(nodes).to(self.device)

    def create_pyg_edges(self) -> tuple[Tensor, Tensor]:
        """_summary_

        Returns
        -------
        tuple[Tensor, Tensor]
            _description_
        """
        self.edges_df = self.replace_string_nans(self.edges_df)
        self.nodes_df = self.replace_string_nans(self.nodes_df)
        self.edges_df = self.cast_as_float(self.edges_df, self.edge_feature_names)
        self.nodes_df = self.cast_as_float(self.nodes_df, self.node_feature_names)

        src_indices = torch.from_numpy(
            self.edges_df["source_company"]
            .map(self.company_index_lookup, na_action="ignore")
            .to_numpy()
        ).to(self.device)
        tgt_indices = torch.from_numpy(
            self.edges_df["target_company"]
            .map(self.company_index_lookup, na_action="ignore")
            .to_numpy()
        ).to(self.device)
        edge_indices = torch.stack([src_indices, tgt_indices])

        edge_attr = torch.from_numpy(
            self.edges_df[self.edge_feature_names].to_numpy()
        ).to(self.device)
        # rev_percent = torch.from_numpy(self.edges_df["revenue_percentage"].to_numpy())
        # edge_attr = torch.stack([rel_size, rev_percent], dim=1)

        return edge_attr, edge_indices

    def create_pyg_labels(
        self, label_type: Literal["classification", "regression"] = "regression"
    ) -> Tensor:
        """_summary_

        Parameters
        ----------
        label_type : Literal[&quot;classification&quot;, &quot;regression&quot;], optional
            _description_, by default "regression"

        Returns
        -------
        Tensor
            _description_
        """
        if label_type == "classification":
            return torch.from_numpy(self.get_classification_labels()).to(self.device)
        if label_type == "regression":
            return torch.from_numpy(self.get_regression_labels()).to(self.device)

    def get_pytorch_graph(self) -> PygData:
        """_summary_

        Returns
        -------
        PygData
            _description_
        """
        if (
            self.x is None
            or self.edge_index is None
            or self.edge_attr is None
            or self.y is None
            or self.ticker_ids is None
        ):
            edge_attr, edge_index = self.create_pyg_edges()
            self.x = self.create_pyg_nodes().float()
            self.edge_index = edge_index
            self.edge_attr = edge_attr.float()
            self.y = self.create_pyg_labels().float()
            self.ticker_ids = self.get_ticker_ids()
        return self

    def get_networkx_graph(self) -> Graph:
        """"""
        pyg_graph = self.get_pytorch_graph()
        return to_networkx(
            pyg_graph,
            node_attrs=["x", "ticker_ids"],
            edge_attrs=["edge_attr"],
            # edge_attrs=["rel_size", "rev_percentage"],
            # to_multi=True,
        )

    # def __repr__(self) -> str:
    #     """Dunder function to return string representation of the dataset."""
    #     return (
    #         f"{self.__class__.__name__} multigraph dataset ({self.mode}) {self.hemisphere} hemisphere with"
    #         f", n_subjects={self.n_subj}, current fold:{self.current_fold + 1}/{self.n_folds}"
    #         f", n_views_source={self.n_views_src}, n_nodes_source={self.n_nodes_src}"
    #         f", n_views_target={self.n_views_tgt}, n_nodes_target={self.n_nodes_tgt}"
    #     )

    @staticmethod
    def get_fold_indices(
        all_data_size: int, n_folds: int, fold_id: int = 0, random_seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        ----------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.
        random_seed: int
            Random seed to be used for randomization.

        Returns
        -------
        train_indices: numpy ndarray
            Indices to get the training dataset.
        val_indices: numpy ndarray
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_indices = kf.split(np.array(range(all_data_size)))
        train_indices, val_indices = [
            (np.array(train), np.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices


class SLPCNetworkGraph(SupplyChainNetworkGraph):
    """"""

    def __init__(
        self,
        edges_df: pd.DataFrame | None = None,
        nodes_df: pd.DataFrame | None = None,
        edge_feature_names: list[str] | None = None,
        node_feature_names: list[str] | None = None,
        label_encoding: dict[str, int] | None = None,
        label_name: str = "esg_score",
        label_type: Literal["classification", "regression"] = "regression",
        node_identifier: str = "id",
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        FILE_PATH = os.path.dirname(__file__)
        DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "bloomberg_cocoa")
        SC_CUSTOMERS_PATH = os.path.join(
            DATA_PATH,
            "cocoa_supply_chain_customers",
            "cocoa_supply_chain_customers.csv",
        )
        COMPANIES_PATH = os.path.join(
            DATA_PATH, "company_list", "companies_filtered.csv"
        )
        FINANCIAL_PATH = os.path.join(
            DATA_PATH, "company_list", "companies_financial.csv"
        )
        ESG_PATH = os.path.join(DATA_PATH, "esg", "cocoa_sc_esg_general.csv")

        esg_df = pd.read_csv(ESG_PATH)
        nodes_df = pd.read_csv(COMPANIES_PATH)
        financial_df = pd.read_csv(FINANCIAL_PATH)
        sc_df = pd.read_csv(SC_CUSTOMERS_PATH)

        nodes_df = nodes_df.merge(
            esg_df,
            left_on="id",
            right_on="Ticker",
            suffixes=(None, None),
            how="left",
        )
        nodes_df = nodes_df.merge(
            financial_df, on="id", suffixes=(None, None), how="left"
        )
        nodes_df = nodes_df[nodes_df["pe_ratio_today"] != "#N/A"]
        nodes_df = nodes_df[~nodes_df["pe_ratio_today"].isna()]
        nodes_df = nodes_df.reset_index()
        sc_df = sc_df[sc_df["relation_size"] != "#N/A Field Not Applicable"]
        sc_df = sc_df[~sc_df["relation_size"].isna()]
        sc_df = sc_df[sc_df["source_company"].isin(nodes_df["id"])]
        sc_df = sc_df[sc_df["target_company"].isin(nodes_df["id"])]
        sc_df = sc_df.reset_index()
        label_name = "ESG Scr"
        super().__init__(
            edges_df=sc_df,
            nodes_df=nodes_df,
            edge_feature_names=edge_feature_names,
            node_feature_names=node_feature_names,
            label_encoding=label_encoding,
            label_name=label_name,
            label_type=label_type,
            node_identifier=node_identifier,
            device=device,
            random_seed=random_seed,
        )


class ISSNetworkGraph(SupplyChainNetworkGraph):
    """"""

    def __init__(
        self,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        edge_feature_names: list[str] | None = None,
        node_feature_names: list[str] | None = None,
        label_encoding: dict[str, int] | None = None,
        label_name: str = "esg_score",
        label_type: Literal["classification", "regression"] = "regression",
        node_identifier: str = "id",
        mode: str = "inference",
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        super().__init__(
            edges_df=edges_df,
            nodes_df=nodes_df,
            edge_feature_names=edge_feature_names,
            node_feature_names=node_feature_names,
            label_encoding=label_encoding,
            label_name=label_name,
            label_type=label_type,
            node_identifier=node_identifier,
            mode=mode,
            device=device,
            random_seed=random_seed,
        )


class MSCINetworkGraph(SupplyChainNetworkGraph):
    """"""

    def __init__(
        self,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        edge_feature_names: list[str] | None = None,
        node_feature_names: list[str] | None = None,
        label_encoding: dict[str, int] | None = None,
        label_name: str = "msci_rating",
        label_type: Literal["classification", "regression"] = "regression",
        node_identifier: str = "id",
        mode: str = "inference",
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        label_encoding = {
            "AAA": 10,
            "AA": 9,
            "A": 8,  # ...
        }
        super().__init__(
            edges_df=edges_df,
            nodes_df=nodes_df,
            edge_feature_names=edge_feature_names,
            node_feature_names=node_feature_names,
            label_encoding=label_encoding,
            label_name=label_name,
            label_type=label_type,
            node_identifier=node_identifier,
            mode=mode,
            device=device,
            random_seed=random_seed,
        )


if __name__ == "__main__":
    sc_network = SLPCNetworkGraph(device="cuda")
    pyg_graph = sc_network.get_pytorch_graph()
    print(pyg_graph)
    print(pyg_graph.y)
    # print(pyg_graph.x)
    # print(pyg_graph.edge_attr)

    # import networkx as nx

    # nx_graph = sc_network.get_networkx_graph()
    # # print(nx.get_node_attributes(nx_graph, "x"))
    # # print(nx.get_edge_attributes(nx_graph, "edge_attr"))

    # from modelname.plotting import Graphplot

    # plotter = Graphplot(nx_graph)
    # plotter.plot()
