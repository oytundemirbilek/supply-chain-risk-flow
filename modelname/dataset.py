"""Utility module for supply chain graph construction functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch_geometric.data import Data as PygData
from torch_geometric.utils import to_networkx

from modelname.utils import read_company_data, read_supply_chain_data, select_companies

if TYPE_CHECKING:
    from networkx import Graph


class SupplyChainNetwork:
    """Class to handle supply chain network graph data."""

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        sc_customers_df: pd.DataFrame,
        sc_suppliers_df: pd.DataFrame,
        node_feature_names: list[str],
        edge_feature_names: list[str],
        node_identifier: str = "Ticker",
        device: str | None = None,
    ) -> None:
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self.node_identifier = node_identifier

        self.nodes_df = nodes_df.reset_index(drop=True)
        self.edges_df = pd.concat([sc_customers_df, sc_suppliers_df]).reset_index(
            drop=True
        )

        self.num_nodes = len(self.nodes_df)
        self.num_edges = len(self.edges_df)
        self.num_node_features = len(self.node_feature_names)
        self.num_edge_features = len(self.edge_feature_names)

        self.pyg_nodes = self.create_pyg_nodes()
        self.pyg_edge_attr, self.pyg_edge_index = self.create_pyg_edges()

        self.graph_data = PygData(
            x=self.pyg_nodes,
            edge_index=self.pyg_edge_index,
            edge_attr=self.pyg_edge_attr,
        )

        self.data_nan_check()
        self.sanity_check()

    def get_pytorch_graph(self) -> PygData:
        """Get the PyTorch Geometric Data object representing the graph."""
        return self.graph_data

    def get_networkx_graph(self) -> Graph:
        """Get the NetworkX graph representation."""
        return to_networkx(self.graph_data, to_undirected=True)

    @staticmethod
    def replace_string_nans(df: pd.DataFrame) -> pd.DataFrame:
        """Replace string 'nan' with actual np.nan in the DataFrame."""
        return df.replace("nan", np.nan)

    @staticmethod
    def cast_as_float(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Cast specified columns of the DataFrame to float type."""
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    @staticmethod
    def normalize_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Normalize specified column using StandardScaler."""
        scaler = StandardScaler()
        df[[column]] = scaler.fit_transform(df[[column]])
        return df

    @staticmethod
    def normalize_with_logscale(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Normalize specified column using log scale followed by MinMaxScaler."""
        df[column] = df[column].apply(lambda x: np.log1p(x) if x > 0 else 0)
        scaler = MinMaxScaler()
        df[[column]] = scaler.fit_transform(df[[column]])
        return df

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
        self.nodes_df = self.replace_string_nans(self.nodes_df)
        self.nodes_df = self.cast_as_float(self.nodes_df, self.node_feature_names)
        for node_feature_name in self.node_feature_names:
            self.nodes_df = self.normalize_features(self.nodes_df, node_feature_name)
        nodes = self.nodes_df[self.node_feature_names].to_numpy()
        return torch.from_numpy(nodes).float().to(self.device)

    def create_pyg_edges(self) -> tuple[Tensor, Tensor]:
        """_summary_

        Returns
        -------
        tuple[Tensor, Tensor]
            _description_
        """
        self.edges_df = self.replace_string_nans(self.edges_df)
        self.edges_df = self.cast_as_float(self.edges_df, self.edge_feature_names)
        # self.edges_df = self.normalize_with_logscale(self.edges_df, "relation_size")

        src_indices = torch.from_numpy(
            self.edges_df["source_company"]
            .map(self.company_index_lookup, na_action="ignore")
            .to_numpy(dtype=np.int64)
        ).to(self.device)
        tgt_indices = torch.from_numpy(
            self.edges_df["target_company"]
            .map(self.company_index_lookup, na_action="ignore")
            .to_numpy(dtype=np.int64)
        ).to(self.device)
        edge_indices = torch.stack([src_indices, tgt_indices])

        edge_attr = (
            torch.from_numpy(self.edges_df[self.edge_feature_names].to_numpy())
            .squeeze()
            .float()
            .to(self.device)
        )

        return edge_attr, edge_indices

    def sanity_check(self) -> None:
        """Perform sanity checks on the graph data."""
        if self.pyg_nodes.shape[0] != self.num_nodes:
            raise ValueError("Node count mismatch.")
        if self.pyg_edge_index.shape[1] != self.num_edges:
            raise ValueError("Edge count mismatch.")
        if self.pyg_edge_attr.shape[0] != self.num_edges:
            raise ValueError("Edge attribute count mismatch.")
        if self.pyg_nodes.shape[1] != self.num_node_features:
            raise ValueError("Node feature dim mismatch.")

        if self.pyg_edge_index.min() < 0:
            raise ValueError("Edge index contains negative values.")
        if self.pyg_edge_index.max() >= self.num_nodes:
            raise ValueError("Edge index exceeds number of nodes.")
        # if self.pyg_edge_attr.shape[1] != self.num_edge_features:
        #     raise ValueError("Edge feature dim mismatch.")
        print("Sanity check passed: Graph data is consistent.")

    def data_nan_check(self) -> None:
        """Check for NaN values in the graph data."""
        if torch.isnan(self.pyg_nodes).any():
            raise ValueError("Node features contain NaN values.")
        if torch.isnan(self.pyg_edge_index).any():
            raise ValueError("Edge indices contain NaN values.")
        if torch.isnan(self.pyg_edge_attr).any():
            raise ValueError("Edge attributes contain NaN values.")
        print("NaN check passed: No NaN values in graph data.")

    def create_disruptions(
        self,
        condition: tuple[str, Any],
        operator: Literal["equals", "not equals", "greater", "smaller"],
        factor: float = 0.2,
    ) -> dict[str, float]:
        """Create a disruption dictionary based on a filter condition.
        Parameters
        ----------
        filter : tuple[str, Any]
            A tuple containing the column name and the value to filter on.
        operator : Literal["equals", "not equals", "greater", "smaller"]
            The operator to apply for filtering.
        factor : float, optional
            The disruption factor to assign to the filtered companies, by default 0.2.
            A factor of 0 means complete disruption (feature set to 0),
            while a factor of 1 means no disruption (feature unchanged).
        Returns
        -------
        dict[str, float]
            A dictionary mapping company tickers to disruption factors.
        """
        column, value = condition
        if column not in self.nodes_df.columns:
            raise ValueError(f"Column {column} not found in nodes DataFrame.")
        if operator == "equals":
            disrupted_companies = self.nodes_df[self.nodes_df[column] == value][
                "Ticker"
            ].tolist()
        elif operator == "not equals":
            disrupted_companies = self.nodes_df[self.nodes_df[column] != value][
                "Ticker"
            ].tolist()
        elif operator == "greater":
            disrupted_companies = self.nodes_df[
                self.nodes_df[column].astype(float) > value
            ]["Ticker"].tolist()
        elif operator == "smaller":
            disrupted_companies = self.nodes_df[
                self.nodes_df[column].astype(float) < value
            ]["Ticker"].tolist()
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        return {company: factor for company in disrupted_companies}

    def apply_disruptions(
        self, disruption_dict: dict[str, float], disruption_features: list[str]
    ) -> None:
        """Apply disruptions to node features based on the disruption dictionary.

        Parameters
        ----------
        disruption_dict : dict[str, float]
            Dictionary mapping company tickers to disruption factors (0 to 1).
            A factor of 0 means complete disruption (feature set to 0),
            while a factor of 1 means no disruption (feature unchanged).
        """
        if disruption_features is None or len(disruption_features) == 0:
            disruption_features = self.node_feature_names
        for ticker, factor in disruption_dict.items():
            if ticker in self.nodes_df[self.node_identifier].values:
                idx = self.nodes_df[
                    self.nodes_df[self.node_identifier] == ticker
                ].index.item()
                for disruption_feature in disruption_features:
                    if disruption_feature not in self.node_feature_names:
                        raise ValueError(
                            f"Disruption feature {disruption_feature} not in node features."
                        )
                    feature_idx = self.node_feature_names.index(disruption_feature)
                    if 0 <= factor <= 1:
                        self.pyg_nodes[idx, feature_idx] *= factor
                        print(idx, feature_idx)
                    else:
                        raise ValueError(
                            f"Disruption factor for {ticker} must be between 0 and 1."
                        )
            else:
                print(f"Ticker {ticker} not found in nodes. Skipping disruption.")

        self.graph_data.x = self.pyg_nodes

    # def drop_missing_nodes(self) -> None:
    #     """Drop nodes with missing features."""
    #     initial_node_count = self.num_nodes
    #     self.nodes_df.dropna(subset=self.node_feature_names, inplace=True)
    #     self.nodes_df.reset_index(drop=True, inplace=True)
    #     self.num_nodes = len(self.nodes_df)
    #     print(
    #         f"Dropped {initial_node_count - self.num_nodes} nodes with missing features."
    #     )
    #     self.pyg_nodes = self.create_pyg_nodes()
    #     self.pyg_edge_attr, self.pyg_edge_index = self.create_pyg_edges()
    #     self.graph_data = PygData(
    #         x=self.pyg_nodes,
    #         edge_index=self.pyg_edge_index,
    #         edge_attr=self.pyg_edge_attr,
    #     )

    # def drop_missing_edges(self) -> None:
    #     """Drop edges with missing features."""
    #     initial_edge_count = self.num_edges
    #     self.edges_df.dropna(subset=self.edge_feature_names, inplace=True)
    #     self.edges_df.reset_index(drop=True, inplace=True)
    #     self.num_edges = len(self.edges_df)
    #     print(
    #         f"Dropped {initial_edge_count - self.num_edges} edges with missing features."
    #     )
    #     self.pyg_nodes = self.create_pyg_nodes()
    #     self.pyg_edge_attr, self.pyg_edge_index = self.create_pyg_edges()
    #     self.graph_data = PygData(
    #         x=self.pyg_nodes,
    #         edge_index=self.pyg_edge_index,
    #         edge_attr=self.pyg_edge_attr,
    #     )


class BBGSupplyChainNetwork(SupplyChainNetwork):
    """Class to handle Bloomberg supply chain network graph data."""

    def __init__(
        self,
        material: Literal["cocoa", "coffee", "palm_oil"] = "cocoa",
        node_identifier: str = "Ticker",
        device: str | None = None,
    ) -> None:
        node_feature_names = [
            "operating_margin",
            "profit_margin",
            "current_ratio",
            "debt_to_equity_ratio",
            "altman_zscore",
            # "at_risk_commodity_revenue",
            # "supplier_count_forest_risk",
            # "cocoa_revenue_percentage",
            # "cocoa_income_supplier_count",
        ]
        # sc_customer_feature_names = ["revenue_percentage", "relation_size"]
        # sc_supplier_feature_names = ["cost_percentage", "relation_size"]
        # edge_feature_names = sc_customer_feature_names + sc_supplier_feature_names
        edge_feature_names = ["relation_size"]
        nodes_df = read_company_data(material=material)
        nodes_df.dropna(subset=node_feature_names, inplace=True)
        nodes_df.reset_index(drop=True, inplace=True)

        sc_customers_df, sc_suppliers_df = read_supply_chain_data(
            normalize=True, alpha=0.5, material=material
        )
        sc_customers_df.dropna(subset=edge_feature_names, inplace=True)
        sc_customers_df.reset_index(drop=True, inplace=True)
        sc_suppliers_df.dropna(subset=edge_feature_names, inplace=True)
        sc_suppliers_df.reset_index(drop=True, inplace=True)

        sc_customers_df = select_companies(
            sc_customers_df, nodes_df["Ticker"].tolist(), "target_company"
        )
        sc_customers_df = select_companies(
            sc_customers_df, nodes_df["Ticker"].tolist(), "source_company"
        )
        sc_suppliers_df = select_companies(
            sc_suppliers_df, nodes_df["Ticker"].tolist(), "target_company"
        )
        sc_suppliers_df = select_companies(
            sc_suppliers_df, nodes_df["Ticker"].tolist(), "source_company"
        )
        print(sc_customers_df.describe())

        nodes_df = select_companies(
            nodes_df,
            list(
                set(
                    sc_suppliers_df["source_company"].tolist()
                    + sc_suppliers_df["target_company"].tolist()
                    + sc_customers_df["source_company"].tolist()
                    + sc_customers_df["target_company"].tolist()
                )
            ),
            "Ticker",
        )

        super().__init__(
            nodes_df,
            sc_customers_df,
            sc_suppliers_df,
            node_feature_names,
            edge_feature_names,
            node_identifier,
            device,
        )


if __name__ == "__main__":
    network = BBGSupplyChainNetwork(material="cocoa")
    original_graph = network.get_pytorch_graph()  # .clone()
    print(original_graph.x[88])
    network.apply_disruptions(
        {"BARN SW Equity": 0.5, "NESN SW Equity": 0.2, "WMT US Equity": 0.0},
        ["operating_margin", "profit_margin"],
    )
    print(original_graph.x[88])
