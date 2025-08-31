"""Module for other utilities."""

from __future__ import annotations

import os
import pandas as pd
import networkx as nx
import torch
from torch import Tensor

FILE_PATH = os.path.dirname(__file__)


class EarlyStopping:
    """Early stopping utility class."""

    def __init__(self, patience: int | None):
        self.patience = patience
        self.trigger = 0.0
        self.last_loss = 100.0

    def step(self, current_loss: float) -> None:
        """Inspect the current loss comparing to previous one."""
        if current_loss > self.last_loss:
            self.trigger += 1
        else:
            self.trigger = 0
        self.last_loss = current_loss

    def check_patience(self) -> bool:
        """Determine whether the training should be stopped."""
        if self.patience is None:
            return False
        return self.trigger >= self.patience


def smoothness_loss(pred, edge_index, edge_weight=None):
    src, dst = edge_index
    diff = pred[src] - pred[dst]
    if edge_weight is not None:
        return (edge_weight * diff.pow(2).sum(dim=1)).mean()
    return diff.pow(2).sum(dim=1).mean()


def supply_risk(model, graph_data) -> Tensor:
    x, edge_index, edge_attr = (
        graph_data.x,
        graph_data.edge_index,
        graph_data.edge_attr,
    )

    # 1. Assuming `model` predicts ESG scores (higher = better)
    supplier_esg = model(x, edge_index, edge_attr)  # Shape: [num_nodes, 1]
    supplier_risk = 1 - (supplier_esg / supplier_esg.max())  # Normalized risk

    # Normalize edge weights (USD/revenue%)
    edge_weight_normalized = edge_attr / edge_attr.max()  # Shape: [num_edges, 1]

    # 2. Edge risk = Supplier risk (src) * Normalized edge weight
    src_nodes = edge_index[0]  # Suppliers
    edge_risk = supplier_risk[src_nodes] * edge_weight_normalized

    # 3. Assuming your GNN outputs node embeddings (`h`) before regression
    # Shape: [num_nodes, embedding_dim]
    h = model.get_embeddings(x, edge_index, edge_attr)
    multi_tier_risk = 1 - (h.mean(dim=1) / h.mean(dim=1).max())  # Convert to risk

    # Weighted combination
    alpha, beta, gamma = 0.6, 0.3, 0.1
    edge_risk_prob = torch.sigmoid(
        alpha * supplier_risk[src_nodes]
        + beta * multi_tier_risk[src_nodes]
        + gamma * edge_weight_normalized
    )
    return edge_risk_prob


def depivot_table(
    df: pd.DataFrame,
    source_column: str = "companies_tier0",
    n_pivot_cols: int = 20,
    pivot_name: str = "customer",
) -> pd.DataFrame:
    """Flatten pivoted 20 customers/suppliers of each company and their sizes to represent each link in one row."""
    source_companies = df[source_column]
    pivot_columns = [pivot_name + str(number) for number in range(1, n_pivot_cols + 1)]
    pivot_relsize = [
        f"relation_size_{pivot_name}" + str(number)
        for number in range(1, n_pivot_cols + 1)
    ]
    # pivot_rev = [
    #     f"revenue_percentage_{pivot_name}" + str(number)
    #     for number in range(1, n_pivot_cols + 1)
    # ]
    pivot_rev = [
        f"cost_percentage_{pivot_name}" + str(number)
        for number in range(1, n_pivot_cols + 1)
    ]
    pivot_reldate = [
        f"relation_date_{pivot_name}" + str(number)
        for number in range(1, n_pivot_cols + 1)
    ]
    pivot_relyear = [
        f"relation_year_{pivot_name}" + str(number)
        for number in range(1, n_pivot_cols + 1)
    ]
    pivot_relperiod = [
        f"relation_period_{pivot_name}" + str(number)
        for number in range(1, n_pivot_cols + 1)
    ]

    target_companies = []
    target_rel_sizes = []
    target_revenues = []
    target_year = []
    target_date = []
    target_period = []
    source_companies = []
    for row, company_row in df.iterrows():
        src_company = company_row[source_column]
        customer_count = 0
        for tgt_company, tgt_rel_size, tgt_rev, tgt_year, tgt_date, tgt_period in zip(
            pivot_columns,
            pivot_relsize,
            pivot_rev,
            pivot_reldate,
            pivot_relyear,
            pivot_relperiod,
        ):
            if company_row[tgt_company].item() == "#N/A N/A":
                break
            if pd.isna(company_row[tgt_company].item()):
                break
            print(company_row[tgt_company])
            target_companies.append(company_row[tgt_company])
            target_rel_sizes.append(company_row[tgt_rel_size])
            target_revenues.append(company_row[tgt_rev])
            target_year.append(company_row[tgt_year])
            target_date.append(company_row[tgt_date])
            target_period.append(company_row[tgt_period])
            customer_count += 1
        source_companies.extend([src_company] * customer_count)

    return pd.DataFrame(
        {
            "source_company": source_companies,
            "target_company": target_companies,
            "relation_size": target_rel_sizes,
            # "revenue_percentage": target_revenues,
            "cost_percentage": target_revenues,
            "relation_date": target_date,
            "relation_year": target_year,
            "relation_period": target_period,
        }
    )


def filter_supply_chain(
    df: pd.DataFrame, included_companies: list[str]
) -> pd.DataFrame:
    """Filter supply chain data to include only the given list of companies.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    include : list[str]
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    df = df[df["source_company"].isin(included_companies)]
    df = df[df["target_company"].isin(included_companies)]
    df = df.drop_duplicates(subset=["source_company", "target_company"], keep="first")
    return df


def inspect_missing_fields(df: pd.DataFrame, features: list[str]) -> None:
    """Create and print a report of missing fields in the provided dataframe and feature list.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    features : list[str]
        _description_
    """
    # Explain empty fields:
    print("-----------------------------------")
    print(f"Data size: {len(df)}")
    for feature in features:
        not_applicable = df[df[feature] == "#N/A Field Not Applicable"]
        nan_df = df[df[feature].isna()]
        print(f"{feature} - Not applicable: {len(not_applicable)}, NaN: {len(nan_df)}")


def read_company_data(root_path: str | None = None) -> pd.DataFrame:
    """Read and merge all company-level data into one dataframe."""
    if root_path is None:
        root_path = os.path.join(FILE_PATH, "..", "datasets", "bloomberg_cocoa")

    esg_general_path = os.path.join(root_path, "esg", "cocoa_sc_esg_general.csv")
    esg_env_path = os.path.join(root_path, "esg", "cocoa_sc_esg_env_v2.csv")
    esg_soc_path = os.path.join(root_path, "esg", "cocoa_sc_esg_soc_v2.csv")
    esg_gov_path = os.path.join(root_path, "esg", "cocoa_sc_esg_gov_v2.csv")
    esg_sdg_path = os.path.join(root_path, "esg", "cocoa_sc_esg_sdg_v2.csv")

    financial_path = os.path.join(
        root_path, "company_list", "companies_financial_v2_2.csv"
    )
    companies_path = os.path.join(root_path, "company_list", "companies_codes.csv")

    esg_general_df = pd.read_csv(esg_general_path)
    esg_env_df = pd.read_csv(esg_env_path)
    esg_soc_df = pd.read_csv(esg_soc_path)
    esg_gov_df = pd.read_csv(esg_gov_path)
    esg_sdg_df = pd.read_csv(esg_sdg_path)

    esg_general_df = esg_general_df.merge(
        esg_env_df,
        left_on="Ticker",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )
    esg_general_df = esg_general_df.merge(
        esg_soc_df,
        left_on="Ticker",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )
    esg_general_df = esg_general_df.merge(
        esg_gov_df,
        left_on="Ticker",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )
    esg_general_df = esg_general_df.merge(
        esg_sdg_df,
        left_on="Ticker",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )

    financial_df = pd.read_csv(financial_path)
    companies_df = pd.read_csv(companies_path)

    companies_df = companies_df.merge(
        esg_general_df,
        left_on="id",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )
    companies_df = companies_df.merge(
        financial_df,
        left_on="id",
        right_on="Query",
        suffixes=(None, None),
        how="left",
    )
    companies_df = companies_df.replace("#N/A Field Not Applicable", pd.NA)
    companies_df = companies_df.replace("#N/A N/A", pd.NA)
    companies_df = companies_df.replace("N.S.", pd.NA)
    companies_df = companies_df.convert_dtypes()
    print(companies_df.dtypes)
    companies_df["registered_in_country"] = companies_df[
        "registered_in_country"
    ].fillna("Unknown")

    return companies_df


def read_supply_chain_data(
    root_path: str | None = None,
    include_companies: list[str] | None = None,
    normalize: bool = False,
    alpha: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read suppliers and customers data and optionally filter the companies."""
    if root_path is None:
        root_path = os.path.join(FILE_PATH, "..", "datasets", "bloomberg_cocoa")

    sc_customers_path = os.path.join(
        root_path,
        "cocoa_supply_chain",
        "cocoa_supply_chain_customers_v2_depivot.csv",
    )
    sc_suppliers_path = os.path.join(
        root_path,
        "cocoa_supply_chain",
        "cocoa_supply_chain_suppliers_v2_depivot.csv",
    )

    sc_customers_df = pd.read_csv(sc_customers_path)
    sc_suppliers_df = pd.read_csv(sc_suppliers_path)
    if include_companies is not None:
        sc_customers_df = filter_supply_chain(sc_customers_df, include_companies)
        sc_suppliers_df = filter_supply_chain(sc_suppliers_df, include_companies)

    # for group_name, group_df in sc_customers_df.groupby("source_company"):
    #     print(group_name)
    #     print(group_df)

    if normalize:
        # Normalize relation sizes
        sc_suppliers_df = sc_suppliers_df[sc_suppliers_df["relation_size"] > 0]
        sc_customers_df = sc_customers_df[sc_customers_df["relation_size"] > 0]
        sc_suppliers_df["relation_size"] = (
            sc_suppliers_df["relation_size"] / sc_suppliers_df["relation_size"].max()
        )
        sc_customers_df["relation_size"] = (
            sc_customers_df["relation_size"] / sc_customers_df["relation_size"].max()
        )

    # Apply power-law transformation to relation sizes to reduce skewness
    sc_suppliers_df["relation_size_normed"] = (
        1 / (sc_suppliers_df["relation_size"]) ** alpha
    )
    sc_customers_df["relation_size_normed"] = (
        1 / (sc_customers_df["relation_size"]) ** alpha
    )

    return sc_customers_df, sc_suppliers_df


def select_companies(
    df: pd.DataFrame, company_names: list[str], column: str
) -> pd.DataFrame:
    """Select the companies from the given dataframe and the company name list.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    company_names : list[str]
        _description_
    column : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    def filter_list(row: pd.Series):
        return row in company_names

    return df[df[column].apply(filter_list)]


def company_name_lookup(df: pd.DataFrame, ticker: str) -> str | None:
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
    if ticker in df["id"].to_numpy():
        return df[df["id"] == ticker]["name"].item()
    return None


def build_graph(
    nodes_df: pd.DataFrame,
    sc_customers_df: pd.DataFrame,
    sc_suppliers_df: pd.DataFrame,
    node_feature_names: list[str] | None = None,
    sc_customer_feature_names: list[str] | None = None,
    sc_supplier_feature_names: list[str] | None = None,
) -> nx.Graph:
    """"""
    if node_feature_names is None:
        node_feature_names = ["id", "name", "industry_group", "industry_sector"]
    if sc_customer_feature_names is None:
        sc_customer_feature_names = ["relation_size"]
    if sc_supplier_feature_names is None:
        sc_supplier_feature_names = ["relation_size"]

    graph = nx.MultiDiGraph()

    for _, row in nodes_df.iterrows():
        graph.add_node(row["id"], **row[node_feature_names].to_dict())

    for _, row in sc_customers_df.iterrows():
        graph.add_edge(
            u_for_edge=row["source_company"],  # supplier
            v_for_edge=row["target_company"],  # customer
            # key="weight",
            **row[sc_customer_feature_names].to_dict(),
        )

    for _, row in sc_suppliers_df.iterrows():
        graph.add_edge(
            u_for_edge=row["target_company"],  # supplier
            v_for_edge=row["source_company"],  # customer
            # key="weight",
            **row[sc_supplier_feature_names].to_dict(),
        )
    return graph


def create_percolation_states(
    graph: nx.Graph, initial_failures: list[str]
) -> dict[str, float]:
    """Create a dictionary to track the failure state of each node in the graph.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    initial_failures : list[str]
        List of node IDs that are initially failed.

    Returns
    -------
    dict[str, float]
        A dictionary mapping node IDs to their risk state (0.0 for non risk-exposure,
        1.0 for full risk-exposure).
    """
    percolation_states = {node: 0.0 for node in graph.nodes()}
    for node in initial_failures:
        if node in percolation_states:
            percolation_states[node] = 1.0
    return percolation_states


if __name__ == "__main__":

    # DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "bloomberg_cocoa")
    # SC_CUSTOMERS_PATH = os.path.join(DATA_PATH, "cocoa_supply_chain_customers")
    # EDGES_T1_PATH = os.path.join(
    #     SC_CUSTOMERS_PATH, "cocoa_supply_chain_template_v3_tier1.csv"
    # )
    # EDGES_T2_PATH = os.path.join(
    #     SC_CUSTOMERS_PATH, "cocoa_supply_chain_template_v3_tier2.csv"
    # )

    # edges_t1_df = pd.read_csv(EDGES_T1_PATH)
    # edges_t2_df = pd.read_csv(EDGES_T2_PATH)

    # t1_depivot = depivot_table(edges_t1_df)
    # t1_depivot["tier"] = "Tier1"
    # t2_depivot = depivot_table(edges_t2_df, "companies_tier1")
    # t2_depivot["tier"] = "Tier2"

    # pd.concat([t1_depivot, t2_depivot]).to_csv(
    #     "cocoa_supply_chain_customers.csv", index=False
    # )

    companies_df = read_company_data()

    sc_customers_df, sc_suppliers_df = read_supply_chain_data()
    sc_customers_df = select_companies(
        sc_customers_df, companies_df["id"].tolist(), "target_company"
    )
    sc_customers_df = select_companies(
        sc_customers_df, companies_df["id"].tolist(), "source_company"
    )
    sc_suppliers_df = select_companies(
        sc_suppliers_df, companies_df["id"].tolist(), "target_company"
    )
    sc_suppliers_df = select_companies(
        sc_suppliers_df, companies_df["id"].tolist(), "source_company"
    )
    inspect_missing_fields(sc_customers_df, ["relation_size"])
    inspect_missing_fields(sc_customers_df, ["revenue_percentage"])
