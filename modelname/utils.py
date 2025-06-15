"""Module for other utilities."""

from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor


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
    df: pd.DataFrame, source_column: str = "companies_tier0", n_pivot_cols: int = 20
) -> pd.DataFrame:
    """Flatten pivoted 20 customers/suppliers of each company and their sizes to represent each link in one row."""
    source_companies = df[source_column]
    pivot_columns = ["customer" + str(number) for number in range(1, n_pivot_cols + 1)]
    pivot_feature1 = [
        "relation_size_customer" + str(number) for number in range(1, n_pivot_cols + 1)
    ]
    pivot_feature2 = [
        "revenue_percentage_customer" + str(number)
        for number in range(1, n_pivot_cols + 1)
    ]

    target_companies = []
    target_rel_sizes = []
    target_revenues = []
    source_companies = []
    for row, company_row in df.iterrows():
        src_company = company_row[source_column]
        customer_count = 0
        for tgt_company, tgt_rel_size, tgt_rev in zip(
            pivot_columns, pivot_feature1, pivot_feature2
        ):
            if company_row[tgt_company] == "#N/A N/A":
                break
            if pd.isna(company_row[tgt_company]):
                break
            print(company_row[tgt_company])
            target_companies.append(company_row[tgt_company])
            target_rel_sizes.append(company_row[tgt_rel_size])
            target_revenues.append(company_row[tgt_rev])
            customer_count += 1
        source_companies.extend([src_company] * customer_count)

    return pd.DataFrame(
        {
            "source_company": source_companies,
            "target_company": target_companies,
            "relation_size": target_rel_sizes,
            "revenue_percentage": target_revenues,
        }
    )


if __name__ == "__main__":

    import os

    FILE_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "bloomberg_cocoa")
    SC_CUSTOMERS_PATH = os.path.join(DATA_PATH, "cocoa_supply_chain_customers")
    EDGES_T1_PATH = os.path.join(
        SC_CUSTOMERS_PATH, "cocoa_supply_chain_template_v3_tier1.csv"
    )
    EDGES_T2_PATH = os.path.join(
        SC_CUSTOMERS_PATH, "cocoa_supply_chain_template_v3_tier2.csv"
    )

    edges_t1_df = pd.read_csv(EDGES_T1_PATH)
    edges_t2_df = pd.read_csv(EDGES_T2_PATH)

    t1_depivot = depivot_table(edges_t1_df)
    t1_depivot["tier"] = "Tier1"
    t2_depivot = depivot_table(edges_t2_df, "companies_tier1")
    t2_depivot["tier"] = "Tier2"

    pd.concat([t1_depivot, t2_depivot]).to_csv(
        "cocoa_supply_chain_customers.csv", index=False
    )
