"""Module for other utilities."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler

FILE_PATH = os.path.dirname(__file__)


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
    pivot_perc = [
        (
            f"cost_percentage_{pivot_name}" + str(number)
            if pivot_name == "supplier"
            else f"revenue_percentage_{pivot_name}" + str(number)
        )
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
    target_percs = []
    target_year = []
    target_date = []
    target_period = []
    source_companies = []
    for row, company_row in df.iterrows():
        src_company = company_row[source_column]
        customer_count = 0
        for tgt_company, tgt_rel_size, tgt_perc, tgt_year, tgt_date, tgt_period in zip(
            pivot_columns,
            pivot_relsize,
            pivot_perc,
            pivot_reldate,
            pivot_relyear,
            pivot_relperiod,
        ):
            if company_row[tgt_company] == "#N/A N/A":  # type: ignore
                break
            if pd.isna(company_row[tgt_company]):  # type: ignore
                break
            target_companies.append(company_row[tgt_company])
            target_rel_sizes.append(company_row[tgt_rel_size])
            target_percs.append(company_row[tgt_perc])
            target_year.append(company_row[tgt_year])
            target_date.append(company_row[tgt_date])
            target_period.append(company_row[tgt_period])
            customer_count += 1
        source_companies.extend([src_company] * customer_count)

    table_data = {
        "source_company": source_companies,
        "target_company": target_companies,
        "relation_size": target_rel_sizes,
        # "revenue_percentage": target_revenues,
        # "cost_percentage": target_costs,
        "relation_date": target_date,
        "relation_year": target_year,
        "relation_period": target_period,
    }
    table_data.update(
        {"cost_percentage": target_percs}
        if pivot_name == "supplier"
        else {"revenue_percentage": target_percs}
    )

    return pd.DataFrame(table_data)


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


def read_company_data(
    root_path: str | None = None, material: str = "cocoa"
) -> pd.DataFrame:
    """Read and merge all company-level data into one dataframe."""
    if root_path is None:
        root_path = os.path.join(FILE_PATH, "..", "datasets", f"bloomberg_{material}")

    esg_general_path = os.path.join(
        root_path, "esg", f"{material}_sc_esg_summary_v2.csv"
    )
    esg_env_path = os.path.join(root_path, "esg", f"{material}_sc_esg_env_v2.csv")
    esg_soc_path = os.path.join(root_path, "esg", f"{material}_sc_esg_soc_v2.csv")
    esg_gov_path = os.path.join(root_path, "esg", f"{material}_sc_esg_gov_v2.csv")
    esg_sdg_path = os.path.join(root_path, "esg", f"{material}_sc_esg_sdg_v2.csv")

    financial_path = os.path.join(
        root_path, "companies", f"{material}_companies_financial_v3.csv"
    )
    companies_path = os.path.join(
        root_path, "companies", f"{material}_companies_info.csv"
    )

    esg_general_df = pd.read_csv(
        esg_general_path, na_values=["#N/A Field Not Applicable", "N.S."]
    )
    esg_env_df = pd.read_csv(esg_env_path, na_values="#N/A Field Not Applicable")
    esg_soc_df = pd.read_csv(esg_soc_path, na_values="#N/A Field Not Applicable")
    esg_gov_df = pd.read_csv(esg_gov_path, na_values="#N/A Field Not Applicable")
    esg_sdg_df = pd.read_csv(esg_sdg_path, na_values="#N/A Field Not Applicable")

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

    financial_df = pd.read_csv(financial_path, na_values="#N/A Field Not Applicable")
    companies_df = pd.read_csv(companies_path, na_values="#N/A Field Not Applicable")

    companies_df = companies_df.merge(
        esg_general_df,
        left_on="Ticker",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )
    companies_df = companies_df.merge(
        financial_df,
        left_on="Ticker",
        right_on="Ticker",
        suffixes=(None, None),
        how="left",
    )
    companies_df["registered_in_country"] = companies_df[
        "registered_in_country"
    ].fillna("Unknown")

    companies_df["operating_state"] = 1.0

    return companies_df


def match_supply_chain_companies(
    sc_customers_df: pd.DataFrame,
    sc_suppliers_df: pd.DataFrame,
    companies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match the companies in the supply chain data to ensure consistency.

    Parameters
    ----------
    sc_customers_df : pd.DataFrame
        _description_
    sc_suppliers_df : pd.DataFrame
        _description_
    companies_df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """

    merged_df = sc_customers_df.merge(
        sc_suppliers_df,
        left_on=["source_company", "target_company"],
        right_on=["target_company", "source_company"],
        how="outer",
        suffixes=("_customer", "_supplier"),
    )
    for col in [
        "relation_size",
        "relation_date",
        "relation_year",
        "relation_period",
    ]:
        combined = merged_df[col + "_customer"].combine_first(
            merged_df[col + "_supplier"]
        )
        merged_df[col + "_supplier"] = combined
        merged_df[col + "_customer"] = combined

    combine_suppliers = merged_df["source_company_supplier"].combine_first(
        merged_df["target_company_customer"]
    )
    merged_df["source_company_supplier"] = combine_suppliers
    merged_df["target_company_customer"] = combine_suppliers

    combine_customers = merged_df["source_company_customer"].combine_first(
        merged_df["target_company_supplier"]
    )
    merged_df["source_company_customer"] = combine_customers
    merged_df["target_company_supplier"] = combine_customers

    for group_name, group_df in merged_df.groupby("source_company_customer"):
        revenue = companies_df.loc[
            companies_df["Ticker"] == group_name, "revenue"
        ].values
        if len(revenue) > 0:
            revenue = revenue[0]
        else:
            revenue = np.nan
        # print(f"Company: {group_name}, Revenue: {revenue}")
        rev_perc = 100 * group_df["relation_size_customer"] / revenue
        # print(f"Revenue percentages: {rev_perc}")
        # Update merged_df directly for the current group
        idx = group_df.index
        merged_df.loc[idx, "revenue_percentage"] = merged_df.loc[
            idx, "revenue_percentage"
        ].fillna(rev_perc)

    for group_name, group_df in merged_df.groupby("source_company_supplier"):
        group_cost_info = group_df.dropna(subset=["cost_percentage"])
        if group_cost_info.empty:
            print(f"Could not found cost info for company: {group_name}")
            continue
        cost_perc_first = group_cost_info["cost_percentage"].to_numpy()[0]
        relation_size = group_cost_info["relation_size_supplier"].to_numpy()[0]
        total_cost = 100 * relation_size / cost_perc_first
        # print(f"Company: {group_name}, Total Cost: {total_cost}")

        cost_perc = 100 * group_df["relation_size_supplier"] / total_cost
        # print(f"Cost percentages: {cost_perc}")
        # Update merged_df directly for the current group
        idx = group_df.index
        merged_df.loc[idx, "cost_percentage"] = merged_df.loc[
            idx, "cost_percentage"
        ].fillna(cost_perc)

    for col in [
        "relation_size",
        "relation_date",
        "relation_year",
        "relation_period",
        "source_company",
        "target_company",
    ]:
        merged_df.rename(columns={f"{col}_customer": col}, inplace=True)

    return merged_df


def process_supply_chain_relations(
    df: pd.DataFrame,
    normalize: bool = True,
    log_transform: bool = True,
    reverse_transform: bool = False,
) -> pd.DataFrame:
    """"""
    # Apply log transformation to relation sizes to reduce skewness
    df["relation_size_normed"] = df["relation_size"]
    if log_transform:
        df["relation_size_normed"] = np.log(df["relation_size_normed"].to_numpy())
    if reverse_transform:
        df["relation_size_normed"] = 1 / (df["relation_size_normed"].to_numpy())
    if normalize:
        # Normalize relation sizes
        df["relation_size_normed"] = (
            df["relation_size_normed"] - df["relation_size_normed"].min()
        ) / (df["relation_size_normed"].max() - df["relation_size_normed"].min())
    df["revenue_percentage"] = df["revenue_percentage"] / 100
    df["cost_percentage"] = df["cost_percentage"] / 100
    return df


def process_company_data(df: pd.DataFrame) -> pd.DataFrame:
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
    negative_possibles = [
        "ROA",
        "operating_margin",
        "profit_margin",
        "current_ratio",
        "debt_to_equity_ratio",
        "altman_zscore",
        "inventory_turnover",
        "cash_conversion_cycle",
        "free_cash_flow",
    ]
    percentages = [
        "asset_damage_rate",
        "cocoa_revenue_percentage",
    ]

    heavy_tailed = [
        "at_risk_commodity_revenue",
        "supplier_count_forest_risk",
        "cocoa_income_supplier_count",
        "revenue",
        "Mkt Cap",
        "total_assets",
    ]

    z_scaler = StandardScaler()
    df[negative_possibles] = z_scaler.fit_transform(df[negative_possibles])

    df[percentages] = df[percentages] / 100

    df["at_risk_commodity_revenue"] = df["at_risk_commodity_revenue"] / 1_000_000

    mm_scaler = MinMaxScaler()
    df[heavy_tailed] = np.log1p(df[heavy_tailed])
    df[heavy_tailed] = mm_scaler.fit_transform(df[heavy_tailed])

    return df


def read_supply_chain_data(
    root_path: str | None = None,
    include_companies: list[str] | None = None,
    material: str = "cocoa",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read suppliers and customers data and optionally filter the companies."""
    if root_path is None:
        root_path = os.path.join(FILE_PATH, "..", "datasets", f"bloomberg_{material}")

    sc_customers_path = os.path.join(
        root_path,
        "supply_chain_network",
        f"{material}_supply_chain_customers_v2_depivot.csv",
    )
    sc_suppliers_path = os.path.join(
        root_path,
        "supply_chain_network",
        f"{material}_supply_chain_suppliers_v2_depivot.csv",
    )

    sc_customers_df = pd.read_csv(
        sc_customers_path, na_values="#N/A Field Not Applicable"
    )
    sc_suppliers_df = pd.read_csv(
        sc_suppliers_path, na_values="#N/A Field Not Applicable"
    )
    if include_companies is not None:
        sc_customers_df = filter_supply_chain(sc_customers_df, include_companies)
        sc_suppliers_df = filter_supply_chain(sc_suppliers_df, include_companies)

    # for idx, row in sc_suppliers_df.iterrows():
    #     try:
    #         row["relation_size"] = float(row["relation_size"])
    #     except ValueError:
    #         print(f"Invalid relation size: {row['relation_size']}")

    # for group_name, group_df in sc_customers_df.groupby("source_company"):
    #     print(group_name)
    #     print(group_df)

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

    companies_df = read_company_data()

    sc_customers_df, sc_suppliers_df = read_supply_chain_data()
    sc_customers_df = select_companies(
        sc_customers_df, companies_df["Ticker"].tolist(), "target_company"
    )
    sc_customers_df = select_companies(
        sc_customers_df, companies_df["Ticker"].tolist(), "source_company"
    )
    sc_suppliers_df = select_companies(
        sc_suppliers_df, companies_df["Ticker"].tolist(), "target_company"
    )
    sc_suppliers_df = select_companies(
        sc_suppliers_df, companies_df["Ticker"].tolist(), "source_company"
    )
    inspect_missing_fields(sc_customers_df, ["relation_size"])
    inspect_missing_fields(sc_customers_df, ["revenue_percentage"])
