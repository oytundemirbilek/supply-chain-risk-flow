"""_summary_"""

import pandas as pd
from networkx.algorithms.centrality import (
    closeness_centrality,
    betweenness_centrality,
    percolation_centrality,
)

from modelname.utils import (
    inspect_missing_fields,
    read_company_data,
    read_supply_chain_data,
    build_graph,
    select_companies,
    create_percolation_states,
)
from modelname.plotting import Graphplot, Boxplot

nodes_df = read_company_data()
sc_customers_df, sc_suppliers_df = read_supply_chain_data(normalize=True, alpha=0.5)
sc_customers_df = select_companies(
    sc_customers_df, nodes_df["id"].tolist(), "target_company"
)
sc_customers_df = select_companies(
    sc_customers_df, nodes_df["id"].tolist(), "source_company"
)
sc_suppliers_df = select_companies(
    sc_suppliers_df, nodes_df["id"].tolist(), "target_company"
)
sc_suppliers_df = select_companies(
    sc_suppliers_df, nodes_df["id"].tolist(), "source_company"
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
    "id",
)

sc_customer_feature_names = ["revenue_percentage", "relation_size"]
sc_supplier_feature_names = ["cost_percentage", "relation_size"]
node_feature_names = [
    "ESG Scr",
    # "operating_margin",
    # "profit_margin",
    # "current_ratio",
    # "debt_to_equity_ratio",
    # "altman_zscore",
    # "at_risk_commodity_revenue",
    # "supplier_count_forest_risk",
    # "cocoa_revenue_percentage",
    # "cocoa_income_supplier_count",
]
node_feature_names = nodes_df.columns.to_list()

inspect_missing_fields(nodes_df, node_feature_names)
inspect_missing_fields(sc_customers_df, sc_customer_feature_names)
inspect_missing_fields(sc_suppliers_df, sc_supplier_feature_names)

graph = build_graph(
    nodes_df,
    sc_customers_df,
    sc_suppliers_df,
    node_feature_names,
    sc_customer_feature_names,
    sc_supplier_feature_names,
)
nodes_df = nodes_df.set_index("id", drop=False)

TOP_CENTRALITY_COUNT = 1000

centralities = betweenness_centrality(graph, weight="relation_size_normed")
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_betweenness_df = nodes_df.sort_values(by=["centrality"], ascending=False).head(
    TOP_CENTRALITY_COUNT
)

centralities = closeness_centrality(graph, distance="relation_size_normed")
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_closeness_df = nodes_df.sort_values(by=["centrality"], ascending=False).head(
    TOP_CENTRALITY_COUNT
)

# the weight is interpreted as the connection strength.
# centralities = katz_centrality(graph, weight="relation_size")
# nodes_df["katz"] = pd.Series(centralities, index=list(centralities.keys()))
# top_katz_df = nodes_df.sort_values(by=["katz"], ascending=False).head(100)

# initial_failures = ["NESN SW Equity"]
# initial_failures = ["WMT US Equity"]
# initial_failures = ["BARN SW Equity"]

# ESG failure based percolation centrality
initial_failures = nodes_df[nodes_df["ESG Scr"].astype(float) < 3.0]["id"].tolist()
print(f"Initial failures (ESG): {initial_failures}")
percolation_states = create_percolation_states(graph, initial_failures=initial_failures)
centralities = percolation_centrality(
    graph, states=percolation_states, weight="relation_size_normed"
)
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_percolation_esg_df = nodes_df.sort_values(by=["centrality"], ascending=False).head(
    TOP_CENTRALITY_COUNT
)

# Financial failure based percolation centrality
initial_failures = nodes_df[nodes_df["altman_zscore"].astype(float) < 1.81][
    "id"
].tolist()
print(f"Initial failures (financial): {initial_failures}")
percolation_states = create_percolation_states(graph, initial_failures=initial_failures)
centralities = percolation_centrality(
    graph, states=percolation_states, weight="relation_size_normed"
)
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_percolation_financial_df = nodes_df.sort_values(
    by=["centrality"], ascending=False
).head(TOP_CENTRALITY_COUNT)

# Geological failure based percolation centrality
initial_failures = nodes_df[nodes_df["registered_in_country"] == "ID"]["id"].tolist()
print(f"Initial failures (geological): {initial_failures}")
percolation_states = create_percolation_states(graph, initial_failures=initial_failures)
centralities = percolation_centrality(
    graph, states=percolation_states, weight="relation_size_normed"
)
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_percolation_geo_df = nodes_df.sort_values(by=["centrality"], ascending=False).head(
    TOP_CENTRALITY_COUNT
)

# print(
#     nodes_df[
#         [
#             "name",
#             "betweenness",
#             "closeness",
#             "percolation",
#         ]
#     ].head(20)
# )

# print(graph.nodes["BARN SW Equity"])

# for u, v, attrs in graph.edges(data=True):
#     print(f"Edge from {u} to {v} has attributes: {attrs}")

# -----------------------------------------------------------------------------------
# print(
#     select_companies(
#         nodes_df, ["BARN SW Equity", "NESN SW Equity", "WMT US Equity"], "id"
#     )
# )
# print(select_companies(sc_customers_df, ["NESN SW Equity"], "target_company"))
# print(select_companies(sc_suppliers_df, ["NESN SW Equity"], "source_company"))

# centr_ser = pd.Series(centralities, index=list(centralities.keys())).nlargest(20)
# centr_df = pd.DataFrame(centr_ser, columns=[centrality_type])
# centr_df = centr_df.reset_index(drop=False, names="id")
# centr_df["company_name"] = centr_df["id"].apply(company_name_lookup)

top_betweenness_df["analysis_type"] = "betweenness_centrality"
top_closeness_df["analysis_type"] = "closeness_centrality"
top_percolation_esg_df["analysis_type"] = "percolation_centrality (ESG)"
top_percolation_financial_df["analysis_type"] = "percolation_centrality (financial)"
top_percolation_geo_df["analysis_type"] = "percolation_centrality (geological)"
combined_df = pd.concat(
    [
        top_betweenness_df,
        # top_closeness_df,
        top_percolation_esg_df,
        top_percolation_financial_df,
        top_percolation_geo_df,
    ]
).reset_index(drop=True)

plotter = Boxplot(combined_df)
plotter.plot(
    # metric="betweenness",
    metric="centrality",
    group_by="analysis_type",
    # split_by="industry_group",
    split_by="registered_in_country",
    reference_by="id",
)

# plotter = Graphplot(graph)
# plotter.plot()
