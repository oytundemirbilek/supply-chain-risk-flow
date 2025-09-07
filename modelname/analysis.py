"""_summary_"""

import pandas as pd
from networkx.algorithms.centrality import (
    betweenness_centrality,
    percolation_centrality,
)

from modelname.dataset import BBGSupplyChainNetwork
from modelname.utils import create_percolation_states
from modelname.plotting import Graphplot, Boxplot

network = BBGSupplyChainNetwork(material="cocoa")
graph = network.get_networkx_graph()
nodes_df = network.nodes_df.copy()

graph_plotter = Graphplot(graph)
graph_plotter.plot()

nodes_df = nodes_df.set_index("Ticker", drop=False)

centralities = betweenness_centrality(graph, weight="relation_size_normed")
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_betweenness_df = nodes_df.sort_values(by=["centrality"], ascending=False)

# ESG failure based percolation centrality
initial_failures = nodes_df[nodes_df["ESG Scr"] < 3.0]["Ticker"].tolist()
print(f"Initial failures (ESG): {initial_failures}")
percolation_states = create_percolation_states(graph, initial_failures=initial_failures)
centralities = percolation_centrality(
    graph, states=percolation_states, weight="relation_size_normed"
)
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_percolation_esg_df = nodes_df.sort_values(by=["centrality"], ascending=False)

# Financial failure based percolation centrality
initial_failures = nodes_df[nodes_df["altman_zscore"] < 1.81]["Ticker"].tolist()
print(f"Initial failures (financial): {initial_failures}")
percolation_states = create_percolation_states(graph, initial_failures=initial_failures)
centralities = percolation_centrality(
    graph, states=percolation_states, weight="relation_size_normed"
)
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_percolation_financial_df = nodes_df.sort_values(by=["centrality"], ascending=False)

# Geological failure based percolation centrality
initial_failures = nodes_df[nodes_df["registered_in_country"] == "ID"][
    "Ticker"
].tolist()
print(f"Initial failures (geological): {initial_failures}")
percolation_states = create_percolation_states(graph, initial_failures=initial_failures)
centralities = percolation_centrality(
    graph, states=percolation_states, weight="relation_size_normed"
)
nodes_df["centrality"] = pd.Series(centralities, index=list(centralities.keys()))
top_percolation_geo_df = nodes_df.sort_values(by=["centrality"], ascending=False)

top_betweenness_df["analysis_type"] = "betweenness_centrality"
top_percolation_esg_df["analysis_type"] = "percolation_centrality (ESG)"
top_percolation_financial_df["analysis_type"] = "percolation_centrality (financial)"
top_percolation_geo_df["analysis_type"] = "percolation_centrality (geological)"
combined_df = pd.concat(
    [
        top_betweenness_df,
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
    reference_by="Ticker",
)
