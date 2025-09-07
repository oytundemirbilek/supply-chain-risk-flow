"""_summary_"""

import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from modelname.dataset import SLPCNetworkGraph
from modelname.utils import filter_supply_chain

# sc_network = SLPCNetworkGraph(device="cpu")
# pyg_graph = sc_network.get_pytorch_graph()

# pyg_graph.transductive_split()


FILE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_PATH, "..", "datasets", "bloomberg_cocoa")
SC_CUSTOMERS_PATH = os.path.join(
    DATA_PATH,
    "cocoa_supply_chain",
    "cocoa_supply_chain_customers_v2_depivot.csv",
)
SC_SUPPLIERS_PATH = os.path.join(
    DATA_PATH,
    "cocoa_supply_chain",
    "cocoa_supply_chain_suppliers_v2_depivot.csv",
)
COMPANIES_PATH = os.path.join(DATA_PATH, "company_list", "companies_tiered.csv")
FINANCIAL_PATH = os.path.join(DATA_PATH, "company_list", "companies_financial_v2_2.csv")
ESG_PATH = os.path.join(DATA_PATH, "esg", "cocoa_sc_esg_general.csv")
SDG_PATH = os.path.join(DATA_PATH, "esg", "cocoa_sc_esg_sdg_v2.csv")

esg_df = pd.read_csv(ESG_PATH)
sdg_df = pd.read_csv(SDG_PATH)
nodes_df = pd.read_csv(COMPANIES_PATH)
financial_df = pd.read_csv(FINANCIAL_PATH)
sc_customers_df = pd.read_csv(SC_CUSTOMERS_PATH)
sc_suppliers_df = pd.read_csv(SC_SUPPLIERS_PATH)

nodes_df = nodes_df.merge(
    esg_df,
    left_on="id",
    right_on="Ticker",
    suffixes=(None, None),
    how="left",
)
nodes_df = nodes_df.merge(
    sdg_df,
    left_on="id",
    right_on="Ticker",
    suffixes=(None, "sdg"),
    how="left",
)
nodes_df = nodes_df.merge(
    financial_df,
    left_on="id",
    right_on="Query",
    suffixes=(None, None),
    how="left",
)
sc_customers_df = filter_supply_chain(sc_customers_df, nodes_df["id"].tolist())
sc_suppliers_df = filter_supply_chain(sc_suppliers_df, nodes_df["id"].tolist())
sc_customer_feature_names = ["revenue_percentage", "relation_size"]
sc_supplier_feature_names = ["cost_percentage", "relation_size"]
node_feature_names = [
    "ROA",
    "operating_margin",
    "profit_margin",
    "current_ratio",
    "debt_to_equity_ratio",
    "altman_zscore",
    "at_risk_commodity_revenue",
    "supplier_count_forest_risk",
    "cocoa_revenue_percentage",
    "cocoa_income_supplier_count",
    # "Net SDG Impact",
]
label_name = "ESG Scr"

model_df = nodes_df[node_feature_names + [label_name]]

model_df[label_name] = model_df[model_df[label_name] != "#N/A Field Not Applicable"][
    label_name
]
model_df[node_feature_names] = model_df[
    model_df[node_feature_names] != "#N/A Field Not Applicable"
][node_feature_names]
model_df = model_df.dropna(subset=node_feature_names)
model_df = model_df.dropna(subset=label_name)
print(model_df)

# normalize ESG range as it was between 0-10
y = model_df[label_name].to_numpy(dtype=float)
X = model_df[node_feature_names].to_numpy(dtype=float)

scaler = StandardScaler()
scaler.fit(X)
X_norm = scaler.transform(X)

X_tr, X_tst, y_tr, y_tst = train_test_split(X_norm, y, train_size=0.7, random_state=0)

regressor = LinearRegression().fit(X_tr, y_tr)
y_pred = regressor.predict(X_tst)
r2 = r2_score(y_tst, y_pred)

y_pred = regressor.predict(X_tr)
r2 = r2_score(y_tr, y_pred)

print(r2)
