import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from modelname.dataset import SLPCNetworkGraph

sc_network = SLPCNetworkGraph(device="cpu")
pyg_graph = sc_network.get_pytorch_graph()

pyg_graph.transductive_split()

y_tr = pyg_graph.y[pyg_graph.train_mask].cpu().numpy()
y_tst = pyg_graph.y[pyg_graph.val_mask].cpu().numpy()
X_tr_1 = pyg_graph.x[pyg_graph.train_mask].cpu().numpy()
X_tr_2 = (
    sc_network.edges_df.groupby("source_company")["relation_size"].mean().to_numpy()
)
X_tst = pyg_graph.x[pyg_graph.val_mask].cpu().numpy()

X_tr = np.vstack((X_tr_1, X_tr_2))

# regressor = LinearRegression().fit(X_tr, y_tr)
# y_pred = regressor.predict(X_tst)
# r2 = r2_score(y_tst, y_pred)

# print(r2)
