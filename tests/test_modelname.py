"""Test graph dataset classes."""

from __future__ import annotations

import os
from logging import Logger

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data as PygData

from modelname.dataset import SLPCNetworkGraph
from modelname.model import SurfgateNet, SurfnnconvNet, SurfrelNet

DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets")
GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "expected")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
DEVICE = "cpu"


@pytest.fixture
def dummy_graph(
    num_nodes: int = 100,
    num_edges: int = 300,
    num_node_features: int = 8,
    num_edge_features: int = 2,
    num_disconnected: int = 5,
    seed: int = 42,
) -> PygData:
    """_summary_

    Returns
    -------
    PygData
        _description_
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Node features (e.g., financials, sfdr, industry encodings, etc.)
    x = torch.randn(num_nodes, num_node_features)

    # ESG labels (some missing = inference)
    y = torch.randn(num_nodes, 1)  # Regression target (ESG score)
    has_label = torch.rand(num_nodes) > 0.2  # 80% nodes have ESG scores
    y[~has_label] = float("nan")  # Simulate missing labels

    # Edge index (directed)
    src = torch.randint(0, num_nodes - num_disconnected, (num_edges,))
    dst = torch.randint(0, num_nodes - num_disconnected, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)

    # Edge features: relation_size (USD), revenue_share (%)
    relation_size = torch.rand(num_edges, 1) * 1e6  # USD
    revenue_share = torch.rand(num_edges, 1)  # percentage (0-1)
    edge_attr = torch.cat([relation_size, revenue_share], dim=1)
    edge_type = torch.randint(0, 1, (num_edges,))

    # Masks
    # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # infer_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # labeled_nodes = torch.where(has_label)[0]
    # num_train = int(0.6 * len(labeled_nodes))
    # num_val = int(0.2 * len(labeled_nodes))

    # train_mask[labeled_nodes[:num_train]] = True
    # val_mask[labeled_nodes[num_train : num_train + num_val]] = True
    # test_mask[labeled_nodes[num_train + num_val :]] = True
    # infer_mask[~has_label] = True

    # Create PyG Data object
    return PygData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        edge_type=edge_type,
        # train_mask=train_mask,
        # val_mask=val_mask,
        # test_mask=test_mask,
        # infer_mask=infer_mask,
    )


@pytest.fixture
def dummy_edges_df(
    num_nodes: int = 100,
    num_edges: int = 300,
    num_edge_features: int = 2,  # Reserved for future use
    num_disconnected: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """_summary_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    np.random.seed(seed)
    node_ids = np.arange(num_nodes)

    connected_nodes = node_ids[:-num_disconnected]
    source = np.random.choice(connected_nodes, size=num_edges)
    target = np.random.choice(connected_nodes, size=num_edges)

    relation_size = np.random.rand(num_edges) * 1e6  # USD
    revenue_share = np.random.rand(num_edges)  # Between 0 and 1

    return pd.DataFrame(
        {
            "source_company": source,
            "target_company": target,
            "relation_size": relation_size,
            "revenue_percentage": revenue_share,
        }
    )


@pytest.fixture
def dummy_nodes_df(
    num_nodes: int = 100, num_node_features: int = 8, seed: int = 42
) -> pd.DataFrame:
    """_summary_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    np.random.seed(seed)
    node_ids = np.arange(num_nodes)

    node_features = np.random.randn(num_nodes, num_node_features)
    esg_scores = np.random.randn(num_nodes)  # Continuous label
    label_mask = np.random.rand(num_nodes) > 0.2  # 80% have labels
    esg_scores[~label_mask] = np.nan  # Simulate missing ESG labels

    node_df = pd.DataFrame(
        node_features, columns=[f"feat_{i}" for i in range(num_node_features)]
    )
    node_df["id"] = node_ids
    node_df["pe_ratio_today"] = node_features[:, 0]
    node_df["esg_score"] = esg_scores
    node_df["has_label"] = label_mask
    return node_df


def test_gat_model_iteration(logger_test: Logger, dummy_graph: PygData) -> None:
    """Test if the model can be iterated - cpu based."""
    logger_test.info("Starting test_model_iteration")
    logger_test.info(dummy_graph)
    model = SurfgateNet(8, 2, 16, 3, 1)
    assert model is not None
    out = model.forward(dummy_graph)
    assert isinstance(out, Tensor)
    assert out.shape == (100, 1)


def test_nnconv_model_iteration(logger_test: Logger, dummy_graph: PygData) -> None:
    """Test if the model can be iterated - cpu based."""
    logger_test.info("Starting test_model_iteration")
    logger_test.info(dummy_graph)
    model = SurfnnconvNet(8, 2, 16, 3, 1)
    assert model is not None
    out = model.forward(dummy_graph)
    assert isinstance(out, Tensor)
    assert out.shape == (100, 1)


def test_rgcn_model_iteration(logger_test: Logger, dummy_graph: PygData) -> None:
    """Test if the model can be iterated - cpu based."""
    logger_test.info("Starting test_model_iteration")
    logger_test.info(dummy_graph)
    model = SurfrelNet(8, 2, 16, 3, 1)
    assert model is not None
    out = model.forward(dummy_graph)
    assert isinstance(out, Tensor)
    assert out.shape == (100, 1)


def test_sample_graph(
    logger_test: Logger, dummy_nodes_df: pd.DataFrame, dummy_edges_df: pd.DataFrame
) -> None:
    """Test if the graph can be built and accessed successfully."""
    logger_test.info("Starting test_sample_graph")
    logger_test.info(dummy_nodes_df)
    logger_test.info(dummy_edges_df)
    network_graph = SLPCNetworkGraph(dummy_edges_df, dummy_nodes_df)
    esgs = network_graph.get_regression_labels()
    assert len(esgs) == 100

    pyg_graph = network_graph.get_pytorch_graph()
    assert isinstance(pyg_graph.y, Tensor)
    assert isinstance(pyg_graph.x, Tensor)
    assert isinstance(pyg_graph.edge_attr, Tensor)
    assert isinstance(pyg_graph.edge_index, Tensor)
    assert pyg_graph.y.shape == (100,)
    assert pyg_graph.x.shape == (100, 1)
    assert pyg_graph.edge_attr.shape == (300, 2)
    assert pyg_graph.edge_index.shape == (2, 300)

    nx_graph = network_graph.get_networkx_graph()


# def test_dataset() -> None:
#     """Test if the model can be iterated - cpu based."""
#     dataset = MockDataset()
#     assert dataset is not None
#     # out = model.forward(data)
#     # assert out


# def test_reproducibility() -> None:
#     """Test if the model can give same results always - compare cpu based results with cuda results."""


# def test_trainer() -> None:
#     """Test if the experiment module works properly."""
#     training_params: dict[str, Any] = {
#         "dataset": "mock_dataset",
#         "timepoint": None,
#         "n_epochs": 5,
#         "learning_rate": 0.005,
#     }
#     trainer = BaseTrainer(**training_params)
#     trainer.train()


# def test_inferer() -> None:
#     """Test if the experiment module works properly."""
#     target_model_path = os.path.join(MODELS_PATH, "default_model_name", "fold0")
#     inference_params: dict[str, Any] = {
#         # "conv_size": 48,
#         "model_params": {
#             "in_features": 3,
#             "out_features": 1,
#             "batch_size": 1,
#             "layer_sizes": (8, 16),
#         },
#         "model_path": target_model_path,
#         "dataset": "mock_dataset",
#     }
#     inferer = BaseInferer(**inference_params)
#     current_results = inferer.run()
#     assert current_results is not None
