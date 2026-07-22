from pathlib import Path
import pytest
import torch
from torch_geometric.data import Data

from src.graph.dataset import ActivityGraphDataset


def test_graph_dataset_initialization():
    activity_dir = Path("data/processed/activities")
    if not activity_dir.exists() or not list(activity_dir.glob("activity_*.json")):
        pytest.skip("No exported activities present in data/processed/activities/")

    dataset = ActivityGraphDataset(root=str(activity_dir))

    # Verify PyG Dataset properties
    assert len(dataset) > 0
    assert dataset.processed_paths[0].endswith("data.pt")
    assert Path(dataset.processed_paths[0]).exists()


def test_graph_dataset_item_properties():
    activity_dir = Path("data/processed/activities")
    if not activity_dir.exists() or not list(activity_dir.glob("activity_*.json")):
        pytest.skip("No exported activities present in data/processed/activities/")

    dataset = ActivityGraphDataset(root=str(activity_dir))

    item = dataset[0]
    assert isinstance(item, Data)
    assert hasattr(item, "x")
    assert hasattr(item, "edge_index")
    assert hasattr(item, "edge_attr")
    assert hasattr(item, "y")
    assert hasattr(item, "activity_id")
    assert hasattr(item, "scenario_id")
    assert hasattr(item, "source")
    assert hasattr(item, "start_time")
    assert hasattr(item, "end_time")
    assert not torch.isnan(item.x).any()

