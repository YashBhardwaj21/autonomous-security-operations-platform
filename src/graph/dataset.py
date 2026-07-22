from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

from src.canon.schema import Activity
from src.graph.graph_builder import GraphBuilder


class ActivityGraphDataset(InMemoryDataset):
    """PyTorch Geometric InMemoryDataset loading serialized activity JSONs into PyG Data objects with caching."""

    def __init__(
        self,
        root: str = "data/processed/activities",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        self.activity_dir = Path(root)
        self.builder = GraphBuilder()
        super().__init__(str(self.activity_dir), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        if not self.activity_dir.exists():
            return []
        return sorted([f.name for f in self.activity_dir.glob("activity_*.json")])

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        raw_files = self.raw_file_names

        # Step 1: Collect unique scenario IDs deterministically in sorted order (Fixes non-deterministic hash() % 10 bug)
        unique_scenarios = set()
        activity_dicts = []

        for filename in raw_files:
            file_path = self.activity_dir / filename
            with file_path.open("r", encoding="utf-8") as f:
                raw_dict = json.load(f)

            if "start_time" not in raw_dict or not raw_dict["start_time"]:
                raw_dict["start_time"] = "2026-01-01T00:00:00"
            if "end_time" not in raw_dict or not raw_dict["end_time"]:
                raw_dict["end_time"] = "2026-01-01T00:05:00"

            scen_id = raw_dict.get("scenario_id") or "unknown"
            unique_scenarios.add(scen_id)
            activity_dicts.append(raw_dict)

        sorted_scenarios = sorted(list(unique_scenarios))
        scenario_label_map: Dict[str, int] = {scen: idx for idx, scen in enumerate(sorted_scenarios)}

        # Step 2: Build PyG Data objects with deterministic label assignments
        graphs: List[Data] = []
        for raw_dict in activity_dicts:
            activity = Activity.model_validate(raw_dict)
            data = self.builder.build(activity)

            scen_id = activity.scenario_id or "unknown"
            y_val = scenario_label_map[scen_id]
            data.y = torch.tensor([y_val], dtype=torch.long)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            graphs.append(data)

        if not graphs:
            now = datetime.now()
            dummy_data = self.builder.build(
                Activity(activity_id="empty", start_time=now, end_time=now)
            )
            dummy_data.y = torch.tensor([0], dtype=torch.long)
            graphs.append(dummy_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
