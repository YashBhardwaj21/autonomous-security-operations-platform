"""Attribution feature pipeline — Activity -> FeatureVector (attribution space).

Ported from asop ``archive/features/engineering.py`` and adapted to the canonical
schema. Feature order is deterministic (sorted) rather than "frozen on first call"
(the old order-dependent bug) — the key set is fully determined by the extractors,
so sorting gives a stable schema without hidden state.

This produces the ATTRIBUTION feature space. The UEBA space is built separately
(src/ueba) and must not share features — REPORT.md M6. In particular no
``anomaly_score`` is ever added here.
"""
from __future__ import annotations

from typing import List, Optional

from src.canon.schema import Activity, FeatureVector, SourceType
from src.features.extractors import ALL_EXTRACTORS


class AttributionFeaturePipeline:
    SPACE = "attribution"

    def extract(self, activity: Activity) -> FeatureVector:
        feats: dict = {}
        for fn in ALL_EXTRACTORS:
            feats.update(fn(activity))
        names = sorted(feats.keys())
        return FeatureVector(
            space=self.SPACE,
            activity_id=activity.activity_id,
            scenario_id=activity.scenario_id,
            source=activity.source,
            feature_names=names,
            features=[float(feats[n]) for n in names],
        )

    def extract_batch(self, activities: List[Activity]) -> List[FeatureVector]:
        return [self.extract(a) for a in activities]

    def feature_names(self, activity: Optional[Activity] = None) -> List[str]:
        """Stable schema. If no sample activity is given, returns the names an
        empty activity would yield (all extractor keys with zero values)."""
        if activity is not None:
            return self.extract(activity).feature_names
        from datetime import datetime
        empty = Activity(activity_id="_schema", start_time=datetime(2000, 1, 1),
                         end_time=datetime(2000, 1, 1), source=SourceType.OTRF)
        return self.extract(empty).feature_names
