#!/usr/bin/env python3
"""Train + evaluate the attribution model on REAL OTRF data.

[YOU — TRAIN/INSPECT] This is intentionally guarded behind --train and is NOT run
by the code assistant. It: builds the real dataset (data/raw/Security-Datasets),
runs nested LOSO with the harness (reporting macro-F1/top-3/ECE/Brier + CIs and
baselines), then — only if you pass --fit-final — fits the calibrated model on all
data and persists the artifact. Inspect the confusion matrix and reliability
diagram before trusting any number, and decide unsupported_class moves yourself.

Usage:
  python scripts/train_attribution.py --train                 # LOSO eval only
  python scripts/train_attribution.py --train --fit-final     # + persist artifact
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="required safety flag — nothing runs without it")
    ap.add_argument("--fit-final", action="store_true",
                    help="after LOSO, fit on all data and persist models/attribution.joblib")
    ap.add_argument("--min-scenarios", type=int, default=3)
    args = ap.parse_args()

    if not args.train:
        print(__doc__)
        print("Refusing to run without --train (safety guard).")
        return 2

    import numpy as np

    from src.attribution.loader import AttributionArtifact
    from src.attribution.model import HierarchicalAttributionModel
    from src.evaluation.harness import nested_loso_cv
    from src.features.hygiene import HygieneTransform
    from src.features.labeling import build_label_space
    from src.ingestion.otrf import build_dataset

    ds = build_dataset()
    print(f"[data] sessions={len(ds.scenario_ids)}  scenarios={len(set(ds.scenario_ids))} "
          f"features={len(ds.feature_names)}  drops={ds.drop_stats}")
    if len(ds.scenario_ids) == 0:
        print("No OTRF data found under data/raw/Security-Datasets. Clone a few "
              "scenarios there first (scripts/fetch_otrf_sample.py).")
        return 1

    supported, unsupported, counts = build_label_space(ds.labels, args.min_scenarios)
    print(f"[labels] supported={len(supported)} unsupported(<{args.min_scenarios})={sorted(unsupported)}")

    # Primary-technique single-label view for the harness, routing unsupported.
    y = [t if t in supported else "unsupported_class" for t in ds.y_primary]

    # Hygiene is fit inside training only; here we fit once on all rows for the LOSO
    # demonstration, but the harness's estimator refits per fold — so pass raw X and
    # let each fold's estimator own its preprocessing in a production run.
    report = nested_loso_cv(
        HierarchicalAttributionModel.factory(calibrate=True),
        ds.X, y, ds.scenario_ids,
    )
    print(report.summary())

    if args.fit_final:
        print("[fit-final] fitting calibrated model on all data + persisting artifact")
        hygiene = HygieneTransform().fit(ds.X, ds.feature_names)
        Xh = hygiene.transform(ds.X)
        model = HierarchicalAttributionModel(calibrate=True).fit(Xh, y, groups_train=ds.scenario_ids)
        art = AttributionArtifact(model=model, hygiene=hygiene,
                                  feature_names_in=ds.feature_names, version="attribution-1.0")
        os.makedirs("models", exist_ok=True)
        art.save(os.path.join("models", "attribution.joblib"))
        print("[fit-final] saved models/attribution.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
