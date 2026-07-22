#!/usr/bin/env python3
"""Train + evaluate the attribution model on REAL OTRF data across session filtering strategies.

Usage:
  python scripts/train_attribution.py --train                             # Evaluates all strategies
  python scripts/train_attribution.py --train --filter-mode evidence       # Evidence filter only
  python scripts/train_attribution.py --train --fit-final                  # Gated candidate persistence
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_strategy(name: str, X: np.ndarray, y: List[str], scenario_ids: List[str],
                      total_single_tech_count: int, feature_names: List[str]):
    from src.attribution.model import HygienicAttributionEstimator
    from src.evaluation.harness import nested_loso_cv

    retained_count = len(y)
    coverage = (retained_count / total_single_tech_count * 100.0) if total_single_tech_count > 0 else 0.0
    unique_scenarios = len(set(scenario_ids))
    unique_classes = len(set(y))
    class_counts = Counter(y)

    print("\n" + "=" * 80)
    print(f"STRATEGY AUDIT & EVALUATION: {name.upper()}")
    print("=" * 80)
    print(f"  - Retained Samples : {retained_count} / {total_single_tech_count} ({coverage:.1f}% coverage)")
    print(f"  - Unique Scenarios : {unique_scenarios}")
    print(f"  - Target Classes   : {unique_classes}")
    print("  - Per-Class Breakdown:")
    for cls_name, cnt in sorted(class_counts.items()):
        print(f"      * {cls_name}: {cnt} samples")

    if retained_count == 0 or unique_scenarios < 2 or unique_classes < 2:
        print(f"[SKIP] Strategy {name} has insufficient samples or classes for LOSO cross-validation.")
        return None

    report = nested_loso_cv(
        HygienicAttributionEstimator.factory(feature_names, calibrate=True),
        X, y, scenario_ids,
    )
    print("\n[LOSO Evaluation Results]")
    print(report.summary())

    macro_f1 = report.aggregate.get("macro_f1", 0.0)
    top3 = report.aggregate.get("mean_fold_top3", 0.0)
    ece = report.aggregate.get("mean_fold_ece", 1.0)
    brier = report.aggregate.get("mean_fold_brier", 1.0)
    maj_baseline = report.baselines.get("majority_class", {}).get("macro_f1", 0.0)
    rand_baseline = report.baselines.get("uniform_random", {}).get("macro_f1", 0.0)

    return {
        "name": name,
        "samples": retained_count,
        "scenarios": unique_scenarios,
        "classes": unique_classes,
        "coverage": coverage,
        "macro_f1": macro_f1,
        "top3": top3,
        "ece": ece,
        "brier": brier,
        "maj_baseline": maj_baseline,
        "rand_baseline": rand_baseline,
        "report": report,
        "X": X,
        "y": y,
        "scenario_ids": scenario_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="required safety flag — nothing runs without it")
    ap.add_argument("--fit-final", action="store_true",
                    help="after LOSO, fit on all data and save candidate model")
    ap.add_argument("--min-scenarios", type=int, default=5)
    ap.add_argument("--filter-mode", type=str, choices=["unfiltered", "evidence", "ueba", "all"], default="all",
                    help="session selection mode: unfiltered, evidence, ueba, or all (comparative)")
    ap.add_argument("--ueba-threshold", type=float, default=0.5,
                    help="UEBA anomaly score threshold for filtering")
    args = ap.parse_args()

    if not args.train:
        print(__doc__)
        print("Refusing to run without --train (safety guard).")
        return 2

    from src.attribution.loader import AttributionArtifact
    from src.attribution.model import HygienicAttributionEstimator
    from src.features.labeling import build_label_space
    from src.ingestion.otrf import build_dataset
    from src.ueba.engine import UEBAEngine

    ds = build_dataset()
    print(f"[data] sessions={len(ds.scenario_ids)}  scenarios={len(set(ds.scenario_ids))} "
          f"features={len(ds.feature_names)}  drops={ds.drop_stats}")
    if len(ds.scenario_ids) == 0:
        print("No OTRF data found under data/raw/Security-Datasets. Clone a few scenarios there first.")
        return 1

    supported, unsupported, counts = build_label_space(ds.labels, args.min_scenarios, scenario_ids=ds.scenario_ids)
    print(f"[labels] supported={len(supported)} unsupported(<{args.min_scenarios})={sorted(unsupported)}")

    # Single-technique target mask
    st_mask = [
        t is not None and t in supported
        for t in ds.train_target
    ]
    st_indices = [i for i, m in enumerate(st_mask) if m]
    if not st_indices:
        print("[error] No single-technique sessions match supported criteria.")
        return 1

    total_st_count = len(st_indices)

    # Compute UEBA scores for UEBA filter strategy
    ueba_engine = UEBAEngine()
    ueba_scores = []
    for idx in range(len(ds.scenario_ids)):
        fmap = dict(zip(ds.feature_names, ds.X[idx]))
        u_feats = {
            "event_count": 1.0,
            "process_access": fmap.get("process_access_count", 0.0),
            "registry_mod": fmap.get("registry_mod_count", 0.0),
            "failed_login": fmap.get("failed_login_count", 0.0),
            "network_flow": fmap.get("network_flow_count", 0.0),
        }
        res = ueba_engine.process("audit_host", u_feats)
        ueba_scores.append(res.score)

    # Build index sets for strategies
    unfiltered_idx = st_indices
    evidence_idx = [i for i in st_indices if ds.has_evidence[i]]
    ueba_idx = [i for i in st_indices if ueba_scores[i] >= args.ueba_threshold]

    # Pre-evaluation Evidence Audit Report
    total_ev_count = sum(1 for i in st_indices if ds.has_evidence[i])
    print("\n" + "=" * 80)
    print("PRE-EVALUATION EVIDENCE FILTER AUDIT REPORT")
    print("=" * 80)
    print(f"Total Single-Technique Sessions : {total_st_count}")
    print(f"Sessions WITH Attack Evidence   : {total_ev_count} ({total_ev_count/total_st_count*100.0:.1f}%)")
    print(f"Sessions WITHOUT Evidence      : {total_st_count - total_ev_count} ({(total_st_count - total_ev_count)/total_st_count*100.0:.1f}%)")

    # Per-class retention breakdown
    st_classes = Counter([ds.train_target[i] for i in st_indices])
    ev_classes = Counter([ds.train_target[i] for i in st_indices if ds.has_evidence[i]])
    print("\nPer-Class Evidence Retention Breakdown:")
    for cls_name in sorted(st_classes.keys()):
        orig = st_classes[cls_name]
        ret = ev_classes.get(cls_name, 0)
        pct = (ret / orig * 100.0) if orig > 0 else 0.0
        print(f"  - {cls_name}: {ret} / {orig} retained ({pct:.1f}%)")

    strategies_to_run = []
    if args.filter_mode in ("unfiltered", "all"):
        strategies_to_run.append(("unfiltered", unfiltered_idx))
    if args.filter_mode in ("evidence", "all"):
        strategies_to_run.append(("evidence", evidence_idx))
    if args.filter_mode in ("ueba", "all"):
        strategies_to_run.append(("ueba", ueba_idx))

    results = []
    for mode_name, idxs in strategies_to_run:
        X_sub = ds.X[idxs]
        y_sub = [ds.train_target[i] for i in idxs]
        scen_sub = [ds.scenario_ids[i] for i in idxs]
        res = evaluate_strategy(mode_name, X_sub, y_sub, scen_sub, total_st_count, ds.feature_names)
        if res:
            results.append(res)

    if not results:
        print("[error] No evaluations completed successfully.")
        return 1

    # Comparative Summary Table
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARATIVE SESSION FILTERING BENCHMARK TABLE")
        print("=" * 80)
        print(f"{'Strategy':<15} | {'Samples':<7} | {'Scen':<5} | {'Class':<5} | {'Coverage':<8} | {'Macro-F1':<8} | {'Top-3':<7} | {'ECE':<6} | {'Maj-F1':<6}")
        print("-" * 88)
        for r in results:
            print(f"{r['name'].upper():<15} | {r['samples']:<7} | {r['scenarios']:<5} | {r['classes']:<5} | {r['coverage']:>6.1f}% | {r['macro_f1']:>8.4f} | {r['top3']:>7.4f} | {r['ece']:>6.4f} | {r['maj_baseline']:>6.4f}")

    # Persistence gate evaluation (evaluates best / requested mode)
    target_res = results[0]
    macro_f1 = target_res["macro_f1"]
    maj_baseline = target_res["maj_baseline"]
    rand_baseline = target_res["rand_baseline"]
    ece = target_res["ece"]

    f1_passed = macro_f1 > max(maj_baseline, rand_baseline) + 0.05
    ece_passed = ece < 0.15

    print("\n" + "=" * 80)
    print(f"CODE-LEVEL PERSISTENCE GATE EVALUATION ({target_res['name'].upper()})")
    print("=" * 80)
    print(f"  - Min Scenario Count (>= {args.min_scenarios}) : PASSED ({target_res['classes']} supported target classes)")
    print(f"  - Macro-F1 Improvement over Baselines : {'PASSED' if f1_passed else 'FAILED'} (Macro-F1: {macro_f1:.4f}, Maj: {maj_baseline:.4f}, Rand: {rand_baseline:.4f})")
    print(f"  - Calibration ECE (< 0.15)              : {'PASSED' if ece_passed else 'FAILED'} (ECE: {ece:.4f})")

    gate_passed = f1_passed and ece_passed

    if args.fit_final:
        if not gate_passed:
            print("\n[GATE BLOCKED] Candidate artifact persistence rejected: performance gate failed.")
            return 1

        cand_dir = os.path.join("models", "candidates")
        os.makedirs(cand_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cand_path = os.path.join(cand_dir, f"attribution-{ts}.joblib")

        print(f"\n[fit-final] Gate PASSED. Fitting calibrated model on all data -> {cand_path}")
        estimator = HygienicAttributionEstimator(ds.feature_names, calibrate=True)
        estimator.fit(target_res["X"], target_res["y"], groups_train=target_res["scenario_ids"])
        art = AttributionArtifact(model=estimator.model, hygiene=estimator.hygiene,
                                  feature_names_in=ds.feature_names, version=f"attribution-{ts}")
        art.save(cand_path)
        print(f"[fit-final] saved candidate artifact: {cand_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
