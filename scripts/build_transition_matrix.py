#!/usr/bin/env python3
"""Build the data-derived ATT&CK transition matrix from real OTRF metadata.

[FABLE — CODE ONLY, deterministic: safe to run — pure counting, no model, no tuning.]
Reads data/raw/Security-Datasets/**/_metadata/*.yaml (fetch via
scripts/fetch_otrf_metadata.py) and writes models/transition_matrix.json with true,
reproducible provenance. Prints per-source rows and their support counts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.transition import build_transition_matrix  # noqa: E402


def main() -> int:
    tm = build_transition_matrix()
    print(f"scenarios={tm.n_scenarios} transitions={tm.n_transitions}")
    if tm.n_scenarios == 0:
        print("No multi-stage scenario metadata found. Run scripts/fetch_otrf_metadata.py first.")
        return 1
    for a in sorted(tm.matrix):
        print(f"  {a} -> {tm.next_techniques(a, 3)}")
    out = os.path.join("models", "transition_matrix.json")
    tm.save(out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
