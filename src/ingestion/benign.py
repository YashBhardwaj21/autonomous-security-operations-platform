"""Benign-corpus loaders for the UEBA baseline — REPORT.md C1/M2/M3.

Maps the REAL, verified schemas of LANL / CICIDS / TON_IoT into UEBA-space
behavioural feature records (per-entity counts). Hard boundary: these sources are
BENIGN-ONLY and feed the UEBA baseline path exclusively — they NEVER carry an
ATT&CK label and never reach the attribution classifier (enforced by SourceType
membership in BENIGN_ONLY_SOURCES and by producing space="ueba" vectors only).

Verified column schemas (from public sources during planning):
  LANL cyber1 auth.txt : time,src_user@dom,dst_user@dom,src_comp,dst_comp,auth_type,
                         logon_type,auth_orientation,success/failure
  LANL cyber1 flows.txt: time,duration,src_comp,src_port,dst_comp,dst_port,proto,
                         pkt_count,byte_count
  CICIDS MachineLearningCSV: ~79 CICFlowMeter cols + 'Label' (filter Label==BENIGN)
  TON_IoT network: many cols + 'type' (filter type==normal)

These loaders are consumed by the write-only scripts (scripts/fetch_lanl.py etc.);
they parse REAL files. Given no file, they yield nothing — never synthetic rows.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional

from src.canon.schema import BENIGN_ONLY_SOURCES, FeatureVector, SourceType


def _assert_benign(source: SourceType) -> None:
    if source not in BENIGN_ONLY_SOURCES:
        raise ValueError(f"{source} is not a benign-only source; "
                         "benign loaders feed UEBA baseline only (REPORT.md C1).")


# ---- LANL auth.txt -> per-(source computer) behavioural counts ---------------
LANL_AUTH_COLUMNS = ["time", "src_user", "dst_user", "src_comp", "dst_comp",
                     "auth_type", "logon_type", "auth_orientation", "success"]


def lanl_auth_features(lines: Iterable[str]) -> Iterator[FeatureVector]:
    """Aggregate LANL auth rows into per-source-computer UEBA feature vectors."""
    _assert_benign(SourceType.LANL)
    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for line in lines:
        parts = line.rstrip("\n").split(",")
        if len(parts) < 9:
            continue
        rec = dict(zip(LANL_AUTH_COLUMNS, parts))
        key = rec["src_comp"]
        agg[key]["auth_count"] += 1.0
        if rec["success"].lower().startswith("fail"):
            agg[key]["auth_fail"] += 1.0
        if rec["src_comp"] != rec["dst_comp"]:
            agg[key]["remote_auth"] += 1.0
    for key, feats in agg.items():
        names = sorted(feats)
        yield FeatureVector(space="ueba", source=SourceType.LANL,
                            activity_id=f"lanl_auth::{key}",
                            feature_names=names, features=[feats[n] for n in names])


# ---- CICIDS MachineLearningCSV (BENIGN only) -> network UEBA features --------
def cicids_benign_features(rows: Iterable[Dict[str, str]],
                           label_col: str = "Label") -> Iterator[FeatureVector]:
    """Yield UEBA network feature vectors from CICIDS rows, BENIGN only."""
    _assert_benign(SourceType.CICIDS)
    numeric_cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                    "Flow Bytes/s", "Flow Packets/s"]
    for i, row in enumerate(rows):
        label = (row.get(label_col) or row.get(label_col.strip()) or "").strip().upper()
        if label != "BENIGN":
            continue                       # network features only from benign rows
        feats = {}
        for c in numeric_cols:
            v = row.get(c) or row.get(c.strip())
            try:
                feats[c] = float(v)
            except (TypeError, ValueError):
                feats[c] = 0.0
        names = sorted(feats)
        yield FeatureVector(space="ueba", source=SourceType.CICIDS,
                            activity_id=f"cicids::{i}",
                            feature_names=names, features=[feats[n] for n in names])


def cicids_from_csv(path: str) -> Iterator[FeatureVector]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader((ln.lstrip() for ln in f))
        yield from cicids_benign_features(reader)


# ---- TON_IoT network (type==normal) -----------------------------------------
def toniot_normal_features(rows: Iterable[Dict[str, str]],
                           type_col: str = "type") -> Iterator[FeatureVector]:
    _assert_benign(SourceType.TONIOT)
    for i, row in enumerate(rows):
        if (row.get(type_col) or "").strip().lower() != "normal":
            continue
        feats = {}
        for k, v in row.items():
            if k == type_col:
                continue
            try:
                feats[k] = float(v)
            except (TypeError, ValueError):
                continue
        if not feats:
            continue
        names = sorted(feats)
        yield FeatureVector(space="ueba", source=SourceType.TONIOT,
                            activity_id=f"toniot::{i}",
                            feature_names=names, features=[feats[n] for n in names])
