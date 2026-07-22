"""OTRF Security-Datasets loader — real telemetry -> sessions -> features + labels.

Reads a local clone of OTRF/Security-Datasets under ``data/raw/Security-Datasets``
(exactly where notebooks/explore.ipynb expects it). No download happens here; if
the directory is absent, ``build_dataset`` yields nothing (honest — never synthetic).

Layout (verified from the public repo):
  datasets/{atomic,compound}/.../_metadata/*.yaml   # attack_mappings (labels)
  datasets/{atomic,compound}/.../*.zip              # ndjson event bundles
Each metadata yaml references its data file(s) via a ``files: [{type, link}]`` list;
we resolve the Host/Windows link to the local zip path.
"""
from __future__ import annotations

import glob
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import yaml

from src.canon.schema import SourceType
from src.features.labeling import ScenarioLabel, label_from_metadata
from src.features.pipeline import AttributionFeaturePipeline
from src.ingestion.parser import DropStats, ParserFactory
from src.sessions.session_builder import SessionBuilder

DEFAULT_ROOT = os.path.join("data", "raw", "Security-Datasets")


def _resolve_local_zip(link: str, root: str) -> Optional[str]:
    # link like https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/...
    marker = "Security-Datasets/master/"
    if marker in link:
        rel = link.split(marker, 1)[1]
        cand = os.path.join(root, rel)
        if os.path.exists(cand):
            return cand
    # fall back: match by basename anywhere under root
    base = os.path.basename(link)
    hits = glob.glob(os.path.join(root, "datasets", "**", base), recursive=True)
    return hits[0] if hits else None


def iter_scenarios(root: str = DEFAULT_ROOT) -> Iterator[Tuple[str, dict, List[str]]]:
    """Yield (scenario_id, metadata_dict, [local_zip_paths]) for each host dataset."""
    if not os.path.isdir(root):
        return
    for meta_path in glob.glob(os.path.join(root, "datasets", "**", "_metadata", "*.yaml"),
                               recursive=True):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
        except Exception:
            continue
        scenario_id = str(meta.get("id") or os.path.splitext(os.path.basename(meta_path))[0])
        zips: List[str] = []
        for fref in (meta.get("files") or []):
            if str(fref.get("type", "")).lower() in ("host", "windows", "endpoint"):
                local = _resolve_local_zip(str(fref.get("link", "")), root)
                if local:
                    zips.append(local)
        if zips:
            yield scenario_id, meta, zips


def read_raw_events(zip_path: str) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith((".json", ".ndjson", ".jsonl")):
                continue
            with z.open(name) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # counted upstream if needed; never fabricated


@dataclass
class BuiltDataset:
    X: np.ndarray
    feature_names: List[str]
    labels: List[ScenarioLabel]           # multi-label per session (scenario's labels)
    scenario_ids: List[str]               # group key for LOSO
    y_primary: List[str]                  # primary technique per session (first mapping)
    drop_stats: Dict[str, int] = field(default_factory=dict)


def build_dataset(root: str = DEFAULT_ROOT, source: SourceType = SourceType.OTRF) -> BuiltDataset:
    """Assemble the real attribution dataset. Runs on real OTRF only; empty if absent."""
    factory = ParserFactory()
    builder = SessionBuilder(factory)
    fp = AttributionFeaturePipeline()
    stats = DropStats()

    rows: List[List[float]] = []
    labels: List[ScenarioLabel] = []
    groups: List[str] = []
    y_primary: List[str] = []
    names = fp.feature_names()  # stable schema

    for scenario_id, meta, zips in iter_scenarios(root):
        scen_label = label_from_metadata(meta, scenario_id, source)
        if scen_label.is_empty():
            continue
        events = []
        for zp in zips:
            for raw in read_raw_events(zp):
                ev = factory.parse(raw, source, stats)
                if ev is not None:
                    events.append(ev)
        if not events:
            continue
        for sess in builder.build_sessions(events, scenario_id=scenario_id):
            print(type(sess))
            print(sess)

            try:
                print(sess.__dict__.keys())
            except AttributeError:
                print(dir(sess))

            break

            fv = fp.extract(sess)
            fmap = dict(zip(fv.feature_names, fv.features))
            rows.append([fmap.get(n, 0.0) for n in names])
            labels.append(scen_label)
            groups.append(scenario_id)

            first = sorted(scen_label.techniques)[0]
            y_primary.append(first)

    X = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(names)))
    return BuiltDataset(
        X=X, feature_names=names, labels=labels, scenario_ids=groups,
        y_primary=y_primary,
        drop_stats={
            "no_timestamp": stats.no_timestamp, "bad_timestamp": stats.bad_timestamp,
            "no_parser": stats.no_parser, "malformed": stats.malformed,
            "total_dropped": stats.total(),
        },
    )
