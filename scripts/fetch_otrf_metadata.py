#!/usr/bin/env python3
"""Fetch OTRF scenario METADATA (small yaml files) for label + transition building.

[FABLE — CODE ONLY, but safe to run: metadata is KBs, not the GB event bundles.]
Downloads the compound and atomic _metadata/*.yaml from OTRF/Security-Datasets into
data/raw/Security-Datasets/... These carry attack_mappings (labels) and drive the
data-derived transition matrix (scripts/build_transition_matrix.py). Event ZIPs are
NOT fetched here — see scripts/fetch_otrf_sample.py for a few real scenarios.
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master"
ROOT = os.path.join("data", "raw", "Security-Datasets")
GROUPS = ("compound", "atomic")


def _list_metadata(group: str):
    # Use public GitHub API to list directory contents instead of gh CLI.
    url = f"https://api.github.com/repos/OTRF/Security-Datasets/contents/datasets/{group}/_metadata"
    try:
        import json
        req = urllib.request.Request(url, headers={'User-Agent': 'Python/urllib'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return [item['name'] for item in data if item['name'].endswith((".yaml", ".yml"))]
    except Exception as e:
        print(f"[{group}] could not list metadata via API ({e}). Rate limited?")
        return []


def main() -> int:
    total = 0
    for group in GROUPS:
        dest = os.path.join(ROOT, "datasets", group, "_metadata")
        os.makedirs(dest, exist_ok=True)
        for name in _list_metadata(group):
            url = f"{RAW}/datasets/{group}/_metadata/{name}"
            try:
                urllib.request.urlretrieve(url, os.path.join(dest, name))
                total += 1
            except Exception as e:
                print(f"  skip {name}: {e}")
        print(f"[{group}] fetched metadata into {dest}")
    print(f"done: {total} metadata files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
