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
