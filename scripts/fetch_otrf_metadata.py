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


KNOWN_SCENARIOS = [
    "SDWIN-190518210652", "SDWIN-200805034820", "SDWIN-200806012009", "SDWIN-200806015757",
    "SDWIN-190518221344", "SDWIN-201019232515", "SDWIN-201012183248", "SDWIN-201022002145",
    "SDWIN-220630130349", "SDWIN-220703123711", "SDWIN-220705170038", "SDWIN-220708104215",
    "SDWIN-200724174200", "SDWIN-200806035621", "SDWIN-200914080546"
]


def fetch_metadata_file(group: str, name: str, dest_dir: str) -> bool:
    url = f"{RAW}/datasets/{group}/_metadata/{name}"
    target = os.path.join(dest_dir, name)
    if os.path.exists(target):
        return True
    try:
        urllib.request.urlretrieve(url, target)
        print(f"  [OK] {group}/_metadata/{name}")
        return True
    except Exception:
        return False


def main() -> int:
    total = 0
    for group in GROUPS:
        dest = os.path.join(ROOT, "datasets", group, "_metadata")
        os.makedirs(dest, exist_ok=True)
        names = _list_metadata(group)
        if names:
            for name in names:
                if fetch_metadata_file(group, name, dest):
                    total += 1
            print(f"[{group}] fetched metadata via API list into {dest}")
        else:
            print(f"[{group}] API rate limited. Attempting direct download of known scenario metadata...")
            for scen in KNOWN_SCENARIOS:
                fname = f"{scen}.yaml"
                if fetch_metadata_file(group, fname, dest):
                    total += 1
    print(f"done: {total} metadata files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
