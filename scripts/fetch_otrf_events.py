from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import requests
import yaml

ROOT = Path("data/raw/Security-Datasets")

METADATA_DIRS = [
    ROOT / "datasets" / "compound" / "_metadata",
    ROOT / "datasets" / "atomic" / "_metadata",
]

MANIFEST = ROOT / "otrf_download_manifest.json"

URL_PREFIX = "https://raw.githubusercontent.com/OTRF/Security-Datasets/"
URL_MARKER = "Security-Datasets/master/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def discover_metadata():

    metadata = {}

    for directory in METADATA_DIRS:

        if not directory.exists():
            continue

        for yaml_file in directory.glob("*.yaml"):
            metadata[yaml_file.stem] = yaml_file

    return metadata


def load_manifest():

    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as f:
            return json.load(f)

    return []


def save_manifest(manifest):

    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def download_archive(url: str, output: Path):

    temp = output.with_suffix(output.suffix + ".part")

    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()

    output.parent.mkdir(parents=True, exist_ok=True)

    with temp.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    temp.rename(output)


def process_scenario(name: str, metadata_path: Path, manifest: list):

    print(f"\n=== {name} ===")

    with metadata_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    files = meta.get("files", [])

    for item in files:

        if str(item.get("type", "")).lower() != "host":
            continue

        url = item["link"]

        if not url.startswith(URL_PREFIX):
            print(f"[SKIP] Untrusted URL: {url}")
            continue

        rel = url.split(URL_MARKER, 1)[1]

        output = ROOT / rel

        if output.exists():
            print(f"[SKIP] {output}")

        else:

            print(f"[DOWNLOAD] {url}")

            download_archive(url, output)

            print(f"[OK] {output}")

        manifest.append(
            {
                "scenario": name,
                "url": url,
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "downloaded_at": datetime.utcnow().isoformat() + "Z",
            }
        )


def fetch_metadata_for_scenario(name: str) -> Optional[Path]:
    groups = ["atomic", "compound"]
    for group in groups:
        url = f"https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/{group}/_metadata/{name}.yaml"
        target_dir = ROOT / "datasets" / group / "_metadata"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{name}.yaml"
        if target_file.exists():
            return target_file
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with target_file.open("wb") as f:
                    f.write(r.content)
                print(f"[METADATA OK] Downloaded {group}/_metadata/{name}.yaml")
                return target_file
        except Exception:
            pass
    return None


def main():

    parser = argparse.ArgumentParser(
        description="Download OTRF Security-Datasets host archives."
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenario IDs (LSASS_campaign_01, SDWIN-190301174830, ...)",
    )

    args = parser.parse_args()

    metadata = discover_metadata()

    manifest = load_manifest()

    for scenario in args.scenarios:

        if scenario not in metadata:
            fetched_path = fetch_metadata_for_scenario(scenario)
            if fetched_path:
                metadata[scenario] = fetched_path
            else:
                print(f"[ERROR] Metadata not found: {scenario}")
                continue

        process_scenario(
            scenario,
            metadata[scenario],
            manifest,
        )

    save_manifest(manifest)

    print("\nDone.")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()