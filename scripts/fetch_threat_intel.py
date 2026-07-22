from __future__ import annotations

import os
import sys
import urllib.request

REF = os.path.join("data", "reference")
ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)


def fetch_attack_stix() -> None:
    os.makedirs(REF, exist_ok=True)
    dest = os.path.join(REF, "enterprise-attack.json")
    print(f"fetching ATT&CK STIX -> {dest} (~35MB, canonical MITRE descriptions)")
    try:
        urllib.request.urlretrieve(ATTACK_STIX_URL, dest)
        print("  ok")
    except Exception as e:
        print(f"  failed: {e}")


def scaffold_advisory_dirs() -> None:
    for sub in ("advisories/cert-in", "advisories/cisa", "cves"):
        d = os.path.join(REF, sub)
        os.makedirs(d, exist_ok=True)
        readme = os.path.join(d, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w", encoding="utf-8") as f:
                f.write(
                    "# Real advisories only.\n\n"
                    "Drop one JSON per advisory: {\"ref\": \"<real-id>\", \"text\": \"...\", "
                    "\"technique_ids\": [\"T1078\"]}.\n"
                    "NEVER invent advisory IDs — an empty dir is honest (REPORT.md H7).\n"
                )
    print("advisory/cve directories scaffolded (real content is yours to add)")


def main() -> int:
    fetch_attack_stix()
    scaffold_advisory_dirs()
    print("done. Rebuild the index via RetrievalService (reads data/reference/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
