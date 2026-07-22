#!/usr/bin/env python3
"""Isolation guard: ensure no synthetic/dummy test data leaks into the runtime app.

Ground rule (user-set): dummy data may exist ONLY under tests/_fixtures and
tests/harness_selftest, and src/** must never import it. Fails loudly (exit 1)
if any src/ module imports the test packages or references a dummy generator.

Run: python scripts/check_no_dummy_in_src.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Match ACTUAL imports/usages of test packages or dummy generators — not prose
# references in docstrings/comments (those are fine).
FORBIDDEN = [
    re.compile(r"^\s*(?:import|from)\s+tests\b"),
    re.compile(r"\btests\.(?:_fixtures|harness_selftest)\b"),
    re.compile(r"\b(?:make_synthetic|make_dummy|fake_features|synthetic_training)\s*\("),
]


def _strip_comment(line: str) -> str:
    # Best-effort: drop a trailing/inline comment so "# see tests/..." is ignored.
    return line.split("#", 1)[0]


def main() -> int:
    violations = []
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = _strip_comment(raw)
            for pat in FORBIDDEN:
                if pat.search(line):
                    violations.append(f"{py.relative_to(ROOT)}:{lineno}: {raw.strip()}")
    if violations:
        print("ISOLATION VIOLATION — dummy/test data referenced in src/:", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 1
    print("OK: src/ contains no dummy-data or test-fixture references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
