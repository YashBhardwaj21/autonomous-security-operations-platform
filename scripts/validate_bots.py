import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.canon.schema import SourceType
from src.ingestion.parser import ParserFactory, DropStats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join("data", "raw", "bots"))
    args = ap.parse_args()
    files = glob.glob(os.path.join(args.path, "**", "*.json"), recursive=True) + \
            glob.glob(os.path.join(args.path, "**", "*.ndjson"), recursive=True)
    if not files:
        print(f"No BOTS event files under {args.path}. Clone splunk/botsv3 (git-lfs) and export events.")
        return 1
    fac, stats, parsed = ParserFactory(), DropStats(), 0
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if fac.parse(raw, SourceType.BOTS, stats) is not None:
                    parsed += 1
    print(f"BOTS validation: parsed={parsed} dropped={stats.total()} "
          f"(no_parser={stats.no_parser}, no_timestamp={stats.no_timestamp})")
    print("Sanity check only — BOTS is never trained on.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
