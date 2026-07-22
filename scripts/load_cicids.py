#!/usr/bin/env python3
"""Load CICIDS2017/2018 BENIGN network flows into UEBA features.

[FABLE wrote it; YOU provide the CSVs.] Use the cleaned Kaggle mirrors (NOT the
450GB AWS bucket): https://www.kaggle.com/datasets/dhoogla/cicids2017 (and .../csecicids2018).
Place CSVs under data/raw/cicids/. Only Label==BENIGN rows are used.
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.benign import cicids_from_csv

def main():
    paths = glob.glob(os.path.join("data", "raw", "cicids", "*.csv"))
    if not paths:
        print("No CSVs under data/raw/cicids/. Download the Kaggle mirror first.")
        return 1
    n = 0
    for p in paths:
        for _fv in cicids_from_csv(p):
            n += 1
    print(f"loaded {n} BENIGN network feature vectors (UEBA space) from {len(paths)} files")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
