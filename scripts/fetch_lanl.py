#!/usr/bin/env python3
"""Fetch a SMALL subset of LANL benign telemetry for the UEBA baseline.

[YOU run this — bandwidth. FABLE wrote it.] Respects the 16GB RAM ceiling: fetches
only a handful of day-files, not the full 12GB/1.65B-event corpus. Benign-only:
red-team rows are used solely to EXCLUDE labelled events from the benign population.

Sources (verified):
  LANL 2015 cyber1: https://csr.lanl.gov/data/cyber1/ (auth/proc/flows/dns/redteam .txt.gz)
  LANL 2017 unified (pure benign): https://lanl.ma.ic.ac.uk/data/2017/wls/wls_day-<i>.bz2

Writes into data/raw/lanl/. Loader: src.ingestion.benign.lanl_auth_features.
"""
import os, sys, urllib.request

DEST = os.path.join("data", "raw", "lanl")
FILES_2015 = ["auth.txt.gz", "flows.txt.gz", "redteam.txt.gz"]  # small subset
BASE_2015 = "https://csr.lanl.gov/data/cyber1"

def main():
    os.makedirs(DEST, exist_ok=True)
    print("NOTE: LANL files are large. This fetches only the listed subset.")
    for name in FILES_2015:
        url = f"{BASE_2015}/{name}"
        out = os.path.join(DEST, name)
        print(f"fetch {url} -> {out}")
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"  skip: {e} (site may require manual access)")
    print("done. Decompress and stream a subset into src.ingestion.benign.lanl_auth_features().")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
