# Autonomous Security Operations Platform

A behavioural-intelligence layer for critical-infrastructure SOCs: it ingests
security telemetry, detects behavioural anomalies (UEBA), attributes activity to
MITRE ATT&CK techniques with **calibrated** confidence, predicts likely next steps,
retrieves supporting threat-intel, estimates blast radius over a digital twin, and
gates automated response — deterministically and auditably.

This repository is the **single merged codebase** unifying two prior prototypes
(`YashBhardwaj21/autonomous-security-operations-platform` = ML/data pipeline;
`vikramhls/etbackend` = FastAPI backend). The merge simultaneously remediated the
findings from the audit in `../sentinelgrid-audit/REPORT.md`; see `MERGE_REPORT.md`
for the finding → resolution map and `BUILD_LOG.md` for the phase-by-phase record.

## Honest scope (what is real vs. what you must run)

- **Real & data-driven now:** canonical parser (Sysmon 1/3/7/10/11/12/13 + PowerShell
  4103/4104), entity-keyed sessions, attribution feature space, fitted feature hygiene,
  nested-LOSO evaluation harness, calibrated hierarchical attribution **scaffold**,
  **data-derived** ATT&CK transition matrix (`models/transition_matrix.json`, built
  from real OTRF compound scenarios), embedding retrieval (offline TF-IDF default),
  online UEBA engine, digital-twin Dijkstra simulator, hardened SOAR blast-radius gate
  with response-mode policy, EPSS-aware vuln scorer, JWT-authenticated ingest.
- **No synthetic data** ever enters the runtime path. Dummy data exists only under
  `tests/_fixtures` and `tests/harness_selftest`, and `scripts/check_no_dummy_in_src.py`
  fails the build if `src/` references it.
- **You (the operator) run the training and downloads** — every `.fit()` is behind a
  guarded CLI flag and is never run by the code assistant. See `docs/ML_SPECIFICATION.md`.

## Layout

```
src/canon        canonical event/entity/session schema (single source of truth)
src/ingestion    parsers (+OTRF/benign loaders); DropStats (no fabricated timestamps)
src/sessions     entity+logon-keyed session builder
src/features     extractors, attribution feature pipeline, fitted hygiene, labeling
src/evaluation   nested-LOSO harness + calibration/ranking metrics
src/attribution  calibrated hierarchical model (scaffold), SHAP, artifact loader
src/prediction   data-derived transition matrix + next-step engine
src/retrieval    embedding retrieval over real threat-intel corpus (non-gating)
src/ueba         online Welford+IsolationForest anomaly engine (own feature space)
src/twin         digital-twin Dijkstra simulator (real topology required)
src/soar         blast-radius gate (response modes) + orchestrator
src/vuln         EPSS-aware risk scorer
src/api          FastAPI app, JWT auth, IncidentPipeline (the cross-boundary chain)
src/config       single settings surface
scripts          dataset/threat-intel fetchers, transition builder, guarded trainer
tests            unit / integration (real inputs) ; _fixtures, harness_selftest (dummy)
```

## Quickstart (dev)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/check_no_dummy_in_src.py         # isolation guard
python -m pytest tests -q                        # 52 tests
```

## Getting to real numbers (operator steps)

```bash
python scripts/fetch_otrf_metadata.py            # labels + transitions (KB metadata)
python scripts/build_transition_matrix.py        # deterministic, safe to run
# clone a few OTRF scenarios into data/raw/Security-Datasets/ for real events, then:
python scripts/train_attribution.py --train      # LOSO eval — YOU inspect results
python scripts/fetch_threat_intel.py             # ATT&CK STIX + advisory scaffolding
export JWT_SECRET=... && uvicorn src.api.app:app # run the API
```
