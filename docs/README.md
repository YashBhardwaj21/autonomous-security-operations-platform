# Autonomous Security Operations Platform — Documentation Index

This directory contains the authoritative technical specification and execution guides for the Autonomous Security Operations Platform.

> **Specification vs. Evidence**: The documents in `docs/` define the design contracts, evaluation standards, and runtime invariants. Directories such as `reports/`, `runs/`, and `archive/` contain generated data artifacts, metrics runs, and historical experimental evidence.

---

## Core Documentation

* **[Hackathon Execution Runbook](HACKATHON_RUNBOOK.md)** — Step-by-step workflow for data preflight, dataset auditing, label contract configuration, model fitting, and API demo execution.
* **[Architecture Specification](ARCHITECTURE.md)** — Core runtime component boundaries, `IncidentPipeline` integration chain, data isolation rules, and artifact surfaces.
* **[Machine Learning Specification](ML_SPECIFICATION.md)** — Dataset contracts (`BuiltDataset`), OTRF label boundaries, single-label (`y_primary`) limitations, LOSO cross-validation rules, and feature hygiene requirements.
* **[API Surface Specification](API.md)** — Verified REST API endpoints (`/health`, `/auth/token`, `/ingest/events`), schema contracts, and `model_unavailable` degraded-mode behavior.
* **[Security Architecture & Threat Model](SECURITY.md)** — Authentication standards (JWT/bcrypt), ingest validation, fail-closed SOAR blast-radius gates, and data integrity guarantees.
* **[Product Requirements Document (PRD)](PRD.md)** — Functional capability status matrix (implemented vs. in-progress vs. archived) and explicit non-goals.

---

## Experimental & Historical Evidence References

* **[GNN Experimental Investigation Package](../experiments/gnn/README.md)** — Active experimental directory containing the isolated GraphSAGE model, graph builders, and experimental scripts.
* **[GNN Historical Archive (2026-07)](../archive/gnn-2026-07/README.md)** — Historical evidence snapshot documenting why GraphSAGE was retired from the primary runtime path (macro-F1 = 0.075, scenario-identity formulation, graph label ambiguity).
* **Historical Audit Logs**:
  * [BUILD_LOG.md](history/BUILD_LOG.md) — Historical phase-by-phase record of the initial repository merge.
  * [MERGE_REPORT.md](history/MERGE_REPORT.md) — Historical mapping of audit findings to initial code resolutions.
