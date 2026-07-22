# Product Requirements — Autonomous Security Operations Platform

> Restored to REAL current-state (these docs were deleted from both source repos'
> git history). Every requirement notes its implementation status with a file path.

## Problem
Public-sector/critical-infrastructure SOCs detect breaches weeks-to-months late.
APTs operate low-and-slow to evade signatures. We provide a behavioural-intelligence
layer on top of the SIEM that compresses compromise→detection→response.

## Capabilities & status
| PS item | Status | Where |
|---|---|---|
| B1/T2 Behavioural anomaly detection (UEBA) | Real online engine; benign baseline pending real corpus | `src/ueba/engine.py`; loaders `src/ingestion/benign.py` |
| B2/T1 ATT&CK attribution (classifier-first, deliberate deviation) | Calibrated scaffold; **operator trains** on real OTRF | `src/attribution/*`, `scripts/train_attribution.py` |
| B2 Next-step prediction | Real, data-derived transition matrix | `src/prediction/*`, `models/transition_matrix.json` |
| B3/T6 Autonomous response w/ blast-radius gates | Real gate + response modes; connectors simulated | `src/soar/*` |
| B4 Vulnerability prioritisation | EPSS-aware scorer; operator supplies real CVE feed | `src/vuln/scorer.py` |
| B5 Digital Twin | Real Dijkstra sim; **real topology required** | `src/twin/simulator.py` |
| T3 Graph AI | Classical graph traversal; **no GNN** (honest) | `src/twin`, `src/graph` |
| T4 RAG over threat-intel | Real embedding retrieval; operator supplies advisories | `src/retrieval/*` |
| T5 Knowledge graph (ATT&CK) | STIX loader + curated mapping | `src/ml/attck_loader.py` |

## Non-goals (explicit)
Agentic-AI security-critical path (deterministic by design); GNN-from-scratch; live
OT/ICS telemetry (OT exists only as twin topology — no OT telemetry ingested); real
EDR/firewall/IdP execution (simulated); multi-week live capture.

## §7.2 note
The retrieval layer NEVER fabricates advisory identifiers. Absent a real advisory
corpus it returns none (honest), never invented CIAD/Sigma IDs (was REPORT.md H7).
