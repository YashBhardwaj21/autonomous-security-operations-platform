# Product Requirements Document (PRD) — Autonomous Security Operations Platform

---

## 1. Problem Statement & Objective

Modern Security Operations Centers (SOCs) face overwhelming telemetry volumes, high false-positive rates, and prolonged adversary dwell times. Advanced Persistent Threats (APTs) execute low-and-slow tactics to bypass static signature rules.

The **Autonomous Security Operations Platform** provides a behavioral-intelligence layer above traditional SIEM telemetry to automate event parsing, sessionisation, ATT&CK technique attribution, next-step prediction, threat-intel retrieval, and blast-radius-aware response gating.

---

## 2. Feature & Capability Status Matrix

| Capability ID | Requirement Description | Implementation Status | Location |
|---|---|---|---|
| **CAP-01** | Canonical Event Parsing & Drop Tracking | **Implemented** | `src/ingestion/parser.py` |
| **CAP-02** | Entity- and Logon-Keyed Session Builder | **Implemented** | `src/sessions/session_builder.py` |
| **CAP-03** | Tabular Attribution Feature Extraction | **Implemented** | `src/features/pipeline.py` |
| **CAP-04** | Calibrated ATT&CK Attribution Model | **In-Progress (Scaffold)** | `src/attribution/model.py`, `scripts/train_attribution.py` |
| **CAP-05** | Online UEBA Anomaly Engine | **Implemented** | `src/ueba/engine.py` |
| **CAP-06** | Data-Derived Next-Step Prediction | **Implemented** | `src/prediction/transition.py` |
| **CAP-07** | Threat-Intel Embedding Retrieval | **Implemented** | `src/retrieval/retrieval_engine.py` |
| **CAP-08** | Digital Twin Dijkstra Blast-Radius Sim | **Implemented** | `src/twin/simulator.py` |
| **CAP-09** | Deterministic SOAR Blast-Radius Gate | **Implemented** | `src/soar/gate.py` |
| **CAP-10** | GraphSAGE GNN Activity Classification | **Archived (Experimental)** | `experiments/gnn/`, `archive/gnn-2026-07/` |

---

## 3. Explicit Non-Goals & Out-of-Scope Items

* **Live EDR / Firewall Remediation**: The platform generates response proposals; active execution against live active directory or security appliances is out of scope.
* **Production Model Performance Claims**: Pre-packaged trained weights (`attribution.joblib`) are omitted. Models must be trained and audited by operators using real telemetry archives.
* **OT / ICS Network Telemetry Ingestion**: Operational Technology assets are represented as digital twin topology nodes; raw OT protocol telemetry is not ingested.
* **Synthetic Data Generation in Production**: Dummy data is strictly prohibited from runtime modules.
* **Presenting GNN as Primary Attribution**: GraphSAGE is retired to an experimental baseline due to scenario-identity mapping limitations (macro-F1 = 0.075).
