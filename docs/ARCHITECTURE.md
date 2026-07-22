# Architecture — Autonomous Security Operations Platform

## Data flow (per-event)
```
raw telemetry -> ingestion/parser (canonical events; drops counted, never faked)
             -> sessions (entity + logon keyed; never merges hosts)
             -> features (attribution space)  ─┐
             -> ueba (separate feature space)  │ M6: spaces are independent
attribution (calibrated) -> soar gate (response mode + REAL twin blast radius)
             -> prediction (data-derived transitions) -> retrieval (non-gating)
```
Implemented as `src/api/pipeline.IncidentPipeline` — the cross-boundary chain that
never existed in one place before the merge (REPORT.md H12).

## §6.7 Retrieval
Embedding index over a REAL corpus (ATT&CK STIX + operator advisories/CVEs). Runs
AFTER attribution, is non-authoritative, and holds no gate/model handle — it cannot
influence any decision (asserted in tests). ADR-001: retrieval is non-gating.

## §6.11 Digital Twin
Static asset-topology graph (networkx), rebuilt on inventory change — **NOT** mutated
per event (the SOC-workflow "every event updates the twin" framing was wrong;
REPORT.md contradiction #1). `build()` requires REAL edges; fabricated topology was
removed (H10). Attack-path simulation is Dijkstra over ATT&CK technique costs.
The entity-relationship graph (`src/graph`) is a SEPARATE, session-scoped analytic
structure, not the twin.

## §8 SOAR blast-radius gate (ADR-002)
Deterministic, fails closed. `auto_execute = mode-policy AND calibrated_confidence >=
threshold AND real_reachability_impact <= tier_limit`. Tier-0 never auto-executes.
Response mode (MANUAL/SEMI/FULL) is org policy (REPORT.md W1). The executor re-checks
the gate for ALL tiers on approval (H3).
