# API — Autonomous Security Operations Platform

Base: FastAPI app `src/api/app.py`. Run: `JWT_SECRET=... uvicorn src.api.app:app`.

| Method | Path | Auth | Description |
|---|---|---|---|
| GET  | `/health` | none | liveness + model/matrix availability |
| POST | `/auth/token` | none | issue access token (bcrypt-verified) |
| POST | `/ingest/events` | **bearer** | run the incident pipeline over a batch of canonical events |

### POST /ingest/events (API-5)
Body: `{ events: [ {EventID, UtcTime, Hostname, ...} ], scenario_id?, source="OTRF",
asset_tier=3, twin_start_node? }`. Returns `{ parsed_events, dropped, incidents: [
{ host, logon_id, anomaly, attribution{status|technique|confidence|top_k},
response{action,status,gate_reason,response_mode}|null, predicted_next, evidence_available } ] }`.

Notes: if the attribution model is untrained, `attribution.status="model_unavailable"`
and no response is proposed — the pipeline degrades honestly (no fabricated technique
or confidence; REPORT.md C3). Incidents are per (host, logon) session, not one global
incident (H9).

A DB-backed incident store, WebSocket stream, and full RBAC surface are out of scope
for this pass (documented in BUILD_LOG.md), pending the frontend phase.
