# API Specification — Autonomous Security Operations Platform

Base application: FastAPI app defined in `src/api/app.py`.  
Launch server: `$env:JWT_SECRET="your-secret-key"; uvicorn src.api.app:app --port 8000`

---

## Verified Endpoints

| Method | Path | Authentication | Description |
|---|---|---|---|
| **GET** | `/health` | None | Returns liveness status, attribution status (`attribution_available`), and transition matrix status (`prediction_available`). |
| **POST** | `/auth/token` | None | Authenticates username/password (JSON payload) and returns a JWT access token. |
| **POST** | `/ingest/events` | Bearer Token | Ingests raw event batches, runs `IncidentPipeline`, and returns sessionised incidents. |
| **POST** | `/twin/topology` | Bearer Token | Ingests asset nodes and network reachability edges for digital twin path analysis. |
| **POST** | `/vuln/inventory` | Bearer Token | Ingests vulnerability scan data (CVE, CVSS, EPSS, asset tier). |
| **GET** | `/vuln/remediation-queue` | Bearer Token | Returns risk-prioritized vulnerability remediation queue scored by `src/vuln/scorer.py`. |

---

## Endpoint Specifications

### 1. `GET /health`
Returns system component availability.

```json
{
  "status": "ok",
  "attribution_available": false,
  "prediction_available": true
}
```

---

### 2. `POST /auth/token`
Accepts JSON payload: `{ "username": "admin", "password": "yourpassword" }`.

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### 3. `POST /ingest/events`
Requires header `Authorization: Bearer <access_token>`.

#### Request JSON Body:
```json
{
  "events": [
    {
      "EventID": 1,
      "UtcTime": "2026-01-01T12:00:00.000Z",
      "Hostname": "WORKSTATION-01",
      "Image": "C:\\Windows\\System32\\cmd.exe",
      "CommandLine": "cmd.exe /c whoami"
    }
  ],
  "scenario_id": "DEMO-001",
  "asset_tier": 2
}
```

#### Response:
```json
{
  "parsed_events": 1,
  "dropped_events": 0,
  "incidents": [
    {
      "host": "WORKSTATION-01",
      "logon_id": "0x3e7",
      "anomaly": {
        "score": 0.42,
        "is_anomalous": false
      },
      "attribution": {
        "status": "model_unavailable",
        "technique": null,
        "confidence": null,
        "top_k": []
      },
      "predicted_next": [],
      "response": null,
      "evidence_available": false
    }
  ]
}
```

---

### 4. `POST /twin/topology`
Requires header `Authorization: Bearer <access_token>`.

#### Request JSON Body:
```json
{
  "assets": [
    {"asset_id": "web_srv", "name": "Web Gateway", "type": "server", "criticality_tier": 2},
    {"asset_id": "dc01", "name": "Domain Controller", "type": "server", "criticality_tier": 0}
  ],
  "edges": [
    {"from_asset": "web_srv", "to_asset": "dc01"}
  ]
}
```

---

### 5. `POST /vuln/inventory` & `GET /vuln/remediation-queue`
Requires header `Authorization: Bearer <access_token>`.

#### Ingest Inventory (`POST /vuln/inventory`):
```json
{
  "items": [
    {
      "cve": "CVE-2024-21413",
      "cvss": 9.8,
      "epss": 0.75,
      "asset_criticality_tier": 0,
      "attack_path_exposure": 0.5,
      "ttp_overlap": 0.2
    }
  ]
}
```

#### Retrieve Remediation Queue (`GET /vuln/remediation-queue`):
```json
{
  "remediation_queue": [
    {
      "cve": "CVE-2024-21413",
      "risk_score": 0.7425,
      "components": {
        "epss": 0.75,
        "cvss": 0.98,
        "asset_criticality": 1.0,
        "attack_path_exposure": 0.5,
        "ttp_overlap": 0.2
      },
      "tier": 0
    }
  ]
}
```
