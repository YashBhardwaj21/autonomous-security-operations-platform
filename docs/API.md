# API Specification — Autonomous Security Operations Platform

Base application: FastAPI app defined in `src/api/app.py`.  
Launch server: `$env:JWT_SECRET="your-secret-key"; uvicorn src.api.app:app --port 8000`

---

## Verified Endpoints

| Method | Path | Authentication | Description |
|---|---|---|---|
| **GET** | `/health` | None | Returns liveness status, model load status (`model_loaded`), and transition matrix availability. |
| **POST** | `/auth/token` | None | Authenticates username/password (bcrypt) and returns a JWT access token. |
| **POST** | `/ingest/events` | Bearer Token | Ingests raw event batches, runs `IncidentPipeline`, and returns sessionised incidents with UEBA, attribution, predictions, and SOAR response proposals. |

---

## Endpoint Details

### 1. `GET /health`
Returns system status diagnostics.

#### Example Response:
```json
{
  "status": "ok",
  "model_loaded": false,
  "transition_matrix_loaded": true
}
```

---

### 2. `POST /auth/token`
Body format: Form data (`username`, `password`). Verifies credentials against configured user provider.

#### Example Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### 3. `POST /ingest/events`
Requires header `Authorization: Bearer <access_token>`.

#### Example Request:
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

#### Example Response:
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

## Degraded Runtime & Boundary Notes

* **Model Availability (`model_unavailable`)**: If `models/attribution.joblib` is missing, `attribution.status` returns `"model_unavailable"`. The pipeline degrades gracefully without producing fabricated probabilities.
* **Non-Executing SOAR Gate**: The API evaluates response rules and returns response proposals. Active network/EDR execution is simulated; no external EDR or firewall API calls are executed.
