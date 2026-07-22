# Hackathon Execution Runbook

This runbook outlines the step-by-step procedure for dataset acquisition, preflight verification, data auditing, model fitting, and live API demonstration.

---

## Step 0: Preflight Verification & Integrity Check

Before performing data acquisition or model training, verify local environment integrity:

```powershell
# 1. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Confirm no test fixtures are imported in production code
python scripts/check_no_dummy_in_src.py

# 3. Execute unit test suite and log test count and timestamp
python -m pytest tests -q --basetemp data\pytest_tmp
```

> **CRITICAL PREFLIGHT BLOCKER**:
> `src/ingestion/otrf.py` currently contains temporary debug instrumentation (`print(...)` and a `break` statement inside `for sess in builder.build_sessions(...)`). This causes `build_dataset()` to return after inspecting the first session, yielding zero appended feature rows.
> **Action Required**: You must clean up the debug `print`/`break` block in `src/ingestion/otrf.py` prior to dataset construction or training.

---

## Step 1: Telemetry Data Acquisition

Acquire real OTRF (Security-Datasets) telemetry for training:

```powershell
# Download scenario metadata YAMLs and attack mappings
python scripts/fetch_otrf_metadata.py

# Display parameters for downloading host event zips
python scripts/fetch_otrf_events.py --help

# Generate the data-derived ATT&CK transition matrix
python scripts/build_transition_matrix.py
```

### Target Scenario Diversity Requirements
- **Total Scenarios**: Target a minimum of **20–30 distinct Windows host scenarios** under `data/raw/Security-Datasets/`.
- **Class Balance**: Ensure each evaluated ATT&CK technique is represented across at least **5 distinct scenarios** to support Leave-One-Scenario-Out (LOSO) cross-validation and calibration.

---

## Step 2: Dataset Auditing (Do Not Train Until Audited)

Never initiate model training without verifying dataset properties. Run an explicit audit over the extracted session feature space:

```powershell
# Run feature statistics, drop rates, and duplicate analysis
python scripts/check_zio_content.py
```

### Mandatory Audit Checks:
1. **Parser Drop Statistics**: Review `DropStats` (e.g., `no_parser`, `no_timestamp`). Inspect dropped Windows Event IDs before expanding parser coverage.
2. **Session Vector Duplicates**: Identify duplicate session feature vectors across different scenarios. Understand whether duplicates represent benign background activity or weak label assignment.

---

## Step 3: Label Contract & Formulation Setup

Understand the current label assignment contract in the repository:

* **Weak Supervision Broadcast**: `label_from_metadata()` assigns scenario-level ATT&CK techniques to *every* session window extracted from that scenario.
* **Single-Label Projection (`y_primary`)**: `src/ingestion/otrf.py` constructs `y_primary` by selecting `sorted(scen_label.techniques)[0]` (alphabetically first technique). `scripts/train_attribution.py` trains against `y_primary`.
* **Session vs. Scenario Support Bug**: `build_label_space()` currently counts total *sessions* per technique rather than distinct *scenarios*. Ensure class support filtering is evaluated against unique `scenario_id` counts.

---

## Step 4: Model Evaluation & Artifact Fitting

Evaluate model performance using grouped Leave-One-Scenario-Out (LOSO) cross-validation:

```powershell
# Run nested LOSO evaluation without persisting artifacts
python scripts/train_attribution.py --train

# Run LOSO evaluation AND fit/persist final model artifact
python scripts/train_attribution.py --train --fit-final
```

> **Evaluation Invariant**: Evaluation must compare model performance against `majority-class` and `uniform-random` baselines. Ensure feature hygiene (`HygieneTransform`) is fitted only on training folds during evaluation to prevent data leakage.

---

## Step 5: Live API Integration & Demonstration

Demonstrate API ingestion and graceful degraded-mode handling:

```powershell
# 1. Set secret key and launch API server
$env:JWT_SECRET="hackathon-production-secret-key-12345"
uvicorn src.api.app:app --port 8000
```

### Testing API Endpoints

#### Liveness & Model Status Check:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
```
*Expected response when model is not fitted*:
`{"status": "ok", "model_loaded": false, ...}`

#### Authentication Token Request:
```powershell
$body = @{ username = "analyst"; password = "password123" }
$auth = Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/token" -Method POST -Body $body
$token = $auth.access_token
```

#### Batch Event Ingest:
```powershell
$headers = @{ Authorization = "Bearer $token" }
$payload = @{
    events = @(
        @{ EventID = 1; UtcTime = "2026-01-01T12:00:00Z"; Hostname = "HOST-01"; Image = "C:\Windows\System32\cmd.exe" }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/ingest/events" -Method POST -Headers $headers -Body $payload -ContentType "application/json"
```

---

## Step 6: Evidence Package Checklist

For hackathon submissions, compile an evidence package containing:

1. **Dataset Manifest**: List of OTRF scenario IDs, raw archive hashes, event counts, and `DropStats`.
2. **Audit Output**: Duplicate vector frequency reports and event ID drop statistics.
3. **Evaluation Report**: Macro-F1, Precision, Recall, Top-3 Accuracy, ECE, Brier Score, and baseline comparisons.
4. **Reliability & Calibration Evidence**: Expected Calibration Error (ECE) plots.
5. **API Response Log**: JSON response demonstrating `POST /ingest/events` pipeline execution.
