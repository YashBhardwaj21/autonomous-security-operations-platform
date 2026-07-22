# Security Architecture & Threat Model — Autonomous Security Operations Platform

This document outlines authentication standards, ingestion access controls, data integrity safeguards, and fail-safe SOAR response boundaries.

---

## 1. Authentication & Access Control (`src/api/auth.py`)

* **JWT Secret Enforcement**: The application requires `JWT_SECRET` to be configured in the environment. Hardcoded fallback secrets are prohibited.
* **Credential Verification**: User authentication uses `bcrypt` password hashing (`bcrypt.checkpw`). Passwords are never stored or logged in plaintext.
* **Token Typing**: Access tokens carry `type="access"` claims. Token validation rejects refresh tokens or altered claims on protected endpoints.
* **Ingestion Endpoint Protection**: `POST /ingest/events` requires a valid HTTP Bearer token. Unauthenticated ingestion attempts are rejected with `401 Unauthorized`.

---

## 2. Telemetry Parsing & Data Integrity

* **No Synthetic Runtime Data**: Production parsing paths (`src/ingestion/`) operate strictly on valid event objects. Synthetic data generators exist only under `tests/_fixtures/` and `tests/harness_selftest/`. The build script `scripts/check_no_dummy_in_src.py` enforces this boundary.
* **Parser Drop Auditing**: Events missing valid timestamps or required fields are skipped and counted in `DropStats`. Timestamps are never silently fabricated.

---

## 3. Honest Degraded Runtime Mode

* **Missing Model Handling**: If `models/attribution.joblib` is absent, the attribution engine returns `status = "model_unavailable"`. The platform degrades gracefully without outputting fabricated confidence scores or technique labels.
* **Threat-Intel Evidence Isolation**: The retrieval layer (`src/retrieval/`) provides contextual threat intelligence documents to security analysts. Retrieval results are non-authoritative and **never influence automated SOAR gate decisions**.

---

## 4. SOAR Response Gate Boundaries (`src/soar/`)

* **Fail-Safe Gate Policy**: Automated response actions are governed by `BlastRadiusGate`.
* **Crown-Jewel Asset Protection**: Actions targeting **Tier-0 assets** (e.g., Domain Controllers, Critical Infrastructure Nodes) strictly require manual human approval (`manual_approval_required`).
* **Unverified Reachability**: If digital-twin reachability analysis cannot be evaluated (e.g., missing network topology links), the SOAR gate defaults to manual approval.
* **Non-Executing Safety**: SOAR output provides structured response recommendations (`ResponseProposal`). Direct execution against external EDR, Active Directory, or network firewalls is omitted in this build.
