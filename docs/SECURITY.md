# Security — Autonomous Security Operations Platform

## AuthN/Z (§5)
- JWT (HS256) via `python-jose`. **JWT_SECRET has no baked default** — the app refuses
  to run auth without it (was a hardcoded secret; REPORT.md M-5).
- Passwords are bcrypt-hashed and verified (`bcrypt` directly). No plaintext creds
  ship in the app; operators supply users (`InMemoryUserProvider`). Demo users live in
  `scripts/demo_seed.py`, OUTSIDE the app path.
- Access tokens carry `type="access"`; verification rejects any other token type
  (refresh tokens no longer accepted on protected paths).

## Ingestion
- `POST /ingest/events` REQUIRES a valid access token (was unauthenticated; REPORT.md C4).

## Response gate (§5)
- Fails closed; tier-0 crown-jewel assets never auto-execute; blast radius is real
  twin reachability, and absent reachability the gate requires approval (fail-safe),
  never substituting a fabricated user count (H2).

## Data integrity
- No synthetic data in the runtime path (`scripts/check_no_dummy_in_src.py` enforces).
- Threat-intel is real or absent — advisory identifiers are never fabricated (H7).
