"""FastAPI application — rewired to consume src.* (single repo), ingest AUTHENTICATED.

REPORT.md fixes wired here:
* C4 — POST /ingest/events REQUIRES a valid access token (was unauthenticated).
* H9 — incidents are keyed per (host, logon) session by the pipeline, not collapsed
  into one global incident.
* W1 — the SOAR response-mode policy governs auto-execute vs approval via the gate.

This is a thin transport layer over src/api/pipeline.py (the real chain). A DB and
full incident store are intentionally out of scope for this pass (documented);
the pipeline returns incidents in-memory.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.api.auth import authenticate, current_user, issue_access_token, InMemoryUserProvider
from src.api.pipeline import IncidentPipeline
from src.canon.schema import SourceType

app = FastAPI(title="Autonomous Security Operations Platform", version="1.0")

# Operator supplies real users (bcrypt hashes) at startup; empty by default so no
# demo credentials ship in the app (REPORT.md M-5). See scripts/demo_seed.py.
_user_provider = InMemoryUserProvider(users={})
_pipeline = IncidentPipeline()


def set_user_provider(provider: InMemoryUserProvider) -> None:
    global _user_provider
    _user_provider = provider


class TokenRequest(BaseModel):
    username: str
    password: str


class IngestRequest(BaseModel):
    events: List[Dict]
    scenario_id: Optional[str] = None
    source: str = "OTRF"
    asset_tier: int = 3
    twin_start_node: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "attribution_available": _pipeline.attribution.available,
            "prediction_available": _pipeline.predictor.available}


@app.post("/auth/token")
def token(req: TokenRequest):
    user = authenticate(_user_provider, req.username, req.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": issue_access_token(user["username"], user.get("roles", [])),
            "token_type": "bearer"}


@app.post("/ingest/events")
def ingest(req: IngestRequest, user: Dict = Depends(current_user)):
    try:
        source = SourceType(req.source)
    except ValueError:
        raise HTTPException(400, f"Unknown source {req.source}")
    return _pipeline.process_events(
        req.events, scenario_id=req.scenario_id, source=source,
        asset_tier=req.asset_tier, twin_start_node=req.twin_start_node)
