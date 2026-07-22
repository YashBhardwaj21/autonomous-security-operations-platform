"""Cross-boundary integration — the chain that never existed in one place (H12).

Uses real parsing/session/feature/UEBA/gate code end to end. The attribution model
is intentionally absent (untrained), so attribution reports 'model_unavailable' —
proving the pipeline degrades honestly (no fabricated technique/confidence, C3).
Event inputs are Sysmon-shaped fixtures (test inputs, isolated folder), not app data.
"""
import os

from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app, set_user_provider
from src.api.auth import InMemoryUserProvider, issue_access_token
from src.api.pipeline import IncidentPipeline
from src.config.settings import Settings, get_settings
from tests._fixtures import sysmon_events as fx


def test_pipeline_runs_end_to_end_and_degrades_honestly():
    pipe = IncidentPipeline()
    out = pipe.process_events(fx.TWO_HOST_BURST, scenario_id="demo", asset_tier=3)
    assert out["parsed_events"] == 4
    assert out["incidents"]                       # one per (host,logon) session, not global
    hosts = {i["host"] for i in out["incidents"]}
    assert hosts == {"HOSTA", "HOSTB"}
    # attribution model not trained -> honest model_unavailable, no response proposed
    for inc in out["incidents"]:
        assert inc["attribution"]["status"] == "model_unavailable"
        assert inc["attribution"]["confidence"] is None
        assert inc["response"] is None


def test_ingest_requires_authentication(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-a-default")
    get_settings.cache_clear()
    client = TestClient(app)
    # no token -> 403 (HTTPBearer) ; C4: ingest is not anonymous
    r = client.post("/ingest/events", json={"events": []})
    assert r.status_code in (401, 403)


def test_ingest_with_valid_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-a-default")
    get_settings.cache_clear()
    provider = InMemoryUserProvider({
        "analyst": {"username": "analyst",
                    "password_hash": InMemoryUserProvider.hash_password("pw"),
                    "roles": ["soc_analyst"]}})
    set_user_provider(provider)
    client = TestClient(app)

    tok = client.post("/auth/token", json={"username": "analyst", "password": "pw"})
    assert tok.status_code == 200
    token = tok.json()["access_token"]

    r = client.post("/ingest/events",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"events": fx.TWO_HOST_BURST, "scenario_id": "demo"})
    assert r.status_code == 200
    assert r.json()["parsed_events"] == 4


def test_wrong_password_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-a-default")
    get_settings.cache_clear()
    set_user_provider(InMemoryUserProvider({
        "analyst": {"username": "analyst",
                    "password_hash": InMemoryUserProvider.hash_password("right"),
                    "roles": []}}))
    client = TestClient(app)
    r = client.post("/auth/token", json={"username": "analyst", "password": "wrong"})
    assert r.status_code == 401
