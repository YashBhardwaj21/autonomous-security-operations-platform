import pytest
from fastapi.testclient import TestClient

from src.api.app import app, set_user_provider
from src.api.auth import InMemoryUserProvider
from src.config.settings import get_settings


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-a-default")
    get_settings.cache_clear()
    provider = InMemoryUserProvider(users={
        "admin": {
            "username": "admin",
            "password_hash": InMemoryUserProvider.hash_password("secret123"),
            "roles": ["admin"]
        }
    })
    set_user_provider(provider)
    client = TestClient(app)
    # Get bearer token
    res = client.post("/auth/token", json={"username": "admin", "password": "secret123"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "attribution_available" in body
    assert "prediction_available" in body


def test_authenticated_topology_import(auth_client):
    payload = {
        "assets": [
            {"asset_id": "srv1", "name": "Web Server", "type": "server", "criticality_tier": 2},
            {"asset_id": "db1", "name": "Database Server", "type": "database", "criticality_tier": 0}
        ],
        "edges": [
            {"from_asset": "srv1", "to_asset": "db1"}
        ]
    }
    res = auth_client.post("/twin/topology", json=payload)
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "nodes": 2, "edges": 1}


def test_topology_invalid_edge_rejected(auth_client):
    payload = {
        "assets": [{"asset_id": "srv1"}],
        "edges": [{"from_asset": "srv1", "to_asset": "unknown_asset"}]
    }
    res = auth_client.post("/twin/topology", json=payload)
    assert res.status_code == 400
    assert "unknown asset" in res.json()["detail"].lower()


def test_vulnerability_inventory_and_queue(auth_client):
    inv_payload = {
        "items": [
            {"cve": "CVE-2024-0001", "cvss": 7.5, "epss": 0.2, "asset_criticality_tier": 3},
            {"cve": "CVE-2024-0002", "cvss": 9.8, "epss": 0.9, "asset_criticality_tier": 0}
        ]
    }
    res = auth_client.post("/vuln/inventory", json=inv_payload)
    assert res.status_code == 200
    assert res.json()["total_items"] == 2

    res_q = auth_client.get("/vuln/remediation-queue")
    assert res_q.status_code == 200
    queue = res_q.json()["remediation_queue"]
    assert len(queue) == 2
    # Critical tier 0 with high CVSS/EPSS should be ranked first
    assert queue[0]["cve"] == "CVE-2024-0002"
