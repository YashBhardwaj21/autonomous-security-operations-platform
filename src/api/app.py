"""FastAPI application — rewired to consume src.* (single repo), ingest AUTHENTICATED.
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


class TopologyRequest(BaseModel):
    assets: List[Dict]
    edges: List[Dict]


class VulnItemRequest(BaseModel):
    cve: str
    cvss: float
    epss: float
    asset_criticality_tier: int = 3
    attack_path_exposure: float = 0.0
    ttp_overlap: float = 0.0


class VulnInventoryRequest(BaseModel):
    items: List[VulnItemRequest]


_vuln_inventory: List[VulnItemRequest] = []


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


@app.post("/twin/topology")
def update_topology(req: TopologyRequest, user: Dict = Depends(current_user)):
    normalized_assets = []
    asset_ids = set()
    for a in req.assets:
        if not isinstance(a, dict):
            continue
        aid = a.get("asset_id") or a.get("id")
        if not aid:
            raise HTTPException(400, "Asset missing required 'asset_id' or 'id'")
        asset_copy = dict(a)
        asset_copy["asset_id"] = str(aid)
        normalized_assets.append(asset_copy)
        asset_ids.add(str(aid))

    normalized_edges = []
    for e in req.edges:
        if not isinstance(e, dict):
            continue
        fa = e.get("from_asset") or e.get("source_id") or e.get("from") or e.get("source")
        ta = e.get("to_asset") or e.get("target_id") or e.get("to") or e.get("target")
        if not fa or not ta or str(fa) not in asset_ids or str(ta) not in asset_ids:
            raise HTTPException(400, f"Edge from '{fa}' to '{ta}' references unknown asset")
        edge_copy = dict(e)
        edge_copy["from_asset"] = str(fa)
        edge_copy["to_asset"] = str(ta)
        normalized_edges.append(edge_copy)

    try:
        if _pipeline.twin is None:
            from src.twin.simulator import DigitalTwinSimulator
            _pipeline.twin = DigitalTwinSimulator()
        _pipeline.twin.build(normalized_assets, normalized_edges)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    return {"status": "ok", "nodes": len(normalized_assets), "edges": len(normalized_edges)}



@app.post("/vuln/inventory")
def update_vuln_inventory(req: VulnInventoryRequest, user: Dict = Depends(current_user)):
    global _vuln_inventory
    _vuln_inventory = req.items
    return {"status": "ok", "total_items": len(_vuln_inventory)}


@app.get("/vuln/remediation-queue")
def get_remediation_queue(user: Dict = Depends(current_user)):
    from src.vuln.scorer import score_vulnerability
    scored = []
    for item in _vuln_inventory:
        score = score_vulnerability(
            cve=item.cve, cvss=item.cvss, epss=item.epss,
            asset_criticality_tier=item.asset_criticality_tier,
            attack_path_exposure=item.attack_path_exposure,
            ttp_overlap=item.ttp_overlap,
        )
        scored.append({
            "cve": score.cve,
            "risk_score": score.risk,
            "components": score.components,
            "tier": item.asset_criticality_tier,
        })
    scored.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"remediation_queue": scored}

