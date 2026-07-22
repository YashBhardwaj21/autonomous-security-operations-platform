"""IncidentPipeline — the end-to-end chain that never existed in one place (H12).

ingest(raw events) -> parse -> per-entity sessions -> UEBA (own feature space) ->
attribution (calibrated loader; model_unavailable if untrained) -> SOAR gate
(response-mode + real twin blast radius) -> next-step prediction -> retrieval evidence.

This is the seam the two former repos never tested together. It is pure library
code (no web framework) so it is exercised by tests/integration directly.
No synthetic data: it consumes real parsed events; if the attribution model or
matrix or corpus is absent, those stages report unavailable — never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.canon.schema import SourceType
from src.features.pipeline import AttributionFeaturePipeline
from src.ingestion.parser import DropStats, ParserFactory
from src.sessions.session_builder import SessionBuilder
from src.attribution.loader import AttributionService
from src.prediction.engine import NextStepPredictor
from src.retrieval.service import RetrievalService
from src.soar.orchestrator import Orchestrator
from src.twin.simulator import DigitalTwinSimulator
from src.ueba.engine import UEBAEngine


@dataclass
class IncidentPipeline:
    factory: ParserFactory = field(default_factory=ParserFactory)
    sessions: SessionBuilder = None
    features: AttributionFeaturePipeline = field(default_factory=AttributionFeaturePipeline)
    ueba: UEBAEngine = field(default_factory=UEBAEngine)
    attribution: AttributionService = field(default_factory=AttributionService)
    predictor: NextStepPredictor = field(default_factory=NextStepPredictor)
    orchestrator: Orchestrator = field(default_factory=Orchestrator)
    twin: Optional[DigitalTwinSimulator] = None
    retrieval: Optional[RetrievalService] = None

    def __post_init__(self):
        if self.sessions is None:
            self.sessions = SessionBuilder(self.factory)

    def process_events(self, raw_events: List[Dict[str, Any]], scenario_id: Optional[str] = None,
                       source: SourceType = SourceType.OTRF,
                       asset_tier: int = 3, twin_start_node: Optional[str] = None) -> Dict[str, Any]:
        stats = DropStats()
        events = [ev for r in raw_events if (ev := self.factory.parse(r, source, stats))]
        incidents = []
        for sess in self.sessions.build_sessions(events, scenario_id=scenario_id):
            fv = self.features.extract(sess)
            fmap = dict(zip(fv.feature_names, fv.features))

            # UEBA on its OWN feature space (behavioural counts per session/host)
            ueba_features = {
                "event_count": float(len(sess.events)),
                "process_access": fmap.get("process_access_count", 0.0),
                "registry_mod": fmap.get("registry_mod_count", 0.0),
                "failed_login": fmap.get("failed_login_count", 0.0),
                "network_flow": fmap.get("network_flow_count", 0.0),
            }
            anomaly = self.ueba.process(sess.host or "unknown", ueba_features)

            attr = self.attribution.predict(fmap)  # calibrated OR model_unavailable

            proposal = None
            prediction = []
            evidence = None
            if attr.status == "ok":
                reach = None
                crit = None
                if self.twin and twin_start_node:
                    br = self.twin.blast_radius(twin_start_node)
                    reach, crit = br["reachable_count"], br["critical_reachable"]
                proposal = self.orchestrator.propose(
                    attr.technique, attr.confidence or 0.0, asset_tier, reach, crit)
                prediction = self.predictor.predict_next(attr.technique)
                if self.retrieval:
                    evidence = self.retrieval.retrieve(attr.technique)

            incidents.append({
                "host": sess.host,
                "logon_id": sess.logon_id,
                "anomaly": {"is_anomalous": anomaly.is_anomalous, "score": anomaly.score,
                            "contributing_features": anomaly.contributing_features},
                "attribution": {"status": attr.status, "technique": attr.technique,
                                "confidence": attr.confidence, "top_k": attr.top_k},
                "response": None if proposal is None else {
                    "action": proposal.action, "status": proposal.status,
                    "gate_reason": proposal.gate_reason, "response_mode": proposal.response_mode},
                "predicted_next": prediction,
                "evidence_available": bool(evidence and evidence.available),
            })
        return {"parsed_events": len(events), "dropped": stats.total(),
                "incidents": incidents}
