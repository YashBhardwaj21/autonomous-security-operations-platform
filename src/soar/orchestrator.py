"""SOAR orchestrator — maps an attributed technique to a proposed response action,
then defers the auto-execute/approval decision to the blast-radius gate.

Actions are SIMULATED (no EDR/firewall/IdP integration exists — scoped honestly in
docs). The orchestrator PROPOSES; the gate DECIDES; on approval the caller re-checks
the gate for all tiers (REPORT.md H3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.soar.gate import BlastRadiusGate

# Technique -> recommended containment action (real ATT&CK-to-response mapping).
TECHNIQUE_ACTION: Dict[str, str] = {
    "T1003": "isolate_endpoint",     # credential dumping
    "T1078": "revoke_credential",    # valid accounts
    "T1110": "force_mfa_reauth",     # brute force
    "T1021": "isolate_endpoint",     # remote services / lateral
    "T1486": "isolate_endpoint",     # ransomware
    "T1059": "isolate_endpoint",     # command/scripting
    "T1055": "isolate_endpoint",     # process injection
    "T1105": "block_ip",             # ingress tool transfer
    "T1041": "block_ip",             # exfil over C2
}
DEFAULT_ACTION = "snapshot_vm"       # non-disruptive default (evidence preservation)


@dataclass
class ActionProposal:
    action: str
    technique: str
    auto_executed: bool
    status: str                  # "auto_executed" | "pending_approval"
    gate_reason: str
    response_mode: str


class Orchestrator:
    def __init__(self, gate: Optional[BlastRadiusGate] = None):
        self.gate = gate or BlastRadiusGate()

    def propose(self, technique: str, calibrated_confidence: float,
                asset_tier: int, reachable_count: Optional[int] = None,
                critical_reachable: Optional[int] = None) -> ActionProposal:
        action = TECHNIQUE_ACTION.get(technique, DEFAULT_ACTION)
        result = self.gate.evaluate(action, asset_tier, calibrated_confidence,
                                    reachable_count, critical_reachable)
        return ActionProposal(
            action=action, technique=technique, auto_executed=result.ok,
            status="auto_executed" if result.ok else "pending_approval",
            gate_reason=result.reason, response_mode=result.response_mode,
        )

    def approve(self, proposal: ActionProposal, asset_tier: int,
                calibrated_confidence: float, reachable_count: Optional[int] = None,
                critical_reachable: Optional[int] = None) -> bool:
        """Re-check the gate on approval for ALL tiers before executing (H3)."""
        result = self.gate.recheck(proposal.action, asset_tier, calibrated_confidence,
                                   reachable_count, critical_reachable)
        return result.ok
