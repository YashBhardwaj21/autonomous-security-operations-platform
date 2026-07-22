"""SOAR blast-radius gate — ported from etbackend and hardened.

Preserved (it was genuinely good): deterministic, fails closed, tier-0 never
auto-executes, unit-tested.

Hardened per REPORT.md:
* H1 — consumes a CALIBRATED confidence (produced by the calibrated attribution
  model; the gate does not know or care, but docs/callers must pass the calibrated
  value, never raw predict_proba).
* H2 — blast radius is REAL reachability from the digital twin (reachable_count,
  critical_reachable), not `user_count x static multiplier`. If no twin reachability
  is supplied, the impact defaults to maximum (fails safe), never a fake user_count.
* W1 — org-level ResponseMode policy: MANUAL forces approval always; SEMI is the
  tiered behaviour; FULL auto-executes up to soar_auto_execute_max_tier (tier-0 still
  never auto-executes).
* H3 — re-check helper `recheck()` is enforced for ALL tiers by callers on approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.config.settings import ResponseMode, Settings, get_settings

ACTION_BASE_IMPACT: Dict[str, float] = {
    "isolate_endpoint": 0.4, "revoke_credential": 0.3, "block_ip": 0.5,
    "snapshot_vm": 0.05, "force_mfa_reauth": 0.15, "restore_endpoint_network": 0.1,
    "reinstate_credential": 0.1, "unblock_ip": 0.2,
}
TIER_AUTO_EXECUTE_LIMITS: Dict[int, float] = {0: 0.0, 1: 0.2, 2: 0.5, 3: 1.0}


@dataclass
class BlastRadiusResult:
    ok: bool
    tier: str
    estimated_impact: float
    reason: str
    response_mode: str


class BlastRadiusGate:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()

    def evaluate(self, action: str, target_criticality_tier: int,
                 calibrated_confidence: float,
                 reachable_count: Optional[int] = None,
                 critical_reachable: Optional[int] = None) -> BlastRadiusResult:
        try:
            return self._inner(action, target_criticality_tier, calibrated_confidence,
                               reachable_count, critical_reachable)
        except Exception as e:  # fail closed
            return BlastRadiusResult(False, f"tier-{target_criticality_tier}", 1.0,
                                     f"Gate error — failing closed: {e}",
                                     self._settings.response_mode.value)

    def _inner(self, action: str, tier: int, confidence: float,
               reachable_count: Optional[int], critical_reachable: Optional[int]) -> BlastRadiusResult:
        mode = self._settings.response_mode
        tier = max(0, min(3, tier))
        tier_str = f"tier-{tier}"

        # Tier-0: never auto-execute, in any mode.
        if tier == 0:
            return BlastRadiusResult(False, tier_str, 1.0,
                                     "Tier-0 crown-jewel: human approval always required", mode.value)

        # MANUAL mode: everything requires approval.
        if mode == ResponseMode.MANUAL:
            return BlastRadiusResult(False, tier_str, 1.0,
                                     "Response mode MANUAL: all actions require approval", mode.value)

        # Confidence gate (calibrated).
        if confidence < self._settings.soar_confidence_threshold:
            return BlastRadiusResult(False, tier_str, 1.0,
                                     f"Calibrated confidence {confidence:.2f} < "
                                     f"{self._settings.soar_confidence_threshold}", mode.value)

        # FULL mode: auto-execute up to the configured max tier; beyond it, approval.
        if mode == ResponseMode.FULL:
            if tier <= self._settings.soar_auto_execute_max_tier:
                return BlastRadiusResult(True, tier_str, 0.0,
                                         f"Response mode FULL: auto-execute allowed for {tier_str}", mode.value)
            return BlastRadiusResult(False, tier_str, 1.0,
                                     f"Response mode FULL: {tier_str} exceeds max auto-execute tier "
                                     f"{self._settings.soar_auto_execute_max_tier}", mode.value)

        # SEMI mode: real reachability-based impact vs tier limit.
        # H2 fail-safe: with NO twin reachability, require approval regardless of tier
        # (never substitute a fake user_count and never auto-execute blind).
        if reachable_count is None:
            return BlastRadiusResult(False, tier_str, 1.0,
                                     "no twin reachability supplied — approval required (fail-safe)",
                                     mode.value)
        base = ACTION_BASE_IMPACT.get(action, 0.5)
        reach_factor = min(1.0, reachable_count / 50.0) if reachable_count > 0 else 0.1
        crit_boost = 1.0 if (critical_reachable or 0) > 0 else 0.0
        tier_mult = [1.5, 1.2, 0.8, 0.5][tier]
        impact = min(1.0, max(base * reach_factor * tier_mult, crit_boost))
        reason_radius = f"reachable={reachable_count} critical_reachable={critical_reachable or 0}"

        limit = TIER_AUTO_EXECUTE_LIMITS[tier]
        ok = impact <= limit
        reason = (f"{'auto-execute' if ok else 'approval required'}: impact {impact:.2f} "
                  f"{'<=' if ok else '>'} {tier_str} limit {limit:.2f} ({reason_radius})")
        return BlastRadiusResult(ok, tier_str, round(impact, 3), reason, mode.value)

    # H3: callers MUST re-run this on approval, for ALL tiers, before execution.
    def recheck(self, *args, **kwargs) -> BlastRadiusResult:
        return self.evaluate(*args, **kwargs)
