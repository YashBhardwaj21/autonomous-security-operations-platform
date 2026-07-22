from src.config.settings import ResponseMode, Settings
from src.soar.gate import BlastRadiusGate


def _gate(mode=ResponseMode.SEMI, threshold=0.85, max_tier=3):
    return BlastRadiusGate(Settings(response_mode=mode, soar_confidence_threshold=threshold,
                                    soar_auto_execute_max_tier=max_tier))


def test_tier0_never_auto_executes_in_any_mode():
    for mode in (ResponseMode.MANUAL, ResponseMode.SEMI, ResponseMode.FULL):
        r = _gate(mode).evaluate("snapshot_vm", 0, 0.99, reachable_count=0, critical_reachable=0)
        assert r.ok is False


def test_manual_mode_forces_approval():
    r = _gate(ResponseMode.MANUAL).evaluate("snapshot_vm", 3, 0.99, reachable_count=0)
    assert r.ok is False and "MANUAL" in r.reason


def test_low_calibrated_confidence_blocks():
    r = _gate(ResponseMode.SEMI, threshold=0.85).evaluate("snapshot_vm", 3, 0.50, reachable_count=0)
    assert r.ok is False and "confidence" in r.reason.lower()


def test_semi_mode_missing_twin_reachability_fails_safe():
    # H2: no fake user_count — absent reachability maxes impact -> approval required
    r = _gate(ResponseMode.SEMI).evaluate("isolate_endpoint", 2, 0.99, reachable_count=None)
    assert r.ok is False and "fail-safe" in r.reason


def test_semi_mode_low_impact_auto_executes_on_tier3():
    r = _gate(ResponseMode.SEMI).evaluate("snapshot_vm", 3, 0.99, reachable_count=1, critical_reachable=0)
    assert r.ok is True


def test_critical_reachable_forces_approval():
    r = _gate(ResponseMode.SEMI).evaluate("snapshot_vm", 2, 0.99, reachable_count=1, critical_reachable=1)
    assert r.ok is False


def test_full_mode_respects_max_tier():
    g = _gate(ResponseMode.FULL, max_tier=2)
    assert g.evaluate("isolate_endpoint", 2, 0.99).ok is True   # within max tier
    assert g.evaluate("isolate_endpoint", 3, 0.99).ok is False  # tier 3 > max tier 2


def test_gate_fails_closed_on_bad_input():
    r = _gate().evaluate("x", "not-an-int", 0.9)  # type error inside -> fail closed
    assert r.ok is False
