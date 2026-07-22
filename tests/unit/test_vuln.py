from src.vuln.scorer import score_vulnerability


def test_epss_is_a_first_class_input():
    low = score_vulnerability("CVE-x", cvss=9.0, epss=0.01, asset_criticality_tier=1)
    high = score_vulnerability("CVE-x", cvss=9.0, epss=0.97, asset_criticality_tier=1)
    assert high.risk > low.risk                    # EPSS materially changes ranking
    assert "epss" in high.components


def test_real_attack_path_exposure_raises_risk():
    base = score_vulnerability("CVE-y", cvss=7.0, epss=0.5, asset_criticality_tier=2,
                               attack_path_exposure=0.0)
    exposed = score_vulnerability("CVE-y", cvss=7.0, epss=0.5, asset_criticality_tier=2,
                                  attack_path_exposure=1.0)
    assert exposed.risk > base.risk


def test_risk_bounded_0_1():
    s = score_vulnerability("CVE-z", cvss=10.0, epss=1.0, asset_criticality_tier=0,
                            attack_path_exposure=1.0, ttp_overlap=1.0)
    assert 0.0 <= s.risk <= 1.0
