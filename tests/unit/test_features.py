from src.canon.schema import SourceType
from src.ingestion.parser import ParserFactory, DropStats
from src.features.pipeline import AttributionFeaturePipeline
from src.sessions.session_builder import SessionBuilder
from tests._fixtures import sysmon_events as fx


def _session(raws):
    fac = ParserFactory()
    st = DropStats()
    events = [e for r in raws if (e := fac.parse(r, SourceType.OTRF, st))]
    return SessionBuilder(fac).build_sessions(events, "demo")[0]


def test_event10_signal_populates_lsass_feature():
    # Previously EventID 10 was dropped, so these features were structurally zero (H4).
    s = _session([fx.HOSTA_PROCESS, fx.HOSTA_LSASS_ACCESS])
    fv = AttributionFeaturePipeline().extract(s)
    d = dict(zip(fv.feature_names, fv.features))
    assert d["lsass_targeted_count"] == 1.0
    assert d["process_access_count"] == 1.0
    assert d["proc_powershell"] >= 1.0


def test_feature_space_is_attribution_and_has_no_anomaly_score():
    s = _session([fx.HOSTA_PROCESS])
    fv = AttributionFeaturePipeline().extract(s)
    assert fv.space == "attribution"
    assert "anomaly_score" not in fv.feature_names  # M6 separation


def test_feature_names_are_stable_and_sorted():
    fp = AttributionFeaturePipeline()
    names = fp.feature_names()
    assert names == sorted(names)
    assert len(names) > 50
