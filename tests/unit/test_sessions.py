from src.canon.schema import SourceType
from src.ingestion.parser import ParserFactory, DropStats
from src.sessions.session_builder import SessionBuilder
from tests._fixtures import sysmon_events as fx


def _events(raws):
    fac = ParserFactory()
    st = DropStats()
    return [e for r in raws if (e := fac.parse(r, SourceType.OTRF, st))], fac


def test_sessions_never_merge_two_hosts():
    events, fac = _events(fx.TWO_HOST_BURST)
    sessions = SessionBuilder(fac).build_sessions(events, scenario_id="demo")
    for s in sessions:
        hosts = {e.host for e in s.events}
        assert len(hosts) == 1, f"session merged hosts: {hosts}"
    assert {s.host for s in sessions} == {"HOSTA", "HOSTB"}


def test_wellknown_logon_id_falls_through_to_gap_window():
    events, fac = _events([fx.SYSTEM_LOGON])
    sessions = SessionBuilder(fac).build_sessions(events, "demo")
    # 0x3e7 must NOT form a logon-keyed session
    assert all(s.logon_id is None for s in sessions)


def test_real_logon_id_forms_keyed_session():
    events, fac = _events([fx.HOSTA_LOGON])
    sessions = SessionBuilder(fac).build_sessions(events, "demo")
    assert any(s.logon_id == "0x111" and s.host == "HOSTA" for s in sessions)


def test_inactivity_gap_splits_sessions():
    # two HOSTB logons far apart in time -> gap window splits them
    late = dict(fx.HOSTB_LOGON, UtcTime="2021-01-01 12:00:00.000", TargetLogonId="0x0")
    early = dict(fx.HOSTB_LOGON, TargetLogonId="0x0")  # both well-known -> gap path
    events, fac = _events([early, late])
    sessions = SessionBuilder(fac, inactivity_gap_seconds=300).build_sessions(events, "demo")
    assert len(sessions) == 2


def test_activity_relationships_invariants():
    from src.canon.schema import RelationshipType

    events, fac = _events(fx.TWO_HOST_BURST)
    sessions = SessionBuilder(fac).build_sessions(events, "demo")
    valid_rel_types = {r.value for r in RelationshipType}

    for s in sessions:
        valid_nodes = (
            set(s.processes)
            | set(s.users)
            | set(s.files)
            | set(s.registry)
            | set(s.network)
            | set(s.services)
        )
        assert len(s.relationships) == len(set(s.relationships)), "Duplicate edges found"
        for src, rel, dst in s.relationships:
            assert src in valid_nodes, f"Dangling src edge node: {src}"
            assert dst in valid_nodes, f"Dangling dst edge node: {dst}"
            assert rel in valid_rel_types, f"Invalid relationship type: {rel}"

