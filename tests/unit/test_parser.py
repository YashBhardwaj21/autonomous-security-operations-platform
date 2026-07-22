from src.canon.schema import SourceType
from src.ingestion.parser import ParserFactory, DropStats
from tests._fixtures import sysmon_events as fx


def test_supported_event_ids_include_h4_additions():
    fac = ParserFactory()
    ids = set(fac.supported_event_ids())
    # H4: Sysmon 10 (ProcessAccess) + 7 (ImageLoad) + PowerShell 4103/4104 added
    assert {10, 7, 4103, 4104}.issubset(ids)
    assert {1, 3, 11, 12, 13, 4624}.issubset(ids)


def test_parse_event10_extracts_lsass_target():
    fac = ParserFactory()
    ev = fac.parse(fx.HOSTA_LSASS_ACCESS, SourceType.OTRF, DropStats())
    assert ev is not None and ev.event_id == 10
    ents = fac.get_parser(10).extract_entities(ev)
    assert any("lsass.exe" in e.image.lower() for e in ents.values())


def test_missing_timestamp_is_counted_not_fabricated():
    fac = ParserFactory()
    stats = DropStats()
    ev = fac.parse(fx.BAD_TIMESTAMP, SourceType.OTRF, stats)
    assert ev is None                    # never fabricated with utcnow()
    assert stats.no_timestamp == 1
    assert stats.by_event_id[1] == 1


def test_unsupported_event_counted():
    fac = ParserFactory()
    stats = DropStats()
    assert fac.parse(fx.UNSUPPORTED_EVENT, SourceType.OTRF, stats) is None
    assert stats.no_parser == 1


def test_parse_event10_extracts_relationship():
    fac = ParserFactory()
    ev = fac.parse(fx.HOSTA_LSASS_ACCESS, SourceType.OTRF, DropStats())
    parser = fac.get_parser(10)
    ents = parser.extract_entities(ev)
    rels = parser.extract_relationships(ev, ents)
    assert len(rels) == 1
    src, rel_type, dst = rels[0]
    assert src == "proc_{proc-a}"
    assert rel_type == "accessed_process"
    assert dst == "proc_{proc-lsass}"

