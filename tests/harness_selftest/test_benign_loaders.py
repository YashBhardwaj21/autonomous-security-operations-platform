"""Benign-loader shape tests — tiny in-memory samples, isolated folder.

These verify the loaders map the REAL verified schemas and enforce the benign-only
boundary. Samples are 2-3 rows written inline purely to exercise parsing shape;
no dataset is downloaded and nothing here is application data.
"""
from src.canon.schema import SourceType
from src.ingestion.benign import (
    cicids_benign_features,
    lanl_auth_features,
    toniot_normal_features,
)


def test_lanl_auth_aggregates_per_source_computer_ueba_space():
    lines = [
        "1,U1@D,U2@D,C1,C2,Kerberos,2,LogOn,Success",
        "2,U1@D,U1@D,C1,C1,Kerberos,2,LogOn,Fail",
    ]
    fvs = list(lanl_auth_features(lines))
    assert fvs and all(fv.space == "ueba" and fv.source == SourceType.LANL for fv in fvs)
    c1 = next(fv for fv in fvs if fv.activity_id.endswith("C1"))
    d = dict(zip(c1.feature_names, c1.features))
    assert d["auth_count"] == 2.0 and d["auth_fail"] == 1.0 and d["remote_auth"] == 1.0


def test_cicids_uses_benign_rows_only():
    rows = [
        {"Flow Duration": "10", "Total Fwd Packets": "3", "Label": "BENIGN"},
        {"Flow Duration": "99", "Total Fwd Packets": "9", "Label": "DDoS"},
    ]
    fvs = list(cicids_benign_features(rows))
    assert len(fvs) == 1                     # the DDoS row is excluded
    assert fvs[0].source == SourceType.CICIDS and fvs[0].space == "ueba"


def test_toniot_uses_normal_rows_only():
    rows = [{"pkts": "5", "bytes": "100", "type": "normal"},
            {"pkts": "7", "bytes": "200", "type": "ddos"}]
    fvs = list(toniot_normal_features(rows))
    assert len(fvs) == 1 and fvs[0].source == SourceType.TONIOT
