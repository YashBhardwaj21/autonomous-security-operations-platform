"""Transition-matrix + next-step prediction tests (deterministic, real-shaped input)."""
from src.prediction.transition import TransitionMatrix, build_transition_matrix, technique_sequence
from src.prediction.engine import NextStepPredictor


def test_technique_sequence_collapses_duplicates_and_strips_subtechniques():
    md = {"attack_mappings": [
        {"technique": "T1552", "sub-technique": "004"},
        {"technique": "T1552", "sub-technique": "001"},  # same parent -> collapsed
        {"technique": "T1078", "sub-technique": "004"},
    ]}
    assert technique_sequence(md) == ["T1552", "T1078"]


def test_matrix_roundtrip_and_lookup(tmp_path):
    tm = TransitionMatrix(provenance="test", matrix={"T1055": {"T1003": 1.0}},
                          support={"T1055": {"T1003": 3}}, n_scenarios=3, n_transitions=3)
    p = tmp_path / "m.json"
    tm.save(str(p))
    loaded = TransitionMatrix.load(str(p))
    assert loaded.next_techniques("T1055") == [("T1003", 1.0)]
    # sub-technique input resolves to parent row
    assert loaded.next_techniques("T1055.001") == [("T1003", 1.0)]


def test_predictor_honest_empty_when_no_matrix(tmp_path):
    pred = NextStepPredictor(matrix_path=str(tmp_path / "absent.json"))
    assert pred.available is False
    assert pred.predict_next("T1003") == []


def test_predictor_returns_support_and_provenance(tmp_path):
    tm = TransitionMatrix(provenance="derived from real scenarios",
                          matrix={"T1055": {"T1003": 1.0}}, support={"T1055": {"T1003": 4}})
    p = tmp_path / "m.json"
    tm.save(str(p))
    pred = NextStepPredictor(matrix_path=str(p))
    out = pred.predict_next("T1055")
    assert out[0]["technique"] == "T1003"
    assert out[0]["support"] == 4
    assert "derived" in out[0]["provenance"]
