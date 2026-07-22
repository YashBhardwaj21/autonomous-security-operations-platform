# Machine Learning Specification — Autonomous Security Operations Platform

This document specifies the current-state machine learning contracts, data boundaries, evaluation protocols, and serving artifacts.

---

## 1. Dataset Contracts & Schema

### Data Ingestion Object (`BuiltDataset`)
Dataset extraction via `src/ingestion/otrf.py` yields a `BuiltDataset` dataclass containing:

```python
@dataclass
class BuiltDataset:
    X: np.ndarray                 # Shape (n_sessions, n_features) - feature matrix
    feature_names: List[str]      # Stable feature schema column names
    labels: List[ScenarioLabel]   # Multi-label sets per session (from scenario metadata)
    scenario_ids: List[str]       # Grouping keys for LOSO cross-validation
    y_primary: List[str]          # Single-label projection (alphabetically first technique)
    drop_stats: Dict[str, int]    # Parse failure and missing timestamp counters
```

### Data Boundaries & Source Types (`SourceType`)
- **Attack Population (`OTRF`, `WINDOWS`, `LAB`)**: Ingested from real Security-Datasets telemetry under `data/raw/Security-Datasets/`. May receive ATT&CK technique labels.
- **Benign Population (`LANL`, `CICIDS`, `TONIOT`)**: Reserved exclusively for the UEBA baseline path (`SourceType.LANL`, etc.). Enforced in code by `label_from_metadata()` (`src/features/labeling.py`); non-empty attack labels on benign sources trigger an explicit `ValueError`.
- **Held-out Validation (`BOTS`)**: External validation population (`SourceType.BOTS`), never used during model training.

---

## 2. Current Label Assignment Contracts & Limitations

### Scenario-Level Weak Supervision
Scenario metadata (`_metadata/*.yaml`) contains `attack_mappings`. `label_from_metadata()` constructs a `ScenarioLabel` containing all tactics, parent techniques, and sub-techniques. Every session extracted from a scenario receives the full scenario-level technique set.

### Single-Label Projection (`y_primary`)
While `ScenarioLabel` stores multi-label sets, `build_dataset()` populates `y_primary` by choosing `sorted(scen_label.techniques)[0]` (alphabetically first technique). `scripts/train_attribution.py` currently trains single-class classifiers against `y_primary`. This is a known formulation limitation to be addressed in future multi-label extensions.

### Label Space & Support Counting
`build_label_space()` in `src/features/labeling.py` filters techniques that appear in fewer than `MIN_SCENARIOS_PER_CLASS` scenarios.
* **Known Implementation Limitation**: Current code passes `ds.labels` (one label per session) to `build_label_space()`, counting *sessions* per technique rather than unique *scenarios*. This must be evaluated against distinct `scenario_id` counts.

---

## 3. Evaluation Contract & Harness Discipline

Model evaluation is governed by `nested_loso_cv()` in `src/evaluation/harness.py`:

```
Outer Loop: Leave-One-Scenario-Out (LOGO) grouped by scenario_id
  ├── Group Leakage Assertion: assert_no_group_leakage(train_groups, test_groups)
  ├── Fit HygieneTransform on Train Fold
  ├── Fit Model + CalibratedClassifierCV on Train Fold
  └── Evaluate on Held-Out Scenario Fold
```

### Required Metrics & Baselines
- **Classification Performance**: Macro-F1, Precision, Recall, Top-3 Accuracy.
- **Uncertainty & Calibration**: Expected Calibration Error (ECE), Brier Score.
- **Baselines**: Every evaluation must report comparisons against `majority_class` and `uniform_random` baselines.
- **Feature Hygiene Requirement**: `HygieneTransform` (constant dropping, high correlation removal, lab artifact scrubbing) must be fitted strictly inside training folds. Passing raw `X` globally without fold-local hygiene is a known script limitation in `train_attribution.py`.

---

## 4. Serving Artifact Contract (`AttributionArtifact`)

Model serialization uses `AttributionArtifact` (`src/attribution/loader.py`), persisting:

```python
@dataclass
class AttributionArtifact:
    model: HierarchicalAttributionModel  # Fitted, calibrated model
    hygiene: HygieneTransform             # Fitted feature hygiene transformer
    feature_names_in: List[str]           # Input feature schema expected at serving
    version: str                          # Schema/model version string
```

* **Degraded Runtime Mode**: If `models/attribution.joblib` does not exist, `POST /ingest/events` safely returns `attribution.status = "model_unavailable"`. No fabricated probabilities or dummy techniques are produced.

---

## 5. Hackathon Non-Goals & Out-of-Scope Items

* **Live EDR / Firewall Action Execution**: Response actions are evaluated and gated logically; real API execution against EDR platforms is out of scope.
* **Multi-Label Model Training**: The executable trainer operates on single-label `y_primary` targets.
* **Graph AI / GNN in Production**: GraphSAGE is retired to `experiments/gnn/` and is excluded from `src/` runtime execution.
