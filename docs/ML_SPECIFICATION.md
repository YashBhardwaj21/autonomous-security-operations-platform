# ML Specification — Autonomous Security Operations Platform

## Datasets (data-driven; operator fetches)
- **OTRF Security-Datasets** — attribution (real Windows telemetry). 135 processed
  ndjson scenarios / 1.93M events; multi-stage: 32 compound + 18 APT29-evals (audit M1).
  Labels: multi-label from full `attack_mappings` (`src/features/labeling.py`).
- **LANL / CICIDS / TON_IoT** — BENIGN/NORMAL only, UEBA baseline path ONLY (never an
  attribution label; enforced by `SourceType` in `src/ingestion/benign.py`).
- **Splunk BOTS** — held-out validation only, never trained on (`scripts/validate_bots.py`).

## Pipeline
UEBA (IsolationForest+z, own space) → tactic/technique attribution (RF/XGBoost,
hierarchical, tactic-constrained) → **CalibratedClassifierCV(sigmoid)** → SHAP →
data-derived transition matrix → retrieval → (optional narrative, not implemented).

## Validation (the discipline)
**Outer:** Leave-One-Scenario-Out CV (grouped by scenario) — reported metrics.
**Inner:** grouped 3-fold CV within each outer-train set — hyperparameter search and
calibration fitting ONLY; never touches the outer-test fold. A loud assertion fails
if any scenario spans train and test (`src/evaluation/harness.assert_no_group_leakage`).
Metrics: macro-F1, precision/recall, top-3, **ECE**, **Brier**, confusion matrix,
bootstrap CIs; baselines: majority-class + uniform-random. Calibration matters because
the SOAR gate consumes the calibrated probability (audit H1).

## Operator boundary
`scripts/train_attribution.py --train` runs LOSO; `--fit-final` persists the artifact.
The assistant never runs `.fit()`. Inspect the confusion matrix + reliability diagram
and choose `unsupported_class` (<3-scenario) moves before trusting any number.

## Feature hygiene
Fit-on-train-only transform (`src/features/hygiene.py`): drop constants, one of each
|rho|>=0.98 pair, and lab-artifact columns; persisted with the model (no train/serve skew).

## Honest ceilings (state these in any results write-up)
Single-source attribution (OTRF only); excluded `unsupported_class` techniques; UEBA
baseline quality bounded by the benign subset size streamed under the 16GB ceiling.
