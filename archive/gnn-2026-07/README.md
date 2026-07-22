# GNN / GraphSAGE Experiment Archive (2026-07)

## Provenance & Rationale for Retirement

This archive contains the generated graph datasets, GNN run checkpoints, and GNN-specific evaluation reports produced during the investigation of GraphSAGE for security telemetry modeling.

### Why GraphSAGE Was Retired from the Primary Path:

1. **Target Formulation Misalignment**:
   The GNN pipeline mapped graph representations of activities directly to `scenario_id` class indices (`scenario_id -> class ID`). This models scenario identity rather than fine-grained ATT&CK technique attribution.

2. **Performance Floor**:
   The reproducible macro-F1 score achieved on multi-class scenario classification was **0.075**, indicating near-random performance across complex graph topologies.

3. **Graph Label Contamination**:
   Audit revealed **23 identical graph content hash groups** carrying conflicting scenario labels. The largest identical graph tensor appeared 251 times across 27 distinct scenarios, demonstrating that structural event graphs in OTRF are non-discriminative for scenario identification.

4. **Stale Report Cache**:
   Generated graph reports reflected a legacy 16-feature schema cache, whereas current source definitions specify 29 features.

### Archived Contents:
- `reports/`: GNN confusion matrices, graph summaries, replica hash analysis, graph visualizations.
- `runs/`: Training runs and PyTorch model checkpoints.
- `activities/`: Serialized activity JSON objects (804 files / ~77MB).
