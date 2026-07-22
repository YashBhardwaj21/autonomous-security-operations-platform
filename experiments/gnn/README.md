# GraphSAGE / GNN Experimental Investigation

This directory contains the archived GraphSAGE model and activity graph construction pipeline.

## Status: Archived Experimental Baseline

GraphSAGE was evaluated as a candidate for ATT&CK attribution but was moved to an experimental investigation based on empirical findings:

1. **Target Mismatch**: The GraphSAGE pipeline maps `scenario_id -> class_id` (scenario-identity classification) rather than performing fine-grained ATT&CK technique attribution.
2. **Performance Floor**: Evaluated macro-F1 reached **0.075** on scenario identification.
3. **Graph Label Ambiguity**: 23 graph-content hash groups contain conflicting labels, with identical graph tensors appearing across 27 distinct scenarios (showing identical structural activity across different scenarios).
4. **Feature Schema Stale Artifacts**: Prior reports referenced a 16-feature schema, whereas source code defines a 29-feature schema.

## Usage

To run experiments from this folder, ensure PyTorch Geometric dependencies are installed from `experiments/gnn/requirements.txt`:

```bash
pip install -r experiments/gnn/requirements.txt
python -m experiments.gnn.scripts.train_gnn
```

Core `src/` modules do NOT depend on code inside `experiments/`.
