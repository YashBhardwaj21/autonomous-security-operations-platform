from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.graph.dataset import ActivityGraphDataset
from src.graph.types import FEATURE_DIM
from src.ml.gnn_detector import ActivityGraphDetector

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 8


def compute_level3_hash(data) -> str:
    hasher = hashlib.sha256()
    hasher.update(data.x.cpu().numpy().tobytes())
    hasher.update(data.edge_index.cpu().numpy().tobytes())
    hasher.update(data.edge_attr.cpu().numpy().tobytes())
    return hasher.hexdigest()


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total_samples / (num_classes * counts[i])
        else:
            weights[i] = 1.0

    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device, randomize_edges: bool = False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)

        edge_index = batch.edge_index
        if randomize_edges and edge_index.shape[1] > 0:
            edge_index = torch.randint_like(edge_index, high=batch.num_nodes)

        optimizer.zero_grad()
        out = model(batch.x, edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, randomize_edges: bool = False):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []

    for batch in loader:
        batch = batch.to(device)

        edge_index = batch.edge_index
        if randomize_edges and edge_index.shape[1] > 0:
            edge_index = torch.randint_like(edge_index, high=batch.num_nodes)

        out = model(batch.x, edge_index, batch.batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(out, dim=-1)
        preds = out.argmax(dim=-1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

    val_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    return val_loss, np.array(y_true), np.array(y_pred), np.array(y_probs)


def main():
    parser = argparse.ArgumentParser(description="5-Fold StratifiedGroupKFold Cross-Validation for GraphSAGE")
    parser.add_argument("--randomize-edges", action="store_true", help="Graph Representation Ablation: Randomize edge indices")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path("data/processed/activities")
    processed_pt = dataset_path / "processed" / "data.pt"
    if processed_pt.exists():
        processed_pt.unlink()

    raw_dataset = ActivityGraphDataset(root=str(dataset_path))
    dataset = [raw_dataset[i] for i in range(len(raw_dataset)) if raw_dataset[i].num_nodes > 0]
    num_graphs = len(dataset)

    content_hashes = [compute_level3_hash(g) for g in dataset]
    raw_labels = [dataset[i].y.item() for i in range(num_graphs)]

    # Filter out classes with < MIN_SAMPLES_PER_CLASS
    label_counts = Counter(raw_labels)
    keep_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= MIN_SAMPLES_PER_CLASS}

    filtered_idx = [i for i in range(num_graphs) if raw_labels[i] in keep_labels]
    dataset = [dataset[i] for i in filtered_idx]
    raw_labels = [raw_labels[i] for i in filtered_idx]
    content_hashes = [content_hashes[i] for i in filtered_idx]

    num_graphs = len(dataset)
    unique_labels = sorted(list(set(raw_labels)))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    mapped_labels = [label_to_idx[lbl] for lbl in raw_labels]

    for i, g in enumerate(dataset):
        g.y = torch.tensor(mapped_labels[i], dtype=torch.long)

    print("=" * 90)
    print(f"5-FOLD STRATIFIED GROUP CROSS-VALIDATION (RandomizeEdges={args.randomize_edges})")
    print("=" * 90)
    print(f"Total Graphs           : {num_graphs}")
    print(f"Unique Content Hashes  : {len(set(content_hashes))}")
    print(f"Num Target Classes     : {num_classes}")

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(range(num_graphs), mapped_labels, groups=content_hashes), 1):
        train_dataset = [dataset[idx].clone() for idx in train_idx]
        val_dataset = [dataset[idx].clone() for idx in val_idx]

        train_labels = [g.y.item() for g in train_dataset]
        val_labels = [g.y.item() for g in val_dataset]

        class_weights = compute_class_weights(train_labels, num_classes).to(device)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = ActivityGraphDetector(
            in_channels=FEATURE_DIM,
            hidden_channels=128,
            num_classes=num_classes,
            dropout=0.3,
            use_batch_norm=True,
            use_residual=True,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        patience = 12
        patience_counter = 0

        for epoch in range(1, 81):
            t_loss = train_epoch(model, train_loader, optimizer, criterion, device, randomize_edges=args.randomize_edges)
            v_loss, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, device, randomize_edges=args.randomize_edges)

            scheduler.step(v_loss)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(model.state_dict(), RUNS_DIR / f"best_model_fold_{fold}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(torch.load(RUNS_DIR / f"best_model_fold_{fold}.pt", map_location=device))
        v_loss, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, device, randomize_edges=args.randomize_edges)

        acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        b_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) > 0 else 0.0
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) > 0 else 0.0

        fold_metrics.append({"fold": fold, "acc": acc, "b_acc": b_acc, "f1_macro": f1_m, "f1_weighted": f1_w})
        print(f"Fold {fold}/{args.n_splits} | Val Acc: {acc:.4f} | Bal Acc: {b_acc:.4f} | Macro F1: {f1_m:.4f} | Weighted F1: {f1_w:.4f}")

    accs = [m["acc"] for m in fold_metrics]
    b_accs = [m["b_acc"] for m in fold_metrics]
    f1_macros = [m["f1_macro"] for m in fold_metrics]
    f1_weighteds = [m["f1_weighted"] for m in fold_metrics]

    print("\n" + "=" * 90)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 90)
    print(f"Accuracy           : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Balanced Accuracy  : {np.mean(b_accs):.4f} ± {np.std(b_accs):.4f}")
    print(f"Macro F1           : {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
    print(f"Weighted F1        : {np.mean(f1_weighteds):.4f} ± {np.std(f1_weighteds):.4f}")

    results = {
        "randomize_edges": args.randomize_edges,
        "n_splits": args.n_splits,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_balanced_accuracy": float(np.mean(b_accs)),
        "std_balanced_accuracy": float(np.std(b_accs)),
        "mean_macro_f1": float(np.mean(f1_macros)),
        "std_macro_f1": float(np.std(f1_macros)),
        "mean_weighted_f1": float(np.mean(f1_weighteds)),
        "std_weighted_f1": float(np.std(f1_weighteds)),
        "folds": fold_metrics,
    }

    out_name = "cross_validation_results_random_edges.json" if args.randomize_edges else "cross_validation_results_baseline.json"
    with (REPORTS_DIR / out_name).open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved cross-validation results to: {(REPORTS_DIR / out_name).resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()
