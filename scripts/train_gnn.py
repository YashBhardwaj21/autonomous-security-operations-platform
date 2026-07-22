from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from src.graph.dataset import ActivityGraphDataset
from src.graph.types import FEATURE_DIM, FEATURE_SPEC
from src.ml.gnn_detector import ActivityGraphDetector

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# New configuration: minimum samples per class after filtering
# ------------------------------------------------------------
MIN_SAMPLES_PER_CLASS = 8


def compute_level3_hash(data) -> str:
    """Computes exact Level 3 content hash (x, edge_index, edge_attr) for group-aware replica-safe splitting."""
    hasher = hashlib.sha256()
    hasher.update(data.x.cpu().numpy().tobytes())
    hasher.update(data.edge_index.cpu().numpy().tobytes())
    hasher.update(data.edge_attr.cpu().numpy().tobytes())
    return hasher.hexdigest()


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification to focus on hard imbalanced examples."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def set_reproducibility_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Computes inverse frequency class weights for balanced loss."""
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total_samples / (num_classes * counts[i])
        else:
            weights[i] = 1.0

    return torch.tensor(weights, dtype=torch.float)


def plot_confusion_matrix(cm: np.ndarray, num_classes: int, save_path: Path, normalize: bool = False, label_names: Optional[List[str]] = None) -> None:
    """Renders and saves confusion matrix plot."""
    plt.figure(figsize=(10, 8))
    display_cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if (normalize and cm.sum() > 0) else cm
    display_cm = np.nan_to_num(display_cm)

    plt.imshow(display_cm, interpolation="nearest", cmap=plt.cm.Blues)
    title_str = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    plt.title(f"{title_str} - GraphSAGE Detector", fontsize=12, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    ticks_labels = label_names if label_names and len(label_names) == num_classes else [f"C{i}" for i in range(num_classes)]
    plt.xticks(tick_marks, ticks_labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, ticks_labels, fontsize=8)

    thresh = display_cm.max() / 2.0 if display_cm.max() > 0 else 1.0
    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            val = display_cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(
                j, i, txt,
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=7,
            )

    plt.ylabel("True Scenario Class", fontsize=10, fontweight="bold")
    plt.xlabel("Predicted Scenario Class", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

        preds = out.argmax(dim=-1)
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    train_loss = total_loss / len(loader.dataset)
    train_acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    train_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) > 0 else 0.0

    return train_loss, train_acc, train_f1_macro


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []
    scenario_ids = []
    activity_ids = []
    node_counts = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(out, dim=-1)
        preds = out.argmax(dim=-1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

        sizes = torch.bincount(batch.batch).cpu().numpy()
        node_counts.extend(sizes)

        if hasattr(batch, "scenario_id"):
            scenario_ids.extend(batch.scenario_id if isinstance(batch.scenario_id, list) else [batch.scenario_id])
        if hasattr(batch, "activity_id"):
            activity_ids.extend(batch.activity_id if isinstance(batch.activity_id, list) else [batch.activity_id])

    val_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    return val_loss, np.array(y_true), np.array(y_pred), np.array(y_probs), scenario_ids, activity_ids, np.array(node_counts)


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE GNN Training with Group-Aware Replica Splitting & Rich Features")
    parser.add_argument("--use-sampler", action="store_true", help="Use WeightedRandomSampler")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use Focal Loss")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden channels for GraphSAGE")
    args = parser.parse_args()

    set_reproducibility_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path("data/processed/activities")

    # Force regenerate cached data.pt to reflect updated 26-dim feature spec and deterministic labels
    processed_pt = dataset_path / "processed" / "data.pt"
    if processed_pt.exists():
        print("[CACHE RESET] Removing legacy data.pt cache to rebuild rich 26-dim feature graph dataset...")
        processed_pt.unlink()

    raw_dataset = ActivityGraphDataset(root=str(dataset_path))
    dataset = [raw_dataset[i] for i in range(len(raw_dataset)) if raw_dataset[i].num_nodes > 0]
    num_graphs = len(dataset)

    # Level 3 Group Hashes to prevent exact tensor replica leakage
    content_hashes = [compute_level3_hash(g) for g in dataset]

    raw_labels = [dataset[i].y.item() for i in range(num_graphs)]
    scenarios = [dataset[i].scenario_id if hasattr(dataset[i], "scenario_id") else "unknown" for i in range(num_graphs)]

    # ------------------------------------------------------------
    # 1. Bayes‑bound check: identical graphs with different labels?
    # ------------------------------------------------------------
    hash_to_labels = defaultdict(set)
    for h, lbl in zip(content_hashes, raw_labels):
        hash_to_labels[h].add(lbl)

    ambiguous = {h: lbls for h, lbls in hash_to_labels.items() if len(lbls) > 1}
    if ambiguous:
        print(f"\n[WARNING] {len(ambiguous)} identical graphs map to multiple labels — task may be unlearnable.")
        for h, lbls in list(ambiguous.items())[:5]:
            print(f"  hash={h[:12]}... labels={lbls}")
        print("  (These will remain in the dataset; consider fixing labels upstream.)\n")
    else:
        print("[OK] No ambiguous graph‑label mappings found.\n")

    # ------------------------------------------------------------
    # 2. Filter out classes with fewer than MIN_SAMPLES_PER_CLASS
    # ------------------------------------------------------------
    label_counts = Counter(raw_labels)
    keep_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= MIN_SAMPLES_PER_CLASS}
    if len(keep_labels) < 2:
        print(f"ERROR: After filtering for >= {MIN_SAMPLES_PER_CLASS} samples per class, only {len(keep_labels)} classes remain. "
              "Cannot train a classifier. Exiting.")
        return

    filtered_idx = [i for i in range(num_graphs) if raw_labels[i] in keep_labels]
    dataset = [dataset[i] for i in filtered_idx]
    raw_labels = [raw_labels[i] for i in filtered_idx]
    scenarios = [scenarios[i] for i in filtered_idx]
    content_hashes = [content_hashes[i] for i in filtered_idx]

    num_graphs = len(dataset)
    print(f"After filtering: {num_graphs} graphs, {len(keep_labels)} classes.")

    # Re‑map labels to contiguous indices
    unique_labels = sorted(list(set(raw_labels)))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    mapped_labels = [label_to_idx[lbl] for lbl in raw_labels]

    # *** CRITICAL FIX: Update each graph's y to the mapped label ***
    for i, graph in enumerate(dataset):
        graph.y = torch.tensor(mapped_labels[i], dtype=torch.long)

    class_counts = np.bincount(mapped_labels, minlength=num_classes)

    # Build index to scenario string label map
    idx_to_scen = {}
    for idx, scen in zip(mapped_labels, scenarios):
        if idx not in idx_to_scen:
            idx_to_scen[idx] = scen
    label_names = [idx_to_scen.get(i, f"Class_{i}") for i in range(num_classes)]

    print("=" * 90)
    print(f"REPLICA-SAFE GRAPHSAGE DETECTOR TRAINING PIPELINE (FeatureDim={FEATURE_DIM}, HiddenDim={args.hidden_channels})")
    print("=" * 90)
    print(f"Total Graphs           : {num_graphs}")
    print(f"Unique Content Hashes  : {len(set(content_hashes))}")
    print(f"Num Target Classes     : {num_classes}")
    print(f"Class Distribution     : {dict(enumerate(class_counts))}")

    # ------------------------------------------------------------
    # 3. StratifiedGroupKFold for group‑aware, class‑balanced split
    # ------------------------------------------------------------
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # Take the first fold as train/val (or you could loop over folds for cross‑validation)
    train_idx, val_idx = next(sgkf.split(range(num_graphs), mapped_labels, groups=content_hashes))

    train_dataset = [dataset[idx].clone() for idx in train_idx]
    val_dataset = [dataset[idx].clone() for idx in val_idx]

    train_labels = [g.y.item() for g in train_dataset]
    val_labels = [g.y.item() for g in val_dataset]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)

    print(f"Train Dataset Split    : {len(train_dataset)} graphs | Class Counts: {dict(Counter(train_labels))}")
    print(f"Val Dataset Split      : {len(val_dataset)} graphs | Class Counts: {dict(Counter(val_labels))}")

    if args.use_sampler:
        counts = Counter(train_labels)
        class_weights_sample = {c: 1.0 / counts[c] for c in counts}
        sample_weights = [class_weights_sample[y] for y in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ActivityGraphDetector(
        in_channels=FEATURE_DIM,
        hidden_channels=args.hidden_channels,
        num_classes=num_classes,
        dropout=0.3,
        use_batch_norm=True,
        use_residual=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    if args.use_focal_loss:
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1_macro": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    print("\nTRAINING LOOP")
    print("-" * 90)
    for epoch in range(1, 101):
        train_loss, train_acc, train_f1_macro = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred, y_probs, scens, acts, n_counts = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        val_acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
        val_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) > 0 else 0.0
        val_f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) > 0 else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(float(train_loss), 4))
        history["train_acc"].append(round(float(train_acc), 4))
        history["train_f1_macro"].append(round(float(train_f1_macro), 4))
        history["val_loss"].append(round(float(val_loss), 4))
        history["val_acc"].append(round(float(val_acc), 4))
        history["val_f1_macro"].append(round(float(val_f1_macro), 4))
        history["val_f1_weighted"].append(round(float(val_f1_weighted), 4))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1_macro:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1_macro:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": float(best_val_loss),
                "label_map": label_to_idx,
                "scenario_map": idx_to_scen,
            }, RUNS_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] Stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    # Save training history
    with (RUNS_DIR / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best model checkpoint
    checkpoint = torch.load(RUNS_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, y_true, y_pred, y_probs, scens, acts, n_counts = evaluate(model, val_loader, criterion, device)

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ------------------------------------------------------------
    # 4. ROC‑AUC with proper exception logging (no silent 0)
    # ------------------------------------------------------------
    try:
        present_classes = np.unique(y_true)
        if len(present_classes) > 1:
            roc_auc = float(roc_auc_score(y_true, y_probs[:, present_classes], multi_class="ovr", average="macro", labels=present_classes))
        else:
            roc_auc = 0.0
    except Exception as e:
        print(f"[ROC-AUC ERROR] {type(e).__name__}: {e}")
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plot_confusion_matrix(cm, num_classes, RUNS_DIR / "confusion_matrix.png", normalize=False, label_names=label_names)
    plot_confusion_matrix(cm, num_classes, RUNS_DIR / "confusion_matrix_normalized.png", normalize=True, label_names=label_names)

    print("\n" + "=" * 90)
    print("FINAL EVALUATION METRICS (REPLICA-SAFE SPLIT & RICH FEATURES)")
    print("=" * 90)
    print(f"Accuracy           : {acc:.4f}")
    print(f"Balanced Accuracy  : {b_acc:.4f}")
    print(f"Precision (Macro)  : {prec_macro:.4f}")
    print(f"Recall (Macro)     : {rec_macro:.4f}")
    print(f"F1 Score (Macro)   : {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"ROC-AUC (Macro OVR): {roc_auc:.4f}")

    metrics = {
        "best_epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_loss", 0.0),
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_ovr": float(roc_auc),
        "label_names": label_names,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = RUNS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved best model checkpoint to : {(RUNS_DIR / 'best_model.pt').resolve()}")
    print(f"Saved evaluation metrics to     : {metrics_path.resolve()}")


if __name__ == "__main__":
    main()