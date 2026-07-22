from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, global_mean_pool

from experiments.gnn.gnn_experiment.graph.types import FEATURE_DIM


class ActivityGraphDetector(nn.Module):
    """Enhanced Residual GraphSAGE detector with BatchNorm and configurable capacity for multi-class ATT&CK activity classification."""

    def __init__(
        self,
        in_channels: int = FEATURE_DIM,
        hidden_channels: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Input projection layer
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # 4-Layer GraphSAGE blocks
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)

        if use_batch_norm:
            self.bn1 = BatchNorm(hidden_channels)
            self.bn2 = BatchNorm(hidden_channels)
            self.bn3 = BatchNorm(hidden_channels)
            self.bn4 = BatchNorm(hidden_channels)

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through input projection, 4 residual GraphSAGE blocks, BatchNorm, pooling, and linear head."""
        h = F.relu(self.input_proj(x))

        # Block 1
        residual = h
        h_conv = self.conv1(h, edge_index)
        if self.use_batch_norm:
            h_conv = self.bn1(h_conv)
        h = F.relu(h_conv)
        if self.use_residual:
            h = h + residual
        h = self.dropout(h)

        # Block 2
        residual = h
        h_conv = self.conv2(h, edge_index)
        if self.use_batch_norm:
            h_conv = self.bn2(h_conv)
        h = F.relu(h_conv)
        if self.use_residual:
            h = h + residual
        h = self.dropout(h)

        # Block 3
        residual = h
        h_conv = self.conv3(h, edge_index)
        if self.use_batch_norm:
            h_conv = self.bn3(h_conv)
        h = F.relu(h_conv)
        if self.use_residual:
            h = h + residual
        h = self.dropout(h)

        # Block 4
        residual = h
        h_conv = self.conv4(h, edge_index)
        if self.use_batch_norm:
            h_conv = self.bn4(h_conv)
        h = F.relu(h_conv)
        if self.use_residual:
            h = h + residual

        pooled = global_mean_pool(h, batch)
        logits = self.classifier(pooled)
        return logits
