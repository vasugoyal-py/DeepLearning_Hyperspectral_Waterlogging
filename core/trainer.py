"""
Stage 4 - Training loop for CNN (classification) and U-Net (segmentation).

Handles optimisation, early stopping, checkpointing, and per-epoch metric
logging. Framework: PyTorch.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger("deepwaterlog.trainer")


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    history: dict[str, list[float]] = field(default_factory=dict)
    checkpoint_path: str | None = None


def _get_loss(name: str, num_classes: int, device: torch.device) -> nn.Module:
    name = (name or "cross_entropy").lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "focal":
        return FocalLoss(num_classes=num_classes, device=device)
    if name == "dice":
        return DiceLoss(num_classes=num_classes)
    if name == "combined":
        return CombinedLoss(num_classes=num_classes)
    raise ValueError(f"Unknown loss: {name}")


def _get_optimizer(name: str, params: Any, lr: float) -> optim.Optimizer:
    name = (name or "adam").lower()
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Optional losses
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, device: torch.device, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = torch.full((targets.size(0),), self.alpha, device=self.device)
            loss = alpha_t * loss
        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation (U-Net)."""

    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) logits; targets: (B, H, W) class ids
        probs = torch.softmax(inputs, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int, device: torch.device, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(inputs, targets) + self.dice_weight * self.dice(inputs, targets)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _save_checkpoint(model: nn.Module, path: str, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": meta or {},
        },
        path,
    )
    LOGGER.info("Checkpoint saved to %s", path)


# ---------------------------------------------------------------------------
# CNN training
# ---------------------------------------------------------------------------
def train_cnn(
    model: nn.Module,
    train_patches: torch.Tensor,
    train_labels: torch.Tensor,
    val_patches: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    loss_name: str | None = None,
    optimizer_name: str | None = None,
    checkpoint_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
) -> TrainResult:
    """
    Train a CNN for patch-level waterlogging classification.

    Returns a TrainResult with the best validation F1 and history.
    """
    epochs = config.CNN_EPOCHS if epochs is None else epochs
    batch_size = config.CNN_BATCH_SIZE if batch_size is None else batch_size
    lr = config.CNN_LEARNING_RATE if lr is None else lr
    patience = config.CNN_EARLY_STOPPING_PATIENCE if patience is None else patience
    loss_name = config.CNN_LOSS if loss_name is None else loss_name
    optimizer_name = config.CNN_OPTIMIZER if optimizer_name is None else optimizer_name
    checkpoint_dir = checkpoint_dir or str(config.DATA_DIR / "models")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(seed)
    model.to(device)

    criterion = _get_loss(loss_name, config.CNN_NUM_CLASSES, device)
    optimizer = _get_optimizer(optimizer_name, model.parameters(), lr)

    train_ds = TensorDataset(train_patches, train_labels)
    val_ds = TensorDataset(val_patches, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = -math.inf
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoint_dir, "cnn_best.pt")

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validate ----
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu())

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        val_f1 = _macro_f1(preds, targets, config.CNN_NUM_CLASSES)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        LOGGER.info(
            "CNN epoch %d/%d - train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch, epochs, train_loss, val_loss, val_f1,
        )

        # ---- early stopping / checkpoint ----
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(model, checkpoint_path, {"epoch": epoch, "val_f1": val_f1})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience)
                break

    return TrainResult(
        best_metric=best_f1,
        best_epoch=best_epoch,
        history=history,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# U-Net training
# ---------------------------------------------------------------------------
def train_unet(
    model: nn.Module,
    train_patches: torch.Tensor,
    train_labels: torch.Tensor,
    val_patches: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    loss_name: str | None = None,
    optimizer_name: str | None = None,
    checkpoint_dir: str | None = None,
    device: str | None = None,
    seed: int = 42,
) -> TrainResult:
    """
    Train a U-Net for pixel-level waterlogging segmentation.

    ``train_labels`` / ``val_labels`` should be integer class-id maps of shape
    (N, H, W).
    """
    epochs = config.UNET_EPOCHS if epochs is None else epochs
    batch_size = config.UNET_BATCH_SIZE if batch_size is None else batch_size
    lr = config.UNET_LEARNING_RATE if lr is None else lr
    patience = config.UNET_EARLY_STOPPING_PATIENCE if patience is None else patience
    loss_name = config.UNET_LOSS if loss_name is None else loss_name
    optimizer_name = config.UNET_OPTIMIZER if optimizer_name is None else optimizer_name
    checkpoint_dir = checkpoint_dir or str(config.DATA_DIR / "models")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(seed)
    model.to(device)

    criterion = _get_loss(loss_name, config.UNET_NUM_CLASSES, device)
    optimizer = _get_optimizer(optimizer_name, model.parameters(), lr)

    train_ds = TensorDataset(train_patches, train_labels)
    val_ds = TensorDataset(val_patches, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = -math.inf
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoint_dir, "unet_best.pt")

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validate ----
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu())

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        val_iou = _mean_iou(preds, targets, config.UNET_NUM_CLASSES)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        LOGGER.info(
            "U-Net epoch %d/%d - train_loss=%.4f val_loss=%.4f val_iou=%.4f",
            epoch, epochs, train_loss, val_loss, val_iou,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(model, checkpoint_path, {"epoch": epoch, "val_iou": val_iou})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, patience)
                break

    return TrainResult(
        best_metric=best_iou,
        best_epoch=best_epoch,
        history=history,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp) / denom if denom > 0 else 0.0)
    return sum(f1s) / len(f1s)


def _mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    ious = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        ious.append(inter / union if union > 0 else 0.0)
    return sum(ious) / len(ious)
