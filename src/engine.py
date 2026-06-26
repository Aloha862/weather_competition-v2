"""Training / evaluation loops, EMA, optimizer and scheduler builders."""
import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from .config import Config


class ModelEMA:
    """Exponential moving average of model weights for steadier validation."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        for name, value in model_params.items():
            ema_params[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.ema.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, value in model_buffers.items():
            ema_buffers[name].copy_(value)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.epochs * steps_per_epoch
    min_factor = cfg.min_lr / cfg.lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional["torch.amp.GradScaler"],
    device: torch.device,
    cfg: Config,
    ema: Optional[ModelEMA] = None,
) -> float:
    model.train()
    running = 0.0
    use_amp = cfg.use_amp and device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())
    return np.array(all_true), np.array(all_preds)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: Tuple[str, ...],
) -> Dict[str, object]:
    y_true, y_pred = predict_logits(model, loader, device)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=list(classes), output_dict=True, zero_division=0,
    )
    pred_counts = {classes[i]: int((y_pred == i).sum()) for i in range(len(classes))}
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "pred_distribution": pred_counts,
    }
