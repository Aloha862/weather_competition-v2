"""Training entry point.

Run from the project root:

    python train.py

Or from the notebook:

    from train import run_training
    summary = run_training(cfg)

It performs a stratified train/val/test split, fine-tunes the backbone, selects
the best epoch by validation macro-F1 (on EMA weights), then reports the final
macro-F1 on the held-out test split and writes all artefacts to results/outputs.
"""
import csv
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config, DEFAULT_CONFIG
from src.data import (
    WeatherDataset,
    build_transforms,
    compute_class_weights,
    label_distribution,
    scan_image_folder,
    stratified_split,
)
from src.engine import (
    ModelEMA,
    build_optimizer,
    build_scheduler,
    evaluate,
    train_one_epoch,
)
from src.model import build_model
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def _make_loader(paths, labels, cfg, train):
    transform = build_transforms(cfg, train=train)
    dataset = WeatherDataset(paths, labels, transform)
    workers = cfg.num_workers if train else 0  # eval on 0 workers: cheap + avoids OOM
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )


def run_training(cfg: Optional[Config] = None) -> dict:
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)
    device = get_device()
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.outputs_dir)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    paths, labels = scan_image_folder(cfg.train_dir, cfg.classes)
    print(f"class_to_idx={cfg.class_to_idx()}")
    print(f"dataset_total={len(paths)} dist={label_distribution(labels, cfg.classes)}")

    splits = stratified_split(paths, labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    for name in ("train", "val", "test"):
        p, l = splits[name]
        print(f"{name}_total={len(p)} dist={label_distribution(l, cfg.classes)}")

    train_loader = _make_loader(*splits["train"], cfg=cfg, train=True)
    val_loader = _make_loader(*splits["val"], cfg=cfg, train=False)
    test_loader = _make_loader(*splits["test"], cfg=cfg, train=False)

    model = build_model(cfg.model_name, cfg.num_classes, cfg.pretrained).to(device)
    print(f"model={cfg.model_name} trainable_params={count_parameters(model):,}")

    if cfg.use_class_weight:
        weights = compute_class_weights(splits["train"][1], cfg.num_classes, cfg.class_weight_power).to(device)
        print(f"class_weights={[round(w, 3) for w in weights.tolist()]}")
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.use_amp and device.type == "cuda")
    ema = ModelEMA(model, cfg.ema_decay) if cfg.use_ema else None

    # Persist class mappings so inference can stay in sync.
    save_json(cfg.class_to_idx(), cfg.class_to_idx_path)
    save_json({str(i): name for i, name in cfg.idx_to_class().items()}, cfg.idx_to_class_path)

    best_f1 = -1.0
    best_epoch = -1
    patience = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg, ema
        )
        eval_model = ema.ema if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, cfg.classes)
        elapsed = time.time() - start

        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d}/{cfg.epochs} | loss={train_loss:.4f} | "
            f"val_f1={val_f1:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.2e} | "
            f"{elapsed:.1f}s | val_pred={val_metrics['pred_distribution']}",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_macro_f1": round(val_f1, 5),
            "val_accuracy": round(val_acc, 5),
            "lr": lr_now,
        })

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            patience = 0
            save_checkpoint(
                eval_model,
                cfg.best_model_path,
                extra={
                    "model_name": cfg.model_name,
                    "img_size": cfg.img_size,
                    "mean": list(cfg.mean),
                    "std": list(cfg.std),
                    "classes": list(cfg.classes),
                    "val_macro_f1": val_f1,
                    "epoch": epoch,
                },
            )
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (best val_f1={best_f1:.4f} @ epoch {best_epoch})")
                break

    # Write training log.
    if history:
        with open(cfg.train_log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # Reload best checkpoint and report the final held-out test macro-F1.
    best_model = build_model(cfg.model_name, cfg.num_classes, pretrained=False).to(device)
    ckpt = torch.load(cfg.best_model_path, map_location=device)
    best_model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(best_model, test_loader, device, cfg.classes)

    print("\n==== Held-out TEST results (best checkpoint) ====")
    print(f"test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"test_accuracy={test_metrics['accuracy']:.4f}")
    print(f"test_pred_distribution={test_metrics['pred_distribution']}")

    summary = {
        "model_name": cfg.model_name,
        "img_size": cfg.img_size,
        "best_val_macro_f1": round(best_f1, 5),
        "best_epoch": best_epoch,
        "test_macro_f1": round(test_metrics["macro_f1"], 5),
        "test_accuracy": round(test_metrics["accuracy"], 5),
        "test_per_class": {
            name: {
                "precision": round(test_metrics["report"][name]["precision"], 4),
                "recall": round(test_metrics["report"][name]["recall"], 4),
                "f1": round(test_metrics["report"][name]["f1-score"], 4),
                "support": int(test_metrics["report"][name]["support"]),
            }
            for name in cfg.classes
        },
    }
    save_json(summary, cfg.training_summary_path)
    return summary


if __name__ == "__main__":
    run_training()
