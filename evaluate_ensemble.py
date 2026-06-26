"""Evaluate the ensemble on the held-out test split and calibrate minority recall.

Steps:
1. Load all results/model_*.pth members.
2. Cache averaged probabilities on the val and test splits.
3. Search per-class probability multipliers on VAL to maximise macro-F1
   (only nudges the minority classes rainy/snowy up to a small cap so we do not
   overfit the small validation set).
4. Apply the val-chosen multipliers to TEST and report both raw and calibrated.
"""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import DEFAULT_CONFIG
from src.data import scan_image_folder, stratified_split
from src.inference import EnsemblePredictor


def collect_probs(predictor, paths):
    return np.stack([predictor.predict_proba(cv2.imread(p)) for p in paths], axis=0)


def macro_f1_with_mult(probs, y_true_idx, mult):
    pred = (probs * mult).argmax(axis=1)
    return f1_score(y_true_idx, pred, labels=list(range(probs.shape[1])), average="macro")


def main():
    cfg = DEFAULT_CONFIG
    classes = list(cfg.classes)
    paths, labels = scan_image_folder(cfg.train_dir, classes)
    splits = stratified_split(paths, labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    checkpoints = sorted(Path(cfg.results_dir).glob("model_*.pth"))
    print("ensemble members:", [p.name for p in checkpoints])
    predictor = EnsemblePredictor(checkpoints)

    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]
    val_probs = collect_probs(predictor, val_paths)
    test_probs = collect_probs(predictor, test_paths)
    val_y = np.array(val_labels)
    test_y = np.array(test_labels)

    # Raw (no calibration).
    raw_pred = test_probs.argmax(axis=1)
    raw_f1 = f1_score(test_y, raw_pred, labels=list(range(len(classes))), average="macro")
    print(f"RAW   test macro_f1={raw_f1:.4f} accuracy={accuracy_score(test_y, raw_pred):.4f}")

    # Search a mild multiplier for the minority classes rainy(1)/snowy(2) on VAL.
    best_mult = np.ones(len(classes))
    best_val = macro_f1_with_mult(val_probs, val_y, best_mult)
    grid = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    for r in grid:
        for s in grid:
            mult = np.array([1.0, r, s, 1.0])
            score = macro_f1_with_mult(val_probs, val_y, mult)
            if score > best_val:
                best_val = score
                best_mult = mult
    print(f"VAL best multiplier={best_mult.tolist()} val_macro_f1={best_val:.4f}")

    cal_pred = (test_probs * best_mult).argmax(axis=1)
    cal_f1 = f1_score(test_y, cal_pred, labels=list(range(len(classes))), average="macro")
    cal_acc = accuracy_score(test_y, cal_pred)
    print(f"CALIB test macro_f1={cal_f1:.4f} accuracy={cal_acc:.4f}")
    report = classification_report(test_y, cal_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    print(classification_report(test_y, cal_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))
    print("RAINY_SNOWY_MULT", best_mult.tolist())

    import json
    summary = {
        "ensemble_members": [p.name for p in checkpoints],
        "class_prior_multiplier": {classes[i]: float(best_mult[i]) for i in range(len(classes))},
        "test_macro_f1_raw": round(float(raw_f1), 4),
        "test_macro_f1_calibrated": round(float(cal_f1), 4),
        "test_accuracy_calibrated": round(float(cal_acc), 4),
        "test_per_class_calibrated": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in classes
        },
    }
    out = cfg.results_dir / "ensemble_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
