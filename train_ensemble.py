"""Train a small ensemble of diverse torchvision backbones and evaluate it.

Each member is saved to results/model_<name>.pth. The ensemble averages softmax
(+ hflip TTA) over members, which reliably lifts macro-F1 on the minority
rainy/snowy classes versus any single model.
"""
from dataclasses import replace

import cv2
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import DEFAULT_CONFIG
from src.data import scan_image_folder, stratified_split
from src.inference import EnsemblePredictor
from train import run_training

MEMBERS = [
    ("convnext_small", "model_convnext_small"),
    ("efficientnet_v2_s", "model_efficientnet_v2_s"),
    ("convnext_tiny", "model_convnext_tiny"),
    ("resnet50", "model_resnet50"),
]


def main():
    base = replace(DEFAULT_CONFIG, num_workers=2)
    checkpoints = []
    for model_name, ckpt_name in MEMBERS:
        print(f"\n########## Training {model_name} ##########")
        cfg = replace(base, model_name=model_name, checkpoint_name=ckpt_name)
        summary = run_training(cfg)
        print(f"{model_name} summary: {summary['test_macro_f1']=} {summary['best_val_macro_f1']=}")
        checkpoints.append(cfg.best_model_path)

    print("\n########## Ensemble evaluation ##########")
    cfg = base
    paths, labels = scan_image_folder(cfg.train_dir, cfg.classes)
    splits = stratified_split(paths, labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    test_paths, test_labels = splits["test"]
    predictor = EnsemblePredictor(checkpoints)
    y_true = [cfg.classes[i] for i in test_labels]
    y_pred = [predictor.predict(cv2.imread(p)) for p in test_paths]
    macro_f1 = f1_score(y_true, y_pred, labels=list(cfg.classes), average="macro")
    acc = accuracy_score(y_true, y_pred)
    print(f"ENSEMBLE_TEST macro_f1={macro_f1:.4f} accuracy={acc:.4f}")
    print(classification_report(y_true, y_pred, labels=list(cfg.classes), zero_division=0))


if __name__ == "__main__":
    main()
