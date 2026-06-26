"""Evaluate the trained checkpoint on a labeled folder using cv2 -> predict.

This reproduces the grading platform's call path (``cv2.imread`` BGR ndarray ->
``predict`` -> class string) so the reported macro-F1 reflects what the platform
would compute.

Usage:
    python scripts/evaluate_folder.py --data-dir data/external_test
"""
import argparse
import sys
from pathlib import Path

import cv2
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Allow running as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_CONFIG

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Single checkpoint. Default: ensemble of results/model_*.pth")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    classes = list(DEFAULT_CONFIG.classes)
    if args.checkpoint:
        from src.inference import WeatherPredictor
        predictor = WeatherPredictor(args.checkpoint)
    else:
        from src.inference import EnsemblePredictor
        members = sorted(DEFAULT_CONFIG.results_dir.glob("model_*.pth"))
        if not members:
            members = [DEFAULT_CONFIG.results_dir / "best_model.pth"]
        print("ensemble members:", [p.name for p in members])
        predictor = EnsemblePredictor(members)

    y_true, y_pred = [], []
    for ci, name in enumerate(classes):
        class_dir = data_dir / name
        if not class_dir.exists():
            continue
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in IMG_EXTS:
                continue
            image = cv2.imread(str(file))  # BGR ndarray, exactly like the platform
            if image is None:
                continue
            label = predictor.predict(image)
            y_true.append(name)
            y_pred.append(label)

    if not y_true:
        print("No images evaluated. Did you prepare the folder?", file=sys.stderr)
        return 1

    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro")
    acc = accuracy_score(y_true, y_pred)
    print(f"Evaluated {len(y_true)} images from {data_dir}")
    print(f"macro_f1={macro_f1:.4f}  accuracy={acc:.4f}")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
