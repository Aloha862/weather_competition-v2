"""Simulate the platform: extract the self-contained predict cell to main.py,
then evaluate the held-out test split through cv2.imread -> main.predict.
"""
import json
import subprocess
import sys
from pathlib import Path


def extract_main_py():
    nb = json.loads(Path("main.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def predict(X" in src and "_find_checkpoints" in src:
            Path("main.py").write_text(src, encoding="utf-8")
            return True
    return False


VERIFY_SNIPPET = r"""
import cv2
from sklearn.metrics import f1_score, accuracy_score, classification_report
import main
from src.config import DEFAULT_CONFIG
from src.data import scan_image_folder, stratified_split

cfg = DEFAULT_CONFIG
paths, labels = scan_image_folder(cfg.train_dir, cfg.classes)
splits = stratified_split(paths, labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
test_paths, test_labels = splits["test"]
y_true = [cfg.classes[i] for i in test_labels]
y_pred = [main.predict(cv2.imread(p)) for p in test_paths]
print("types:", set(type(x).__name__ for x in y_pred))
macro = f1_score(y_true, y_pred, labels=list(cfg.classes), average="macro")
print("PLATFORM_PATH test macro_f1 = %.4f" % macro)
print("PLATFORM_PATH accuracy = %.4f" % accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, labels=list(cfg.classes), zero_division=0))
"""


def main():
    if not extract_main_py():
        print("ERROR: could not find self-contained predict cell in main.ipynb")
        return 1
    print("Wrote main.py from notebook predict cell.")
    # Run verification in a fresh interpreter (like the platform importing main).
    result = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET], cwd=".")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
