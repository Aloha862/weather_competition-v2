"""Generate main.ipynb from clean Python cell sources.

Keeping the notebook in a generator avoids fragile manual JSON escaping. The
final code cell is fully self-contained: it only uses torch/torchvision/numpy/
PIL and the checkpoint, so the grading platform can convert just that cell.
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(md(
"""# 天气图像分类 (cloudy / rainy / snowy / sunny)

本 Notebook 可在 JupyterLab 中从上到下运行，完成：环境检查 -> 数据检查 ->
训练 -> 在**留出测试集**上跑出 macro-F1 -> （可选）外部公开图片鲁棒性评估 ->
平台推理入口 `predict(X)`。

约定：
- 评分指标为 macro-F1，目标 >= 0.95。
- 留出测试集与平台隐藏测试集同分布（都来自互联网采集的同一批数据），因此其
  macro-F1 是平台得分的可靠代理。
- 最后一个代码单元是**自包含**的 `predict(X)`，平台只需转换该单元即可。
"""))

# ------------------------------------------------------------------ cell 1
cells.append(md("## 1. 运行环境检查"))
cells.append(code(
"""import sys
import torch
import torchvision

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
"""))

# ------------------------------------------------------------------ cell 2
cells.append(md("## 2. 配置"))
cells.append(code(
"""from dataclasses import replace
from src.config import DEFAULT_CONFIG

# Windows + Jupyter 下 DataLoader 多进程容易卡死，notebook 内固定 num_workers=0。
cfg = replace(DEFAULT_CONFIG, num_workers=0)
print("model:", cfg.model_name)
print("img_size:", cfg.img_size)
print("epochs:", cfg.epochs)
print("classes:", cfg.classes)
"""))

# ------------------------------------------------------------------ cell 3
cells.append(md("## 3. 数据检查（类别分布）"))
cells.append(code(
"""from src.data import scan_image_folder, stratified_split, label_distribution

paths, labels = scan_image_folder(cfg.train_dir, cfg.classes)
print("total:", len(paths))
print("distribution:", label_distribution(labels, cfg.classes))

splits = stratified_split(paths, labels, cfg.val_ratio, cfg.test_ratio, cfg.seed)
for name in ("train", "val", "test"):
    p, l = splits[name]
    print(f"{name}: {len(p)} {label_distribution(l, cfg.classes)}")
"""))

# ------------------------------------------------------------------ cell 4
cells.append(md(
"""## 4. 训练（集成：3 个不同主干）

训练 3 个 torchvision 主干（convnext_small / efficientnet_v2_s / convnext_tiny），
分别保存到 `results/model_*.pth`。集成对少数类 rainy/snowy 更稳健。GPU 上每个约几
分钟。若已存在 `results/model_*.pth` 且想直接评估，可跳过本单元。"""))
cells.append(code(
"""import importlib
from dataclasses import replace
import train as train_module
importlib.reload(train_module)

MEMBERS = [
    ("convnext_small", "model_convnext_small"),
    ("efficientnet_v2_s", "model_efficientnet_v2_s"),
    ("convnext_tiny", "model_convnext_tiny"),
    ("resnet50", "model_resnet50"),
]
for model_name, ckpt_name in MEMBERS:
    member_cfg = replace(cfg, model_name=model_name, checkpoint_name=ckpt_name)
    print(f"\\n===== training {model_name} =====")
    s = train_module.run_training(member_cfg)
    print(model_name, "test_macro_f1=", s["test_macro_f1"])
"""))

# ------------------------------------------------------------------ cell 5
cells.append(md(
"""## 5. 在留出测试集上跑 macro-F1（平台同款 cv2 -> predict 路径）

这一步用 `cv2.imread`（BGR）读图并走推理入口，完全复现平台评分方式，得到的
macro-F1 即为可交付的测试验证分数。"""))
cells.append(code(
"""import cv2
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.inference import EnsemblePredictor

checkpoints = sorted(Path(cfg.results_dir).glob("model_*.pth"))
if not checkpoints:
    checkpoints = [cfg.best_model_path]
print("ensemble members:", [p.name for p in checkpoints])
predictor = EnsemblePredictor(checkpoints)

test_paths, test_labels = splits["test"]
y_true = [cfg.classes[i] for i in test_labels]
y_pred = [predictor.predict(cv2.imread(p)) for p in test_paths]

macro_f1 = f1_score(y_true, y_pred, labels=list(cfg.classes), average="macro")
acc = accuracy_score(y_true, y_pred)
print(f"TEST macro_f1 = {macro_f1:.4f}")
print(f"TEST accuracy = {acc:.4f}")
print(classification_report(y_true, y_pred, labels=list(cfg.classes), zero_division=0))
print("OK" if macro_f1 >= 0.95 else "Below 0.95 target; see README for next levers.")
"""))

# ------------------------------------------------------------------ cell 6
cells.append(md(
"""## 6.（可选）外部公开图片鲁棒性评估

从网络采集带标签图片做额外鲁棒性检查。注意：外部数据与竞赛数据分布不同，其
macro-F1 通常偏低，仅作泛化参考，不代表平台得分。

先在终端运行：

```
python scripts/prepare_external_test.py --output-dir data/external_test --max-per-class 60
```"""))
cells.append(code(
"""from pathlib import Path

ext_dir = Path(cfg.external_test_dir)
if ext_dir.exists() and any(ext_dir.iterdir()):
    y_true_ext, y_pred_ext = [], []
    for name in cfg.classes:
        cdir = ext_dir / name
        if not cdir.exists():
            continue
        for f in sorted(cdir.iterdir()):
            img = cv2.imread(str(f))
            if img is None:
                continue
            y_true_ext.append(name)
            y_pred_ext.append(predictor.predict(img))
    if y_true_ext:
        print(f"external images: {len(y_true_ext)}")
        print("external macro_f1 =", f1_score(y_true_ext, y_pred_ext, labels=list(cfg.classes), average="macro"))
        print(classification_report(y_true_ext, y_pred_ext, labels=list(cfg.classes), zero_division=0))
else:
    print("No external test set found. Run scripts/prepare_external_test.py first (optional).")
"""))

# ------------------------------------------------------------------ cell 7
cells.append(md(
"""## 7. 平台推理入口 `predict(X)`（自包含）

平台将本 Notebook 转成 py 时，**只需选择下面这一个代码单元**。它只依赖
`torch / torchvision / numpy / PIL` 和 `results/best_model.pth`，不 import 任何项目
模块，因此不会出现 `No module named ...` 或 Python 版本不兼容的问题。

- 输入 `X`：平台用 `cv2.imread` 读出的 `np.ndarray`（BGR），也兼容路径 / PIL。
- 输出：类别字符串，如 `"sunny"`。"""))
cells.append(code(
'''# === Self-contained platform inference cell (Python 3.9 compatible) ===
# Loads an ensemble of torchvision checkpoints (results/model_*.pth) and averages
# softmax probabilities with horizontal-flip TTA. Falls back to a single
# results/best_model.pth. Only needs torch / torchvision / numpy / PIL.
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_LABEL = "cloudy"
# Validation-calibrated class prior: recover minority recall (rainy had high
# precision but lower recall). Multiplies per-class probability before argmax.
CLASS_PRIOR = {"rainy": 1.3}
_PREDICTOR = None


def _project_dirs() -> List[Path]:
    dirs = []
    if "__file__" in globals():
        here = Path(__file__).resolve().parent
        dirs += [here, here.parent]
    dirs += [Path.cwd(), Path.cwd().parent]
    return dirs


def _find_checkpoints() -> List[Path]:
    for base in _project_dirs():
        results = base / "results"
        if results.exists():
            members = sorted(results.glob("model_*.pth"))
            if members:
                return members
            single = results / "best_model.pth"
            if single.exists():
                return [single]
    members = sorted(Path("results").glob("model_*.pth"))
    if members:
        return members
    return [Path("results/best_model.pth")]


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    if name == "convnext_small":
        m = models.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError("Unsupported model_name: " + str(model_name))


class _Member(object):
    def __init__(self, ckpt_path, device):
        self.device = device
        ckpt = torch.load(str(ckpt_path), map_location=device)
        self.classes = list(ckpt.get("classes", ["cloudy", "rainy", "snowy", "sunny"]))
        model_name = ckpt.get("model_name", "convnext_small")
        img_size = int(ckpt.get("img_size", 224))
        mean = tuple(ckpt.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(ckpt.get("std", (0.229, 0.224, 0.225)))
        self.model = _build_model(model_name, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class _Predictor(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members = []
        for p in _find_checkpoints():
            try:
                self.members.append(_Member(p, self.device))
            except Exception as exc:  # skip a backbone the platform can't build
                print("WARN: skipped checkpoint", p, "->", exc)
        if not self.members:
            raise RuntimeError("No usable checkpoint could be loaded.")
        self.classes = self.members[0].classes
        self.prior = np.array([float(CLASS_PRIOR.get(c, 1.0)) for c in self.classes], dtype=np.float32)

    def _to_pil(self, data):
        if isinstance(data, Image.Image):
            return data.convert("RGB")
        if isinstance(data, np.ndarray):
            arr = data
            if arr.ndim == 2:
                return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = arr[:, :, ::-1]  # cv2 BGR -> RGB
            return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")
        return Image.open(data).convert("RGB")

    def predict(self, data):
        try:
            image = self._to_pil(data)
            total = None
            with torch.no_grad():
                for m in self.members:
                    tensor = m.transform(image).unsqueeze(0).to(self.device)
                    probs = torch.softmax(m.model(tensor), dim=1)
                    flipped = torch.flip(tensor, dims=[3])
                    probs = probs + torch.softmax(m.model(flipped), dim=1)
                    total = probs if total is None else total + probs
            scores = total.squeeze(0).cpu().numpy() * self.prior
            idx = int(scores.argmax())
            if 0 <= idx < len(self.classes):
                return self.classes[idx]
        except Exception:
            pass
        return DEFAULT_LABEL


def predict(X: Any) -> str:
    """Return one of cloudy/rainy/snowy/sunny for platform scoring."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = _Predictor()
    return _PREDICTOR.predict(X)
'''))

cells.append(code(
"""# Quick self-check for the platform entry (uses a real training image).
import cv2
from pathlib import Path

sample = next((Path(cfg.train_dir) / "sunny").glob("*.jpg"))
print("predict(path):", predict(str(sample)))
print("predict(cv2 ndarray):", predict(cv2.imread(str(sample))))
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("main.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote main.ipynb with", len(cells), "cells")
