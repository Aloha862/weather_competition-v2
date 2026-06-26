# === Self-contained platform inference cell (Python 3.9 compatible) ===
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
