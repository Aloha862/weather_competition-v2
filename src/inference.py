"""Canonical inference logic.

The self-contained ``predict`` cell in main.ipynb mirrors this module but inlines
everything so the grading platform only needs torch + torchvision + the
checkpoint. Keeping a tested reference here lets scripts reuse the same path.
"""
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import DEFAULT_CONFIG
from .model import build_model
from .utils import get_device, load_checkpoint, load_json

DEFAULT_LABEL = "cloudy"


class WeatherPredictor:
    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None):
        self.device = device or get_device()
        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model_name = ckpt.get("model_name", DEFAULT_CONFIG.model_name)
        self.img_size = int(ckpt.get("img_size", DEFAULT_CONFIG.img_size))
        self.mean = tuple(ckpt.get("mean", DEFAULT_CONFIG.mean))
        self.std = tuple(ckpt.get("std", DEFAULT_CONFIG.std))
        self.classes = list(ckpt.get("classes", list(DEFAULT_CONFIG.classes)))

        self.model = build_model(self.model_name, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.use_tta = True

    def _to_pil(self, data: Any) -> Image.Image:
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
            # cv2.imread returns BGR; convert to RGB.
            arr = arr[:, :, ::-1]
            return Image.fromarray(arr).convert("RGB")
        if isinstance(data, (str, Path)):
            return Image.open(data).convert("RGB")
        raise TypeError(f"Unsupported input type for prediction: {type(data)}")

    @torch.no_grad()
    def predict(self, data: Any) -> str:
        image = self._to_pil(data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)
        if self.use_tta:
            flipped = torch.flip(tensor, dims=[3])
            probs = probs + torch.softmax(self.model(flipped), dim=1)
        idx = int(probs.argmax(dim=1).item())
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return DEFAULT_LABEL


class EnsemblePredictor:
    """Average softmax probabilities of several checkpoints (+ hflip TTA).

    ``class_prior`` multiplies per-class probabilities before argmax. It is
    calibrated on the validation split to recover minority-class recall (the
    minority classes had high precision but lower recall). Default boosts the
    ``rainy`` class, which had the most recall headroom.
    """

    DEFAULT_PRIOR = {"rainy": 1.3}

    def __init__(self, checkpoint_paths, device: Optional[torch.device] = None, class_prior=None):
        self.device = device or get_device()
        self.members = []
        for p in checkpoint_paths:
            try:
                self.members.append(WeatherPredictor(p, self.device))
            except Exception as exc:
                print(f"WARN: skipped checkpoint {p} -> {exc}")
        if not self.members:
            raise ValueError("EnsemblePredictor needs at least one usable checkpoint")
        self.classes = self.members[0].classes
        prior_map = self.DEFAULT_PRIOR if class_prior is None else class_prior
        self.prior = np.array([float(prior_map.get(c, 1.0)) for c in self.classes], dtype=np.float32)

    @torch.no_grad()
    def predict_proba(self, data: Any):
        """Return averaged class-probability vector (numpy) for one image."""
        total = None
        for m in self.members:
            image = m._to_pil(data)
            tensor = m.transform(image).unsqueeze(0).to(m.device)
            probs = torch.softmax(m.model(tensor), dim=1)
            flipped = torch.flip(tensor, dims=[3])
            probs = probs + torch.softmax(m.model(flipped), dim=1)
            total = probs if total is None else total + probs
        return (total / (2 * len(self.members))).squeeze(0).cpu().numpy()

    def predict(self, data: Any) -> str:
        probs = self.predict_proba(data) * self.prior
        idx = int(probs.argmax())
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return DEFAULT_LABEL
