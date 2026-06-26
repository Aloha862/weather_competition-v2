"""Data loading: folder scanning, transforms, stratified splits, weighting.

Images are read with PIL in RGB. The platform feeds ``cv2.imread`` arrays
(BGR) at inference time, and the inference code converts BGR -> RGB before
applying the exact same transforms, so train/serve preprocessing stays aligned.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import Config


def scan_image_folder(train_dir: Path, classes: Tuple[str, ...]) -> Tuple[List[str], List[int]]:
    """Return (paths, label indices) for ``train_dir/<class>/*.jpg`` layout."""
    train_dir = Path(train_dir)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    paths: List[str] = []
    labels: List[int] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for name in classes:
        class_dir = train_dir / name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() in exts:
                paths.append(str(file))
                labels.append(class_to_idx[name])
    if not paths:
        raise RuntimeError(f"No images found under {train_dir}")
    return paths, labels


def label_distribution(labels: List[int], classes: Tuple[str, ...]) -> Dict[str, int]:
    counter = Counter(labels)
    return {classes[i]: int(counter.get(i, 0)) for i in range(len(classes))}


def stratified_split(
    paths: List[str],
    labels: List[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Class-balanced split into train / val / test."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for label, indices in by_class.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test_part = indices[:n_test]
        val_part = indices[n_test:n_test + n_val]
        train_part = indices[n_test + n_val:]
        test_idx.extend(test_part.tolist())
        val_idx.extend(val_part.tolist())
        train_idx.extend(train_part.tolist())

    def gather(idx_list: List[int]) -> Tuple[List[str], List[int]]:
        return [paths[i] for i in idx_list], [labels[i] for i in idx_list]

    return {
        "train": gather(train_idx),
        "val": gather(val_idx),
        "test": gather(test_idx),
    }


def build_transforms(cfg: Config, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=cfg.rrc_scale, ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=cfg.color_jitter,
                contrast=cfg.color_jitter,
                saturation=cfg.color_jitter,
            ),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
            transforms.RandomErasing(p=cfg.random_erasing_p),
        ])
    # Eval: match the competition's stated 224x224 square resize (no crop) so we
    # keep the whole scene (edge sky/cloud cues matter for weather).
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std),
    ])


class WeatherDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        label = self.labels[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, label


def compute_class_weights(labels: List[int], num_classes: int, power: float = 1.0) -> torch.Tensor:
    """sklearn-style 'balanced' weights raised to ``power`` then renormalised.

    power=1.0 -> standard balanced weights (aggressive on minorities)
    power=0.5 -> square-root softening (better precision/recall balance)
    """
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = counter.get(c, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append((total / (num_classes * count)) ** power)
    w = torch.tensor(weights, dtype=torch.float32)
    # Renormalise so the mean weight is ~1 (keeps loss scale stable).
    w = w * (num_classes / w.sum())
    return w
