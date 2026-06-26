"""Central configuration for the weather image classification project.

All tunable knobs live here. The notebook (main.ipynb) and the training entry
point import a single ``Config`` instance so behaviour stays consistent between
interactive runs and scripted runs.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    train_dir: Path = Path("train")
    results_dir: Path = Path("results")
    outputs_dir: Path = Path("outputs")
    external_test_dir: Path = Path("data/external_test")

    # --------------------------------------------------------------- classes
    # Fixed, sorted class order. The index of each class is its label id.
    classes: Tuple[str, ...] = ("cloudy", "rainy", "snowy", "sunny")

    # ----------------------------------------------------------------- model
    # Backbone names are torchvision-native so inference needs only
    # torch + torchvision (no timm dependency on the grading platform).
    # Supported: convnext_tiny, convnext_small, efficientnet_v2_s, resnet50
    model_name: str = "convnext_small"
    pretrained: bool = True
    img_size: int = 224
    checkpoint_name: str = "best_model"

    # -------------------------------------------------------------- training
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    use_class_weight: bool = True
    class_weight_power: float = 0.5  # 1.0=balanced, 0.5=sqrt (softer on minorities)
    use_amp: bool = True
    early_stopping_patience: int = 10
    use_ema: bool = True
    ema_decay: float = 0.999
    max_grad_norm: float = 1.0

    # ----------------------------------------------------- augmentation knobs
    rrc_scale: Tuple[float, float] = (0.8, 1.0)
    color_jitter: float = 0.2
    random_erasing_p: float = 0.15

    # ------------------------------------------------------------- normalize
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ----------------------------------------------------- derived file paths
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def best_model_path(self) -> Path:
        return self.results_dir / f"{self.checkpoint_name}.pth"

    @property
    def class_to_idx_path(self) -> Path:
        return self.results_dir / "class_to_idx.json"

    @property
    def idx_to_class_path(self) -> Path:
        return self.results_dir / "idx_to_class.json"

    @property
    def training_summary_path(self) -> Path:
        return self.results_dir / "training_summary.json"

    @property
    def train_log_path(self) -> Path:
        return self.outputs_dir / "train_log.csv"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.outputs_dir / "confusion_matrix.png"

    @property
    def training_curves_path(self) -> Path:
        return self.outputs_dir / "training_curves.png"

    def class_to_idx(self):
        return {name: i for i, name in enumerate(self.classes)}

    def idx_to_class(self):
        return {i: name for i, name in enumerate(self.classes)}


DEFAULT_CONFIG = Config()
