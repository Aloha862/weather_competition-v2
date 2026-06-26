"""Small, dependency-light helpers shared across the project.

Kept Python 3.9 compatible (no ``X | Y`` type unions) because the grading
platform runs Python 3.9.
"""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make a run as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(
    model: torch.nn.Module,
    path: Union[str, Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], map_location: Any = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
