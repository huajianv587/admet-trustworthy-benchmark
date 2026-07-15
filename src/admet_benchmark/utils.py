from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def figures(self) -> Path:
        return self.results / "figures"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def require_module(module_name: str, install_hint: str | None = None) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        hint = install_hint or module_name
        raise ImportError(
            f"Missing optional dependency '{module_name}'. Install it with: {hint}"
        ) from exc


def as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    try:
        return values.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(values)
