from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
ImageInput = Image.Image | str | PathLike[str]


def select_device(device: str | torch.device | None = None) -> torch.device:
    """Return an explicitly requested device or the best available local device."""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_id(
    encoder_name: str,
    config_path: str | PathLike[str] = DEFAULT_CONFIG_PATH,
) -> str:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file) or {}

    try:
        model_id = config[encoder_name]["model_id"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Missing 'model_id' for encoder '{encoder_name}' in {path}"
        ) from exc

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            f"Invalid 'model_id' for encoder '{encoder_name}' in {path}"
        )
    return model_id


def load_image(image: ImageInput) -> Image.Image:
    """Load an image path or normalize a PIL image to RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, PathLike)):
        with Image.open(image) as opened_image:
            return opened_image.convert("RGB")
    raise TypeError("image must be a PIL.Image.Image or a filesystem path")
