from __future__ import annotations

from os import PathLike

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from ._utils import DEFAULT_CONFIG_PATH, ImageInput, load_image, load_model_id, select_device


class CLIPEncoder:
    def __init__(
        self,
        device: str | torch.device | None = None,
        config_path: str | PathLike[str] = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.device = select_device(device)
        self.model_id = load_model_id("clip", config_path)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_image(self, image: ImageInput) -> torch.Tensor:
        inputs = self.processor(images=load_image(image), return_tensors="pt")
        inputs = inputs.to(self.device)
        embedding = self.model.get_image_features(**inputs)
        return F.normalize(embedding, p=2, dim=-1).squeeze(0)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must contain at least one prompt")
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        embeddings = self.model.get_text_features(**inputs)
        return F.normalize(embeddings, p=2, dim=-1)
