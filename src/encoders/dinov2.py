from __future__ import annotations

from os import PathLike

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model

from ._utils import DEFAULT_CONFIG_PATH, ImageInput, load_image, load_model_id, select_device


class DINOv2Encoder:
    def __init__(
        self,
        device: str | torch.device | None = None,
        config_path: str | PathLike[str] = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.device = select_device(device)
        self.model_id = load_model_id("dinov2", config_path)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = Dinov2Model.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_image(self, image: ImageInput) -> torch.Tensor:
        inputs = self.processor(images=load_image(image), return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        # The first token is DINOv2's CLS token, used as the global representation.
        embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embedding, p=2, dim=-1).squeeze(0)
