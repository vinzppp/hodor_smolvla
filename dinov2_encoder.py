"""DINOv2 visual backbone for the Section 3.1 scene graph.

The paper computes object appearance embeddings by average-pooling DINO-v2
features inside each object mask, and the global scene node from the DINO-v2
CLS token. The original code release ships a DINO-v1 ViT-B/16 encoder; this
module adds the DINO-v2 option while keeping the exact ``(global, patch)``
interface the HODOR processor expects:

    global_repr, patch_repr = encoder(image_tensor)   # (B, D), (B, N, D)

``get_vit`` in ``vit.py`` returns one of these for any ``model.vit_model``
name that starts with ``dinov2`` (see ``DINOV2_HUB`` for the options).

Weights load from ``torch.hub`` (facebookresearch/dinov2) by default, which
needs network access on first use; ``setup_gpu.sh`` pre-fetches them. To run
fully offline, drop a state-dict at ``pretrained/<vit_model>.pth`` and it is
loaded instead.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

# name -> (torch.hub entrypoint, embed_dim, patch_size)
DINOV2_HUB = {
    "dinov2-vit-s14": ("dinov2_vits14", 384, 14),
    "dinov2-vit-b14": ("dinov2_vitb14", 768, 14),
    "dinov2-vit-l14": ("dinov2_vitl14", 1024, 14),
    "dinov2-vit-g14": ("dinov2_vitg14", 1536, 14),
}


class DINOv2Encoder(nn.Module):
    """Wrap a DINOv2 ViT to match the (global, patch) contract of the HODOR
    processor. Returns the CLS token as the global scene representation and the
    normalized patch tokens for masked average pooling."""

    def __init__(
        self,
        model_name: str,
        img_size: int = 224,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        if model_name not in DINOV2_HUB:
            raise KeyError(
                f"Unknown DINOv2 model {model_name!r}; options: {sorted(DINOV2_HUB)}"
            )
        hub_name, embed_dim, patch_size = DINOV2_HUB[model_name]
        if img_size % patch_size != 0:
            raise ValueError(
                f"{model_name} needs model.vit_image_size divisible by "
                f"{patch_size}; got {img_size}. Use 224, 322, 448, ..."
            )
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # If a local checkpoint is provided we still build the architecture via
        # torch.hub (pretrained=False) and load weights from disk, so the run is
        # offline once the hub repo is cached.
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=(checkpoint is None),
        )
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[dinov2] loaded {checkpoint} with missing={list(missing)} "
                    f"unexpected={list(unexpected)}"
                )
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self.model.forward_features(x)
        cls_token = out["x_norm_clstoken"]        # (B, D)
        patch_tokens = out["x_norm_patchtokens"]  # (B, N, D)
        return cls_token, patch_tokens

    def freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
