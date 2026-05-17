from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UNetConfig:
    sample_size: int = 128
    in_channels: int = 3
    out_channels: int = 3
    layers_per_block: int = 2
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512)
    down_block_types: tuple[str, ...] = (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    )
    up_block_types: tuple[str, ...] = (
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
    cross_attention_dim: int = 768
    attention_head_dim: int = 8


def build_unet(cfg: UNetConfig | dict[str, Any]):
    """Construct a diffusers ``UNet2DConditionModel`` from our config.

    Imported lazily so that the package can be imported without diffusers
    installed (handy for docs / tests that don't touch the model).
    """
    from diffusers import UNet2DConditionModel

    if isinstance(cfg, dict):
        cfg = UNetConfig(**cfg)

    return UNet2DConditionModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=cfg.block_out_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
        cross_attention_dim=cfg.cross_attention_dim,
        attention_head_dim=cfg.attention_head_dim,
    )


def build_text_encoder(name: str = "t5-base"):
    """Return ``(tokenizer, encoder, embed_dim)`` for a frozen text encoder.

    Supports any HF model id; ``t5-base`` matches the thesis baseline.
    Encoder is set to eval mode and returned with grads disabled.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    embed_dim = getattr(encoder.config, "d_model", None) or getattr(
        encoder.config, "hidden_size", None
    )
    if embed_dim is None:
        raise RuntimeError(f"Could not infer embed dim for {name}")

    return tokenizer, encoder, int(embed_dim)
