from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    name: str = "rsicd"
    root: str = "data/RSICD_optimal"
    image_size: int = 128
    caption_idx: int | None = None
    num_workers: int = 4


@dataclass
class TextEncoderConfig:
    name: str = "t5-base"
    max_length: int = 128


@dataclass
class ModelConfig:
    arch: str = "unet2d_cond"
    sample_size: int = 128
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512)
    cross_attention_dim: int = 768


@dataclass
class DiffusionConfig:
    scheduler: str = "ddpm"
    num_train_timesteps: int = 1000
    prediction_type: str = "epsilon"
    beta_schedule: str = "scaled_linear"


@dataclass
class OptimConfig:
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    max_steps: int = 200_000
    grad_accum: int = 1
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs/run"
    batch_size: int = 32
    mixed_precision: str = "fp16"  # no | fp16 | bf16
    log_every: int = 50
    sample_every: int = 1000
    save_every: int = 5000
    gradient_checkpointing: bool = False


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> RunConfig:
    base = OmegaConf.structured(RunConfig)
    override = OmegaConf.load(path)
    merged = OmegaConf.merge(base, override)
    OmegaConf.resolve(merged)
    # Cast back to dataclass for static-typing friendliness.
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    OmegaConf.save(OmegaConf.structured(cfg), path)


def as_dict(cfg: RunConfig) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]
