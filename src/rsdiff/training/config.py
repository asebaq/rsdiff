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
class UnetArch:
    """imagen-pytorch Unet hyperparameters for one cascade stage.

    Field names mirror ``imagen_pytorch.Unet`` so a stage can be built directly.
    ``num_resnet_blocks`` is int or per-stage tuple. ``params_m`` is the measured
    parameter count (millions) — documentation only, not consumed at build time.
    """

    dim: int = 128
    cond_dim: int = 256
    dim_mults: tuple[int, ...] = (1, 2, 2, 2)
    num_resnet_blocks: Any = 0
    layer_attns: tuple[bool, ...] = (False, True, True, True)
    layer_cross_attns: tuple[bool, ...] = (False, True, True, True)
    optimizer: str = "adam"  # adam | adafactor
    params_m: float | None = None


@dataclass
class CascadeConfig:
    """Cascaded LR (+ optional SR) imagen-style pipeline (the thesis baseline).

    Parallel to ``model`` (the diffusers rewrite). ``enabled: false`` by default so
    existing single-stage configs are unaffected. ``sr: null`` = LR-only.
    """

    enabled: bool = False
    image_sizes: tuple[int, ...] = (128, 256)
    timesteps: int = 1000
    cond_drop_prob: float = 0.1
    text_encoder: str = "t5-base"
    base: UnetArch = field(default_factory=UnetArch)
    sr: UnetArch | None = None


@dataclass
class MetaConfig:
    """Preset provenance — which model line, expected/measured FID, notes."""

    version: str = ""          # rsdiff1 | rsdiff1.5-light | ...
    reflects: str = ""         # "paper §3.3 prose" | "thesis actual run" | ...
    params_m: float | None = None
    fid_expected: float | None = None
    fid_measured: float | None = None
    notes: str = ""


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)


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
