"""Cascade configuration and the packaged presets (``rsdiff1.5``, ``rsdiff1``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PRESETS = ("rsdiff1.5", "rsdiff1")
DEFAULT_PRESET = "rsdiff1.5"


@dataclass
class DataConfig:
    name: str = "rsicd"
    root: str = "data/RSICD_optimal"
    image_size: int = 128
    num_workers: int = 4


@dataclass
class UnetArch:
    """imagen-pytorch ``Unet`` hyperparameters for one cascade stage.

    ``dim`` through ``layer_cross_attns`` are passed to ``imagen_pytorch.Unet`` unchanged.
    ``num_resnet_blocks`` is an int or a per-stage tuple. ``optimizer`` and ``params_m``
    record the training recipe and the measured parameter count (millions).
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
    """Imagen-style text-conditioned cascade: a base UNet plus an optional SR UNet."""

    image_sizes: tuple[int, ...] = (128, 256)
    timesteps: int = 1000
    cond_drop_prob: float = 0.1
    text_encoder: str = "t5-base"
    base: UnetArch = field(default_factory=UnetArch)
    sr: UnetArch | None = None


@dataclass
class OptimConfig:
    lr: float = 1.0e-4
    warmup_steps: int = 10000
    weight_decay: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs/run"
    batch_size: int = 64
    mixed_precision: str = "bf16"  # no | fp16 | bf16


@dataclass
class MetaConfig:
    """Preset provenance: model line, parameter count, expected and measured FID."""

    version: str = ""
    reflects: str = ""
    params_m: float | None = None
    fid_expected: float | None = None
    fid_measured: float | None = None
    notes: str = ""


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config, validating every key against :class:`RunConfig`."""
    merged = OmegaConf.merge(OmegaConf.structured(RunConfig), OmegaConf.load(path))
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def load_preset(name: str = DEFAULT_PRESET) -> RunConfig:
    """Load a packaged preset by name (see :data:`PRESETS`)."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {', '.join(PRESETS)}")
    with resources.as_file(resources.files("rsdiff.configs") / f"{name}.yaml") as path:
        return load_config(path)


def resolve_config(spec: str | Path) -> RunConfig:
    """Return the preset named ``spec``, or load ``spec`` as a YAML path."""
    return load_preset(str(spec)) if str(spec) in PRESETS else load_config(spec)
