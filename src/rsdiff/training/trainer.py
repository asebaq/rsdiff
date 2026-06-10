"""Single-file training loop for the v0 text-conditional cascade.

Intentionally minimal — once the scope is locked after lit review we will
either keep this loop or switch to a HF ``diffusers`` example-style trainer.
"""

from __future__ import annotations

from pathlib import Path

from rsdiff.training.config import RunConfig


def train(cfg: RunConfig) -> None:  # pragma: no cover - skeleton
    raise NotImplementedError(
        "Trainer not implemented yet. Use the cascade scripts in ddpm/ as the "
        "reference until the diffusers-native trainer lands."
    )


def sample(cfg: RunConfig, prompt: str, n: int = 1, out_dir: str | Path | None = None) -> None:  # pragma: no cover
    raise NotImplementedError("Sampling pipeline pending milestone scope lock.")
