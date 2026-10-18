from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from rsdiff.cascade import build_imagen, build_unet
from rsdiff.config import load_config, load_preset, resolve_config


def _params_m(module) -> float:
    return round(sum(p.numel() for p in module.parameters()) / 1e6, 1)


def test_released_preset_matches_its_recorded_parameter_counts():
    # Recorded counts are standalone UNets; inside the cascade the SR stage gains
    # low-res conditioning layers (92.7M -> 105.0M), which the release checkpoint holds.
    cascade = load_preset("rsdiff1.5").cascade
    assert [_params_m(build_unet(a)) for a in (cascade.base, cascade.sr)] == [
        cascade.base.params_m,
        cascade.sr.params_m,
    ]
    imagen = build_imagen(cascade, text_embed_dim=768)
    assert [_params_m(u) for u in imagen.unets] == [27.2, 105.0]
    assert imagen.image_sizes == (128, 256)


def test_unknown_yaml_key_is_rejected(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("cascade:\n  enabled: true\n")
    with pytest.raises(ConfigKeyError):
        load_config(p)


def test_unknown_preset_is_rejected():
    with pytest.raises(KeyError, match="rsdiff1.5"):
        load_preset("rsdiff2")


def test_resolve_config_prefers_preset_name_then_path(tmp_path: Path):
    assert resolve_config("rsdiff1").meta.version == "rsdiff1"
    p = tmp_path / "c.yaml"
    p.write_text("cascade:\n  timesteps: 250\n  sr: null\n")
    cfg = resolve_config(p)
    assert cfg.cascade.timesteps == 250
    assert cfg.cascade.sr is None
