"""Model factories for rsdiff.

v0 wraps ``diffusers`` UNet2DConditionModel and exposes a thin builder that
takes our config dict. DiT/MMDiT backends will plug in here once the v0 baseline
is reproducing thesis numbers.
"""

from rsdiff.models.builders import build_unet, build_text_encoder

__all__ = ["build_unet", "build_text_encoder"]
