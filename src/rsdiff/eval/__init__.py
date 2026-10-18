"""Evaluation metrics for generated remote-sensing imagery.

- ``fid``: Fréchet Inception Distance (torchmetrics, Inception feature 2048 by default);
  the metric behind the published FID 65.70.
- ``clip_score``: caption-image CLIP similarity with a shuffled-caption baseline.
"""

from rsdiff.eval.clip import CLIPResult, clip_score
from rsdiff.eval.fid import FIDResult, fid

__all__ = ["CLIPResult", "FIDResult", "clip_score", "fid"]
