"""Evaluation metrics for RS-generation models.

v0 plan:
- ``fid``       — clean-fid + Inception-V3 backbone (and CLIP backbone variant)
- ``clip_score`` — text/image alignment with OpenCLIP
- ``zeroshot_oa`` — zero-shot classification accuracy of generated images
                    against dataset class labels
"""

from rsdiff.eval.fid import FIDResult, fid
from rsdiff.eval.zeroshot_oa import zeroshot_oa

__all__ = ["fid", "FIDResult", "zeroshot_oa"]
