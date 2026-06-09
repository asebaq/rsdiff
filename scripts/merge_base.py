"""Reinject the frozen base UNet into a base-stripped SR milestone.

strip_base.py drops the frozen base (unets.0 + ema 0.* + optim0/scaler0) from SR
checkpoints to slim them. Cascade sampling/FID needs the base back, so this merges
the base tensors from a full LR-base checkpoint into a slim SR milestone, producing
a checkpoint that trainer.load accepts for two-unet cascade sampling.

Usage: python merge_base.py <slim_sr_ckpt> <base_ckpt> <out_ckpt>
"""
import sys

import torch

slim_path, base_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
slim = torch.load(slim_path, map_location="cpu", weights_only=False)
base = torch.load(base_path, map_location="cpu", weights_only=False)

# base ckpt is an LR-only trainer.save: its single unet is index 0.
m0 = len(slim["model"]), len(slim["ema"])
for k, v in base["model"].items():
    if k.startswith("unets.0."):
        slim["model"][k] = v
for k, v in base["ema"].items():
    if k.startswith("0."):
        slim["ema"][k] = v
for k in ("optim0", "scaler0"):
    if k in base:
        slim[k] = base[k]

torch.save(slim, out_path)
print(f"merged base: model {m0[0]}->{len(slim['model'])}, ema {m0[1]}->{len(slim['ema'])}")
