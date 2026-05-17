#!/usr/bin/env python3
"""Strip an ImagenTrainer checkpoint down to inference-only weights.

Drops optimizer + scaler + step counter. Keeps EMA (default) and/or the online
model. Optionally casts weights to fp16 to shave another ~2x off the file.

Use cases:
    - Public release on HuggingFace Hub (smaller, no optimizer leak)
    - Backups when disk is tight
    - Shipping inference snapshots between machines

Usage:
    python scripts/strip_checkpoint.py IN.pt OUT.pt [--keep ema|model|both] [--fp16]

Loading the stripped file:
    ck = torch.load("OUT.pt", map_location="cpu", weights_only=False)
    imagen.load_state_dict(ck["ema"], strict=False)  # if keep=ema
    # or
    imagen.load_state_dict(ck["model"])              # if keep=model

For sample_grid.py / ImagenTrainer.load(), keep the full ema dict shape;
imagen-pytorch's ImagenTrainer.load expects 'model' + 'ema' keys.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

KEEP_CHOICES = ("ema", "model", "both")


def cast_state_dict(sd: dict, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=pathlib.Path, help="Source checkpoint.pt")
    p.add_argument("output", type=pathlib.Path, help="Destination .pt")
    p.add_argument("--keep", choices=KEEP_CHOICES, default="ema",
                   help="Which weight stream to retain (default: ema).")
    p.add_argument("--fp16", action="store_true",
                   help="Cast float tensors to float16 for smaller file size.")
    args = p.parse_args()

    if not args.input.is_file():
        sys.exit(f"no such file: {args.input}")

    src_size = args.input.stat().st_size / 1e6
    ck = torch.load(args.input, map_location="cpu", weights_only=False)

    out: dict = {"version": ck.get("version", "unknown")}
    if "steps" in ck:
        out["steps"] = ck["steps"]

    if args.keep in ("ema", "both"):
        if "ema" not in ck:
            sys.exit("checkpoint has no 'ema' key")
        out["ema"] = cast_state_dict(ck["ema"], torch.float16) if args.fp16 else ck["ema"]
    if args.keep in ("model", "both"):
        if "model" not in ck:
            sys.exit("checkpoint has no 'model' key")
        out["model"] = cast_state_dict(ck["model"], torch.float16) if args.fp16 else ck["model"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    dst_size = args.output.stat().st_size / 1e6
    print(f"in : {args.input}  ({src_size:.1f} MB)")
    print(f"out: {args.output}  ({dst_size:.1f} MB)  keep={args.keep} fp16={args.fp16}")
    print(f"reduction: {src_size - dst_size:.1f} MB  ({(1 - dst_size/src_size)*100:.1f}%)")
    print(f"top-level keys: {list(out.keys())}")


if __name__ == "__main__":
    main()
