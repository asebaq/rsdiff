"""``rsdiff`` command line: sample the released cascade and score generation bundles."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rsdiff import __version__
from rsdiff.config import DEFAULT_PRESET, PRESETS


def cmd_sample(args: argparse.Namespace) -> int:
    from rsdiff.cascade import download_release, generate, load_cascade, seed_everything
    from rsdiff.config import resolve_config
    from rsdiff.datasets.rsicd import sample_captions

    if args.prompt:
        captions = args.prompt * (args.n or 1)
    else:
        captions = sample_captions(args.rsicd, split=args.split, n=args.n, seed=args.seed)
    cfg = resolve_config(args.config)

    if args.checkpoint is None:
        if args.config != DEFAULT_PRESET:
            sys.exit("--checkpoint is required unless --config is the released rsdiff1.5")
        print("fetching released checkpoint (1.9 GB on first use)", file=sys.stderr)
        checkpoint = download_release()
    else:
        checkpoint = args.checkpoint

    seed_everything(args.seed)
    trainer = load_cascade(cfg, checkpoint, cpu=args.cpu)
    print(f"sampling {len(captions)} image(s) on {trainer.device}", file=sys.stderr)
    paths = generate(
        trainer.sample,
        captions,
        args.out_dir,
        cond_scale=args.cond_scale,
        batch_size=args.batch_size,
        stop_at_unet=1 if args.lr_only else None,
        on_batch=lambda done, total: print(f"  {done}/{total}", file=sys.stderr),
    )
    print(f"wrote {len(paths)} PNGs to {args.out_dir}", file=sys.stderr)
    return 0


def cmd_eval_fid(args: argparse.Namespace) -> int:
    from rsdiff.eval.fid import fid

    res = fid(
        gen_dir=args.gen_dir,
        real_csv=args.rsicd / "dataset_rsicd.csv",
        real_dir=args.rsicd / "RSICD_images",
        split=args.split,
        feature=args.feature,
        image_size=args.size,
        max_n=args.max_n,
    )
    print(json.dumps(asdict(res), indent=2))
    return 0


def cmd_eval_clip(args: argparse.Namespace) -> int:
    from rsdiff.eval.clip import clip_score

    print(json.dumps(asdict(clip_score(args.gen_dir, model_id=args.model)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rsdiff", description="Text-to-satellite-image diffusion (RSDiff)."
    )
    p.add_argument("--version", action="version", version=f"rsdiff {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "sample",
        help="Generate images from text with the cascade.",
        description="Generate synthetic 256x256 images. Downloads the released "
        "rsdiff1.5 checkpoint unless --checkpoint is given.",
    )
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", action="append", help="Caption to render (repeatable).")
    src.add_argument("--rsicd", type=Path, help="RSICD root; sample captions from its CSV instead.")
    s.add_argument("--split", default="test", help="RSICD split for --rsicd (default: test).")
    s.add_argument(
        "-n",
        type=int,
        default=None,
        help="Images per --prompt (default 1), or captions drawn from --rsicd (default: all).",
    )
    s.add_argument("--out-dir", type=Path, default=Path("generated"))
    s.add_argument(
        "--config",
        default=DEFAULT_PRESET,
        help=f"Preset ({', '.join(PRESETS)}) or YAML path (default: {DEFAULT_PRESET}).",
    )
    s.add_argument("--checkpoint", type=Path, help="ImagenTrainer checkpoint to load.")
    s.add_argument("--cond-scale", type=float, default=5.0, help="CFG scale (default: 5).")
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--batch-size", type=int, default=2)
    s.add_argument("--lr-only", action="store_true", help="Stop at the 128x128 base UNet.")
    s.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    s.set_defaults(fn=cmd_sample)

    e = sub.add_parser("eval", help="Score a generation bundle.")
    esub = e.add_subparsers(dest="metric", required=True)

    f = esub.add_parser("fid", help="FID against the real RSICD split.")
    f.add_argument("gen_dir", type=Path, help="Directory of generated PNGs.")
    f.add_argument("--rsicd", type=Path, required=True, help="RSICD root (CSV + RSICD_images).")
    f.add_argument("--split", default="test")
    f.add_argument("--feature", type=int, choices=[64, 192, 768, 2048], default=2048)
    f.add_argument("--size", type=int, default=256, help="Resize both sides to this first.")
    f.add_argument("--max-n", type=int, default=None, help="Cap images per side (debug only).")
    f.set_defaults(fn=cmd_eval_fid)

    c = esub.add_parser("clip", help="CLIP score with a shuffled-caption baseline.")
    c.add_argument("gen_dir", type=Path, help="Bundle written by `rsdiff sample`.")
    c.add_argument(
        "--model", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model id."
    )
    c.set_defaults(fn=cmd_eval_clip)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
