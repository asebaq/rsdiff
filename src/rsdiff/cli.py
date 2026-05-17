"""rsdiff CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", type=Path, required=True, help="YAML config path")


def cmd_train(args: argparse.Namespace) -> int:
    from rsdiff.training.config import load_config
    from rsdiff.training.trainer import train

    cfg = load_config(args.config)
    train(cfg)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from rsdiff.training.config import load_config
    from rsdiff.training.trainer import sample

    cfg = load_config(args.config)
    sample(cfg, prompt=args.prompt, n=args.n, out_dir=args.out_dir)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from rsdiff.eval import zeroshot_oa
    from rsdiff.training.config import load_config

    cfg = load_config(args.config)
    csv = Path(cfg.data.root) / "dataset_rsicd.csv"
    images_dir = args.images_dir or (Path(cfg.data.root) / "RSICD_images")

    if "zeroshot_oa" in args.metric:
        res = zeroshot_oa(csv, images_dir)
        print(f"zeroshot_oa: {res.accuracy:.2f}%  ({res.correct}/{res.total}, missing={res.missing})")
    if "fid" in args.metric or "clip_score" in args.metric:
        print("fid / clip_score not implemented yet — pending v0 scope lock.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsdiff", description="Diffusion models for remote sensing.")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model from a config.")
    _add_common(p_train)
    p_train.set_defaults(fn=cmd_train)

    p_sample = sub.add_parser("sample", help="Sample images from a trained checkpoint.")
    _add_common(p_sample)
    p_sample.add_argument("--prompt", type=str, required=True)
    p_sample.add_argument("-n", type=int, default=1)
    p_sample.add_argument("--out-dir", type=Path, default=Path("generated"))
    p_sample.set_defaults(fn=cmd_sample)

    p_eval = sub.add_parser("eval", help="Run evaluation metrics on a set of images.")
    _add_common(p_eval)
    p_eval.add_argument("--images-dir", type=Path, default=None)
    p_eval.add_argument(
        "--metric",
        nargs="+",
        choices=["fid", "clip_score", "zeroshot_oa"],
        default=["zeroshot_oa"],
    )
    p_eval.set_defaults(fn=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
