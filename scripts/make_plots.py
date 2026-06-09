#!/usr/bin/env python3
"""Generate publication figures from the rsdiff thesis reproduction run.

Reads TSVs + JSON results, writes PNG figures to docs/figures/.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VAST = REPO / "outputs" / "vast"
FIG = REPO / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SR_TSV = VAST / "fid" / "fid_curve_full_sr_gdm.tsv"
CFG_TSV = VAST / "fid" / "fid_cfg_full_sr_gdm_step89050.tsv"
LR_TSV = VAST / "fid" / "fid_curve_full_lr_gdm.tsv"
LOCAL_SR_TSV = REPO / "outputs" / "fid_curve_full_sr_gdm.tsv"
FINAL_DIR = VAST / "legacy/DDPM/logs/full_sr_gdm/generated_images/final_test_step89050_cs5"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_sr_curve() -> None:
    df = pd.read_csv(SR_TSV, sep="\t").sort_values("epoch")
    # Merge in the local ep50/ep100 rows scored at same protocol.
    extra = pd.read_csv(LOCAL_SR_TSV, sep="\t") if LOCAL_SR_TSV.exists() else pd.DataFrame()
    if not extra.empty:
        df = pd.concat([extra, df], ignore_index=True).drop_duplicates("epoch").sort_values("epoch")
    winner = df.loc[df.fid.idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.epoch, df.fid, "o-", color="#1f77b4", lw=1.6)
    ax.scatter([winner.epoch], [winner.fid], color="#d62728", s=80, zorder=5,
               label=f"winner ep{int(winner.epoch)}  FID {winner.fid:.2f}")
    ax.set_xlabel("SR epoch")
    ax.set_ylabel("FID  (cascade-256, N=128, feature=2048, cs=4)")
    ax.set_title("SR milestone FID sweep — RSICD test split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIG / "sr_fid_curve.png"
    fig.savefig(out)
    print(f"wrote {out}")


def plot_cfg_curve() -> None:
    df = pd.read_csv(CFG_TSV, sep="\t").sort_values("cond_scale")
    winner = df.loc[df.fid.idxmin()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.cond_scale, df.fid, "o-", color="#2ca02c", lw=1.6)
    ax.scatter([winner.cond_scale], [winner.fid], color="#d62728", s=80, zorder=5,
               label=f"winner cs={int(winner.cond_scale)}  FID {winner.fid:.2f}")
    ax.set_xlabel("CFG cond_scale")
    ax.set_ylabel("FID  (cascade-256, N=64)")
    ax.set_title(f"CFG sweep on ep{int(df.epoch.iloc[0])}  (step {int(df.step.iloc[0])})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIG / "cfg_sweep.png"
    fig.savefig(out)
    print(f"wrote {out}")


def plot_lr_curve() -> None:
    df = pd.read_csv(LR_TSV, sep="\t").sort_values("epoch")
    winner = df.loc[df.fid.idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.epoch, df.fid, "o-", color="#9467bd", lw=1.6)
    ax.scatter([winner.epoch], [winner.fid], color="#d62728", s=80, zorder=5,
               label=f"SR seed: ep{int(winner.epoch)}  FID {winner.fid:.2f}")
    ax.set_xlabel("LR base epoch")
    ax.set_ylabel("FID  (128, N=64, feature=2048, cs=4)")
    ax.set_title("LR base milestone FID sweep — RSICD test split")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = FIG / "lr_fid_curve.png"
    fig.savefig(out)
    print(f"wrote {out}")


def plot_headline() -> None:
    fid = json.loads((FINAL_DIR / "fid_result.json").read_text())
    clip = json.loads((FINAL_DIR / "clip_result.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # FID side
    ax = axes[0]
    bars = ax.bar(["thesis 2024", "rsdiff repro"], [66.49, fid["fid"]],
                  color=["#9aa7b3", "#1f77b4"], width=0.55)
    for b, v in zip(bars, [66.49, fid["fid"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("FID (lower is better)")
    ax.set_title(f"FID parity — N={fid['n_gen']}, cascade-{fid['size']}, feature={fid['feature']}")
    ax.set_ylim(0, max(70, fid["fid"] + 10))

    # CLIP side
    ax = axes[1]
    bars = ax.bar(["shuffled\nbaseline", "real captions"],
                  [clip["clip_score_shuffled"], clip["clip_score"]],
                  color=["#9aa7b3", "#2ca02c"], width=0.55)
    for b, v in zip(bars, [clip["clip_score_shuffled"], clip["clip_score"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("CLIP cosine similarity")
    ax.set_title(f"CLIP-score — delta +{clip['delta']:.3f} ({clip['model'].split('/')[-1]})")
    ax.set_ylim(0, max(0.35, clip["clip_score"] + 0.05))

    fig.suptitle(f"rsdiff cascade-256: ep650 × cs=5 — N={fid['n_gen']} (RSICD test)")
    fig.tight_layout()
    out = FIG / "headline.png"
    fig.savefig(out)
    print(f"wrote {out}")


def plot_sample_montage(n: int = 9, seed: int = 11) -> None:
    pngs = sorted(p for p in FINAL_DIR.glob("[0-9]*.png"))
    captions = (FINAL_DIR / "captions.txt").read_text().splitlines()
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(pngs)), n))

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = axes.ravel() if rows * cols > 1 else [axes]
    for ax, idx in zip(axes, idxs):
        ax.imshow(Image.open(pngs[idx]))
        ax.set_xticks([]); ax.set_yticks([])
        cap = captions[idx]
        if len(cap) > 60:
            cap = cap[:57] + "..."
        ax.set_title(cap, fontsize=8)
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.suptitle("ep650 × cs=5 — 9 random samples (RSICD test captions)")
    fig.tight_layout()
    out = FIG / "sample_montage.png"
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    plot_sr_curve()
    plot_cfg_curve()
    plot_lr_curve()
    plot_headline()
    plot_sample_montage()


if __name__ == "__main__":
    main()
