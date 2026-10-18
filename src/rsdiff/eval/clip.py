"""CLIP score of a generation bundle, with a shuffled-caption null baseline.

The score is the mean cosine similarity between each image and its caption under OpenAI
CLIP ViT-B/32. The baseline pairs every image with another image's caption (fixed shuffle,
seed 17); ``delta`` is how much the real pairing beats chance. Published release values:
0.278 vs 0.232 (delta +0.046) on the 1,093-image RSICD test bundle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from rsdiff.eval.fid import _auto_device

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class CLIPResult:
    clip_score: float
    clip_score_std: float
    clip_score_shuffled: float
    delta: float
    n: int
    model: str


def bundle_pairs(gen_dir: str | Path) -> tuple[list[str], list[Path]]:
    """Return the captions and PNGs of a bundle written by ``rsdiff sample``.

    Captions come from ``captions.txt``; PNGs are paired with them in sorted file order.
    """
    gen_dir = Path(gen_dir)
    captions = (gen_dir / "captions.txt").read_text(encoding="utf-8").splitlines()
    pngs = sorted(p for p in gen_dir.glob("*.png") if p.name != "_grid.png")
    if len(pngs) != len(captions):
        raise ValueError(f"{len(pngs)} PNGs but {len(captions)} captions in {gen_dir}")
    return captions, pngs


def clip_score(
    gen_dir: str | Path,
    model_id: str = DEFAULT_CLIP_MODEL,
    shuffle_seed: int = 17,
    device: str | None = None,
) -> CLIPResult:
    from transformers import CLIPModel, CLIPProcessor

    captions, pngs = bundle_pairs(gen_dir)
    device = device or _auto_device()
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    with torch.no_grad():
        text = []
        for i in range(0, len(captions), 64):
            tk = proc(
                text=captions[i : i + 64], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text.append(model.get_text_features(**tk).cpu())
        image = []
        for i in range(0, len(pngs), 32):
            imgs = [Image.open(p).convert("RGB") for p in pngs[i : i + 32]]
            image.append(
                model.get_image_features(**proc(images=imgs, return_tensors="pt").to(device)).cpu()
            )

    te = torch.cat(text)
    te = te / te.norm(dim=-1, keepdim=True)
    ie = torch.cat(image)
    ie = ie / ie.norm(dim=-1, keepdim=True)
    sim = (te * ie).sum(-1)

    order = list(range(len(captions)))
    random.Random(shuffle_seed).shuffle(order)
    sim_shuffled = (te[order] * ie).sum(-1)

    return CLIPResult(
        clip_score=sim.mean().item(),
        clip_score_std=sim.std().item(),
        clip_score_shuffled=sim_shuffled.mean().item(),
        delta=(sim.mean() - sim_shuffled.mean()).item(),
        n=len(pngs),
        model=model_id,
    )
