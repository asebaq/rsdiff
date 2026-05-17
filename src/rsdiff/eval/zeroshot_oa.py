"""Zero-shot classification overall accuracy using CLIP.

Port of the legacy ``evaluate_model.py`` from the thesis repo, cleaned up:
- repo-relative paths
- mps/cuda/cpu auto-detect
- single text-feature pass (was inside the loop in the original)
- batched image encoding
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ZeroShotResult:
    total: int
    correct: int
    missing: int
    accuracy: float
    per_class: dict[str, tuple[int, int]]  # label -> (correct, total)


def zeroshot_oa(
    csv_path: str | Path,
    images_dir: str | Path,
    split: str = "test",
    model_id: str = "openai/clip-vit-base-patch32",
    template: str = "a satellite image of a {label}",
    batch_size: int = 32,
    device: str | None = None,
) -> ZeroShotResult:
    from transformers import CLIPModel, CLIPProcessor

    device = device or _auto_device()
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for split={split!r} in {csv_path}")

    classes = sorted(df["label"].dropna().unique().tolist())
    images_dir = Path(images_dir)

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)

    with torch.no_grad():
        prompts = [template.format(label=c) for c in classes]
        text_in = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        text_feats = model.get_text_features(**text_in)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

    correct = 0
    total = 0
    missing = 0
    per_class: dict[str, list[int]] = {c: [0, 0] for c in classes}

    buf_imgs: list[Image.Image] = []
    buf_labels: list[str] = []

    def _flush() -> None:
        nonlocal correct, total
        if not buf_imgs:
            return
        with torch.no_grad():
            img_in = proc(images=buf_imgs, return_tensors="pt").to(device)
            img_feats = model.get_image_features(**img_in)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            logits = (model.logit_scale.exp() * img_feats @ text_feats.t()).softmax(dim=-1)
            preds = logits.argmax(dim=-1).tolist()
        for true_label, pred_idx in zip(buf_labels, preds):
            pred_label = classes[pred_idx]
            per_class[true_label][1] += 1
            total += 1
            if pred_label == true_label:
                per_class[true_label][0] += 1
                correct += 1
        buf_imgs.clear()
        buf_labels.clear()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="zeroshot_oa"):
        path = images_dir / row["filename"]
        if not path.exists():
            missing += 1
            continue
        try:
            buf_imgs.append(Image.open(path).convert("RGB"))
            buf_labels.append(row["label"])
        except Exception:
            missing += 1
            continue
        if len(buf_imgs) >= batch_size:
            _flush()
    _flush()

    acc = (correct / total * 100.0) if total else 0.0
    return ZeroShotResult(
        total=total,
        correct=correct,
        missing=missing,
        accuracy=acc,
        per_class={k: (v[0], v[1]) for k, v in per_class.items()},
    )
