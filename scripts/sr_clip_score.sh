#!/usr/bin/env bash
# CLIP-score: mean cosine similarity between caption embeddings and image
# embeddings on a generation bundle. Also computes shuffled-baseline as a
# null control. Uses openai/clip-vit-base-patch32 (HF transformers).
#
# Reads captions.txt (one per line, parallel to sorted PNG files) and writes
# clip_result.json into the same dir.
set -euo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
GEN_DIR="${GEN_DIR:?set GEN_DIR (eg .../final_test_step102750_cs4)}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"

cd "$ROOT"
python - "$GEN_DIR" "$CLIP_MODEL" <<'PY'
import sys, json, os, random
from glob import glob
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

gen_dir, model_name = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"

captions = open(os.path.join(gen_dir, "captions.txt")).read().splitlines()
pngs = [p for p in sorted(glob(os.path.join(gen_dir, "*.png"))) if os.path.basename(p) != "_grid.png"]
assert len(pngs) == len(captions), f"{len(pngs)} pngs vs {len(captions)} captions"

model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
proc = CLIPProcessor.from_pretrained(model_name)

@torch.no_grad()
def embed_text(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        tk = proc(text=texts[i:i+bs], return_tensors="pt", padding=True, truncation=True).to(device)
        e = model.get_text_features(**tk); e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu())
    return torch.cat(out)

@torch.no_grad()
def embed_img(paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+bs]]
        tk = proc(images=imgs, return_tensors="pt").to(device)
        e = model.get_image_features(**tk); e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu())
    return torch.cat(out)

print(f"embedding {len(pngs)} pairs on {device}", flush=True)
te = embed_text(captions)
ie = embed_img(pngs)
sim = (te * ie).sum(-1)

random.seed(17)
shuf = list(range(len(captions))); random.shuffle(shuf)
sim_shuf = (te[shuf] * ie).sum(-1)

result = {
    "clip_score": sim.mean().item(),
    "clip_score_std": sim.std().item(),
    "clip_score_shuffled": sim_shuf.mean().item(),
    "delta": (sim.mean() - sim_shuf.mean()).item(),
    "n": len(pngs),
    "model": model_name,
}
print(json.dumps(result, indent=2))
out_path = os.path.join(gen_dir, "clip_result.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"saved: {out_path}", flush=True)
PY
