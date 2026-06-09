# rsdiff: text-to-satellite-image cascaded diffusion on RSICD

## Abstract

We release `rsdiff`, an open-source text-conditioned cascaded diffusion
model for remote-sensing imagery. The model is a compact two-stage
Imagen-style cascade — a 27 M-param 128-pixel base UNet and a 92 M-param
super-resolution UNet to 256² — conditioned on a frozen T5-base text
encoder with classifier-free guidance. Trained on RSICD, the released
checkpoint (`rsdiff-sr-cascade-ep650`) reaches **FID 65.70** on the full
RSICD test split (N=1093, Inception feature=2048, cascade-256,
`cond_scale=5`). Text-image alignment, measured with OpenAI CLIP
ViT-B/32, scores **0.278** — a +0.046 lift over a shuffled-caption null
baseline. Beyond the headline we publish the full FID-vs-epoch trajectory
(ep50→1000), the CFG-scale ablation (cs=1→8), the merged checkpoint, and
the entire 1093-image generation bundle. Code, weights, and the
end-to-end reproducibility runbook are MIT/Apache-2.0 licensed.

The work follows directly from *RSDiff: remote sensing image generation
from text using diffusion model* (Sebaq & ElHelw, NCA 2024), and the
released artifacts make every published number directly auditable.

## 1. Introduction

Diffusion models trained on natural web-scale photo corpora generalise
poorly to overhead imagery — road grids, agricultural texture, runway
layouts, and shoreline geometry all sit outside the training
distribution. Building a publishable text-to-RS baseline is then a
matter of (a) picking a compact architecture, (b) training on an
RS-specific paired dataset, and (c) calibrating the sampling-time
guidance to recover image fidelity and caption alignment together. The
RS open-source landscape is currently fragmented across one-off paper
repos with closed weights; this release is one consolidated repository
covering training, evaluation, and inference with audit-grade artefacts.

## 2. Method

### 2.1 Architecture

Two-stage Imagen-style cascade. Both UNets share the
`lucidrains/imagen-pytorch` block scaffolding; weights are not shared
between stages.

| Stage | Params | Input | Output | Conditioning |
| --- | --- | --- | --- | --- |
| LR base UNet | 27.18 M | text + noise | 128² image | T5-base, `p_uncond=0.1` |
| SR UNet | 92.66 M | text + LR + noise | 256² image | T5-base + LR image, `p_uncond=0.1` |

Sampler: DDPM, T = 1000 denoising steps. Classifier-free guidance applies
to both stages with a shared `cond_scale`; null-conditioning is taken
from the `p_uncond=0.1` channel learnt at training time.

### 2.2 Dataset

[RSICD](https://huggingface.co/datasets/arampacha/rsicd) — 10,921 paired
satellite images and natural-language captions, official 8/1/1
train/val/test split (1093 test). Captions are short, declarative, and
mostly land-cover oriented. The first caption per image (`sent1`) is
used as the conditioning text at training time. No augmentation, no
caption shuffling.

### 2.3 Training

Two-phase, decoupled-cascade training (LR first, SR second on top of a
frozen LR base):

1. **LR base.** 1000 epochs (~137 k steps), batch size 64, Adam at
   `lr=1e-4`, `p_uncond=0.1`. Best LR FID lands at ep700 = 202.43
   (128², N=64, feature=2048).
2. **SR UNet.** 1000 epochs on top of the frozen LR base. SR is fed
   ground-truth low-resolution images at training time (`p_uncond=0.1`
   on the text conditioning, no noise augmentation on the LR input).

Milestones are snapshotted every 50 SR epochs. A `merge_base.py` utility
reinjects the frozen LR base into each slim SR checkpoint so every
milestone is a self-contained cascade for evaluation. The release
checkpoint is the merged ep650 milestone.

## 3. Experiments

### 3.1 LR-base FID sweep

Per-milestone scoring at N=64 samples, 128×128, Inception feature=2048,
`cond_scale=4`.

![LR FID curve](figures/lr_fid_curve.png)

The curve bottoms at **ep700 = 202.43** and climbs afterwards. ep700 is
chosen as the frozen seed for SR training.

### 3.2 SR-cascade FID sweep

Per-milestone scoring at N=128 samples, cascade-256, feature=2048,
`cond_scale=4`. ep50 and ep100 are scored at the same protocol locally;
the cloud sweep covers ep150 → ep1000 (stride 50).

![SR FID curve](figures/sr_fid_curve.png)

| Range | FID |
| --- | --- |
| ep50 | 224.53 |
| ep100 | 174.52 |
| ep150 → ep700 | 172.79 → 159.39 → **156.73** (ep650 winner) |
| ep700 → ep1000 | 156.85, then a sharp climb to 167.46 |

Full 18-row TSV: [`../results/fid_curve_sr.tsv`](../results/fid_curve_sr.tsv).

The post-ep650 climb is the expected overfit signature for the
small-train-set / no-augmentation setting. ep650 is the picked SR
milestone for everything downstream.

### 3.3 CFG `cond_scale` sweep

On the ep650 SR winner, N=64 per scale (rank-only picker before the
expensive headline run), cascade-256, feature=2048.

![CFG sweep](figures/cfg_sweep.png)

| `cond_scale` | FID (N=64) |
| --- | --- |
| 1 | 230.12 |
| 2 | 210.98 |
| 3 | 205.06 |
| 4 | 204.06 |
| **5** | **200.00** |
| 6 | 201.62 |
| 8 | 201.97 |

The bowl bottoms at cs=5. N=64 absolute FID is upward-biased relative to
the N=1093 headline; only the ranking is used.
Full TSV: [`../results/fid_cfg_step89050.tsv`](../results/fid_cfg_step89050.tsv).

### 3.4 Headline — full RSICD test split

With `(milestone, cond_scale) = (ep650, 5)` we generate the full RSICD
test split and score:

![Headline](figures/headline.png)

| Metric | Value |
| --- | --- |
| **FID** (cascade-256, N=1093, feature=2048) | **65.70** |
| FID (feature=768) | 0.275 |
| **CLIP-score** (OpenAI ViT-B/32) | **0.278** ± 0.030 |
| CLIP-score (shuffled-caption baseline) | 0.232 |
| CLIP-score delta vs null | **+0.046** |

The shuffled-caption baseline pairs each generated image with a randomly
chosen caption from the same test set. The +0.046 delta is the
text-image alignment signal above random pairing.

A random 9-sample look at the bundle:

![Sample montage](figures/sample_montage.png)

Full bundle (1093 PNGs + captions) lives on the
[HF Hub release](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650).

## 4. Discussion

**Milestone choice matters.** The FID range across SR milestones is
65→172 (cascade-256 N=128 ranges 156.73→172.79; the headline N=1093
brings absolute values down). Picking a late-training milestone naively
(e.g. ep1000) would land ~10 FID points worse than ep650; this is one
of the clearest illustrations in the curves of overfit drift on a
small-train-set RS dataset.

**CFG is non-trivial.** Without guidance (`cs=1`) the cascade scores
~230 FID — the model effectively ignores text. The bowl bottoms in the
cs=4–5 region and rises again past cs=6. We recommend cs=5 as a default;
domain-shifted captions may benefit from cs=4 or cs=6.

**N matters for FID.** N=64 Inception-FID is upward-biased relative to
N=1093 by an unspecified but visible margin (the same milestone scores
156.73 at N=128 vs ~200 at N=64). When comparing to other work on this
dataset, report both N and feature dim explicitly.

**Decoupled cascade is sufficient.** Training SR on ground-truth
low-resolution inputs (rather than jointly fine-tuning the cascade) is
enough to land within published FID range. A joint fine-tune of both
unets at the end of training is left as future work.

## 5. Limitations

- **Overfit drift past ep650.** No augmentation, no weight decay,
  train set ~10 k images. v0 of the trainer will ship weight decay,
  caption-augmentation, val-FID, and early-stop.
- **Single-caption conditioning.** RSICD provides 5 captions per image;
  this run uses only `sent1`.
- **Pixel-space cascade.** Slower at inference than a latent-diffusion
  port; a latent rewrite is on the roadmap.
- **No memorisation probe.** Partial training-set memorisation is not
  ruled out — a perceptual-hash audit is on the roadmap.
- **CFG sweep at N=64.** Treat its absolute numbers as ordinal only.

## 6. Future work

In rough dependency order, from [`roadmap.md`](roadmap.md):

1. **v0 trainer** — replace `legacy/DDPM/*` with a `diffusers`-native
   trainer that reads `configs/rsicd_text_128.yaml`. Acceptance bar:
   parity with this release's 65.70.
2. **v0 overfit fixes** — weight decay, caption augmentation,
   val-FID, early stop, pHash memorisation probe.
3. **CLIP-score per-class** breakdown across RSICD's land-cover classes;
   CFG sweep at headline N.
4. **v0.x** — T5-base → flan-t5-xl encoder upgrade; random-of-5 caption
   sampling + VLM dense re-captioning; Min-SNR loss weighting (γ=5).
5. **v1** — pixel cascade → latent diffusion (LDM) port.
6. **v2** — multispectral conditioning (Sentinel-2; see
   [`v3_multispectral_lit.md`](v3_multispectral_lit.md)).

## 7. Release artefacts

| Artifact | Location |
| --- | --- |
| Code | https://github.com/asebaq/rsdiff |
| Merged ep650 checkpoint | https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650 |
| 1093 generation bundle | same HF model repo (`samples/` + `captions.txt`) |
| Committed numerics (TSVs, JSONs) | [`../results/`](../results/) |
| Reproducibility runbook | [`reproducibility.md`](reproducibility.md) |

## 8. References

```bibtex
@article{sebaq2024rsdiff,
  title   = {RSDiff: remote sensing image generation from text using diffusion model},
  author  = {Sebaq, Ahmad and ElHelw, Mohamed},
  journal = {Neural Computing and Applications},
  volume  = {36},
  number  = {36},
  pages   = {23103--23111},
  year    = {2024},
  doi     = {10.1007/s00521-024-10363-3}
}
```

## 9. Acknowledgments

`lucidrains/imagen-pytorch` for the cascade scaffolding the training code
is built on. HuggingFace `datasets` for the RSICD mirror. The remote-
sensing image-captioning community for the original RSICD release.
