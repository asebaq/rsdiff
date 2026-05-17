# rsdiff roadmap

## v0 — parity baseline (target: 2026 Q3)

Locked after literature review. Tentative plan:

- **Dataset**: RSICD (10 921 images, captions, 30 classes). Same split as the thesis CSV.
- **Architecture**: diffusers `UNet2DConditionModel`, cascaded 128 → 256.
  Frozen T5-base text encoder.
- **Schedulers**: DDPM training, DDIM/EulerDiscrete sampling.
- **Training**: `accelerate` single-GPU bf16, EMA, gradient checkpointing optional.
- **Eval**: FID (Inception backbone + CLIP backbone), CLIP-score, zero-shot OA.
- **Hardware target**: 1× A100/H100 on vast.ai; documented runbook.

Acceptance criterion: ≤ FID 70 on RSICD test split using the same protocol as
the thesis. Anything lower than 66.49 = beats the thesis.

## v0.x — broader baseline

- Class-conditional generation on EuroSAT, NWPU-RESISC45, AID.
- Released checkpoints on HF Hub.
- `rsdiff sample` and `rsdiff eval` CLIs feature-complete.

## v1 — conditioning expansion

- ControlNet for segmentation-mask → image.
- IP-Adapter for image-prompt → image.
- SAR ↔ optical translation pipeline (SEN12MS).

## v1.x — foundation-model fine-tuning

- LoRA fine-tunes of SDXL / SD3 / FLUX.1 on RS data.
- ComfyUI nodes.

## v2 — research surface

- Multispectral (>3 channel) diffusion.
- Conditioning on elevation / land-cover priors.
- Downstream-task probes (segmentation, detection) using generated data
  as augmentation.

## Open questions blocked on lit review

1. Should v0 reuse the imagen-style cascade or move to a single-stage latent
   diffusion model (LDM) approach? Latent saves compute at the cost of an
   extra VAE.
2. Is `clean-fid` + Inception-V3 still acceptable, or should v0 ship RSFID
   (FID with an RS-pretrained backbone) from day one?
3. Which conditioning extensions have the strongest evidence in 2024–2026
   RS-gen papers? Pick 2 for v1, not all 5.
