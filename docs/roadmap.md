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

### Overfit mitigations (must ship in v0)

The legacy thesis run was effectively over-parameterized (27M params / 8 734
training images ≈ 3 100 params/image) and trained for 1 000 epochs with
neither weight decay nor data augmentation nor validation-FID tracking.
EMA + CFG dropout did most of the work. v0 must do better:

- **Weight decay 0.01** on UNet params (matches the paper recipe, undoes the
  imagen-pytorch default of zero).
- **Light augmentation**: random horizontal flip; small random crop with
  reflection padding. No rotation (breaks "north-up" satellite priors).
- **Validation FID every N epochs** on the 1 094-image RSICD val split.
  Log to TensorBoard + WandB. Use the same Inception backbone we report on test.
- **Early-stop policy**: stop if val FID has not improved by ≥ 1.0 over
  the last `patience=10` validation rounds. Keep the best-FID checkpoint.
- **Sample diversity probe**: every K epochs, sample 32 fixed test captions
  with two different seeds; report mean pairwise LPIPS. Hard drop = mode
  collapse signal.
- **Memorization probe**: at each milestone, run 16 random training-set
  captions and compare nearest-neighbor (LPIPS) against the actual training
  image. If `min LPIPS < 0.1` for a meaningful share of samples, model is
  reproducing the training set.

These checks compose into a single `rsdiff eval --during-training` hook;
needed both for v0 acceptance and for any v1 / v2 model selection.

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

## v2 — higher resolution (latent diffusion)

See "Higher resolution" under the future-work plan below. LDM at 512/1024 on a
higher-res RS dataset; the 256 cascade stays the efficient baseline.

## v3 — multispectral generation (Sentinel-2)

Generate multi-band imagery, not just RGB — the main research differentiator.
Target: **Sentinel-2** (13 bands B1–B12 + B8A; native 10/20/60 m GSD; reflectance,
not uint8).

- **Architecture**: UNet `in/out_channels = 12–13` (trivial). The real lift is the
  multispectral latent space. Lit (see [v3_multispectral_lit.md](v3_multispectral_lit.md))
  says native >3-band *generation* is largely unsolved and the 13-band VAE is the
  bottleneck — so **don't train a 13-ch VAE from scratch**. Stand on an existing
  MS-VAE (**EO-VAE**, S1+S2) + `diffusers` UNet on its latents, OR ship the
  **DiffFuSR hybrid** (RGB latent diffusion + learned band-fusion head). "Hard but smart."
- **Data + conditioning**: **BigEarthNet** (S2 L2A, 12 bands, multi-label,
  ~590k patches) is the best fit; also SEN12MS, EuroSAT-MS (13-band, class-cond).
  Condition on class / multi-label, or text via label templating (S2 captions scarce).
- **Preprocessing**: resample bands to a common GSD (or multi-res handling);
  per-band reflectance normalization with dataset stats.
- **Eval**: Inception-FID is invalid for >3 bands. Use **SAM** (spectral angle),
  per-band FID on an MS/RS backbone, band-wise statistics, spectral-index
  plausibility (NDVI/NDWI ranges), and downstream land-cover classification.
- **Physical validity bar**: generated spectra must be physically plausible
  (vegetation NDVI in range, water NIR absorption), not just RGB-realistic.

**Phasing (decided):**
- **v3.0 — pure generation first.** Class/metadata-conditional MS generation via
  frozen EO-VAE latent diffusion + spectral eval (SAM, per-band, SatFID). Get a
  working generator + recon sanity check before any application loop.
- **v3.x — application loop (deferred).** Add a measurable downstream-utility
  use-case to fix the weak "application pull". Lead candidate: **rare-class
  augmentation** (generate scarce land-cover/event classes, augment a seg/cls
  model, report ΔmIoU/ΔOA on rare classes — ties future-work #2). Alternatives:
  cloud/gap infilling, S1→S2 cross-sensor translation.

- Conditioning on elevation / land-cover priors.
- Downstream-task probes (segmentation, detection) using generated data
  as augmentation.

## Thesis future-work → rsdiff plan

The thesis (`ch5_conclusion_future.tex`) lists four future-work items. Mapping
each to a concrete rsdiff version so we actually address them:

1. **Improved spatial reasoning** (model struggles with multi-element layouts /
   relational prompts). → **v1**: layout/region conditioning (boxes or region
   prompts) + segmentation-mask ControlNet; investigate stronger / multi-level
   cross-attention. Measure with a spatial-relation prompt suite.
2. **Handling rare geographical features** (scarce in RSICD). → **v1.x**:
   class-balanced sampling + external/more RS data + plain full fine-tune (cheap
   at our model sizes). No LoRA — models are small enough to fine-tune fully.
3. **Computational efficiency** (thesis: 1.8 s/image inference, 1000 train
   epochs). → **v0.x**: few-step samplers (DPM-Solver++, DDIM) — enough on their
   own given small models; **no distillation** (rsdiff1.5 is already 119.9M).
   Training: cut epochs via the val-FID early-stop above (expect ~200–400 ep vs
   1000 once weight-decay + aug land) — drive by val-FID, not loss (diffusion
   loss saturates early while FID keeps improving).
4. **Zero-shot classification probe** of learned representations. → **partly
   shipped**: `rsdiff eval --metric zeroshot_oa` (CLIP backbone). Extend with an
   RS-pretrained backbone (ties to the RSFID question below).

### Higher resolution (>256) — v2, latent diffusion

RSICD is natively 224² (→256), so the pixel cascade is at its data ceiling; you
cannot honestly train text-to-image >256 on RSICD. Going to 512/1024 is a **v2
data + paradigm change**, not a tweak:

- **Move to latent diffusion (LDM)** — VAE encodes 512/1024 → ~64–128 latent,
  text-conditional UNet/DiT in latent space. This is how SDXL/SD3 reach 1024
  cheaply; pixel-space cascade at 1024 is 16× the pixels of 256 and too costly.
- **Needs a higher-res RS dataset** (DOTA, fMoW, Million-AID, or high-res tiles)
  for >256 ground truth. Survey during lit review.
- Resolves roadmap open-Q#1 (cascade vs LDM) in LDM's favour *for the high-res
  track*; the 256 cascade (rsdiff1 / 1.5) stays as the efficient baseline.

## Open questions blocked on lit review

1. Should v0 reuse the imagen-style cascade or move to a single-stage latent
   diffusion model (LDM) approach? Latent saves compute at the cost of an
   extra VAE.
2. Is `clean-fid` + Inception-V3 still acceptable, or should v0 ship RSFID
   (FID with an RS-pretrained backbone) from day one?
3. Which conditioning extensions have the strongest evidence in 2024–2026
   RS-gen papers? Pick 2 for v1, not all 5.
