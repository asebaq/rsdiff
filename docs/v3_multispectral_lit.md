# v3 multispectral (Sentinel-2) — literature scope

Survey of 2024–2026 work on generative diffusion for multispectral / Sentinel-2
imagery, to scope v3. See [roadmap.md](roadmap.md) "v3 — multispectral generation".

> Some citations (esp. EO-VAE arXiv id, GeoSynth venue/URL) are **unverified** —
> confirm before formal citation.

## Headline finding

**True native >3-band S2 *generation* is largely unsolved.** The big RS
generators are RGB latent-diffusion models reusing the Stable Diffusion 3-channel
VAE, and take multispectral only as *conditioning*, not as generation output.
Genuine 13-band work lives in super-resolution / fusion, not generation. The
**13-band VAE is the unsolved bottleneck** everyone dodges.

## Three patterns in the field

1. **RGB-diffuse + learned band-fusion** — DiffFuSR (TGRS 2025). Diffuse RGB,
   propagate to remaining bands via a fusion head. Cheapest; trainable small.
2. **Custom MS-VAE → latent-diffuse all bands** — EO-VAE (2026, S1+S2 tokenizer),
   ESA OpenSR MS-LDM (RGB-NIR). The proper "hard but smart" path.
3. **Physics-constrained reduced manifold** — hyperspectral unmixing-guided
   diffusion in abundance space (non-negativity, sum-to-one). Cleanest theory for
   many-band; HSI not S2.

## Key works

| Work | Year | Modality | Conditioning | Arch | Note |
|---|---|---|---|---|---|
| DiffusionSat | ICLR 2024 | RGB gen; MS as *cond* (10ch, drops B1/B9/B10) | text + metadata (lat/lon, GSD, time) | latent, SD RGB VAE, diffusers | closest to our stack; weights released |
| Text2Earth | GRSM 2025 | RGB | text + resolution | latent, 1.3B | Git-10M (10M img-text, RGB) |
| CRS-Diff | 2024 | RGB | text+metadata+image (roads/SAR/seg) | latent + ControlNet | multi-condition |
| MetaEarth | TPAMI 2024 | RGB multi-res | resolution-guided cascade | pixel, 600M | unbounded tiling |
| EarthSynth | 2025 | RGB | text + semantic mask | latent (SD) | for downstream seg/det/cls |
| EO-VAE | 2026 | **S1+S2 MS** | — (tokenizer) | multi-sensor VAE | **the MS-latent unblocker** (verify cite) |
| ESA OpenSR MS-LDM | 2025 | **MS S2 (RGB-NIR)** | lowres | latent diffusion SR | proof MS latent works at scale |
| DiffFuSR | TGRS 2025 | RGB→all bands | — | diffusion SR + fusion net | band-grouping pattern |
| HS unmixing-guided | 2025 | hyperspectral | abundance | diffusion in abundance manifold | physics-constrained |

## Datasets

- **SSL4EO-S12 v1.1** — ~250k global, S1+S2, all 13 bands, 4 seasons, HF-hosted.
  Best raw-pixel S2 corpus. **No captions.**
- **BigEarthNet** — 590k S1+S2 patches, all S2 bands, multi-label land-cover.
  Standard for class-conditional.
- **fMoW-Sentinel** — 13-band S2 + categories + metadata. DiffusionSat's set.
- Captions for true 13-band S2 are scarce → **condition on class/metadata, not text.**

## Evaluation

- **No accepted FID substitute for >3 bands** (Inception is RGB-only).
- Spectral fidelity: **SAM** (spectral angle), per-band **PSNR/SSIM/ERGAS**.
- Spectral-index plausibility (NDVI/NDWI ranges); per-band statistics.
- Downstream-task proxy: scene classification / segmentation (extends our `zeroshot_oa`).
- Opportunity: **SatCLIP-feature Fréchet distance ("SatFID")** — novel, none standardized.

## Why is MS-LDM (with a real VAE) near-unclaimed?

Timing + incentives, not impossibility — supports v3 as a genuine contribution:

1. **Enabling VAE only just landed** — EO-VAE is Feb 2026; before it, no good
   general MS tokenizer existed to build latent diffusion on.
2. **MS-VAE is hard** — faithful 12–13 band reconstruction (spectral fidelity,
   per-band SNR/resolution, reflectance range) → most reused the SD RGB VAE and
   stayed RGB.
3. **No eval / demo / reward** — no standard >3-band FID; reviewers can't eyeball
   12 bands. Field optimized for measurable, demoable RGB.
4. **Data + captions starved** — captioned MS data barely exists; text-to-MS is
   data-poor, so MS work stayed discriminative.
5. **Weak application pull** — utility went to SR / cloud-removal / fusion;
   de-novo synthetic MS has unclear demand vs real data.
6. **EO foundation models went discriminative** — Prithvi/SatMAE/Clay/DOFA/
   TerraMind are encoders, not generators.

Risk of being early: EO-VAE is unproven at scale → recon sanity check first.

## EO-VAE — verified, chosen as the v3 VAE

Confirmed real (github.com/nilsleh/eo-vae, Apache-2.0, paper arXiv:2602.12177):

- Supports **S2 L2A (12-band)**, L2A L1C (13-band), S2 RGB, S1 RTC.
- **256² → latent 32×32×32** (8× spatial downsample, 32 latent channels).
- PyTorch + Lightning (standalone, wrap encode/decode — not diffusers).
- Pretrained weights via HF `nilsleh/eo-vae` (config-driven loading).

**Plan:** freeze EO-VAE → encode S2 256² to 32×32×32 latent → train conditional
diffusion UNet (`in/out_channels=32`) in latent space → decode to 12-band S2.

**Dependency risk:** frozen EO-VAE bounds our generation fidelity by its
reconstruction quality. First v3 step = recon sanity check (encode→decode a
held-out S2 set, per-band SAM/PSNR/MS-SSIM). If weak → fine-tune EO-VAE or fall
back to the DiffFuSR hybrid.

## Recommended approach (small open-source MS-S2 model)

Go **latent, not pixel.** **Do not train a 13-ch VAE from scratch** (the unsolved
part). Either:
- **(a)** adopt/fine-tune **EO-VAE** as MS tokenizer, train a `diffusers` UNet on
  its latents; or
- **(b)** ship the **DiffFuSR-style hybrid** — RGB latent diffusion (DiffusionSat-
  derived) + lightweight learned band-fusion head to the full S2 stack. Cheaper,
  single-GPU trainable.

Pretrain on **SSL4EO-S12 v1.1**; add class/metadata conditioning from
**BigEarthNet** + DiffusionSat-style geo/GSD/time tokens. Eval **SAM + per-band
PSNR/SSIM/ERGAS + downstream**, and optionally a **SatCLIP-FID** as a contribution.

## Sources

- DiffusionSat: https://arxiv.org/abs/2312.03606 · https://github.com/samar-khanna/DiffusionSat
- Text2Earth: https://arxiv.org/abs/2501.00895 · https://github.com/chen-yang-liu/Text2Earth
- CRS-Diff: https://arxiv.org/abs/2403.11614
- MetaEarth: https://arxiv.org/abs/2405.13570
- EarthSynth: https://arxiv.org/abs/2505.12108 · https://github.com/jaychempan/EarthSynth
- EO-VAE (verify): https://github.com/nilsleh/eo-vae
- DiffFuSR: https://arxiv.org/abs/2506.11764 · https://github.com/NorskRegnesentral/DiffFuSR
- ESA OpenSR: https://github.com/ESAOpenSR/opensr-model
- HS unmixing-guided: https://arxiv.org/abs/2506.02601
- SSL4EO-S12: https://arxiv.org/pdf/2211.07044 · https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1
- BigEarthNet: https://bigearth.net/
- fMoW-Sentinel: https://purl.stanford.edu/vg497cb6002
- SatCLIP: https://arxiv.org/abs/2311.17179
