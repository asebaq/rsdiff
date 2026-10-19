<h1 align="center">
  <a href="https://github.com/asebaq/rsdiff">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/asebaq/rsdiff/main/docs/assets/cover_dark.svg">
      <img alt="rsdiff" src="https://raw.githubusercontent.com/asebaq/rsdiff/main/docs/assets/cover_light.svg" width="640">
    </picture>
  </a>
</h1>

<p align="center">
  <a href="https://pypi.org/project/rsdiff/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rsdiff?color=1D4ED8"></a>
  <a href="https://pypi.org/project/rsdiff/"><img alt="Python" src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fasebaq%2Frsdiff%2Fmain%2Fpyproject.toml"></a>
  <a href="https://github.com/asebaq/rsdiff/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/asebaq/rsdiff/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/asebaq/rsdiff/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/asebaq/rsdiff"></a>
  <a href="https://doi.org/10.1007/s00521-024-10363-3"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1007%2Fs00521--024--10363--3-blue"></a>
  <a href="https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650"><img alt="Hugging Face model" src="https://img.shields.io/badge/Hugging%20Face-rsdiff--sr--cascade--ep650-yellow"></a>
</p>

<p align="center">
  <a href="#install"><strong>Install</strong></a> &middot;
  <a href="#use"><strong>Use</strong></a> &middot;
  <a href="#results"><strong>Results</strong></a> &middot;
  <a href="#how-it-works"><strong>How it works</strong></a> &middot;
  <a href="#limitations"><strong>Limitations</strong></a> &middot;
  <a href="#training"><strong>Training</strong></a> &middot;
  <a href="#roadmap"><strong>Roadmap</strong></a> &middot;
  <a href="#citation"><strong>Citation</strong></a>
</p>

**Text-to-satellite-image diffusion, packaged.** rsdiff is the reference implementation
of RSDiff ([Neural Computing and Applications, 2024](https://doi.org/10.1007/s00521-024-10363-3)):
a T5-conditioned, Imagen-style cascaded diffusion model that turns a caption such as
"many buildings and green trees are around a playground ." into a 256x256 synthetic
remote-sensing image. One command downloads the released RSICD-trained checkpoint,
verifies it, and samples; a second scores the result with the same FID and CLIP-score
protocol used for the published numbers (FID 65.70, CLIP-score 0.278 on the full
RSICD test split).

Every image rsdiff produces is synthetic. It is not an observation or a prediction of
any real place, and each PNG says so in its metadata.

<p align="center">
  <img src="https://raw.githubusercontent.com/asebaq/rsdiff/main/docs/figures/sample_montage.png" width="780" alt="Synthetic 256x256 samples generated from RSICD test captions">
</p>

## Install

```bash
pip install rsdiff            # sampling + CLIP-score
pip install "rsdiff[eval]"    # adds torchmetrics for FID
```

Python 3.10+. Sampling runs on CUDA when available and otherwise on CPU. Apple MPS is
not supported: imagen-pytorch 2.1.0 and the torchmetrics FID both use float64.

## Use

Generate images from your own captions. The first run downloads the released checkpoint
(1.9 GB, pinned Hugging Face revision) and verifies its SHA-256:

```bash
rsdiff sample --prompt "many buildings and green trees are around a playground ." -n 4 --out-dir generated
```

This writes 256x256 PNGs and a `captions.txt` to `generated/`. `--prompt` can be
repeated. Each PNG carries `Software`, `Description` (a synthetic-content notice), and
`Prompt` text chunks.

Or draw captions from RSICD (first caption, `sent1`, shuffled with `--seed`, default 17):

```bash
rsdiff sample --rsicd data/RSICD_optimal --split test -n 16 --out-dir generated
```

Useful options: `--cond-scale` (default 5), `--batch-size` (default 2), `--lr-only`
(stop at the 128x128 base stage), `--cpu`, `--config rsdiff1.5|rsdiff1|path.yaml`, and
`--checkpoint PATH` (required for non-default configs). An interrupted run resumes:
batches whose PNGs already exist are skipped.

Score a generated directory against the RSICD test split:

```bash
rsdiff eval fid generated --rsicd data/RSICD_optimal    # --feature 2048 --size 256 by default
rsdiff eval clip generated                             # clip_score, clip_score_shuffled, delta
```

Both print JSON. The same pieces are available from Python:

```python
from rsdiff.cascade import download_release, generate, load_cascade
from rsdiff.config import load_preset
from rsdiff.eval import clip_score

cfg = load_preset("rsdiff1.5")
trainer = load_cascade(cfg, download_release())
paths = generate(trainer.sample, ["a river runs through the green farmland ."], "generated")
print(clip_score("generated"))
```

`rsdiff.datasets.rsicd.sample_captions` reproduces the CLI's RSICD caption draw, and
`rsdiff.eval.fid` computes FID.

## Results

Released checkpoint, full RSICD test split (N=1,093), cascade output at 256x256,
`cond_scale=5`:

| Metric | Value |
| --- | --- |
| FID (torchmetrics Inception, feature 2048, both sides at 256) | **65.70** |
| CLIP-score (OpenAI CLIP ViT-B/32) | **0.278** |
| CLIP-score, shuffled-caption baseline | 0.232 |
| Parameters (base UNet 27.2 M + SR UNet 92.7 M) | 119.9 M[^params] |

[^params]: 119.9 M counts the two standalone UNets, as in the paper and model card.
    Loaded inside the cascade, the SR UNet gains low-resolution conditioning layers
    (105.0 M), so the cascade holds 132.2 M trainable parameters, plus the frozen T5-base
    text encoder.

The CLIP-score sits 0.046 above the shuffled-caption baseline: the images follow their
captions, not just the RSICD domain.

| Curve | Figure |
| --- | --- |
| SR FID versus epoch (ep150 to ep1000) | [`docs/figures/sr_fid_curve.png`](docs/figures/sr_fid_curve.png) |
| CFG `cond_scale` sweep on the selected SR milestone | [`docs/figures/cfg_sweep.png`](docs/figures/cfg_sweep.png) |
| LR base FID versus epoch | [`docs/figures/lr_fid_curve.png`](docs/figures/lr_fid_curve.png) |

The numbers behind every figure are committed in [`results/`](results/). Methodology and
discussion: [`docs/REPORT.md`](docs/REPORT.md). Model card:
[`docs/model_card.md`](docs/model_card.md).

## How it works

1. A frozen **T5-base** encoder turns the caption into token embeddings.
2. A **128x128 base UNet** denoises from noise over 1,000 DDPM steps, cross-attending to
   the text.
3. A **256x256 super-resolution UNet** denoises again, conditioned on the text and the
   upsampled base image.

Both stages use classifier-free guidance (trained with 10% caption dropout) with a shared
`cond_scale`; 5 is the value selected by the CFG sweep and used for every headline
number. Sampling uses the EMA weights of the merged epoch-650 checkpoint. The presets
ship inside the package: [`rsdiff1.5`](src/rsdiff/configs/rsdiff1.5.yaml) (released) and
[`rsdiff1`](src/rsdiff/configs/rsdiff1.yaml) (the 723 M paper configuration; weights not
released).

## Limitations

- **Synthetic only.** Outputs are plausible-looking imagery, not observed or predicted
  geography. Do not use them as evidence about a real location.
- **RSICD domain.** The model was trained on the 30 RSICD scene types and their short,
  lowercase captions. Prompts unlike those captions have not been evaluated.
- **256 px.** The cascade stops at 256x256.
- **No Apple MPS.** float64 operations in imagen-pytorch 2.1.0 and torchmetrics FID are
  not supported on MPS.
- **CPU sampling is slow.** 2,000 denoising steps per image took about an hour on an
  Apple-silicon CPU; use a CUDA GPU for more than a handful of images.
- **Weights not commercially cleared.** RSICD's commercial terms are unresolved; see
  [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md).

## Training

Training runs through the original training code in [`ddpm/`](ddpm/):

```bash
bash ddpm/scripts/run_training_local.sh   # stage 1: 128x128 base UNet
bash ddpm/scripts/run_sr.sh               # stage 2: 256x256 SR UNet on the frozen base
```

[`docs/reproducibility.md`](docs/reproducibility.md) is the complete runbook: environment,
dataset preparation, training, milestone sweeps, sampling the 1,093-image bundle, and
scoring.

## Roadmap

1. **Commercial relevance.** Test whether synthetic augmentation improves a downstream
   model on an untouched real-data test set.
2. **Training in the package.** Move the two-stage training loop out of `ddpm/` into
   `rsdiff`.

Details and acceptance criteria: [`docs/PLAN.md`](docs/PLAN.md).

## Citation

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

## License

Code: Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The released weights
were trained on RSICD, whose commercial terms are unresolved; dataset, dependency, and
derived-weight rights are tracked in [`docs/LICENSE_AUDIT.md`](docs/LICENSE_AUDIT.md).
Built on [`lucidrains/imagen-pytorch`](https://github.com/lucidrains/imagen-pytorch);
developed at the Center for Informatics Science, Nile University.
