---
title: RSDiff
hide:
  - navigation
---

# RSDiff: Remote Sensing Image Generation from Text

[![Paper](https://img.shields.io/badge/DOI-10.1007%2Fs00521--024--10363--3-blue)](https://doi.org/10.1007/s00521-024-10363-3)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/asebaq/rsdiff)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/asebaq/rsdiff/blob/main/LICENSE)

> **RSDiff: Remote Sensing Image Generation from Text Using Diffusion Model**
> [Ahmad Sebaq](https://github.com/asebaq), Mohamed ElHelw
> Center for Informatics Science, Nile University
> *Neural Computing and Applications*, 2024
> [[Paper](https://doi.org/10.1007/s00521-024-10363-3)] · [[Code](https://github.com/asebaq/rsdiff)] · [[Model](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650)]

A T5-conditioned cascaded diffusion model for text-to-satellite-image generation at 256×256, trained on RSICD. The released checkpoint reaches **FID 65.70** and **CLIP-score 0.278** on the full RSICD test split (N=1,093, `cond_scale=5`).

The cascade is text-conditioned via a frozen T5-base encoder, classifier-free-guided at inference, and split across a 27 M-param 128² base UNet and a 92 M-param 256² super-resolution UNet (120 M total parameters, 1000-step DDPM). Pre-trained weights are released on the HuggingFace Hub at [`asebaq/rsdiff-sr-cascade-ep650`](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650).

## Headline

<p align="center">
  <img src="figures/headline.png" width="720" alt="Headline FID + CLIP">
</p>

| Metric | Value |
|---|---|
| **FID** (cascade-256, N=1093, feature=2048) | **65.70** |
| **CLIP-score** (OpenAI ViT-B/32) | **0.278** ± 0.030 |
| CLIP-score (shuffled-caption null) | 0.232 |
| CLIP-score delta vs null | **+0.046** |

See [Results](results.md) for the full FID-vs-epoch sweep, CFG-scale ablation, and discussion.

## Samples

Nine 256×256 samples from the RSICD test split, generated with the released cascade at `cond_scale=5`. Captions shown verbatim — no truncation.

<div class="sample-grid" markdown>

<figure markdown>
  <img src="samples/sample_0000.png" width="256" height="256" alt="sample 0000">
  <figcaption>"there's a green pool like an airplane closing the house with a grey roof ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0096.png" width="256" height="256" alt="sample 0096">
  <figcaption>"there are many large trees on both sides of the wide road ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0199.png" width="256" height="256" alt="sample 0199">
  <figcaption>"the lake cover the most area of the lake ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0349.png" width="256" height="256" alt="sample 0349">
  <figcaption>"the waves are crushing on the wet sand ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0499.png" width="256" height="256" alt="sample 0499">
  <figcaption>"roads and rivers can be seen in the valleys ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0649.png" width="256" height="256" alt="sample 0649">
  <figcaption>"the airport covers a large area and has many planes ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0799.png" width="256" height="256" alt="sample 0799">
  <figcaption>"the rectangular pond close to the trapezoidal pond is next to a building with a parking lot surrounded by trees ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_0949.png" width="256" height="256" alt="sample 0949">
  <figcaption>"some streets divide advertising into several pieces ."</figcaption>
</figure>

<figure markdown>
  <img src="samples/sample_1049.png" width="256" height="256" alt="sample 1049">
  <figcaption>"white advertising with surrounding trees is next to a main road and some apartments ."</figcaption>
</figure>

</div>

Full 1,093-image generation bundle on the [HF Hub release](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650) under `samples/`.

## Acknowledgments

Built on `lucidrains/imagen-pytorch` for the cascade scaffolding and the HuggingFace `datasets` mirror of RSICD. Developed at the Center for Informatics Science, Nile University.
