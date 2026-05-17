# rsdiff

Open-source diffusion models for remote sensing imagery.

Successor to the master's thesis "RSDiff: A Diffusion-Based Framework for
Text-to-Satellite-Image Generation" (Nile University, 2024). Clean rewrite on
top of HuggingFace `diffusers` + `accelerate`, designed for modular datasets
and multi-modal conditioning beyond text.

> Status: **v0 scaffold**. Milestone scope is being locked after a short
> literature review pass. See `docs/roadmap.md`.

## Why

Existing image diffusion models are trained on natural images and degrade on
overhead views: roads as parallel lines, buildings as repeating rectangles,
agricultural texture. RS-specific generation is needed for data augmentation,
simulation, change detection priors, and educational visualisation — and the
RS-gen open-source landscape is fragmented across one-off paper repos with
incompatible APIs.

`rsdiff` aims to be the `diffusers`-equivalent for satellite/aerial imagery:
one repo, one CLI, multiple datasets and conditioning modes.

## Planned scope (v0)

- **Architectures** — diffusers `UNet2DConditionModel`, optional DiT.
  Cascaded LR + SR pipeline matching the thesis.
- **Conditioning** — text (T5 / CLIP / OpenCLIP), class labels,
  segmentation masks (ControlNet), low-res → high-res.
- **Datasets** — RSICD (parity baseline), then NWPU-RESISC45, EuroSAT,
  AID, BigEarthNet v2, fMoW.
- **Eval suite** — FID (Inception + CLIP backbones), CLIP-score,
  zero-shot OA against class names, downstream-task probes.
- **Runbooks** — local single-GPU, vast.ai / RunPod, NGC container on
  DGX Spark (aarch64).

## Quick start (placeholder)

```bash
pip install -e ".[dev,eval]"
accelerate config
rsdiff train --config configs/rsicd_text_128.yaml
rsdiff sample --config configs/rsicd_text_128.yaml --prompt "an airport with several planes"
rsdiff eval --config configs/rsicd_text_128.yaml --metric fid clip_score zeroshot_oa
```

## License

Apache-2.0. See `LICENSE`.

## Citation

If you build on this work please cite the underlying thesis:

```bibtex
@mastersthesis{sebaq2024rsdiff,
  title  = {RSDiff: A Diffusion-Based Framework for Text-to-Satellite-Image Generation},
  author = {Sebaq, Ahmad},
  school = {Nile University},
  year   = {2024}
}
```
