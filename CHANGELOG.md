# Changelog

## 0.1.0

First PyPI release. The package now samples and scores the released `rsdiff1.5` cascade
instead of raising `NotImplementedError`.

- `rsdiff sample --prompt "..."` downloads the released checkpoint
  (`asebaq/rsdiff-sr-cascade-ep650`, pinned revision, SHA-256 verified) and writes 256×256
  PNGs plus `captions.txt`. `--rsicd ROOT` draws RSICD captions instead; with seed 17 and
  the full test split they are the captions of the published bundle. Interrupted runs
  resume. Every PNG carries the prompt and a synthetic-content notice in its text chunks.
- `rsdiff eval fid` and `rsdiff eval clip` score a bundle with the published protocol.
  On the published 1,093-image bundle they return FID 65.688 (released 65.698) and
  CLIP-score 0.27771 (released 0.27771). FID must be scored against real images built by
  `scripts/build_rsicd_csv.py --split-csv`; the original RSICD JPEGs give 70.15.
- `rsdiff.cascade`, `rsdiff.config` (packaged `rsdiff1.5` and `rsdiff1` presets),
  `rsdiff.eval` and `rsdiff.datasets` form the Python API.
- Sampling and FID run on CUDA or CPU. Apple MPS is not used: imagen-pytorch 2.1.0 and
  torchmetrics' FID need float64.
- Removed the unused `diffusers` scaffold (`rsdiff train`, `rsdiff.models`,
  `rsdiff.pipelines`, `rsdiff.conditioning`, `zeroshot_oa`) and its dependencies.
  Training remains in `ddpm/`.
- Corrected the documented metric protocol: the headline FID uses torchmetrics'
  `FrechetInceptionDistance` (feature 2048), not clean-fid.
- The commercial monitoring experiments moved to a separate repository.
