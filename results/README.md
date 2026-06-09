# Committed numerics

Small evaluation artefacts that back every figure and table in
[`../docs/REPORT.md`](../docs/REPORT.md) and the README. Generation
bundles (PNGs) are too large for git and live on the
[HF Hub release](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650).

| File | What it is | Schema |
| --- | --- | --- |
| `fid_curve_sr.tsv` | SR milestone FID sweep, 18 rows ep150–1000 stride 50 | `epoch step fid feature size n_gen` |
| `fid_curve_lr.tsv` | LR base milestone FID sweep, 9 rows ep100–900 stride 100 | same |
| `fid_cfg_step89050.tsv` | CFG `cond_scale` sweep on ep650 winner, 7 rows cs ∈ {1,2,3,4,5,6,8} | `step epoch cond_scale fid feature size n_gen` |
| `fid_result.json` | Headline FID — ep650 × cs=5 on full RSICD test split (1093) | `fid n_gen feature size split` |
| `fid_result_f768.json` | Same generations, Inception feature=768 head | same |
| `clip_result.json` | CLIP cosine sim (OpenAI ViT-B/32) — real vs shuffled-caption baseline | `clip_score clip_score_shuffled delta n model` |
| `final_test_captions.txt` | 1093-line caption index matching the headline PNG bundle | one per line |

## Headline

| Metric | Value |
| --- | --- |
| **FID** (cascade-256, N=1093, feature=2048) | **65.70** |
| FID (feature=768) | 0.275 |
| **CLIP-score** (OpenAI ViT-B/32) | **0.278** |
| CLIP-score shuffled baseline | 0.232 |
| CLIP delta | **+0.046** |

The SR sweep and LR sweep are scored at `cond_scale=4`. The CFG sweep was
scored at N=64 (a cheap pre-headline picker — ranking-only; N=64 FID is
upward-biased compared to the N=1093 headline).
