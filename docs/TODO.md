# rsdiff TODO

Working task list. Strategic milestones live in [roadmap.md](roadmap.md);
model-config provenance lives in `configs/` (`rsdiff1.yaml`, `rsdiff1.5.yaml`).

## North star — the end state

**One clean `rsdiff` package. No `legacy/` folder.** Different YAML configs run
different versions of the system:

- `configs/rsdiff1.yaml` — paper-faithful cascade, 723M (≈ abstract 0.75B).
- `configs/rsdiff1.5.yaml` — optimized cascade, 119.9M (27.2M base + 92.7M SR).
- future `rsdiff2.yaml`, etc. — new versions = new configs, same code path.

`rsdiff train -c configs/<version>.yaml` must fully reproduce any version. The
imagen-pytorch `legacy/` scripts are a temporary bridge — deleted once the
`rsdiff` trainer reaches FID parity.

## In flight

- [ ] **LR base run (rsdiff1.5)** — legacy `full_lr_gdm`, 1000 ep on 4090. ~96%
      done (ep963 / 12:58 UTC). Final `checkpoint.pt` seeds the SR stage. (task #26)
- [ ] **SR run (rsdiff1.5, path B)** — after LR ep1000:
      `./scripts/vast_run.sh run-sr 1000 full_sr_gdm`. Base seeded+frozen, train
      92.7M SR only. Then `sample-grid full_sr_gdm` (auto `--sr` 256² cascade). (task #27)
- [ ] **Joint fine-tune (cheap legacy approx)** — after SR converges:
      `./scripts/vast_run.sh run-joint 200 full_joint_gdm`. Both unets, base
      unfrozen, `L = L_LR + 0.8·L_SR` (λ via SR-grad scaling). Seeds both unets
      from `full_sr_gdm/checkpoint.pt`. Code: `Imagen_text_joint_pytorch.py`.
      **Smoke 10ep first** — both unets at 256² may OOM on 24GB; tune batch.
      Two known shortcuts vs paper: both unets see GT low-res (not LR-GDM
      output); separate backward steps not one graph. True end-to-end joint =
      rewrite (task #28). See [[project_joint_finetune_divergence]]. (task #29)

### SR migration plan (current 4090 ephemeral — migrate before destroy)

1. ep1000 → pull final ckpt + grids + milestones local (have ep100–900; need ep1000).
2. Launch new cheap 4090.
3. **Box-to-box** rsync: new box pulls code + weights + RSICD from old box (faster
   than local uplink). Fallback if old box dies first: upload ep1000 from local.
4. Verify new box complete → **then** destroy old.

**Base-milestone selection = plan (a), decided.** SR training is independent of
the base (unet2 conditions on real 256→128 downsamples, not base output; base
frozen, used only at cascade *inference*). So: seed SR with ep1000, train SR,
then select best base milestone (ep700/900/1000) via **full-cascade 256² FID at
inference** — score the shipped output, not LR-only. Skips the costly large-N
LR sweep (~$15–30, wrong metric).

**Budget gate:** SR run ~120h+ (≥ LR, heavier at 256²) ≈ $50–70. Balance ~$28
earlier → **top-up before launch.**

## Utilities

- [ ] **Denoising-trajectory viz** — sample intermediate steps noise→image, save as
      a strip/grid (e.g. t=1000,800,…,0). Legacy: monkeypatch imagen-pytorch
      `p_sample_loop` to snapshot `img` at chosen t. Rewrite: trivial in the
      diffusers scheduler loop → ship as `rsdiff sample --trajectory`. Prefer
      building in the rewrite unless needed sooner for debugging.

## Reproduction + eval

- [ ] Run FID on LR-128 and SR-256 outputs: `rsdiff eval --metric fid
      --fid-feature 2048 --fid-size {128,256}`. Generate full test split first
      (~1093 imgs), not just grids.
- [ ] Fill `meta.fid_measured` in `rsdiff1.yaml` / `rsdiff1.5.yaml` once measured.
- [ ] Decide paper-comparable protocol: confirm feature=2048 + 256² cascade vs
      RSICD test. (66.49 provenance unrecoverable — document, don't chase exactly.)
- [ ] Optional: train `rsdiff1` (paper line, 723M) for a head-to-head FID row.

## De-legacy — port into `rsdiff` package (task #28)

- [ ] `trainer.py`: build base + SR unets from `cascade.*`; seed+freeze base
      (path B); train SR. Replace `NotImplementedError`.
- [ ] `rsdiff sample`: port `sample_grid.py --sr` (2-unet, `stop_at_unet_number=2`).
- [ ] Decide engine: keep imagen-pytorch as a dep, or reimplement cascade on
      diffusers `UNet2DConditionModel` (roadmap open question #1).
- [ ] Verify `rsdiff train -c rsdiff1.5.yaml` matches the legacy run's FID.
- [ ] **Delete `legacy/`** once parity confirmed. Update `CLAUDE.md` + scripts.

## v0 overfit mitigations (must ship in rewrite — see roadmap)

- [ ] weight_decay 0.01, light aug (h-flip + reflect crop, no rotation).
- [ ] val-FID every N ep on 1094-img val split + early stop (patience 10).
- [ ] sample-diversity (LPIPS) + memorization probe at milestones.
- [ ] compose into `rsdiff eval --during-training`.

## Docs / publish (gate: after SR run lands)

README + site rewritten reproduction-first (rsdiff1.5 headline, source install,
diffusers rewrite = roadmap). Site builds `--strict` clean. Fill these `_TBD_`
in one pass once SR + paper-comparable eval done, then publish:

- [ ] `results.md` + `README.md` headline FID: rsdiff1.5 256² (full test split, feat2048) + CLIP-score + zero-shot OA.
- [ ] `results.md` FID-vs-epoch: extend curve ep700→ep1000 (have ep100–600).
- [ ] `results.md` inference cost: SRDM + full-cascade time/img on 4090.
- [ ] `results.md` + `method.md` SR training wall-clock + cost (path B).
- [ ] `results.md` qualitative: real sample grids into `docs/site/docs/assets/` (replace ASCII sketch). ep900 grid pulling now.
- [ ] `configs/rsdiff1.5.yaml` `meta.fid_measured` (+ rsdiff1 if trained).
- [x] citation: NCAA 2024 article bibtex (Sebaq & ElHelw, Neural Comput. Appl. 36(36):23103–23111, doi 10.1007/s00521-024-10363-3) in README + citation.md.

## Infra / housekeeping

- [ ] Periodically pull milestones + fire grids as LR run advances (ep 600–1000).
- [ ] Top up vast.ai before the SR run if balance < SR cost estimate.
- [ ] Release checkpoints to HF Hub under `asebaq/rsdiff-*` once FID passes.

## Future versions (addresses thesis future-work)

See [roadmap.md](roadmap.md) "Thesis future-work → rsdiff plan" for the mapping.

- [ ] **v2 fewer epochs**: drive training length by val-FID early-stop (+ weight
      decay + aug), not fixed 1000 ep. Expect ~200–400 ep. Don't cut on loss alone.
- [ ] v1: spatial-reasoning conditioning (layout/region + mask ControlNet).
- [ ] v1.x: rare-feature handling — balanced sampling + external data + full
      fine-tune (NO LoRA; models small enough to fully fine-tune).
- [ ] v0.x: inference speedup via few-step samplers only (NO distillation —
      rsdiff1.5 already 119.9M).
- [ ] **v2 higher-res (>256): latent diffusion + higher-res RS dataset** (DOTA /
      fMoW / Million-AID). Pixel cascade can't exceed RSICD's 224/256 ceiling.
- [ ] **v3 multispectral (Sentinel-2, 13-band)**: custom MS-VAE (or pixel-space);
      data = BigEarthNet / SEN12MS / EuroSAT-MS; eval = SAM + per-band + downstream
      (Inception-FID invalid >3 bands). See roadmap v3.
- [ ] extend zero-shot eval with RS-pretrained backbone.

## Blocked

- [ ] Lit-review pass — locks v0 scope (cascade vs LDM, RSFID vs Inception-FID,
      which v1 conditioning). (task #14)
