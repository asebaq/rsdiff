# ddpm/

Training implementation of the RSDiff cascade: the patched thesis-era `imagen-pytorch`
code that produced the released checkpoint. The installable `rsdiff` package covers
sampling and evaluation; training lives here until the package gains it.

`rsdiff sample` reproduces `sample_grid.py` pixel for pixel, so new sampling and scoring
work belongs in the package (see the reproduction status in
[`docs/reproducibility.md`](../docs/reproducibility.md#reproduction-status)).

```
ddpm/
  models/Imagen_text_pytorch.py      # stage 1: 128x128 base UNet
  models/Imagen_text_sr_pytorch.py   # stage 2: 128 -> 256 SR UNet
  models/Imagen_text_joint_pytorch.py
  scripts/run_training_local.sh      # stage 1 launcher
  scripts/run_sr.sh                  # stage 2 launcher
  sample_grid.py                     # historical sampler used by scripts/eval/*.sh
  evaluate_model.py                  # CLIP zero-shot OA on the test split
  requirements_dgx.txt               # torch 2.5 + cu124 pins for training boxes
```

Environment, dataset preparation, training, sweeps, and scoring are documented in
[`docs/reproducibility.md`](../docs/reproducibility.md).
