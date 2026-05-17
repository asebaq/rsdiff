"""Load a legacy LR-GDM checkpoint and sample a batch of test-set captions.

Outputs:
    <log_dir>/generated_images/grid_step{step}/
        <imgid>_<sent>.png       per-image PNG
        _grid.png                NxN tiled grid (matplotlib)
        captions.txt             one caption per line in order

Usage:
    python DDPM/sample_grid.py \
        --log_dir DDPM/logs/smoke_lr_gdm \
        --data_root RSICD_optimal \
        --n 16 --img_sz 128 --ts 1000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch
from PIL import Image
from imagen_pytorch import Imagen, ImagenTrainer, Unet

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from utils.seed_everything import seed_everything  # noqa: E402


def pick_device(arg: str) -> str:
    if arg != 'auto':
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_imagen(img_sz: int, ts: int) -> tuple[Imagen, Unet]:
    """Mirror legacy/DDPM/Imagen_text_pytorch.py build_models()."""
    unet = Unet(
        dim=128,
        cond_dim=256,
        dim_mults=(1, 2, 2, 2),
        num_resnet_blocks=0,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
    )
    imagen = Imagen(
        text_encoder_name='t5-base',
        unets=unet,
        image_sizes=img_sz,
        timesteps=ts,
        cond_drop_prob=0.1,
    )
    return imagen, unet


def build_grid(images: list[Image.Image], cols: int = 4) -> Image.Image:
    if not images:
        raise ValueError("no images")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new('RGB', (cols * w, rows * h), color=(0, 0, 0))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


def slugify(text: str, n: int = 40) -> str:
    s = ''.join(ch if ch.isalnum() else '_' for ch in text.strip().lower())
    s = '_'.join(filter(None, s.split('_')))
    return s[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log_dir', required=True, help='Training log dir holding checkpoint.pt')
    p.add_argument('--data_root', required=True, help='RSICD root with dataset_rsicd.csv + RSICD_images')
    p.add_argument('--n', type=int, default=16, help='Number of captions to sample')
    p.add_argument('--cols', type=int, default=4, help='Grid columns')
    p.add_argument('--img_sz', type=int, default=128)
    p.add_argument('--ts', type=int, default=1000)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--split', default='test')
    p.add_argument('--cond_scale', type=float, default=4.0, help='CFG guidance scale (thesis range 3-5)')
    p.add_argument('--out_subdir', default=None, help='Override output subdir name')
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    print(f'device: {device}')

    df = pd.read_csv(os.path.join(args.data_root, 'dataset_rsicd.csv'))
    df = df[df['split'] == args.split].reset_index(drop=True)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    captions = sample['sent1'].tolist()
    print(f'sampling {len(captions)} captions from split={args.split}')

    imagen, _ = build_imagen(args.img_sz, args.ts)
    trainer = ImagenTrainer(imagen=imagen)
    if device == 'cuda':
        trainer = trainer.cuda()

    ckpt = os.path.join(args.log_dir, 'checkpoint.pt')
    trainer.load(ckpt)
    step = int(getattr(trainer, 'steps', torch.tensor(0)).sum().item())
    print(f'loaded {ckpt} (steps={step})')

    out_name = args.out_subdir or f'grid_step{step}'
    out_dir = os.path.join(args.log_dir, 'generated_images', out_name)
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    images = trainer.sample(
        texts=captions,
        cond_scale=args.cond_scale,
        return_pil_images=True,
    )
    print(f'sampled {len(images)} in {time.time() - start:.1f}s')

    for i, (im, cap) in enumerate(zip(images, captions)):
        slug = slugify(cap)
        im.save(os.path.join(out_dir, f'{i:02d}_{slug}.png'))

    build_grid(images, cols=args.cols).save(os.path.join(out_dir, '_grid.png'))
    with open(os.path.join(out_dir, 'captions.txt'), 'w') as f:
        f.write('\n'.join(captions) + '\n')

    print(f'wrote -> {out_dir}')


if __name__ == '__main__':
    main()
