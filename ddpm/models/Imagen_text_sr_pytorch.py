import os.path
import time
import argparse
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from PIL import Image
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from shutil import copyfile

import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from utils.seed_everything import seed_everything
from utils import log
from torch.utils.tensorboard import SummaryWriter


def pick_device(arg):
    if arg != 'auto':
        return arg
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SatDataset(Dataset):
    def __init__(self, df, root, image_size=64, transform=None, split='train'):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.root = root
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.filename[idx])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        txt = self.df.sent1[idx]
        return img, txt


def config(args):
    os.makedirs(args.log_dir, exist_ok=True)
    logger = log.setup_custom_logger(args.log_dir, 'root')
    logger.debug('main')
    writer = SummaryWriter(os.path.join(args.log_dir, 'runs'))
    df = pd.read_csv(os.path.join(args.data_root, 'dataset_rsicd.csv'))
    os.makedirs(os.path.join(args.log_dir, 'generated_images'), exist_ok=True)
    
    py_file = os.path.realpath(__file__)
    copyfile(py_file, os.path.join(args.log_dir, os.path.basename(py_file)))

    bash_file = os.path.realpath(os.path.join(REPO_ROOT, 'scripts', 'lunch_training_sr.sh'))
    if os.path.isfile(bash_file):
        copyfile(bash_file, os.path.join(args.log_dir, os.path.basename(bash_file)))
    return logger, writer, df


def build_models(args):
    # Base 128x128 generator (unet 1). Must match ddpm/Imagen_text_pytorch.py
    # build_models() exactly so the trained `full_lr_gdm` checkpoint loads into it.
    # We seed this from the LR run and freeze it; only the SR unet (unet 2) trains.
    unet_gen = Unet(
        dim=128,
        cond_dim=256,
        dim_mults=(1, 2, 2, 2),
        num_resnet_blocks=0,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True)
    )

    # SR unet (unet 2) = rsdiff1.5 optimized config (~92.7M). Shrunk from the
    # original 462.4M Efficient-U-Net: dim_mults deepest 8x->4x, num_resnet_blocks
    # ->(2,2,2,2). See configs/rsdiff1.5.yaml. configs/rsdiff1.yaml keeps the 462M.
    unet_sr = Unet(
        dim=128,
        cond_dim=512,
        dim_mults=(1, 2, 3, 4),
        num_resnet_blocks=(2, 2, 2, 2),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True)
    )

    # imagen, which contains the unet above
    imagen = Imagen(
        text_encoder_name='t5-base',
        unets=(unet_gen, unet_sr),
        image_sizes=(args.img_sz, args.sr_sz),
        timesteps=args.ts,
        cond_drop_prob=0.1
    )
    if args.device == 'cuda':
        imagen = imagen.cuda()

    return imagen, unet_gen, unet_sr


def build_dataloaders(df, args, image_size=64):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])

    train_dataset = SatDataset(df, os.path.join(args.data_root, 'RSICD_images'), image_size=image_size, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True, num_workers=2, pin_memory=True)

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    test_dataset = SatDataset(df, os.path.join(args.data_root, 'RSICD_images'), image_size=image_size, transform=transform,
                              split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    return train_dataloader, test_dataloader


def seed_base_unet(imagen, lr_ckpt, logger):
    """Load the trained LR base (unet 1) from an ImagenTrainer checkpoint.

    Seeds imagen.unets[0] from the `unets.0.*` slice of the LR run's saved
    state. Must run BEFORE ImagenTrainer() is constructed so the trainer's EMA
    copy is initialised from the seeded weights, not the random init.
    """
    if not os.path.isfile(lr_ckpt):
        raise FileNotFoundError(f'--lr_ckpt not found: {lr_ckpt}')
    ckpt = torch.load(lr_ckpt, map_location='cpu')
    model_sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    prefix = 'unets.0.'
    base_sd = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
    if not base_sd:
        raise RuntimeError(f'no {prefix}* tensors in {lr_ckpt} (wrong checkpoint?)')
    imagen.unets[0].load_state_dict(base_sd, strict=True)
    logger.info(f'seeded base unet (unet 1) from {lr_ckpt}: {len(base_sd)} tensors')


def freeze_base_unet(imagen, logger):
    n = 0
    for p in imagen.unets[0].parameters():
        p.requires_grad_(False)
        n += 1
    imagen.unets[0].eval()
    logger.info(f'froze base unet (unet 1): {n} param tensors, requires_grad=False')


def train(imagen, df, logger, writer, args):
    # Path B: base (unet 1) is seeded from the trained LR run and frozen.
    # Only the super-resolution unet (unet 2) trains here.
    SR_UNET = 2
    model_path = os.path.join(args.log_dir, 'checkpoint.pt')
    resume = os.path.isfile(model_path)

    # Seed the frozen base before constructing the trainer (EMA copies it).
    if not resume:
        seed_base_unet(imagen, args.lr_ckpt, logger)
    freeze_base_unet(imagen, logger)

    trainer = ImagenTrainer(imagen=imagen)
    if args.device == 'cuda':
        trainer = trainer.cuda()
    if resume:
        trainer.load(model_path)
        logger.info(f'resumed SR training from {model_path}')
        freeze_base_unet(imagen, logger)  # re-freeze after restore

    train_dataloader, test_dataloader = build_dataloaders(df, args, args.sr_sz)

    for j in range(args.start_epoch, args.epochs):
        loss = 0
        start = time.time()
        for _, (imgs, txts) in enumerate(tqdm(train_dataloader)):
            loss += trainer(
                imgs,
                texts=txts,
                unet_number=SR_UNET,
                max_batch_size=2
            )
            trainer.update(unet_number=SR_UNET)
        loss = loss / max(len(train_dataloader), 1)
        writer.add_scalar('Imagen SR-UNet Model', round(loss, 3), j)

        logger.info(
            f'Finished epoch {j} for SR-UNet model with loss: {round(loss, 3)} in {round(time.time() - start, 3)} sec')

        if not (j % 5):
            data = next(iter(test_dataloader))
            txt = data[1][0]
            start = time.time()
            images = trainer.sample(texts=[txt], batch_size=1, stop_at_unet_number=SR_UNET, return_pil_images=True)
            logger.info(f'Sampling time: {round(time.time() - start, 3)} sec')
            image_path = os.path.join(args.log_dir, 'generated_images',
                                      f"sample-SR-UNet-{j}-text-{'_'.join(txt.replace('.', '').split())}.png")
            images[0].save(image_path)
        trainer.save(model_path)


def main(args):
    args.device = pick_device(args.device)
    logger, writer, df = config(args)
    logger.info(f'Using device: {args.device}')
    seed_everything()

    imagen, unet_gen, unet_sr = build_models(args)
    params = sum(p.numel() for p in unet_gen.parameters())
    logger.info(f'Number of image generation UNet model parameters : {params:,}')
    params = sum(p.numel() for p in unet_sr.parameters())
    logger.info(f'Number of super-resolution UNet model parameters : {params:,}')
    params = sum(p.numel() for p in imagen.parameters())
    logger.info(f'Number of Imagen model parameters : {params:,}')

    train(imagen, df, logger, writer, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Imagen with super-res')
    parser.add_argument('-i', '--img_sz', type=int,
                        default=128, help='Size of the generated image')
    parser.add_argument('-r', '--sr_sz', type=int,
                        default=256, help='Size of the super-resolved image')
    parser.add_argument('-t', '--ts', type=int,
                        default=1000, help='time steps of the diffusion process')
    parser.add_argument('-b', '--batch_sz', type=int,
                        default=64, help='Size of batch of images')
    parser.add_argument('-e', '--epochs', type=int,
                        default=1000, help='Number of training epochs')
    parser.add_argument('-s', '--start_epoch', type=int,
                        default=0, help='Number of starting epoch')
    parser.add_argument('-d', '--data_root', type=str,
                        default=os.path.join(REPO_ROOT, 'RSICD_optimal'),
                        help='Path to data directory')
    parser.add_argument('-l', '--log_dir', type=str,
                        default=os.path.join(REPO_ROOT, 'logs', 'full_sr_gdm'),
                        help='Path to log directory')
    parser.add_argument('--lr_ckpt', type=str,
                        default=os.path.join(REPO_ROOT, 'logs', 'full_lr_gdm', 'checkpoint.pt'),
                        help='Trained LR base checkpoint to seed + freeze unet 1 (path B).')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Compute device (auto detects).')

    args = parser.parse_args()
    main(args)
