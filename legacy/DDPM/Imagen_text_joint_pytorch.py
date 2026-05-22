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
    return logger, writer, df


def build_models(args):
    # Both unets must match the LR base + SR scripts exactly so a post-SR
    # ImagenTrainer checkpoint loads into them.
    unet_gen = Unet(
        dim=128,
        cond_dim=256,
        dim_mults=(1, 2, 2, 2),
        num_resnet_blocks=0,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True)
    )

    unet_sr = Unet(
        dim=128,
        cond_dim=512,
        dim_mults=(1, 2, 3, 4),
        num_resnet_blocks=(2, 2, 2, 2),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True)
    )

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


def train(imagen, df, logger, writer, args):
    # Joint fine-tune phase (paper ch3/ch4): train BOTH unets, base unfrozen,
    # combined objective L = L_LR + lambda_sr * L_SR with lambda_sr=0.8.
    #
    # Cheap legacy approximation, two known shortcuts vs the paper:
    #   1. Both unets still condition on GT low-res (imagen downsamples the real
    #      image internally). True joint feeds the SR unet the LR-GDM *output*;
    #      imagen-pytorch's train() exposes no hook for that. End-to-end base->SR
    #      conditioning lands in the diffusers rewrite, not here.
    #   2. The combined loss is two separate backward/update steps, not one graph.
    #      lambda_sr is applied by scaling the SR unet grads before its update
    #      (under GradScaler the scale cancels at unscale_, so 0.8x grad == 0.8x
    #      effective loss weight).
    LR_UNET, SR_UNET = 1, 2
    model_path = os.path.join(args.log_dir, 'checkpoint.pt')
    resume = os.path.isfile(model_path)

    trainer = ImagenTrainer(imagen=imagen)
    if args.device == 'cuda':
        trainer = trainer.cuda()

    if resume:
        trainer.load(model_path)
        logger.info(f'resumed joint training from {model_path}')
    else:
        # Seed both unets (+ EMA + optimizer) from the post-SR run checkpoint.
        if not os.path.isfile(args.init_ckpt):
            raise FileNotFoundError(f'--init_ckpt not found: {args.init_ckpt}')
        trainer.load(args.init_ckpt)
        logger.info(f'seeded both unets from post-SR checkpoint {args.init_ckpt}')

    # Both unets train in the joint phase.
    for p in imagen.unets[0].parameters():
        p.requires_grad_(True)
    imagen.unets[0].train()

    train_dataloader, test_dataloader = build_dataloaders(df, args, args.sr_sz)

    for j in range(args.start_epoch, args.epochs):
        loss_lr = 0.0
        loss_sr = 0.0
        start = time.time()
        for _, (imgs, txts) in enumerate(tqdm(train_dataloader)):
            loss_lr += trainer(imgs, texts=txts, unet_number=LR_UNET, max_batch_size=2)
            trainer.update(unet_number=LR_UNET)

            loss_sr += trainer(imgs, texts=txts, unet_number=SR_UNET, max_batch_size=2)
            for p in imagen.unets[1].parameters():
                if p.grad is not None:
                    p.grad.mul_(args.lambda_sr)
            trainer.update(unet_number=SR_UNET)

        n = max(len(train_dataloader), 1)
        loss_lr, loss_sr = loss_lr / n, loss_sr / n
        loss_joint = loss_lr + args.lambda_sr * loss_sr
        writer.add_scalar('joint/L_LR', round(loss_lr, 3), j)
        writer.add_scalar('joint/L_SR', round(loss_sr, 3), j)
        writer.add_scalar('joint/L_joint', round(loss_joint, 3), j)

        logger.info(
            f'Finished joint epoch {j} | L_LR={round(loss_lr, 3)} '
            f'L_SR={round(loss_sr, 3)} L_joint={round(loss_joint, 3)} '
            f'in {round(time.time() - start, 3)} sec')

        if not (j % 5):
            data = next(iter(test_dataloader))
            txt = data[1][0]
            start = time.time()
            images = trainer.sample(texts=[txt], batch_size=1, stop_at_unet_number=SR_UNET, return_pil_images=True)
            logger.info(f'Sampling time: {round(time.time() - start, 3)} sec')
            image_path = os.path.join(args.log_dir, 'generated_images',
                                      f"sample-joint-{j}-text-{'_'.join(txt.replace('.', '').split())}.png")
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

    parser = argparse.ArgumentParser(description='Joint fine-tune of Imagen LR base + SR unets')
    parser.add_argument('-i', '--img_sz', type=int, default=128, help='Size of the generated image')
    parser.add_argument('-r', '--sr_sz', type=int, default=256, help='Size of the super-resolved image')
    parser.add_argument('-t', '--ts', type=int, default=1000, help='time steps of the diffusion process')
    parser.add_argument('-b', '--batch_sz', type=int, default=64, help='Size of batch of images')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of joint fine-tune epochs')
    parser.add_argument('-s', '--start_epoch', type=int, default=0, help='Number of starting epoch')
    parser.add_argument('--lambda_sr', type=float, default=0.8,
                        help='Weight on the SR loss in L = L_LR + lambda_sr * L_SR (paper: 0.8).')
    parser.add_argument('-d', '--data_root', type=str,
                        default=os.path.join(REPO_ROOT, 'RSICD_optimal'),
                        help='Path to data directory')
    parser.add_argument('-l', '--log_dir', type=str,
                        default=os.path.join(REPO_ROOT, 'DDPM', 'logs', 'full_joint_gdm'),
                        help='Path to log directory')
    parser.add_argument('--init_ckpt', type=str,
                        default=os.path.join(REPO_ROOT, 'DDPM', 'logs', 'full_sr_gdm', 'checkpoint.pt'),
                        help='Post-SR ImagenTrainer checkpoint to seed both unets from.')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Compute device (auto detects).')

    args = parser.parse_args()
    main(args)
