"""VAE reconstruction check for RSDiff v0.

Encodes -> decodes RSICD test images through diffusers' SD-1.5 KL-f8 VAE,
measures recon quality (PSNR, LPIPS, recon-FID), and saves a visual grid.

Decision rule baked into the README at top of `.notes/vae_recon_check_*.md`:
  - PSNR <23 dB OR LPIPS >0.15  => VAE is bottleneck for v0
  - PSNR 23-26 dB                => acceptable, run experiments
  - PSNR >26 dB                  => VAE clean, full ahead

Reference numbers (SD-1.5 VAE on COCO): PSNR ~25.7, LPIPS ~0.10.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


def load_rsicd_test(csv_path: Path, img_dir: Path, limit: int | None = None) -> list[Path]:
    rows: list[Path] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                rows.append(img_dir / row["filename"])
    rows = sorted(set(rows), key=lambda p: p.name)
    if limit is not None:
        rows = rows[:limit]
    return rows


def to_tensor_norm(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = T.functional.pil_to_tensor(img).float() / 255.0
    return arr * 2.0 - 1.0


def from_tensor_norm(t: torch.Tensor) -> torch.Tensor:
    return ((t.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def run(
    vae,
    paths: list[Path],
    size: int,
    device: str,
    dtype: torch.dtype,
    lpips_fn,
    out_dir: Path,
    batch: int = 8,
    grid_n: int = 8,
) -> dict:
    from torchvision.utils import save_image

    scaling = vae.config.scaling_factor
    psnrs: list[float] = []
    lpipses: list[float] = []
    orig_dir = out_dir / f"_orig_{size}"
    recon_dir = out_dir / f"_recon_{size}"
    orig_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    # Keep only a small set of tensors for the grid PNG.
    grid_orig: list[torch.Tensor] = []
    grid_recon: list[torch.Tensor] = []
    grid_idx = set(torch.linspace(0, len(paths) - 1, grid_n).long().tolist())
    t0 = time.time()
    counter = 0
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        imgs = [to_tensor_norm(Image.open(p), size) for p in chunk]
        x = torch.stack(imgs).to(device=device, dtype=dtype)
        z = vae.encode(x).latent_dist.sample() * scaling
        xr = vae.decode(z / scaling).sample
        a01 = from_tensor_norm(x.float().cpu())
        b01 = from_tensor_norm(xr.float().cpu())
        for j in range(a01.shape[0]):
            psnrs.append(psnr(a01[j], b01[j]))
            save_image(a01[j].clamp(0, 1), orig_dir / f"{counter:05d}.png")
            save_image(b01[j].clamp(0, 1), recon_dir / f"{counter:05d}.png")
            if counter in grid_idx:
                grid_orig.append(a01[j])
                grid_recon.append(b01[j])
            counter += 1
        if lpips_fn is not None:
            with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                lp = lpips_fn(x.float(), xr.float()).detach().cpu().flatten().tolist()
            lpipses.extend(lp)
        del a01, b01, x, z, xr, imgs
        if (i // batch) % 5 == 0:
            print(f"  [{size}px] {i + len(chunk)}/{len(paths)}  t={time.time() - t0:.1f}s", flush=True)
    return {
        "size": size,
        "n": len(paths),
        "psnr_mean": sum(psnrs) / len(psnrs),
        "psnr_std": (sum((p - sum(psnrs) / len(psnrs)) ** 2 for p in psnrs) / len(psnrs)) ** 0.5,
        "lpips_mean": (sum(lpipses) / len(lpipses)) if lpipses else None,
        "lpips_std": ((sum((l - sum(lpipses) / len(lpipses)) ** 2 for l in lpipses) / len(lpipses)) ** 0.5) if lpipses else None,
        "wallclock_s": round(time.time() - t0, 1),
        "_grid_orig": torch.stack(grid_orig) if grid_orig else None,
        "_grid_recon": torch.stack(grid_recon) if grid_recon else None,
        "_orig_dir": orig_dir,
        "_recon_dir": recon_dir,
    }


def save_grid(orig: torch.Tensor, recon: torch.Tensor, out_path: Path, n: int = 8) -> None:
    from torchvision.utils import make_grid, save_image

    n = min(n, orig.shape[0])
    rows = torch.cat([orig[:n], recon[:n], (orig[:n] - recon[:n]).abs() * 4.0], dim=0)
    grid = make_grid(rows.clamp(0, 1), nrow=n, padding=2)
    save_image(grid, out_path)


def compute_recon_fid(orig_dir: Path, recon_dir: Path) -> float | None:
    try:
        from cleanfid import fid
    except Exception as e:
        print(f"  [warn] clean-fid unavailable, skipping recon-FID: {e}", flush=True)
        return None
    return float(fid.compute_fid(str(orig_dir), str(recon_dir), mode="clean", num_workers=0))


def dump_for_fid(t: torch.Tensor, out_dir: Path) -> None:
    from torchvision.utils import save_image

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(t.shape[0]):
        save_image(t[i].clamp(0, 1), out_dir / f"{i:05d}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--img-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--vae-id", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--sizes", type=int, nargs="+", default=[224, 256, 512])
    p.add_argument("--limit", type=int, default=None, help="cap test images (default: all 1093)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--no-fid", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device={device}  dtype={dtype}", flush=True)

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(args.vae_id, torch_dtype=dtype).to(device).eval()
    print(f"loaded VAE: {args.vae_id}", flush=True)

    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
            print("loaded LPIPS (alex)", flush=True)
        except Exception as e:
            print(f"[warn] LPIPS unavailable: {e}", flush=True)

    paths = load_rsicd_test(args.csv, args.img_dir, limit=args.limit)
    print(f"loaded {len(paths)} RSICD test images", flush=True)

    summary: dict = {"vae_id": args.vae_id, "device": device, "n_images": len(paths), "results": {}}

    for size in args.sizes:
        print(f"\n=== size={size}px ===", flush=True)
        r = run(vae, paths, size, device, dtype, lpips_fn, args.out, batch=args.batch)
        grid_orig = r.pop("_grid_orig")
        grid_recon = r.pop("_grid_recon")
        orig_dir = r.pop("_orig_dir")
        recon_dir = r.pop("_recon_dir")

        if grid_orig is not None and grid_recon is not None:
            grid_path = args.out / f"grid_{size}.png"
            save_grid(grid_orig, grid_recon, grid_path, n=grid_orig.shape[0])
            print(f"  saved grid: {grid_path}", flush=True)

        if not args.no_fid:
            r["recon_fid"] = compute_recon_fid(orig_dir, recon_dir)
            print(f"  recon-FID={r['recon_fid']}", flush=True)

        summary["results"][str(size)] = r
        print(f"  PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f}", flush=True)
        if r.get("lpips_mean") is not None:
            print(f"  LPIPS={r['lpips_mean']:.4f}±{r['lpips_std']:.4f}", flush=True)
        with open(args.out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
