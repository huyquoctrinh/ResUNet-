import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from configs import load_config
from models import MoConvSSMNet
from utils import load_checkpoint


def load_sample(path: Path):
    if path.suffix == ".npz":
        d = np.load(path)
        key_img = "image" if "image" in d else "img"
        img = d[key_img].astype(np.float32)
        return img
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            key = "image" if "image" in f else "img"
            return f[key][...].astype(np.float32)
    raise ValueError(f"unsupported file: {path}")


@torch.no_grad()
def infer_slice_stack(model, img, img_size, device, batch_size: int = 16):
    """img: (D, H, W) or (H, W). Returns pred (D, H, W) or (H, W)."""
    single = img.ndim == 2
    if single:
        img = img[None]
    mean, std = img.mean(), img.std()
    img = (img - mean) / (std + 1e-8)
    D, H, W = img.shape
    x = torch.from_numpy(img).float().unsqueeze(1).to(device)   # (D, 1, H, W)
    if (H, W) != tuple(img_size):
        x = F.interpolate(x, size=tuple(img_size), mode="bilinear", align_corners=False)
    preds = []
    for i in range(0, D, batch_size):
        out = model(x[i:i + batch_size])
        preds.append(out.argmax(dim=1).cpu())
    pred = torch.cat(preds, dim=0)
    if (H, W) != tuple(img_size):
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(H, W), mode="nearest")[:, 0].long()
    pred_np = pred.numpy().astype(np.uint8)
    return pred_np[0] if single else pred_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="file or folder")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MoConvSSMNet(
        in_channels=cfg.data.in_channels,
        num_classes=cfg.data.num_classes,
        base_dim=cfg.model.base_dim,
        depths=cfg.model.depths,
        patch_size=cfg.model.patch_size,
        spatial_dim=cfg.model.spatial_dim,
        kernel_sizes=cfg.model.kernel_sizes,
        n_experts=cfg.model.n_experts,
        ssm_d_state=cfg.model.ssm_d_state,
        ssm_d_conv=cfg.model.ssm_d_conv,
        ssm_expand=cfg.model.ssm_expand,
        ffn_ratio=cfg.model.ffn_ratio,
        router_mode=cfg.model.router_mode,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted([p for p in in_path.iterdir() if p.suffix in {".npz", ".h5", ".hdf5"}])

    for f in files:
        img = load_sample(f)
        pred = infer_slice_stack(model, img, cfg.data.img_size, device)
        out_path = out_dir / f"{f.stem}_pred.npz"
        np.savez_compressed(out_path, pred=pred)
        print(f"saved {out_path}  shape={pred.shape}")


if __name__ == "__main__":
    main()
