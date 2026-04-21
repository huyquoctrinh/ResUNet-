import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import load_config
from data import build_datasets
from metrics import MetricTracker
from models import MoConvSSMNet
from utils import load_checkpoint


@torch.no_grad()
def predict_volume_2d(model, vol_1dhw: torch.Tensor, img_size, device, batch_size: int = 16):
    """vol_1dhw: (1, D, H, W). Returns pred (D, H, W)."""
    D, H, W = vol_1dhw.shape[-3], vol_1dhw.shape[-2], vol_1dhw.shape[-1]
    vol_in = vol_1dhw.permute(1, 0, 2, 3).to(device)  # (D, 1, H, W)
    if (H, W) != tuple(img_size):
        vol_in = F.interpolate(vol_in, size=tuple(img_size), mode="bilinear", align_corners=False)
    preds = []
    for i in range(0, D, batch_size):
        out = model(vol_in[i:i + batch_size])
        preds.append(out.argmax(dim=1).cpu())
    pred = torch.cat(preds, dim=0)
    if (H, W) != tuple(img_size):
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(H, W), mode="nearest")[:, 0].long()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(cfg.logging.log_dir) / cfg.data.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    _, _, test_ds = build_datasets(cfg)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

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

    tracker = MetricTracker(
        cfg.data.num_classes,
        cfg.data.class_names,
        ignore_background=cfg.evaluation.ignore_background,
    )

    pred_dir = out_root / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for vol, label, name in tqdm(test_loader, desc="test"):
        # vol: (1, 1, D, H, W) for volume; (1, 1, H, W) for slices
        if vol.dim() == 5:
            pred = predict_volume_2d(model, vol[0], cfg.data.img_size, device)
            tracker.update(pred, label[0])
            if args.save_predictions:
                np.savez_compressed(pred_dir / f"{Path(name[0]).stem}_pred.npz", pred=pred.numpy().astype(np.uint8))
        else:
            img = vol.to(device)
            logits = model(img)
            pred = logits.argmax(dim=1).cpu()
            tracker.update(pred, label)
            if args.save_predictions:
                np.savez_compressed(pred_dir / f"{Path(name[0]).stem}_pred.npz", pred=pred.numpy().astype(np.uint8))

    title = f"{cfg.data.dataset.upper()} Results"
    print(tracker.format_table(title))

    res = tracker.compute()
    csv_path = out_root / "test_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "dice", "iou"])
        for name, d, i in zip(res["class_names"], res["per_class_dice"], res["per_class_iou"]):
            w.writerow([name, f"{d:.4f}", f"{i:.4f}"])
        w.writerow(["Mean", f"{res['mean_dice']:.4f}", f"{res['mean_iou']:.4f}"])
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
