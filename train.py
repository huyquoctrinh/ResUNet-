import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import load_config
from data import build_datasets
from losses import CombinedLoss
from metrics import MetricTracker
from models import MoConvSSMNet
from utils import build_logger, build_tensorboard, save_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg.training.epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_2d(model, loader, num_classes, class_names, device, ignore_bg=True):
    model.eval()
    tracker = MetricTracker(num_classes, class_names, ignore_background=ignore_bg)
    for batch in tqdm(loader, desc="eval", leave=False):
        img, label, _ = batch
        img = img.to(device, non_blocking=True)
        logits = model(img)
        pred = logits.argmax(dim=1).cpu()
        tracker.update(pred, label)
    return tracker


@torch.no_grad()
def evaluate_volumes_2d_slices(model, loader, cfg, device):
    """Run 2D model over each slice of 3D volumes; aggregate 3D metrics."""
    model.eval()
    tracker = MetricTracker(cfg.data.num_classes, cfg.data.class_names, ignore_background=cfg.evaluation.ignore_background)
    img_size = tuple(cfg.data.img_size)
    for vol, label, _ in tqdm(loader, desc="eval-vol", leave=False):
        vol = vol[0]      # (1, D, H, W)
        label = label[0]  # (D, H, W)
        D, H, W = vol.shape[-3], vol.shape[-2], vol.shape[-1]
        vol_in = vol.permute(1, 0, 2, 3).to(device)    # (D, 1, H, W)
        if (H, W) != img_size:
            vol_in = torch.nn.functional.interpolate(vol_in, size=img_size, mode="bilinear", align_corners=False)
        # batch over slices
        preds = []
        bs = 16
        for i in range(0, D, bs):
            out = model(vol_in[i:i + bs])
            preds.append(out.argmax(dim=1).cpu())
        pred = torch.cat(preds, dim=0)      # (D, h, w)
        if (H, W) != img_size:
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1).float(), size=(H, W), mode="nearest")[:, 0].long()
        tracker.update(pred, label)
    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = True

    dataset_name = cfg.data.dataset
    out_root = Path(cfg.logging.log_dir) / dataset_name
    ckpt_dir = out_root / "checkpoints"
    tb_dir = out_root / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg._config_path, out_root / "config.yaml")

    logger = build_logger("train", out_root)
    writer = build_tensorboard(tb_dir) if cfg.logging.tensorboard else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    # Data
    train_ds, val_ds, _ = build_datasets(cfg)
    logger.info(f"train={len(train_ds)}  val={len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory, drop_last=True,
    )
    val_is_volume = dataset_name == "synapse"
    val_loader = DataLoader(
        val_ds, batch_size=1 if val_is_volume else cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, pin_memory=cfg.training.pin_memory,
    )

    # Model
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
        deep_supervision=cfg.model.deep_supervision,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"params={n_params/1e6:.2f}M")

    criterion = CombinedLoss(alpha=cfg.training.loss_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp and device.type == "cuda")

    best_metric = -1.0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.training.epochs}", leave=False)
        running_loss = 0.0
        for img, label, _ in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.training.amp and device.type == "cuda"):
                logits = model(img)
                loss = criterion(logits, label)
                aux = model.aux_losses()
                loss = loss + 0.01 * aux
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item() if global_step > 0 else loss.item()
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix(loss=f"{running_loss:.4f}")
            global_step += 1

        # --- validation ---
        if val_is_volume:
            tracker = evaluate_volumes_2d_slices(model, val_loader, cfg, device)
        else:
            tracker = evaluate_2d(
                model, val_loader, cfg.data.num_classes, cfg.data.class_names,
                device, ignore_bg=cfg.evaluation.ignore_background,
            )
        res = tracker.compute()
        mean_dice = res["mean_dice"]
        mean_iou = res["mean_iou"]
        logger.info(f"epoch {epoch+1}: val mean_dice={mean_dice*100:.2f}% mean_iou={mean_iou*100:.2f}%")
        if writer is not None:
            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            writer.add_scalar("val/mean_iou", mean_iou, epoch)
            for name, d, i in zip(res["class_names"], res["per_class_dice"], res["per_class_iou"]):
                writer.add_scalar(f"val_dice/{name}", d, epoch)
                writer.add_scalar(f"val_iou/{name}", i, epoch)

        # --- checkpoint ---
        save_checkpoint(ckpt_dir / "latest_model.pth", model, optimizer, scheduler, epoch=epoch + 1, best_metric=best_metric)
        if mean_dice > best_metric:
            best_metric = mean_dice
            save_checkpoint(ckpt_dir / "best_model.pth", model, optimizer, scheduler, epoch=epoch + 1, best_metric=best_metric)
            logger.info(f"→ new best: dice={best_metric*100:.2f}%")
        if (epoch + 1) % cfg.logging.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch+1:03d}.pth", model, optimizer, scheduler, epoch=epoch + 1, best_metric=best_metric)

    logger.info(f"training done. best mean_dice={best_metric*100:.2f}%")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
