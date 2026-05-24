import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import load_config
from data import build_datasets
from distillation import build_dino_kd
from losses import CombinedLoss, DeepSupervisionLoss
from metrics import MetricTracker
from models import MoConvSSMNet
from utils import build_logger, build_tensorboard, save_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg.training.epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _main_logits(out):
    """Deep-supervision returns a list; eval / argmax uses the main logits."""
    return out[0] if isinstance(out, (list, tuple)) else out


@torch.no_grad()
def evaluate_2d(model, loader, num_classes, class_names, device, ignore_bg=True):
    model.eval()
    tracker = MetricTracker(num_classes, class_names, ignore_background=ignore_bg)
    for batch in tqdm(loader, desc="eval", leave=False):
        img, label, _ = batch
        img = img.to(device, non_blocking=True)
        logits = _main_logits(model(img))
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
            out = _main_logits(model(vol_in[i:i + bs]))
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
    # Save the resolved (base-merged) config so it can be reloaded standalone.
    def _plain(obj):
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [_plain(v) for v in obj]
        return obj

    with open(out_root / "config.yaml", "w") as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False)

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
    encoder_type = getattr(cfg.model, "encoder_type", "moconv_ssm")
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
        encoder_type=encoder_type,
        pvt_variant=getattr(cfg.model, "pvt_variant", "b2"),
        pvt_pretrained=getattr(cfg.model, "pvt_pretrained", True),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"params={n_params/1e6:.2f}M")

    # --- Load ImageNet-pretrained encoder weights (MoConvSSM only) ---
    pretrained_path = getattr(cfg.model, "pretrained_encoder", None)
    if pretrained_path is not None and encoder_type != "pvt_v2":
        pretrained_path = Path(pretrained_path)
        if pretrained_path.exists():
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            # If it's a full checkpoint, extract encoder weights
            if isinstance(ckpt, dict) and "model" in ckpt:
                ckpt = {k[len("encoder."):]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
            elif isinstance(ckpt, dict) and "encoder" in ckpt:
                ckpt = ckpt["encoder"]
            # Handle RGB (3ch) → grayscale (1ch) patch embed conversion
            patch_key = "patch_embed.proj.weight"
            if patch_key in ckpt and ckpt[patch_key].shape[1] != cfg.data.in_channels:
                logger.info(f"converting patch_embed: {ckpt[patch_key].shape[1]}ch → {cfg.data.in_channels}ch")
                ckpt[patch_key] = ckpt[patch_key].mean(dim=1, keepdim=True)
            missing, unexpected = model.encoder.load_state_dict(ckpt, strict=False)
            logger.info(f"loaded pretrained encoder from {pretrained_path}")
            if missing:
                logger.info(f"  missing keys: {missing}")
            if unexpected:
                logger.info(f"  unexpected keys: {unexpected}")
        else:
            logger.warning(f"pretrained encoder not found: {pretrained_path}")

    if cfg.model.deep_supervision:
        ds_weights = tuple(
            getattr(cfg.training, "ds_weights", [1.0, 0.5, 0.25, 0.125])
        )
        criterion = DeepSupervisionLoss(
            alpha=cfg.training.loss_alpha, weights=ds_weights,
        )
        logger.info(f"deep supervision ON, weights={ds_weights}")
    else:
        criterion = CombinedLoss(alpha=cfg.training.loss_alpha)
    # --- DINOv3 KD (Strategy B: encoder feature distillation) ---
    kd_module = build_dino_kd(
        cfg, student_dims=model.encoder.stage_dims,
        spatial_dim=cfg.model.spatial_dim,
    )
    kd_enabled = kd_module is not None
    if kd_enabled:
        kd_module = kd_module.to(device)
        n_proj = sum(p.numel() for p in kd_module.trainable_parameters())
        logger.info(
            f"DINO KD ON: model={cfg.distillation.dino_model} "
            f"blocks={tuple(cfg.distillation.extract_blocks)} "
            f"mode={cfg.distillation.kd_loss_mode} "
            f"projector_params={n_proj/1e6:.3f}M"
        )
        # Capture encoder output via forward hook so we don't re-run encoder.
        _cached = {"features": None, "grids": None}

        def _hook(module, inputs, output):
            _cached["features"], _cached["grids"] = output

        model.encoder.register_forward_hook(_hook)
        lambda_kd = float(cfg.distillation.lambda_kd.encoder_feature)
        kd_warmup = int(getattr(cfg.distillation, "kd_warmup_epochs", 0))

    opt_params = list(model.parameters())
    if kd_enabled:
        opt_params += list(kd_module.trainable_parameters())
    optimizer = torch.optim.AdamW(
        opt_params, lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp and device.type == "cuda")

    best_metric = -1.0
    global_step = 0

    es_cfg = cfg.training.get("early_stopping", {}) if hasattr(cfg.training, "get") else cfg.training.__dict__.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False)) if isinstance(es_cfg, dict) else bool(getattr(es_cfg, "enabled", False))
    es_patience = int(es_cfg.get("patience", 30)) if isinstance(es_cfg, dict) else int(getattr(es_cfg, "patience", 30))
    es_min_delta = float(es_cfg.get("min_delta", 0.0)) if isinstance(es_cfg, dict) else float(getattr(es_cfg, "min_delta", 0.0))
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta, mode="max") if es_enabled else None
    if early_stopper is not None:
        logger.info(f"early stopping: patience={es_patience} min_delta={es_min_delta}")

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
                if kd_enabled:
                    w = (
                        min(1.0, (epoch + 1) / kd_warmup)
                        if kd_warmup > 0 else 1.0
                    )
                    kd_loss, kd_per_level = kd_module.compute_loss(
                        img, _cached["features"], _cached["grids"],
                    )
                    loss = loss + w * lambda_kd * kd_loss
                    if writer is not None:
                        writer.add_scalar(
                            "train/kd_loss", kd_loss.item(), global_step,
                        )
                        for i, v in enumerate(kd_per_level):
                            writer.add_scalar(
                                f"train/kd_level_{i}", v, global_step,
                            )
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

        if early_stopper is not None:
            improved = early_stopper.step(mean_dice)
            logger.info(
                f"early-stop: counter={early_stopper.counter}/{early_stopper.patience} "
                f"best={early_stopper.best*100:.2f}% {'improved' if improved else 'no-improve'}"
            )
            if early_stopper.should_stop:
                logger.info(f"early stopping triggered at epoch {epoch+1}")
                break

    logger.info(f"training done. best mean_dice={best_metric*100:.2f}%")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
