import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.data_acdc import ACDCdataset, RandomGenerator as ACDCRandomGenerator
from data_utils.data_synapse import Synapse_dataset, RandomGenerator as SynapseRandomGenerator
from data_utils.data_polyp import PolypDataset, PolypTestDataset
from losses import CombinedLoss, FeatureDistillationLoss
from metrics import MetricTracker
from network import EdgeGuidedSegModel
from network.kd_teacher import TeacherModel
from utils import build_logger, build_tensorboard, save_checkpoint


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg):
    name = cfg['dataset']
    img_size = tuple(cfg['img_size'])
    transform = ACDCRandomGenerator(output_size=list(img_size))

    if name == 'acdc':
        train_ds = ACDCdataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='train',
            transform=transform,
        )
        val_ds = ACDCdataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='valid',
            transform=ACDCRandomGenerator(output_size=list(img_size)),
        )
        test_ds = ACDCdataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='test',
        )
        return train_ds, val_ds, test_ds

    if name == 'synapse':
        syn_transform = SynapseRandomGenerator(output_size=list(img_size))
        nclass = cfg.get('nclass', 9)
        train_ds = Synapse_dataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='train',
            nclass=nclass,
            transform=syn_transform,
        )
        val_ds = Synapse_dataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='test',
            nclass=nclass,
        )
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    if name == 'polyp':
        train_ds = PolypDataset(
            root=cfg['data_root'],
            img_size=img_size[0],
        )
        test_datasets = cfg.get('test_datasets', ['Kvasir'])
        first_test = test_datasets[0]
        val_ds = PolypTestDataset(
            root=str(Path(cfg['test_root']) / first_test),
            img_size=img_size[0],
            original_resolution=False,
        )
        test_ds = val_ds
        return train_ds, val_ds, test_ds

    raise ValueError(f"unknown dataset: {name}")


def build_optimizer(model, cfg, extra_params=None):
    backbone_params = list(model.encoder.parameters())
    if model.input_proj is not None:
        backbone_params += list(model.input_proj.parameters())

    new_params = list(model.decoder.parameters())
    if model.edge_net is not None:
        new_params += list(model.edge_net.parameters())
    if model.mhsa3 is not None:
        new_params += list(model.mhsa3.parameters())
    if model.mhsa4 is not None:
        new_params += list(model.mhsa4.parameters())
    if extra_params:
        new_params += list(extra_params)

    param_groups = [
        {'params': backbone_params, 'lr': cfg['lr_backbone']},
        {'params': new_params, 'lr': cfg['lr_new']},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg['epochs'] * steps_per_epoch
    warmup_steps = cfg['warmup_epochs'] * steps_per_epoch

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
        img, label = batch['image'], batch['label']
        img = img.to(device, non_blocking=True)
        logits = model(img)
        pred = logits.argmax(dim=1).cpu()
        tracker.update(pred, label)
    return tracker


@torch.no_grad()
def evaluate_volumes(model, loader, cfg, device):
    """Evaluate on 3D volumes slice-by-slice, following EMCAD protocol (no normalization)."""
    model.eval()
    num_classes = cfg['num_classes']
    class_names = cfg['class_names']
    ignore_bg = cfg.get('ignore_background', True)
    img_size = tuple(cfg['img_size'])
    tracker = MetricTracker(num_classes, class_names, ignore_background=ignore_bg)

    for batch in tqdm(loader, desc="eval-vol", leave=False):
        vol = batch['image']    # (1, D, H, W)
        label = batch['label']  # (1, D, H, W)
        if isinstance(vol, torch.Tensor):
            vol_np = vol.numpy()
            label_np = label.numpy()
        else:
            vol_np = vol
            label_np = label

        vol_np = vol_np[0].astype(np.float32)    # (D, H, W)
        label_np = label_np[0]                    # (D, H, W)

        D, H, W = vol_np.shape
        prediction = np.zeros_like(label_np)

        for ind in range(D):
            slc = vol_np[ind, :, :]
            x, y = slc.shape
            if x != img_size[0] or y != img_size[1]:
                slc = zoom(slc, (img_size[0] / x, img_size[1] / y), order=3)
            inp = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(inp)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().numpy()
            if x != img_size[0] or y != img_size[1]:
                pred = zoom(out, (x / img_size[0], y / img_size[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred

        pred_t = torch.from_numpy(prediction.astype(np.int64)).long()
        label_t = torch.from_numpy(label_np.astype(np.int64)).long()
        tracker.update(pred_t, label_t)

    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained backbone weights')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True

    dataset_name = cfg['dataset']
    kd_cfg = cfg.get('kd', {})
    kd_enabled = kd_cfg.get('enabled', False)
    ablation_name = cfg.get('ablation', {}).get('name', '')
    if ablation_name:
        run_name = f"{ablation_name}_{dataset_name}"
    elif kd_enabled:
        run_name = f"{dataset_name}_kd_{kd_cfg['teacher']}"
    else:
        run_name = dataset_name
    out_root = Path(cfg['log_dir']) / run_name
    ckpt_dir = out_root / 'checkpoints'
    tb_dir = out_root / 'tensorboard'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / 'config.yaml')

    logger = build_logger('train', out_root)
    writer = build_tensorboard(tb_dir) if cfg.get('tensorboard', False) else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device={device}")

    # Data
    train_ds, val_ds, _ = build_datasets(cfg)
    logger.info(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    val_is_volume = (dataset_name == 'synapse')
    val_loader = DataLoader(
        val_ds,
        batch_size=1 if val_is_volume else cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
    )

    # Model
    ablation = cfg.get('ablation', {})
    deep_supervision = ablation.get('deep_supervision', False)
    model = EdgeGuidedSegModel(
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        backbone=cfg['backbone'],
        num_heads=cfg['num_heads'],
        attn_drop=cfg.get('attn_drop', 0.),
        proj_drop=cfg.get('proj_drop', 0.),
        edge_type=ablation.get('edge_type', 'learned'),
        attention_type=ablation.get('attention_type', 'cross_attention'),
        q_source=ablation.get('q_source', 'encoder'),
        attention_scales=ablation.get('attention_scales', ['enc3', 'enc4']),
        deep_supervision=deep_supervision,
    ).to(device)

    if args.pretrained:
        model.load_pretrained_backbone(args.pretrained)
        logger.info(f"loaded pretrained backbone from {args.pretrained}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"params={n_params / 1e6:.2f}M")

    # --- Knowledge Distillation setup ---
    teacher = None
    kd_loss_fn = None
    kd_weight = 0.0
    if kd_enabled:
        teacher_name = kd_cfg['teacher']
        teacher_model_id = kd_cfg.get('teacher_model',
                                      TeacherModel.TEACHER_CONFIGS[teacher_name]['default_model'])
        teacher_img_size = kd_cfg.get('teacher_img_size', cfg['img_size'][0])
        teacher = TeacherModel(teacher_name, teacher_model_id, teacher_img_size).to(device)
        teacher.eval()

        teacher_dim = teacher.hidden_size
        student_dim = model.channels[2]  # f3 at stride 16
        kd_loss_fn = FeatureDistillationLoss(student_dim, teacher_dim).to(device)
        kd_weight = kd_cfg.get('kd_weight', 1.0)

        logger.info(f"KD: teacher={teacher_name} ({teacher_model_id}), "
                    f"dim={student_dim}->{teacher_dim}, weight={kd_weight}")

    criterion = CombinedLoss(alpha=0.5)
    aux_weight = 0.4 if deep_supervision else 0.0
    extra_params = list(kd_loss_fn.parameters()) if kd_loss_fn is not None else None
    optimizer = build_optimizer(model, cfg, extra_params=extra_params)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(enabled=cfg.get('amp', True) and device.type == 'cuda')

    best_metric = -1.0
    global_step = 0
    patience = cfg.get('early_stopping_patience', 0)
    epochs_no_improve = 0

    for epoch in range(cfg['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}", leave=False)
        running_loss = 0.0

        for batch in pbar:
            img, label = batch['image'], batch['label']
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.get('amp', True) and device.type == 'cuda'):
                if kd_enabled:
                    logits, feats = model(img, return_features=True)
                    seg_loss = criterion(logits, label)
                    teacher_feat = teacher(img)
                    distill_loss = kd_loss_fn(feats['f3'], teacher_feat)
                    loss = seg_loss + kd_weight * distill_loss
                elif deep_supervision:
                    out = model(img)
                    logits, aux_logits = out
                    main_loss = criterion(logits, label)
                    label_ds = F.interpolate(
                        label.float().unsqueeze(1),
                        size=aux_logits.shape[2:], mode='nearest',
                    ).squeeze(1).long()
                    aux_loss = criterion(aux_logits, label_ds)
                    loss = main_loss + aux_weight * aux_loss
                else:
                    logits = model(img)
                    loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * loss.item() if global_step > 0 else loss.item()
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if kd_enabled:
                    writer.add_scalar('train/seg_loss', seg_loss.item(), global_step)
                    writer.add_scalar('train/kd_loss', distill_loss.item(), global_step)
            pbar.set_postfix(loss=f"{running_loss:.4f}")
            global_step += 1

        # --- validation ---
        if (epoch + 1) % cfg.get('val_every', 1) == 0:
            if val_is_volume:
                tracker = evaluate_volumes(model, val_loader, cfg, device)
            else:
                tracker = evaluate_2d(
                    model, val_loader,
                    cfg['num_classes'], cfg['class_names'],
                    device, ignore_bg=cfg.get('ignore_background', True),
                )
            res = tracker.compute()
            mean_dice = res['mean_dice']
            mean_iou = res['mean_iou']
            logger.info(
                f"epoch {epoch + 1}: val mean_dice={mean_dice * 100:.2f}%  "
                f"mean_iou={mean_iou * 100:.2f}%"
            )
            if writer is not None:
                writer.add_scalar('val/mean_dice', mean_dice, epoch)
                writer.add_scalar('val/mean_iou', mean_iou, epoch)
                for name, d, i in zip(res['class_names'], res['per_class_dice'], res['per_class_iou']):
                    writer.add_scalar(f'val_dice/{name}', d, epoch)
                    writer.add_scalar(f'val_iou/{name}', i, epoch)

            # --- checkpoint ---
            save_checkpoint(
                ckpt_dir / 'latest_model.pth', model, optimizer, scheduler,
                epoch=epoch + 1, best_metric=best_metric,
            )
            if mean_dice > best_metric:
                best_metric = mean_dice
                epochs_no_improve = 0
                save_checkpoint(
                    ckpt_dir / 'best_model.pth', model, optimizer, scheduler,
                    epoch=epoch + 1, best_metric=best_metric,
                )
                logger.info(f"-> new best: dice={best_metric * 100:.2f}%")
            else:
                epochs_no_improve += 1

            if patience > 0 and epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} "
                            f"(no improvement for {patience} validations)")
                break

        if (epoch + 1) % cfg.get('save_every', 50) == 0:
            save_checkpoint(
                ckpt_dir / f'epoch_{epoch + 1:03d}.pth', model, optimizer, scheduler,
                epoch=epoch + 1, best_metric=best_metric,
            )

    logger.info(f"training done. best mean_dice={best_metric * 100:.2f}%")
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
