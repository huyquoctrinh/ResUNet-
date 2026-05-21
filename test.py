import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.data_acdc import ACDCdataset
from data_utils.data_synapse import Synapse_dataset
from metrics import MetricTracker
from network import EdgeGuidedSegModel
from utils import load_checkpoint


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_test_dataset(cfg):
    name = cfg['dataset']
    if name == 'acdc':
        return ACDCdataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='test',
        )
    if name == 'synapse':
        return Synapse_dataset(
            base_dir=cfg['data_root'],
            list_dir=cfg['list_dir'],
            split='test',
            nclass=cfg.get('nclass', 9),
        )
    raise ValueError(f"unknown dataset: {name}")


@torch.no_grad()
def predict_volume(model, sample, img_size, device, batch_size=16):
    """Run 2D model slice-by-slice over a 3D volume."""
    vol = sample['image']  # (D, H, W) numpy
    vol = vol.astype(np.float32)
    mean, std = vol.mean(), vol.std()
    vol = (vol - mean) / (std + 1e-8)

    D, H, W = vol.shape
    vol_t = torch.from_numpy(vol).float().unsqueeze(1).to(device)  # (D, 1, H, W)
    if (H, W) != tuple(img_size):
        vol_t = F.interpolate(vol_t, size=tuple(img_size), mode='bilinear', align_corners=False)

    preds = []
    for i in range(0, D, batch_size):
        out = model(vol_t[i:i + batch_size])
        preds.append(out.argmax(dim=1).cpu())
    pred = torch.cat(preds, dim=0)

    if (H, W) != tuple(img_size):
        pred = F.interpolate(pred.unsqueeze(1).float(), size=(H, W), mode='nearest')[:, 0].long()

    label = torch.from_numpy(sample['label'].astype(np.int64)).long()
    return pred, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_root = Path(cfg['log_dir']) / cfg['dataset']
    out_root.mkdir(parents=True, exist_ok=True)

    test_ds = build_test_dataset(cfg)
    img_size = tuple(cfg['img_size'])
    num_classes = cfg['num_classes']
    class_names = cfg['class_names']
    ignore_bg = cfg.get('ignore_background', True)

    model = EdgeGuidedSegModel(
        in_channels=cfg['in_channels'],
        num_classes=num_classes,
        backbone=cfg['backbone'],
        num_heads=cfg['num_heads'],
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    tracker = MetricTracker(num_classes, class_names, ignore_background=ignore_bg)

    pred_dir = out_root / 'predictions'
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(test_ds)), desc='test'):
        sample = test_ds[idx]
        pred, label = predict_volume(model, sample, img_size, device)
        tracker.update(pred, label)

        if args.save_predictions:
            case = sample['case_name']
            np.savez_compressed(
                pred_dir / f"{Path(case).stem}_pred.npz",
                pred=pred.numpy().astype(np.uint8),
            )

    title = f"{cfg['dataset'].upper()} Test Results"
    print(tracker.format_table(title))

    res = tracker.compute()
    csv_path = out_root / 'test_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'dice', 'iou'])
        for name, d, i in zip(res['class_names'], res['per_class_dice'], res['per_class_iou']):
            w.writerow([name, f"{d:.4f}", f"{i:.4f}"])
        w.writerow(['Mean', f"{res['mean_dice']:.4f}", f"{res['mean_iou']:.4f}"])
    print(f"saved: {csv_path}")


if __name__ == '__main__':
    main()
