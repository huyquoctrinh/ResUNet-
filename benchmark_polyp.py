"""Benchmark script for polyp segmentation.

Evaluates per-dataset Dice, IoU, HD95, Jaccard, ASD on 5 standard polyp test sets:
  Kvasir, CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS-LaribPolypDB

Usage:
    python benchmark_polyp.py \
        --checkpoint runs/polyp/checkpoints/best_model.pth \
        [--test_root dataset/TestDataset] \
        [--img_size 352]
"""

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from medpy import metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.data_polyp import PolypTestDataset
from network import EdgeGuidedSegModel
from utils import load_checkpoint

TEST_DATASETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def calculate_metrics(pred, gt):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    if pred_bin.sum() > 0 and gt_bin.sum() > 0:
        dice = metric.binary.dc(pred_bin, gt_bin)
        hd95 = metric.binary.hd95(pred_bin, gt_bin)
        jc = metric.binary.jc(pred_bin, gt_bin)
        asd = metric.binary.assd(pred_bin, gt_bin)
        return dice, hd95, jc, asd
    elif pred_bin.sum() > 0 and gt_bin.sum() == 0:
        return 1.0, 0.0, 1.0, 0.0
    else:
        return 0.0, 0.0, 0.0, 0.0


@torch.no_grad()
def evaluate_dataset(model, dataset_name, test_root, img_size, device):
    ds = PolypTestDataset(
        root=os.path.join(test_root, dataset_name),
        img_size=img_size,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    dice_list, hd95_list, jc_list, asd_list = [], [], [], []

    for batch in tqdm(loader, desc=dataset_name, leave=False):
        img = batch['image'].to(device)
        label = batch['label'].numpy()[0]
        orig_h, orig_w = batch['orig_size'][0].tolist()

        logits = model(img)
        pred = F.interpolate(
            logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False,
        )
        pred = pred.argmax(dim=1).cpu().numpy()[0]

        d, h, j, a = calculate_metrics(pred, label)
        dice_list.append(d)
        hd95_list.append(h)
        jc_list.append(j)
        asd_list.append(a)

    return {
        'dice': np.mean(dice_list),
        'hd95': np.mean(hd95_list),
        'jaccard': np.mean(jc_list),
        'asd': np.mean(asd_list),
        'n': len(ds),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Polyp Benchmark')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_root', type=str, default='dataset/TestDataset')
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='runs/polyp')
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--datasets', type=str, nargs='+', default=TEST_DATASETS)
    args = parser.parse_args()

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, 'benchmark_polyp.log')
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg_path = os.path.join(args.output_dir, 'config.yaml')
    ablation = {}
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
            ablation = cfg.get('ablation', {})

    backbone = cfg.get('backbone', args.backbone)
    num_heads = cfg.get('num_heads', args.num_heads)

    model = EdgeGuidedSegModel(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        backbone=backbone,
        num_heads=num_heads,
        edge_type=ablation.get('edge_type', 'learned'),
        attention_type=ablation.get('attention_type', 'cross_attention'),
        q_source=ablation.get('q_source', 'encoder'),
        attention_scales=ablation.get('attention_scales', ['enc3', 'enc4']),
        deep_supervision=ablation.get('deep_supervision', False),
    ).to(device)

    load_checkpoint(args.checkpoint, model, map_location=device)
    logging.info(f'Loaded checkpoint: {args.checkpoint}')
    model.eval()

    all_results = {}
    for ds_name in args.datasets:
        res = evaluate_dataset(model, ds_name, args.test_root, args.img_size, device)
        all_results[ds_name] = res
        logging.info(
            f'{ds_name} (n={res["n"]}): dice={res["dice"]:.4f}  '
            f'hd95={res["hd95"]:.4f}  jaccard={res["jaccard"]:.4f}  asd={res["asd"]:.4f}'
        )

    print('\n' + '=' * 90)
    print(f'{"Polyp Benchmark Results":^90}')
    print('=' * 90)
    print(f'{"Dataset":<25} {"N":<6} {"Dice":<12} {"HD95":<12} {"Jaccard":<12} {"ASD":<12}')
    print('-' * 90)

    all_dice, all_hd95, all_jc, all_asd = [], [], [], []
    for ds_name in args.datasets:
        r = all_results[ds_name]
        print(f'{ds_name:<25} {r["n"]:<6} {r["dice"]:<12.4f} {r["hd95"]:<12.4f} '
              f'{r["jaccard"]:<12.4f} {r["asd"]:<12.4f}')
        all_dice.append(r['dice'])
        all_hd95.append(r['hd95'])
        all_jc.append(r['jaccard'])
        all_asd.append(r['asd'])

    print('-' * 90)
    mean_dice = np.mean(all_dice)
    mean_hd95 = np.mean(all_hd95)
    mean_jc = np.mean(all_jc)
    mean_asd = np.mean(all_asd)
    print(f'{"Mean":<25} {"":6} {mean_dice:<12.4f} {mean_hd95:<12.4f} '
          f'{mean_jc:<12.4f} {mean_asd:<12.4f}')
    print('=' * 90)

    logging.info(
        f'Overall: mean_dice={mean_dice:.4f}  mean_hd95={mean_hd95:.4f}  '
        f'mean_jaccard={mean_jc:.4f}  mean_asd={mean_asd:.4f}'
    )

    csv_path = os.path.join(args.output_dir, 'benchmark_polyp.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'n', 'dice', 'hd95', 'jaccard', 'asd'])
        for ds_name in args.datasets:
            r = all_results[ds_name]
            w.writerow([ds_name, r['n'], f'{r["dice"]:.4f}', f'{r["hd95"]:.4f}',
                        f'{r["jaccard"]:.4f}', f'{r["asd"]:.4f}'])
        w.writerow(['Mean', '', f'{mean_dice:.4f}', f'{mean_hd95:.4f}',
                     f'{mean_jc:.4f}', f'{mean_asd:.4f}'])
    print(f'Results saved to {csv_path}')
