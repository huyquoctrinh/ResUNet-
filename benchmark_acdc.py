"""Benchmark script for ACDC cardiac segmentation.

Evaluates per-class Dice, HD95, Jaccard, ASD on ACDC test volumes.
Follows the EMCAD / PraNet-V2 evaluation protocol.

Usage:
    python benchmark_acdc.py \
        --checkpoint runs/acdc/checkpoints/best_model.pth \
        [--volume_path dataset/ACDC] \
        [--list_dir dataset/ACDC/lists_ACDC] \
        [--num_classes 4] \
        [--img_size 224] \
        [--save_predictions]
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
import yaml
from medpy import metric
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.data_acdc import ACDCdataset
from network import EdgeGuidedSegModel
from utils import load_checkpoint

ACDC_CLASSES = ['RV', 'Myo', 'LV']


# ---------------------------------------------------------------------------
# Metric helpers (following EMCAD utils.py)
# ---------------------------------------------------------------------------

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size, args):
    """Run slice-by-slice inference on a 3D volume and compute per-class metrics."""
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)
    if image.ndim == 3:
        for ind in range(image.shape[0]):
            slc = image[ind, :, :]
            x, y = slc.shape
            if x != patch_size[0] or y != patch_size[1]:
                slc = zoom(slc, (patch_size[0] / x, patch_size[1] / y), order=3)
            inp = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(inp)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(inp)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def inference(args, model):
    db_test = ACDCdataset(
        base_dir=args.volume_path,
        list_dir=args.list_dir,
        split='test',
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test volumes")

    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['case_name'][0]

        metric_i = test_single_volume(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            args=args,
        )
        metric_list += np.array(metric_i)
        logging.info(
            'idx %d case %s  mean_dice %.4f  mean_hd95 %.4f  mean_jaccard %.4f  mean_asd %.4f'
            % (i_batch, case_name,
               np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
               np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3])
        )

    metric_list = metric_list / len(db_test)

    # Per-class results
    print('\n' + '=' * 80)
    print(f'{"ACDC Benchmark Results":^80}')
    print('=' * 80)
    print(f'{"Class":<15} {"Dice":<12} {"HD95":<12} {"Jaccard":<12} {"ASD":<12}')
    print('-' * 80)

    for i in range(args.num_classes - 1):
        cls_name = ACDC_CLASSES[i] if i < len(ACDC_CLASSES) else f'class_{i + 1}'
        dice_val = metric_list[i][0]
        hd95_val = metric_list[i][1]
        jacc_val = metric_list[i][2]
        asd_val = metric_list[i][3]
        print(f'{cls_name:<15} {dice_val:<12.4f} {hd95_val:<12.4f} {jacc_val:<12.4f} {asd_val:<12.4f}')
        logging.info(
            'Mean class (%d) %s  dice %.4f  hd95 %.4f  jaccard %.4f  asd %.4f'
            % (i + 1, cls_name, dice_val, hd95_val, jacc_val, asd_val)
        )

    print('-' * 80)
    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jaccard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    print(f'{"Mean":<15} {mean_dice:<12.4f} {mean_hd95:<12.4f} {mean_jaccard:<12.4f} {mean_asd:<12.4f}')
    print('=' * 80)

    logging.info(
        'Overall: mean_dice %.4f  mean_hd95 %.4f  mean_jaccard %.4f  mean_asd %.4f'
        % (mean_dice, mean_hd95, mean_jaccard, mean_asd)
    )

    # Save to CSV
    csv_path = os.path.join(args.output_dir, 'benchmark_acdc.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'dice', 'hd95', 'jaccard', 'asd'])
        for i in range(args.num_classes - 1):
            cls_name = ACDC_CLASSES[i] if i < len(ACDC_CLASSES) else f'class_{i + 1}'
            w.writerow([cls_name, f'{metric_list[i][0]:.4f}', f'{metric_list[i][1]:.4f}',
                        f'{metric_list[i][2]:.4f}', f'{metric_list[i][3]:.4f}'])
        w.writerow(['Mean', f'{mean_dice:.4f}', f'{mean_hd95:.4f}',
                     f'{mean_jaccard:.4f}', f'{mean_asd:.4f}'])
    print(f'Results saved to {csv_path}')

    return mean_dice


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACDC Benchmark')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--volume_path', type=str, default='dataset/ACDC',
                        help='Root dir for ACDC data')
    parser.add_argument('--list_dir', type=str, default='dataset/ACDC/lists_ACDC',
                        help='List dir with train/valid/test .txt files')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='runs/acdc')
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--deterministic', type=int, default=1)
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

    log_path = os.path.join(args.output_dir, 'benchmark_acdc.log')
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

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
    ).cuda()

    load_checkpoint(args.checkpoint, model, map_location='cuda')
    logging.info(f'Loaded checkpoint: {args.checkpoint}')

    model.eval()
    inference(args, model)
