#!/usr/bin/env python3
"""Collect ablation benchmark results into RESULTS.md.

Scans runs/<exp_name>/ for benchmark CSVs and training logs,
then generates formatted Markdown tables with per-class metrics
and auto-computed deltas.

Usage:
    python scripts/collect_results.py [--results_dir runs] [--output RESULTS.md]
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime

ABLATION_ORDER = [
    'E0_baseline', 'E1_canny', 'E2_sobel', 'E3_full',
    'A1_concat', 'A2_swap_qkv',
    'D1_dino_only', 'D2_full_dino',
    'S1_single_scale',
]

DATASETS = ['synapse', 'acdc', 'polyp']

PHASE_MAP = {
    'E0_baseline': 1, 'E1_canny': 1, 'E2_sobel': 1,
    'E3_full': 1, 'A1_concat': 1, 'A2_swap_qkv': 1,
    'D1_dino_only': 2, 'D2_full_dino': 2,
    'S1_single_scale': 3,
}

DESCRIPTIONS = {
    'E0_baseline':    'Vanilla U-Net (no edge, no MHSA)',
    'E1_canny':       'Canny edges -> K/V',
    'E2_sobel':       'Sobel edges -> K/V',
    'E3_full':        'Learned EdgeNet -> K/V (**proposed**)',
    'A1_concat':      'Concat fusion (no attention)',
    'A2_swap_qkv':    'MHSA: Q=edge, K/V=encoder (reversed)',
    'D1_dino_only':   'No edge + DINOv3 distillation',
    'D2_full_dino':   'Full model + DINOv3 distillation',
    'S1_single_scale': 'MHSA at Enc4 only (single-scale)',
}


def find_benchmark_csv(run_dir, dataset):
    """Find the benchmark CSV file for a given dataset."""
    candidates = [
        f'benchmark_{dataset}.csv',
        'benchmark_polyp.csv',
        'test_results.csv',
    ]
    for c in candidates:
        path = os.path.join(run_dir, c)
        if os.path.exists(path):
            return path
    return None


def parse_csv(csv_path):
    """Parse a benchmark CSV and return mean dice + per-class data."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    result = {'per_class': [], 'mean_dice': None, 'mean_hd95': None}
    for row in rows:
        name = row.get('class', row.get('dataset', ''))
        dice = float(row.get('dice', 0))
        if name.lower() == 'mean':
            result['mean_dice'] = dice
            result['mean_hd95'] = float(row.get('hd95', 0)) if 'hd95' in row else None
        else:
            result['per_class'].append({
                'name': name,
                'dice': dice,
                'hd95': float(row.get('hd95', 0)) if 'hd95' in row else None,
            })

    if result['mean_dice'] is None and result['per_class']:
        result['mean_dice'] = sum(r['dice'] for r in result['per_class']) / len(result['per_class'])

    return result


def parse_best_dice_from_log(run_dir):
    """Try to extract best val dice from training log."""
    log_files = [f for f in os.listdir(run_dir) if f.endswith('.log')]
    best = 0.0
    for lf in log_files:
        with open(os.path.join(run_dir, lf)) as f:
            for line in f:
                m = re.search(r'new best.*?dice=([\d.]+)%', line)
                if m:
                    best = max(best, float(m.group(1)))
    return best


def collect_all(results_dir):
    """Scan results_dir and collect all available results."""
    data = {}
    for abl in ABLATION_ORDER:
        data[abl] = {}
        for ds in DATASETS:
            run_name = f'{abl}_{ds}'
            run_dir = os.path.join(results_dir, run_name)
            if not os.path.isdir(run_dir):
                data[abl][ds] = None
                continue

            csv_path = find_benchmark_csv(run_dir, ds)
            if csv_path:
                data[abl][ds] = parse_csv(csv_path)
            else:
                best = parse_best_dice_from_log(run_dir)
                if best > 0:
                    data[abl][ds] = {'mean_dice': best / 100.0, 'per_class': [], 'mean_hd95': None}
                else:
                    data[abl][ds] = None
    return data


def fmt_dice(val):
    """Format dice as percentage or dash."""
    if val is None:
        return '—'
    return f'{val * 100:.2f}'


def fmt_hd95(val):
    if val is None:
        return '—'
    return f'{val:.2f}'


def generate_markdown(data, output_path):
    """Generate the RESULTS.md file."""
    lines = []
    lines.append('# Ablation Study Results')
    lines.append('')
    lines.append(f'*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*')
    lines.append('')

    # ── 1. Main Comparison Table ──
    lines.append('## 1. Main Comparison — Mean Dice (%)')
    lines.append('')
    lines.append('| # | Phase | Config | Synapse | ACDC | Polyp |')
    lines.append('|---|-------|--------|---------|------|-------|')

    for i, abl in enumerate(ABLATION_ORDER, 1):
        phase = PHASE_MAP[abl]
        desc = DESCRIPTIONS[abl]
        syn = fmt_dice(data[abl]['synapse']['mean_dice'] if data[abl]['synapse'] else None)
        acdc = fmt_dice(data[abl]['acdc']['mean_dice'] if data[abl]['acdc'] else None)
        polyp = fmt_dice(data[abl]['polyp']['mean_dice'] if data[abl]['polyp'] else None)
        lines.append(f'| {i} | P{phase} | {desc} | {syn} | {acdc} | {polyp} |')

    lines.append('')

    # ── 2. Per-Class Dice on Synapse ──
    lines.append('## 2. Per-Class Dice on Synapse (%)')
    lines.append('')

    syn_classes = ['aorta', 'gallbladder', 'left_kidney', 'right_kidney',
                   'liver', 'pancreas', 'spleen', 'stomach']
    header = '| Config | ' + ' | '.join(syn_classes) + ' | Mean |'
    sep = '|--------|' + '|'.join(['------'] * len(syn_classes)) + '|------|'
    lines.append(header)
    lines.append(sep)

    for abl in ABLATION_ORDER:
        res = data[abl].get('synapse')
        if res and res['per_class']:
            per_class = {r['name']: r['dice'] for r in res['per_class']}
            vals = [fmt_dice(per_class.get(c)) for c in syn_classes]
            mean = fmt_dice(res['mean_dice'])
            lines.append(f'| {abl} | ' + ' | '.join(vals) + f' | {mean} |')
        else:
            dashes = ' | '.join(['—'] * len(syn_classes))
            lines.append(f'| {abl} | {dashes} | — |')

    lines.append('')

    # ── 3. HD95 Comparison (if available) ──
    has_hd95 = any(
        data[abl][ds] and data[abl][ds].get('mean_hd95') is not None
        for abl in ABLATION_ORDER for ds in DATASETS
    )
    if has_hd95:
        lines.append('## 3. HD95 Comparison')
        lines.append('')
        lines.append('| # | Config | Synapse | ACDC | Polyp |')
        lines.append('|---|--------|---------|------|-------|')
        for i, abl in enumerate(ABLATION_ORDER, 1):
            desc = DESCRIPTIONS[abl]
            vals = []
            for ds in DATASETS:
                res = data[abl][ds]
                vals.append(fmt_hd95(res['mean_hd95'] if res else None))
            lines.append(f'| {i} | {desc} | {vals[0]} | {vals[1]} | {vals[2]} |')
        lines.append('')

    # ── 4. Key Findings (auto-computed deltas) ──
    lines.append('## 4. Key Findings')
    lines.append('')

    def get_dice(abl, ds):
        res = data[abl].get(ds)
        if res and res['mean_dice'] is not None:
            return res['mean_dice'] * 100
        return None

    # Axis 1: Edge representation
    lines.append('### Axis 1 — Edge Representation')
    lines.append('')
    for ds in DATASETS:
        e0 = get_dice('E0_baseline', ds)
        e3 = get_dice('E3_full', ds)
        if e0 is not None and e3 is not None:
            delta = e3 - e0
            lines.append(f'- **{ds.capitalize()}**: Learned EdgeNet vs baseline: '
                         f'{e3:.2f}% vs {e0:.2f}% ({"+" if delta >= 0 else ""}{delta:.2f}%)')
    e1_syn = get_dice('E1_canny', 'synapse')
    e2_syn = get_dice('E2_sobel', 'synapse')
    e3_syn = get_dice('E3_full', 'synapse')
    if e1_syn and e2_syn and e3_syn:
        lines.append(f'- Canny: {e1_syn:.2f}%, Sobel: {e2_syn:.2f}%, Learned: {e3_syn:.2f}% (Synapse)')
    lines.append('')

    # Axis 2: Attention design
    lines.append('### Axis 2 — Attention Design')
    lines.append('')
    for ds in DATASETS:
        e3 = get_dice('E3_full', ds)
        a1 = get_dice('A1_concat', ds)
        a2 = get_dice('A2_swap_qkv', ds)
        parts = []
        if e3 is not None and a1 is not None:
            parts.append(f'Cross-attn vs concat: {e3:.2f}% vs {a1:.2f}% '
                         f'({"+" if e3 - a1 >= 0 else ""}{e3 - a1:.2f}%)')
        if e3 is not None and a2 is not None:
            parts.append(f'Q=enc vs Q=edge: {e3:.2f}% vs {a2:.2f}% '
                         f'({"+" if e3 - a2 >= 0 else ""}{e3 - a2:.2f}%)')
        if parts:
            lines.append(f'- **{ds.capitalize()}**: ' + '; '.join(parts))
    lines.append('')

    # Axis 3: Distillation
    lines.append('### Axis 3 — Distillation Interaction')
    lines.append('')
    for ds in DATASETS:
        e3 = get_dice('E3_full', ds)
        d1 = get_dice('D1_dino_only', ds)
        d2 = get_dice('D2_full_dino', ds)
        parts = []
        if e3 is not None:
            parts.append(f'Edge only (E3): {e3:.2f}%')
        if d1 is not None:
            parts.append(f'DINO only (D1): {d1:.2f}%')
        if d2 is not None:
            parts.append(f'Both (D2): {d2:.2f}%')
        if parts:
            lines.append(f'- **{ds.capitalize()}**: ' + ', '.join(parts))
    lines.append('')

    # Axis 4: Scale
    lines.append('### Axis 4 — Multi-Scale Attention')
    lines.append('')
    for ds in DATASETS:
        e3 = get_dice('E3_full', ds)
        s1 = get_dice('S1_single_scale', ds)
        if e3 is not None and s1 is not None:
            delta = e3 - s1
            lines.append(f'- **{ds.capitalize()}**: Dual-scale vs single-scale: '
                         f'{e3:.2f}% vs {s1:.2f}% ({"+" if delta >= 0 else ""}{delta:.2f}%)')
    lines.append('')

    # ── 5. Experiment Tracker ──
    lines.append('## 5. Experiment Tracker')
    lines.append('')
    lines.append('| Ablation | Synapse | ACDC | Polyp |')
    lines.append('|----------|---------|------|-------|')
    for abl in ABLATION_ORDER:
        statuses = []
        for ds in DATASETS:
            if data[abl][ds] is not None:
                statuses.append('done')
            else:
                statuses.append('pending')
        lines.append(f'| {abl} | {statuses[0]} | {statuses[1]} | {statuses[2]} |')
    lines.append('')

    done = sum(1 for abl in ABLATION_ORDER for ds in DATASETS if data[abl][ds] is not None)
    total = len(ABLATION_ORDER) * len(DATASETS)
    lines.append(f'**Progress: {done}/{total} experiments completed.**')
    lines.append('')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Results written to {output_path} ({done}/{total} experiments)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect ablation results')
    parser.add_argument('--results_dir', type=str, default='runs')
    parser.add_argument('--output', type=str, default='RESULTS.md')
    args = parser.parse_args()

    data = collect_all(args.results_dir)
    generate_markdown(data, args.output)
