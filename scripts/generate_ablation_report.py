"""Generate ablation study comparison tables for ACDC and Polyp benchmarks."""

import csv
import os
import sys

ABLATIONS = [
    ('E0_baseline', 'E0: Baseline (no edge/attn)'),
    ('E1_canny', 'E1: Canny edge'),
    ('E2_sobel', 'E2: Sobel edge'),
    ('E3_full', 'E3: Learned EdgeNet (Ours)'),
    ('A1_concat', 'A1: Concat fusion'),
    ('A2_swap_qkv', 'A2: Swap Q/KV'),
    ('D1_dino_only', 'D1: DINO only'),
    ('D2_full_dino', 'D2: Full + DINO'),
    ('S1_single_scale', 'S1: Single scale'),
]

ACDC_CLASSES = ['RV', 'Myo', 'LV']
POLYP_DATASETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def read_acdc_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    per_class = {}
    mean = {}
    for row in rows:
        name = row['class']
        if name == 'Mean':
            mean = {k: float(v) for k, v in row.items() if k != 'class'}
        else:
            per_class[name] = {k: float(v) for k, v in row.items() if k != 'class'}
    return {'per_class': per_class, 'mean': mean}


def read_polyp_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    per_ds = {}
    mean = {}
    for row in rows:
        name = row['dataset']
        if name == 'Mean':
            mean = {k: float(v) for k, v in row.items() if k not in ('dataset', 'n') and v}
        else:
            per_ds[name] = {k: float(v) for k, v in row.items() if k not in ('dataset', 'n')}
            per_ds[name]['n'] = int(row['n'])
    return {'per_dataset': per_ds, 'mean': mean}


def fmt(val, best_val, metric_name, higher_better):
    s = f'{val:.4f}'
    if best_val is not None:
        if higher_better and abs(val - best_val) < 1e-6:
            s = f'**{s}**'
        elif not higher_better and abs(val - best_val) < 1e-6:
            s = f'**{s}**'
    return s


def generate_report(runs_dir, output_path):
    lines = []
    lines.append('# Ablation Study Results\n')
    lines.append('> Bold values indicate the best result per metric.\n')

    # ── ACDC ──
    lines.append('## ACDC Cardiac Segmentation\n')

    acdc_data = {}
    for abl_key, abl_name in ABLATIONS:
        csv_path = os.path.join(runs_dir, f'{abl_key}_acdc', 'benchmark_acdc.csv')
        result = read_acdc_csv(csv_path)
        if result:
            acdc_data[abl_key] = result

    if acdc_data:
        # Mean comparison table
        lines.append('### Overall Comparison\n')
        lines.append('| Ablation | Dice | HD95 | Jaccard | ASD |')
        lines.append('|----------|------|------|---------|-----|')

        metrics = ['dice', 'hd95', 'jaccard', 'asd']
        higher_better = {'dice': True, 'hd95': False, 'jaccard': True, 'asd': False}

        best = {}
        for m in metrics:
            vals = [acdc_data[k]['mean'][m] for k in acdc_data]
            if higher_better[m]:
                best[m] = max(vals)
            else:
                best[m] = min(vals)

        for abl_key, abl_name in ABLATIONS:
            if abl_key not in acdc_data:
                lines.append(f'| {abl_name} | - | - | - | - |')
                continue
            d = acdc_data[abl_key]['mean']
            cols = [fmt(d[m], best[m], m, higher_better[m]) for m in metrics]
            lines.append(f'| {abl_name} | {" | ".join(cols)} |')

        # Per-class table
        lines.append('\n### Per-Class Dice Scores\n')
        header = '| Ablation | ' + ' | '.join(ACDC_CLASSES) + ' | Mean |'
        lines.append(header)
        lines.append('|' + '------|' * (len(ACDC_CLASSES) + 2))

        best_cls = {}
        for cls in ACDC_CLASSES + ['Mean']:
            vals = []
            for k in acdc_data:
                if cls == 'Mean':
                    vals.append(acdc_data[k]['mean']['dice'])
                else:
                    vals.append(acdc_data[k]['per_class'].get(cls, {}).get('dice', 0))
            best_cls[cls] = max(vals) if vals else 0

        for abl_key, abl_name in ABLATIONS:
            if abl_key not in acdc_data:
                lines.append(f'| {abl_name} | ' + ' | '.join(['-'] * (len(ACDC_CLASSES) + 1)) + ' |')
                continue
            parts = []
            for cls in ACDC_CLASSES:
                v = acdc_data[abl_key]['per_class'].get(cls, {}).get('dice', 0)
                parts.append(fmt(v, best_cls[cls], 'dice', True))
            mean_d = acdc_data[abl_key]['mean']['dice']
            parts.append(fmt(mean_d, best_cls['Mean'], 'dice', True))
            lines.append(f'| {abl_name} | {" | ".join(parts)} |')

    # ── Polyp ──
    lines.append('\n## Polyp Segmentation\n')

    polyp_data = {}
    for abl_key, abl_name in ABLATIONS:
        csv_path = os.path.join(runs_dir, f'{abl_key}_polyp', 'benchmark_polyp.csv')
        result = read_polyp_csv(csv_path)
        if result:
            polyp_data[abl_key] = result

    if polyp_data:
        lines.append('### Overall Comparison\n')
        lines.append('| Ablation | Dice | HD95 | Jaccard | ASD |')
        lines.append('|----------|------|------|---------|-----|')

        best = {}
        for m in metrics:
            vals = [polyp_data[k]['mean'][m] for k in polyp_data]
            if higher_better[m]:
                best[m] = max(vals)
            else:
                best[m] = min(vals)

        for abl_key, abl_name in ABLATIONS:
            if abl_key not in polyp_data:
                lines.append(f'| {abl_name} | - | - | - | - |')
                continue
            d = polyp_data[abl_key]['mean']
            cols = [fmt(d[m], best[m], m, higher_better[m]) for m in metrics]
            lines.append(f'| {abl_name} | {" | ".join(cols)} |')

        # Per-dataset Dice table
        lines.append('\n### Per-Dataset Dice Scores\n')
        short_names = ['Kvasir', 'ClinicDB', 'CVC-300', 'ColonDB', 'ETIS']
        header = '| Ablation | ' + ' | '.join(short_names) + ' | Mean |'
        lines.append(header)
        lines.append('|' + '------|' * (len(short_names) + 2))

        best_ds = {}
        for ds in POLYP_DATASETS + ['Mean']:
            vals = []
            for k in polyp_data:
                if ds == 'Mean':
                    vals.append(polyp_data[k]['mean']['dice'])
                else:
                    vals.append(polyp_data[k]['per_dataset'].get(ds, {}).get('dice', 0))
            best_ds[ds] = max(vals) if vals else 0

        for abl_key, abl_name in ABLATIONS:
            if abl_key not in polyp_data:
                lines.append(f'| {abl_name} | ' + ' | '.join(['-'] * (len(short_names) + 1)) + ' |')
                continue
            parts = []
            for ds in POLYP_DATASETS:
                v = polyp_data[abl_key]['per_dataset'].get(ds, {}).get('dice', 0)
                parts.append(fmt(v, best_ds[ds], 'dice', True))
            mean_d = polyp_data[abl_key]['mean']['dice']
            parts.append(fmt(mean_d, best_ds['Mean'], 'dice', True))
            lines.append(f'| {abl_name} | {" | ".join(parts)} |')

    content = '\n'.join(lines) + '\n'
    with open(output_path, 'w') as f:
        f.write(content)
    print(f'Report written to {output_path}')
    print(content)


if __name__ == '__main__':
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'ABLATION_RESULTS.md'
    generate_report(runs_dir, output_path)
