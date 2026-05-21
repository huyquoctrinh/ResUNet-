#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="/home/trnhquchuy/miniconda3/envs/huy/bin/python"
GPU="${1:-0}"
DATASET="${2:-all}"
export CUDA_VISIBLE_DEVICES="$GPU"

ABLATIONS=(E0_baseline E1_canny E2_sobel E3_full A1_concat A2_swap_qkv D1_dino_only D2_full_dino S1_single_scale)
if [[ "$DATASET" == "all" ]]; then
    DS_LIST=(synapse acdc polyp)
else
    DS_LIST=("$DATASET")
fi

for ABL in "${ABLATIONS[@]}"; do
    for DS in "${DS_LIST[@]}"; do
        RUN_NAME="${ABL}_${DS}"
        BEST_CKPT="runs/${RUN_NAME}/checkpoints/best_model.pth"
        RESULTS_CSV="runs/${RUN_NAME}/benchmark_results.csv"

        if [[ ! -f "$BEST_CKPT" ]]; then
            echo "[SKIP] No checkpoint: $RUN_NAME"
            continue
        fi
        if [[ -f "$RESULTS_CSV" ]]; then
            echo "[SKIP] Already done: $RUN_NAME"
            continue
        fi

        echo "[BENCH] $RUN_NAME"
        case $DS in
            synapse) $PYTHON benchmark_synapse.py --checkpoint "$BEST_CKPT" --output_dir "runs/${RUN_NAME}" ;;
            acdc)    $PYTHON benchmark_acdc.py --checkpoint "$BEST_CKPT" --output_dir "runs/${RUN_NAME}" ;;
            polyp)   $PYTHON benchmark_polyp.py --checkpoint "$BEST_CKPT" --output_dir "runs/${RUN_NAME}" ;;
        esac
        echo "[DONE] $RUN_NAME"
    done
done

echo "Collecting results..."
$PYTHON scripts/collect_results.py --results_dir runs --output RESULTS.md
echo "Done."
