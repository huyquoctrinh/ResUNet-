#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# run_ablations.sh — Train + benchmark all ablation experiments
# ──────────────────────────────────────────────────────────────────────
#
# Usage:
#   ./run_ablations.sh                          # all 27 experiments
#   ./run_ablations.sh --phase 1                # Phase 1 only (18 runs)
#   ./run_ablations.sh --phase 1 --dataset synapse  # Phase 1, Synapse (6 runs)
#   ./run_ablations.sh --dry-run                # preview commands
#   ./run_ablations.sh --no-skip                # re-run completed experiments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/home/trnhquchuy/miniconda3/envs/huy/bin/python"

PHASE="all"
DATASET="all"
GPU=0
SEED=42
DRY_RUN=0
NO_SKIP=0
PRETRAINED=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)     PHASE="$2"; shift 2;;
        --dataset)   DATASET="$2"; shift 2;;
        --gpu)       GPU="$2"; shift 2;;
        --seed)      SEED="$2"; shift 2;;
        --pretrained) PRETRAINED="$2"; shift 2;;
        --dry-run)   DRY_RUN=1; shift;;
        --no-skip)   NO_SKIP=1; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

# Phase → ablation ID mapping
PHASE1=(E0_baseline E1_canny E2_sobel E3_full A1_concat A2_swap_qkv)
PHASE2=(D1_dino_only D2_full_dino)
PHASE3=(S1_single_scale)

DATASETS_ALL=(synapse acdc polyp)

# Resolve which ablations to run
ABLATIONS=()
case $PHASE in
    1)   ABLATIONS=("${PHASE1[@]}");;
    2)   ABLATIONS=("${PHASE2[@]}");;
    3)   ABLATIONS=("${PHASE3[@]}");;
    all) ABLATIONS=("${PHASE1[@]}" "${PHASE2[@]}" "${PHASE3[@]}");;
    *)   echo "Invalid phase: $PHASE (use 1, 2, 3, or all)"; exit 1;;
esac

# Resolve which datasets to run
DS_LIST=()
case $DATASET in
    all)     DS_LIST=("${DATASETS_ALL[@]}");;
    synapse) DS_LIST=(synapse);;
    acdc)    DS_LIST=(acdc);;
    polyp)   DS_LIST=(polyp);;
    *)       echo "Invalid dataset: $DATASET"; exit 1;;
esac

TOTAL=0
SKIPPED=0
FAILED=0
SUCCEEDED=0

echo "============================================================"
echo " Ablation Study Runner"
echo " Phase: $PHASE | Dataset: $DATASET | GPU: $GPU | Seed: $SEED"
echo "============================================================"
echo ""

for ABL in "${ABLATIONS[@]}"; do
    for DS in "${DS_LIST[@]}"; do
        CONFIG="configs/ablations/${ABL}_${DS}.yaml"
        RUN_NAME="${ABL}_${DS}"
        CKPT_DIR="runs/${RUN_NAME}/checkpoints"
        BEST_CKPT="${CKPT_DIR}/best_model.pth"
        RESULTS_CSV="runs/${RUN_NAME}/benchmark_results.csv"
        TOTAL=$((TOTAL + 1))

        if [[ ! -f "$CONFIG" ]]; then
            echo "[SKIP] Config not found: $CONFIG"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Skip if results already exist (unless --no-skip)
        if [[ $NO_SKIP -eq 0 && -f "$RESULTS_CSV" ]]; then
            echo "[SKIP] Results exist: $RUN_NAME"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "──────────────────────────────────────────────────────"
        echo "[RUN] $RUN_NAME"
        echo "  Config: $CONFIG"
        echo "  Checkpoint: $BEST_CKPT"
        echo "──────────────────────────────────────────────────────"

        # ── Training ──
        TRAIN_CMD="$PYTHON train.py --config $CONFIG"
        if [[ -n "$PRETRAINED" ]]; then
            TRAIN_CMD="$TRAIN_CMD --pretrained $PRETRAINED"
        fi

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY-RUN] $TRAIN_CMD"
        else
            echo "  Training..."
            if ! $TRAIN_CMD; then
                echo "  [FAIL] Training failed for $RUN_NAME"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi

        # ── Benchmarking ──
        if [[ $DRY_RUN -eq 0 && ! -f "$BEST_CKPT" ]]; then
            echo "  [FAIL] No best checkpoint found at $BEST_CKPT"
            FAILED=$((FAILED + 1))
            continue
        fi

        case $DS in
            synapse)
                BENCH_CMD="$PYTHON benchmark_synapse.py --checkpoint $BEST_CKPT --output_dir runs/${RUN_NAME}"
                ;;
            acdc)
                BENCH_CMD="$PYTHON benchmark_acdc.py --checkpoint $BEST_CKPT --output_dir runs/${RUN_NAME}"
                ;;
            polyp)
                BENCH_CMD="$PYTHON benchmark_polyp.py --checkpoint $BEST_CKPT --output_dir runs/${RUN_NAME}"
                ;;
        esac

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [DRY-RUN] $BENCH_CMD"
        else
            echo "  Benchmarking..."
            if ! $BENCH_CMD; then
                echo "  [FAIL] Benchmark failed for $RUN_NAME"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi

        SUCCEEDED=$((SUCCEEDED + 1))
        echo "  [DONE] $RUN_NAME"
        echo ""
    done
done

echo ""
echo "============================================================"
echo " Summary"
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "============================================================"

if [[ $DRY_RUN -eq 0 && $SUCCEEDED -gt 0 ]]; then
    echo ""
    echo "Collecting results..."
    $PYTHON scripts/collect_results.py --results_dir runs --output RESULTS.md
    echo "Results written to RESULTS.md"
fi
