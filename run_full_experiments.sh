#!/bin/bash
#SBATCH --job-name=tbx11k_maskrcnn
#SBATCH --output=slurm_logs/job_%x_%A_task_%a.out
#SBATCH --error=slurm_logs/job_%x_%A_task_%a.err
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general
#SBATCH --time=0-24:00:00

# ============================================================
# TBX11K Mask R-CNN Training & Evaluation
# Usage: sbatch run_full_experiments.sh [scratch|finetune]
# ============================================================

# Validate argument
if [ $# -eq 0 ]; then
    echo "Error: Specify experiment type (scratch or finetune)"
    echo "Usage: sbatch run_full_experiments.sh [scratch|finetune]"
    exit 1
fi

EXPERIMENT_TYPE=$1

# ============================================================
# Configuration
# ============================================================
BASE_SEED=42
NUM_EPOCHS=20
BATCH_SIZE=2
LEARNING_RATE=0.005

# Set experiment-specific parameters
if [ "$EXPERIMENT_TYPE" == "scratch" ]; then
    EXPERIMENT_NAME="maskrcnn_scratch"
    BASE_OUTPUT_DIR="./experiments/maskrcnn_scratch"
    PRETRAINED_FLAG=""
elif [ "$EXPERIMENT_TYPE" == "finetune" ]; then
    EXPERIMENT_NAME="maskrcnn_finetune"
    BASE_OUTPUT_DIR="./experiments/maskrcnn_finetune"
    PRETRAINED_FLAG="--pretrained-backbone"
else
    echo "Error: Invalid experiment type. Use 'scratch' or 'finetune'"
    exit 1
fi

# ============================================================
# Environment Setup
# ============================================================
echo "======================================================"
echo "TBX11K Mask R-CNN Experiment"
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: 1x GPU"
echo "CPUs: 8"
echo "Memory: 32G"
echo "Time Limit: 24h"
echo "Start time: $(date)"
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "======================================================"
echo ""

# Load environment
module purge
module load cuda/11.8  # CHANGE THIS to match your cluster

# Activate virtual environment
source /path/to/venv/bin/activate  # CHANGE THIS to your venv path
# OR if using conda:
# module load mamba/latest
# source activate tbx11k_venv

# Verify GPU
echo "Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ============================================================
# Run Configuration
# ============================================================
RUN_INDEX=${SLURM_ARRAY_TASK_ID}
SEED=$((BASE_SEED + RUN_INDEX))
RUN_DIR="${BASE_OUTPUT_DIR}/run_$(printf '%02d' ${RUN_INDEX})"

echo "Run configuration:"
echo "  Run ID:       $RUN_INDEX"
echo "  Seed:         $SEED"
echo "  Epochs:       $NUM_EPOCHS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output:       $RUN_DIR"
echo ""

# Create output directory
mkdir -p ${RUN_DIR}

# ============================================================
# STEP 1/2: Training
# ============================================================
echo "=========================================="
echo "STEP 1/2: Training"
echo "=========================================="
echo ">>> Starting training..."
echo "Time: $(date)"
echo ""

python src/train.py \
    --train-json ./data/processed/train.json \
    --val-json ./data/processed/val.json \
    --image-root ./data/TBX11K/imgs \
    ${PRETRAINED_FLAG} \
    --experiment-name ${EXPERIMENT_NAME} \
    --run-id ${RUN_INDEX} \
    --output-dir ${RUN_DIR} \
    --seed ${SEED} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --num-workers 8

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully"
    echo "  Best model: ${RUN_DIR}/checkpoints/best_model.pth"
else
    echo ""
    echo "✗ Training failed for run ${RUN_INDEX}"
    echo "Training failed at $(date)" >> ${RUN_DIR}/error.log
    exit 1
fi

# ============================================================
# STEP 2/2: FROC Evaluation
# ============================================================
echo ""
echo "=========================================="
echo "STEP 2/2: FROC Evaluation"
echo "=========================================="
echo ">>> Starting evaluation..."
echo "Time: $(date)"
echo ""

CHECKPOINT_PATH="${RUN_DIR}/checkpoints/best_model.pth"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "✗ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Checkpoint missing at $(date)" >> ${RUN_DIR}/error.log
    exit 1
fi

python src/evaluate_froc.py \
    --eval-json ./data/processed/val.json \
    --image-root ./data/TBX11K/imgs \
    --checkpoint ${CHECKPOINT_PATH} \
    --output-dir ${RUN_DIR}/eval \
    --batch-size ${BATCH_SIZE} \
    --num-workers 8

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully"
    echo "  FROC data: ${RUN_DIR}/eval/froc_interpolated.json"
else
    echo ""
    echo "✗ Evaluation failed for run ${RUN_INDEX}"
    echo "Evaluation failed at $(date)" >> ${RUN_DIR}/error.log
    exit 1
fi

# ============================================================
# Completion Summary
# ============================================================
echo ""
echo "======================================================"
echo "✅ Run ${RUN_INDEX} Completed Successfully"
echo "======================================================"
echo "End time: $(date)"
echo ""
echo "Outputs saved to: ${RUN_DIR}"
echo ""
echo "Files created:"
echo "  - config.json                    (run configuration)"
echo "  - checkpoints/best_model.pth     (trained model)"
echo "  - logs/training_log.json         (training history)"
echo "  - eval/froc.csv                  (full FROC curve)"
echo "  - eval/froc_interpolated.json    (sensitivity at key FP rates)"
echo "  - eval/froc_curve.png            (visualization)"
echo ""
echo "======================================================"

exit 0
