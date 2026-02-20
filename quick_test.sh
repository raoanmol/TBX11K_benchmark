#!/bin/bash

# Quick test script to verify the full pipeline works
# Runs 1 epoch each for scratch and finetuned models

set -e  # Exit on error

echo "=============================================="
echo "TBX11K Benchmark - Quick Test"
echo "=============================================="
echo ""

# Step 1: Data preparation
echo "[1/5] Dataset discovery..."
python scripts/phase1_dataset_discovery.py

echo ""
echo "[2/5] Dataset standardization..."
python scripts/phase2_standardization.py

echo ""
echo "[3/5] Dataset validation..."
python scripts/phase3_validation.py

# Step 2: Quick training test (1 epoch)
echo ""
echo "[4/5] Running quick training test..."
python run_experiments.py \
    --n-runs 1 \
    --num-epochs 1 \
    --batch-size 2

# Step 3: Statistical analysis
echo ""
echo "[5/5] Statistical analysis..."
python src/statistical_analysis.py \
    --scratch-dir ./experiments/maskrcnn_scratch \
    --finetune-dir ./experiments/maskrcnn_finetune \
    --n-runs 1 \
    --output-dir ./outputs

echo ""
echo "=============================================="
echo "✅ Quick test completed successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  - Review outputs in ./outputs/"
echo "  - Check sample visualizations in ./data/processed/debug_samples/"
echo "  - Run full experiments with: python run_experiments.py --n-runs 10 --num-epochs 20"
echo ""
