#!/bin/bash

# ============================================================
# TBX11K × Mask R-CNN Full Experiment Submission
# Submits 20 training runs + statistical analysis
# ============================================================

echo "======================================================"
echo "TBX11K Benchmark - Full Experiment Submission"
echo "======================================================"
echo "Submission time: $(date)"
echo ""
echo "Configuration:"
echo "  Scratch runs:    10 (array 0-9)"
echo "  Finetune runs:   10 (array 0-9)"
echo "  Total runs:      20"
echo "  Epochs/run:      20"
echo "  Expected time:   40-80 hours (parallel)"
echo ""

# Create logs directory
mkdir -p slurm_logs
echo "✓ Created slurm_logs/ directory"

# ============================================================
# Submit Scratch Experiments
# ============================================================
echo ""
echo "=========================================="
echo "Submitting SCRATCH experiments"
echo "=========================================="
echo ">>> Submitting job array (10 runs)..."

SCRATCH_JOB=$(sbatch --parsable run_full_experiments.sh scratch)

if [ $? -eq 0 ]; then
    echo "✓ Submitted successfully"
    echo "  Job ID: $SCRATCH_JOB"
    echo "  Tasks:  0-9 (10 runs)"
    echo "  Type:   Random initialization (no pretrained weights)"
else
    echo "✗ Submission failed"
    exit 1
fi

# ============================================================
# Submit Finetune Experiments
# ============================================================
echo ""
echo "=========================================="
echo "Submitting FINETUNE experiments"
echo "=========================================="
echo ">>> Submitting job array (10 runs)..."

FINETUNE_JOB=$(sbatch --parsable run_full_experiments.sh finetune)

if [ $? -eq 0 ]; then
    echo "✓ Submitted successfully"
    echo "  Job ID: $FINETUNE_JOB"
    echo "  Tasks:  0-9 (10 runs)"
    echo "  Type:   ImageNet-pretrained backbone"
else
    echo "✗ Submission failed"
    exit 1
fi

# ============================================================
# Create Statistical Analysis Job
# ============================================================
echo ""
echo "=========================================="
echo "Creating STATISTICAL ANALYSIS job"
echo "=========================================="

cat > slurm_analysis.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=tbx11k_analysis
#SBATCH --output=slurm_logs/analysis_%j.out
#SBATCH --error=slurm_logs/analysis_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "======================================================"
echo "TBX11K Statistical Analysis"
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load environment
source /path/to/venv/bin/activate  # CHANGE THIS TO YOUR VENV PATH

echo "Running paired statistical tests..."
echo "  Comparing: scratch vs. finetune"
echo "  Runs: 10 per condition"
echo "  Primary metric: Sensitivity at FP/image = 1.0"
echo ""

python src/statistical_analysis.py \
    --scratch-dir ./experiments/maskrcnn_scratch \
    --finetune-dir ./experiments/maskrcnn_finetune \
    --n-runs 10 \
    --output-dir ./outputs

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Analysis complete!"
    echo ""
    echo "Results saved to ./outputs/"
    ls -lh ./outputs/
    echo ""
    echo "Key files:"
    echo "  - final_statistics.json      (t-test, Wilcoxon, Cohen's d)"
    echo "  - aggregated_results.json    (all FROC data)"
    echo "  - froc_mean_plot.png         (mean curves)"
    echo "  - boxplot_comparison.png     (distribution at FP=1)"
else
    echo "✗ Analysis failed"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "======================================================"
EOF

echo "✓ Created analysis script: slurm_analysis.sh"

# Submit with dependency
echo ""
echo ">>> Submitting analysis job (depends on training completion)..."

ANALYSIS_JOB=$(sbatch --parsable --dependency=afterok:${SCRATCH_JOB}:${FINETUNE_JOB} slurm_analysis.sh)

if [ $? -eq 0 ]; then
    echo "✓ Submitted successfully"
    echo "  Job ID: $ANALYSIS_JOB"
    echo "  Depends on: $SCRATCH_JOB and $FINETUNE_JOB"
    echo "  Note: Will run automatically after all 20 runs complete"
else
    echo "✗ Submission failed"
fi

# ============================================================
# Submission Summary
# ============================================================
echo ""
echo "======================================================"
echo "Submission Complete!"
echo "======================================================"
echo ""
echo "Submitted jobs:"
echo "  1. Scratch:   $SCRATCH_JOB (array: 0-9)"
echo "  2. Finetune:  $FINETUNE_JOB (array: 0-9)"
echo "  3. Analysis:  $ANALYSIS_JOB (depends on 1 & 2)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  ./monitor_jobs.sh"
echo "  watch -n 10 './monitor_jobs.sh'"
echo ""
echo "Check logs:"
echo "  ls -lht slurm_logs/ | head"
echo "  tail -f slurm_logs/job_tbx11k_maskrcnn_${SCRATCH_JOB}_task_0.out"
echo ""
echo "Expected completion:"
echo "  Training: 40-80 hours (20 runs in parallel)"
echo "  Analysis: +1 hour (after training)"
echo ""
echo "Final results will be in:"
echo "  ./outputs/"
echo ""
echo "======================================================"
