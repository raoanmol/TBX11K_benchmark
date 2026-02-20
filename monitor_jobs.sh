#!/bin/bash

# Quick monitoring helper for SLURM jobs

echo "=================================================="
echo "TBX11K Experiment Monitor"
echo "=================================================="

echo ""
echo "Active jobs:"
squeue -u $USER -o "%.18i %.12j %.8T %.10M %.6D %R"

echo ""
echo "Job summary:"
echo "  Running:    $(squeue -u $USER -t RUNNING | tail -n +2 | wc -l)"
echo "  Pending:    $(squeue -u $USER -t PENDING | tail -n +2 | wc -l)"
echo "  Completed:  Check with: sacct -u $USER --starttime=today"

echo ""
echo "Experiment progress:"
echo ""
echo "Scratch experiments:"
for i in {0..9}; do
    run_dir="./experiments/maskrcnn_scratch/run_$(printf '%02d' $i)"
    if [ -f "$run_dir/checkpoints/best_model.pth" ]; then
        echo "  ✓ Run $i: Training complete"
        if [ -f "$run_dir/eval/froc_interpolated.json" ]; then
            echo "    ✓ Evaluation complete"
        else
            echo "    ⏳ Evaluation pending/running"
        fi
    elif [ -d "$run_dir" ]; then
        echo "  ⏳ Run $i: Training in progress..."
    else
        echo "  ⏸  Run $i: Not started"
    fi
done

echo ""
echo "Finetune experiments:"
for i in {0..9}; do
    run_dir="./experiments/maskrcnn_finetune/run_$(printf '%02d' $i)"
    if [ -f "$run_dir/checkpoints/best_model.pth" ]; then
        echo "  ✓ Run $i: Training complete"
        if [ -f "$run_dir/eval/froc_interpolated.json" ]; then
            echo "    ✓ Evaluation complete"
        else
            echo "    ⏳ Evaluation pending/running"
        fi
    elif [ -d "$run_dir" ]; then
        echo "  ⏳ Run $i: Training in progress..."
    else
        echo "  ⏸  Run $i: Not started"
    fi
done

echo ""
echo "Analysis:"
if [ -f "./outputs/final_statistics.json" ]; then
    echo "  ✓ Statistical analysis complete"
    echo "    View results: cat ./outputs/final_statistics.json"
else
    echo "  ⏸  Not run yet"
fi

echo ""
echo "=================================================="
echo "Commands:"
echo "  Watch jobs:     watch -n 10 './monitor_jobs.sh'"
echo "  Check logs:     ls -lht slurm_logs/ | head -20"
echo "  GPU usage:      squeue -u \$USER -o '%R' | grep -v NODELIST | xargs -I {} ssh {} nvidia-smi"
echo "  Cancel all:     scancel -u \$USER"
echo "=================================================="
