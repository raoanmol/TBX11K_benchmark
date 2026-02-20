# SLURM Cluster Deployment Guide

Quick guide for running the full TBX11K experiments on a SLURM cluster.

## Prerequisites

1. **Data prepared** on the cluster:
   ```bash
   python scripts/phase1_dataset_discovery.py
   python scripts/phase2_standardization.py
   python scripts/phase3_validation.py
   ```

2. **Configure scripts** - Edit these paths in the scripts:
   - `run_full_experiments.sh`: Update lines 37-40 (venv/conda path, CUDA module)
   - `submit_all_experiments.sh`: Update line 28 (venv path in analysis script)

## Quick Start

### Option 1: Submit Everything (Recommended)

```bash
# Submit all 20 runs + analysis in one command
./submit_all_experiments.sh
```

This submits:
- 10 scratch runs (Job array 0-9)
- 10 finetune runs (Job array 0-9)
- 1 analysis job (runs after all 20 complete)

### Option 2: Submit Individually

```bash
# Create log directory
mkdir -p slurm_logs

# Submit scratch experiments
sbatch run_full_experiments.sh scratch

# Submit finetune experiments
sbatch run_full_experiments.sh finetune

# After both complete, run analysis manually
python src/statistical_analysis.py \
    --scratch-dir ./experiments/maskrcnn_scratch \
    --finetune-dir ./experiments/maskrcnn_finetune \
    --n-runs 10 \
    --output-dir ./outputs
```

## Monitoring

### Check job status
```bash
# Quick check
squeue -u $USER

# Detailed monitoring
./monitor_jobs.sh

# Watch in real-time
watch -n 10 './monitor_jobs.sh'
```

### Check logs
```bash
# List recent logs
ls -lht slurm_logs/ | head -20

# View specific run
tail -f slurm_logs/job_tbx11k_maskrcnn_<JOB_ID>_task_0.out

# Check for errors
grep -r "Error\|Failed" slurm_logs/
```

### GPU usage
```bash
# SSH to compute node and check GPU
squeue -u $USER -o '%R' | grep -v NODELIST | head -1 | xargs -I {} ssh {} nvidia-smi
```

## Troubleshooting

### Job failed - check why
```bash
# Find failed tasks
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode | grep FAILED

# Check error log
cat experiments/maskrcnn_scratch/run_XX/error.log
cat slurm_logs/job_*_task_XX.err
```

### Resubmit specific failed runs
```bash
# If runs 3 and 7 failed for scratch
sbatch --array=3,7 run_full_experiments.sh scratch

# If run 5 failed for finetune
sbatch --array=5 run_full_experiments.sh finetune
```

### Out of memory (OOM)
```bash
# Edit run_full_experiments.sh and reduce batch size
# Change line: BATCH_SIZE=2
# To:          BATCH_SIZE=1

# Or request more memory
# Change line: #SBATCH --mem=32G
# To:          #SBATCH --mem=64G
```

### Cancel jobs
```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific array job
scancel <JOB_ID>_[0-9]
```

## Resource Requirements

Per run (with default settings):
- **GPU**: 1x GPU (8GB+ VRAM)
- **CPU**: 8 cores
- **Memory**: 32GB RAM
- **Time**: ~2-4 hours per run
- **Storage**: ~1GB per run

Total for 20 runs:
- **Time**: ~40-80 hours (runs in parallel)
- **Storage**: ~20GB

## Configuration

Edit `run_full_experiments.sh` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_SEED` | 42 | Starting seed (run i uses BASE_SEED + i) |
| `NUM_EPOCHS` | 20 | Training epochs |
| `BATCH_SIZE` | 2 | Batch size |
| `LEARNING_RATE` | 0.005 | Initial learning rate |

SLURM directives (lines 2-8):
| Directive | Default | Description |
|-----------|---------|-------------|
| `--gres=gpu:1` | 1 GPU | GPUs per task |
| `--cpus-per-task` | 8 | CPU cores |
| `--mem` | 32G | RAM |
| `--time` | 24:00:00 | Max runtime |

## Output Structure

After completion:
```
experiments/
├── maskrcnn_scratch/
│   ├── run_00/
│   │   ├── checkpoints/best_model.pth
│   │   ├── logs/training_log.json
│   │   └── eval/froc_interpolated.json
│   ├── run_01/ ... run_09/
│
├── maskrcnn_finetune/
│   └── run_00/ ... run_09/

outputs/
├── final_statistics.json
├── aggregated_results.json
├── froc_mean_plot.png
└── boxplot_comparison.png

slurm_logs/
├── job_tbx11k_maskrcnn_<JOB_ID>_task_0.out
└── ...
```

## Post-Processing

After all jobs complete:

```bash
# View results
cat outputs/final_statistics.json | jq .

# Download plots (from cluster to local)
scp cluster:/path/to/TBX11K_benchmark/outputs/*.png ./

# Verify all runs completed
for i in {0..9}; do
    ls experiments/maskrcnn_scratch/run_$(printf '%02d' $i)/eval/froc_interpolated.json
    ls experiments/maskrcnn_finetune/run_$(printf '%02d' $i)/eval/froc_interpolated.json
done
```

## Common Issues

### Issue: Module not found
**Solution:** Verify venv/conda path in script line 37

### Issue: CUDA module not found
**Solution:** Check available modules with `module avail cuda`, update line 40

### Issue: Data not found
**Solution:** Ensure you ran the 3 data preparation scripts first

### Issue: Disk quota exceeded
**Solution:** Clean up old checkpoints or request more storage
```bash
# Remove latest checkpoints (keep only best)
find experiments/ -name "latest_model.pth" -delete
```

---

**Need help?** Check the main [README.md](README.md) and [SETUP.md](SETUP.md) for more details.
