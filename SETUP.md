# Setup Guide

Step-by-step instructions to get the TBX11K benchmark running.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 30GB+ free disk space

## Installation Steps

### 1. Clone Repository

```bash
cd /path/to/your/workspace
git clone <repository-url>
cd TBX11K_benchmark
```

### 2. Download TBX11K Dataset

Option A: Download from official source

```bash
# Download TBX11K dataset from:
# https://mmcheng.net/tb/
# Or from the paper: https://github.com/alexklwong/tbx11k

# Extract to ./data/TBX11K/
mkdir -p data
cd data
# ... extract dataset here ...
cd ..
```

The expected structure should be:
```
data/TBX11K/
├── imgs/
├── annotations/
└── lists/
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**Important:** Adjust the PyTorch installation command for your CUDA version.
See: https://pytorch.org/get-started/locally/

For CPU-only:
```bash
pip install torch torchvision torchaudio
```

### 5. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
```

### 6. Prepare Dataset

```bash
# Automatically discover and standardize dataset
python scripts/phase1_dataset_discovery.py
python scripts/phase2_standardization.py
python scripts/phase3_validation.py
```

Expected final output:
```
✅ ALL VALIDATIONS PASSED - Dataset ready for training
```

### 7. Quick Test (Optional)

Run a quick 1-epoch test to verify everything works:

```bash
./quick_test.sh
# OR
python run_experiments.py --n-runs 1 --num-epochs 1
```

This should complete in ~30-60 minutes and verify:
- ✅ Dataset loading works
- ✅ Model training runs
- ✅ FROC evaluation works
- ✅ Statistical analysis works

---

## Running Full Experiments

### Option 1: Automated Pipeline (Recommended)

```bash
# Run 10 independent experiments for both scratch and finetuned
python run_experiments.py \
    --n-runs 10 \
    --num-epochs 20 \
    --batch-size 2 \
    --learning-rate 0.005
```

**Time estimate:** 40-80 hours on a single GPU

**To run in background:**
```bash
nohup python run_experiments.py --n-runs 10 --num-epochs 20 > experiment.log 2>&1 &
```

### Option 2: Manual Step-by-Step

**Step 1: Train scratch model (Run 0)**
```bash
python src/train.py \
    --train-json ./data/processed/train.json \
    --val-json ./data/processed/val.json \
    --image-root ./data/TBX11K/imgs \
    --experiment-name maskrcnn_scratch \
    --run-id 0 \
    --output-dir ./experiments/maskrcnn_scratch/run_00 \
    --seed 42 \
    --num-epochs 20 \
    --batch-size 2
```

**Step 2: Evaluate scratch model (Run 0)**
```bash
python src/evaluate_froc.py \
    --eval-json ./data/processed/val.json \
    --image-root ./data/TBX11K/imgs \
    --checkpoint ./experiments/maskrcnn_scratch/run_00/checkpoints/best_model.pth \
    --output-dir ./experiments/maskrcnn_scratch/run_00/eval
```

**Step 3: Repeat for runs 1-9**
- Change `--run-id` and `--seed` accordingly
- Seed for run i: `42 + i`

**Step 4: Train finetuned models**
- Same as scratch, but add `--pretrained-backbone` flag
- Use different experiment name: `maskrcnn_finetune`

**Step 5: Statistical analysis**
```bash
python src/statistical_analysis.py \
    --scratch-dir ./experiments/maskrcnn_scratch \
    --finetune-dir ./experiments/maskrcnn_finetune \
    --n-runs 10 \
    --output-dir ./outputs
```

---

## Monitoring Progress

### Check Training Progress

```bash
# View training log for specific run
cat experiments/maskrcnn_scratch/run_00/logs/training_log.json

# Monitor live (if running)
tail -f experiments/maskrcnn_scratch/run_00/training_output.log
```

### Visualize with TensorBoard (Optional)

If you want to add TensorBoard logging, modify `src/train.py` to include:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=logs_dir / 'tensorboard')
```

Then run:
```bash
tensorboard --logdir experiments/
```

---

## Disk Space Management

Full experiments will generate ~20GB of data:
- Checkpoints: ~15GB (20 models × 2 conditions)
- Logs and evaluations: ~5GB

To save space:
```bash
# Remove non-best checkpoints
find experiments/ -name "latest_model.pth" -delete

# Keep only FROC results
find experiments/ -name "*.png" -delete
```

---

## GPU Optimization

### Multi-GPU Training

The current implementation uses single GPU. To use multiple GPUs, modify `src/train.py`:

```python
model = nn.DataParallel(model)
```

### Batch Size Tuning

| GPU Memory | Max Batch Size |
|------------|----------------|
| 8GB        | 2              |
| 11GB       | 4              |
| 16GB       | 6              |
| 24GB+      | 8              |

Adjust `--batch-size` accordingly.

---

## Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 1`
2. Clear GPU cache before running:
   ```bash
   nvidia-smi --gpu-reset
   ```
3. Use CPU-only mode (slower):
   ```python
   device = torch.device('cpu')
   ```

### Issue 2: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue 3: Dataset Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/TBX11K/...'
```

**Solution:**
1. Verify dataset location:
   ```bash
   ls data/TBX11K/
   ```
2. Re-run data preparation:
   ```bash
   python scripts/phase1_dataset_discovery.py
   python scripts/phase2_standardization.py
   ```

### Issue 4: Slow Training

**Possible causes:**
- Running on CPU instead of GPU
- Small batch size
- Disk I/O bottleneck

**Solutions:**
1. Verify GPU usage:
   ```bash
   nvidia-smi
   ```
2. Increase batch size if memory allows
3. Move dataset to SSD if on HDD

### Issue 5: Validation Loss Not Decreasing

**Normal behavior:**
- TBX11K has very sparse annotations (~900 boxes for 6600 images)
- Most images have no lesions
- Validation loss may appear high

**What to check:**
- FROC sensitivity should be increasing
- Check `experiments/*/run_*/eval/froc_curve.png`

---

## Expected Timeline

For 10 runs × 2 conditions × 20 epochs on a single RTX 3090:

| Phase | Time Estimate |
|-------|---------------|
| Data preparation | 5 minutes |
| Scratch training (10 runs) | 20-30 hours |
| Scratch evaluation (10 runs) | 2-3 hours |
| Finetuned training (10 runs) | 20-30 hours |
| Finetuned evaluation (10 runs) | 2-3 hours |
| Statistical analysis | 1 minute |
| **Total** | **44-66 hours** |

**Speedup strategies:**
- Use multiple GPUs in parallel (run different seeds on different GPUs)
- Reduce epochs to 10 (may affect results)
- Use smaller validation set

---

## Next Steps

After successful setup:

1. ✅ Review validation visualizations in `data/processed/debug_samples/`
2. ✅ Run quick test to verify pipeline
3. ✅ Start full experiments
4. ✅ Monitor progress periodically
5. ✅ Analyze results in `outputs/`

---

## Support

If you encounter issues not covered here:

1. Check `experiments/*/run_*/error.log` for detailed stack traces
2. Verify all file paths are correct
3. Ensure sufficient disk space and GPU memory
4. Review README.md for additional documentation
5. Open an issue on GitHub with error logs

---

**Document version:** 1.0
**Last updated:** 2026-02-20
