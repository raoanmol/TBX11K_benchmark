# TBX11K × Mask R-CNN Benchmark

Reproducible experiments comparing **Mask R-CNN** trained from scratch vs. fine-tuned from ImageNet on the TBX11K tuberculosis detection dataset.

## 📋 Overview

This repository implements a rigorous experimental pipeline to answer:

> **Does ImageNet pre-training improve tuberculosis lesion detection compared to training from scratch?**

The pipeline includes:
- ✅ Automatic dataset discovery and standardization
- ✅ 10 independent runs per condition (scratch vs. fine-tuned)
- ✅ Deterministic training with full reproducibility
- ✅ FROC (Free-Response ROC) evaluation
- ✅ Statistical significance testing (paired t-test, Wilcoxon)
- ✅ Complete logging and experiment tracking

---

## 🏗️ Project Structure

```
TBX11K_benchmark/
├── data/
│   ├── TBX11K/                     # Original dataset (auto-discovered)
│   │   ├── imgs/                   # Images organized by category
│   │   ├── annotations/            # JSON and XML annotations
│   │   └── lists/                  # Train/val/test splits
│   └── processed/                  # Standardized COCO format
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       └── debug_samples/          # Validation visualizations
│
├── scripts/
│   ├── phase1_dataset_discovery.py    # Auto-discover dataset structure
│   ├── phase2_standardization.py      # Convert to COCO format
│   └── phase3_validation.py           # Validate with visualizations
│
├── src/
│   ├── dataset.py                  # PyTorch dataset loader
│   ├── model.py                    # Mask R-CNN model setup
│   ├── train.py                    # Training script
│   ├── evaluate_froc.py            # FROC computation
│   └── statistical_analysis.py     # Statistical testing
│
├── experiments/                    # Training outputs (created during runs)
│   ├── maskrcnn_scratch/
│   │   ├── run_00/ ... run_09/
│   └── maskrcnn_finetune/
│       └── run_00/ ... run_09/
│
├── outputs/                        # Final analysis results
│   ├── aggregated_results.json
│   ├── final_statistics.json
│   ├── froc_mean_plot.png
│   └── boxplot_comparison.png
│
├── run_experiments.py              # Master experiment runner
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

The dataset should already be in `./data/TBX11K/`. If not, download it first.

```bash
# Step 1: Discover dataset structure
python scripts/phase1_dataset_discovery.py

# Step 2: Standardize to COCO format
python scripts/phase2_standardization.py

# Step 3: Validate with visualizations
python scripts/phase3_validation.py
```

**Expected output:**
```
✅ ALL VALIDATIONS PASSED - Dataset ready for training
```

You can view validation samples in `data/processed/debug_samples/`.

### 3. Run Full Experiment Pipeline

**Option A: Full pipeline (20 runs total, ~40-80 hours on GPU)**

```bash
python run_experiments.py \
    --n-runs 10 \
    --num-epochs 20 \
    --batch-size 2 \
    --learning-rate 0.005
```

**Option B: Quick test (1 run each, ~2-4 hours)**

```bash
python run_experiments.py \
    --n-runs 1 \
    --num-epochs 5 \
    --batch-size 2
```

**Option C: Run individual experiments**

```bash
# Scratch training (single run)
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

# Fine-tuned training (single run)
python src/train.py \
    --train-json ./data/processed/train.json \
    --val-json ./data/processed/val.json \
    --image-root ./data/TBX11K/imgs \
    --pretrained-backbone \
    --experiment-name maskrcnn_finetune \
    --run-id 0 \
    --output-dir ./experiments/maskrcnn_finetune/run_00 \
    --seed 42 \
    --num-epochs 20 \
    --batch-size 2
```

### 4. Evaluate FROC

```bash
python src/evaluate_froc.py \
    --eval-json ./data/processed/val.json \
    --image-root ./data/TBX11K/imgs \
    --checkpoint ./experiments/maskrcnn_scratch/run_00/checkpoints/best_model.pth \
    --output-dir ./experiments/maskrcnn_scratch/run_00/eval
```

### 5. Statistical Analysis

```bash
python src/statistical_analysis.py \
    --scratch-dir ./experiments/maskrcnn_scratch \
    --finetune-dir ./experiments/maskrcnn_finetune \
    --n-runs 10 \
    --output-dir ./outputs
```

---

## 📊 Expected Results

After running the full pipeline, you'll find:

### Training Logs
- `experiments/maskrcnn_scratch/run_XX/logs/training_log.json` - Loss curves, best epoch
- `experiments/maskrcnn_scratch/run_XX/checkpoints/best_model.pth` - Best model checkpoint

### FROC Evaluation
- `experiments/maskrcnn_scratch/run_XX/eval/froc.csv` - Full FROC curve data
- `experiments/maskrcnn_scratch/run_XX/eval/froc_interpolated.json` - Sensitivity at key FP rates
- `experiments/maskrcnn_scratch/run_XX/eval/froc_curve.png` - Visualization

### Statistical Analysis
- `outputs/final_statistics.json` - Paired t-test, Wilcoxon, Cohen's d
- `outputs/aggregated_results.json` - All run results
- `outputs/froc_mean_plot.png` - Mean FROC curves with std dev
- `outputs/boxplot_comparison.png` - Distribution comparison at FP=1

---

## 🔬 Methodology

### Dataset
- **TBX11K**: 12,279 chest X-ray images
- **Classes**: 3 tuberculosis types (ActiveTB, ObsoletePulmonaryTB, PulmonaryTB)
- **Annotations**: 1,211 bounding boxes (902 train, 309 val)
- **Split**: 6,600 train / 1,800 val / 3,302 test (unlabeled)

### Model
- **Architecture**: Mask R-CNN with ResNet-50 FPN backbone
- **Conditions**:
  1. **Scratch**: Random initialization
  2. **Fine-tuned**: ImageNet-pretrained backbone

### Training
- **Epochs**: 20 (configurable)
- **Batch size**: 2
- **Optimizer**: SGD (lr=0.005, momentum=0.9, weight_decay=0.0005)
- **LR Schedule**: StepLR (step=5, gamma=0.1)
- **Determinism**: Fixed seeds, cudnn.deterministic=True

### Evaluation
- **Metric**: FROC (Free-Response ROC)
- **IoU threshold**: 0.5
- **Key FP rates**: [0.125, 0.25, 0.5, 1, 2, 4] FP per image
- **Primary comparison**: Sensitivity at FP/image = 1.0

### Statistical Testing
- **Paired t-test** (parametric)
- **Wilcoxon signed-rank test** (non-parametric)
- **Effect size**: Cohen's d
- **Significance level**: α = 0.05

---

## 🔧 Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-runs` | 10 | Number of runs per experiment |
| `--base-seed` | 42 | Base random seed (run i = base_seed + i) |
| `--num-epochs` | 20 | Training epochs |
| `--batch-size` | 2 | Batch size |
| `--learning-rate` | 0.005 | Initial learning rate |

### Computational Requirements

**Minimum (1 run, 5 epochs):**
- GPU: 8GB VRAM (e.g., GTX 1080)
- Time: ~2 hours per run

**Recommended (10 runs, 20 epochs):**
- GPU: 16GB+ VRAM (e.g., RTX 3090, V100)
- Time: ~40-80 hours total
- Storage: ~20GB for checkpoints

**CPU-only mode:**
- Possible but ~10-20x slower
- Not recommended for full experiments

---

## 📝 Reproducibility Checklist

This implementation ensures full reproducibility:

- ✅ Fixed random seeds (PyTorch, NumPy, Python)
- ✅ Deterministic CUDA operations
- ✅ Recorded environment info (PyTorch/CUDA versions, GPU model)
- ✅ Saved exact configurations per run
- ✅ Dataset hash verification
- ✅ Deterministic data loader shuffling
- ✅ Complete logging of all hyperparameters
- ✅ Error handling with full stack traces

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory

**Solution:**
```bash
# Reduce batch size
python run_experiments.py --batch-size 1
```

### Issue: Missing dataset files

**Solution:**
```bash
# Re-run discovery and standardization
python scripts/phase1_dataset_discovery.py
python scripts/phase2_standardization.py
python scripts/phase3_validation.py
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: Training crashes mid-run

**Solution:**
- Check `experiments/*/run_XX/error.log` for stack trace
- Training automatically saves checkpoints, can resume if needed
- Each run is independent, failed runs don't block others

---

## 📚 References

- **TBX11K Dataset**: Liu et al., "Rethinking Computer-Aided Tuberculosis Diagnosis" (CVPR 2020)
- **Mask R-CNN**: He et al., "Mask R-CNN" (ICCV 2017)
- **FROC Analysis**: Chakraborty & Berbaum, "Observer studies involving detection and localization" (2004)

---

## 📄 License

This code is provided for research purposes. The TBX11K dataset has its own license terms.

---

## 🙏 Acknowledgments

- TBX11K dataset authors
- PyTorch and torchvision teams
- Medical imaging community

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

**Last updated:** 2026-02-20
