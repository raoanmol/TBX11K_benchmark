"""
Training script for Mask R-CNN on TBX11K dataset.
Supports deterministic training with full reproducibility.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import TBX11KDataset, collate_fn, get_transform
from model import get_maskrcnn_model, get_optimizer, get_lr_scheduler


def set_deterministic(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_environment_info(output_dir):
    """Save environment and dependency information."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Save to file
    with open(output_dir / 'environment_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    return info


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch."""
    model.train()

    epoch_losses = []
    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Track
        epoch_losses.append(losses.item())

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch}] Iter [{i+1}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} | "
                  f"Avg Loss: {np.mean(epoch_losses):.4f} | "
                  f"Time: {elapsed:.1f}s")

    avg_loss = np.mean(epoch_losses)
    return avg_loss, epoch_losses


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Simple evaluation (compute average loss)."""
    model.train()  # Keep in train mode to get losses

    all_losses = []

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        all_losses.append(losses.item())

    avg_loss = np.mean(all_losses) if all_losses else float('inf')
    return avg_loss


def main(args):
    """Main training function."""

    print("=" * 80)
    print(f"TRAINING: {args.experiment_name} - Run {args.run_id}")
    print("=" * 80)

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Set deterministic
    set_deterministic(args.seed)
    print(f"\n✓ Set random seed: {args.seed}")

    # Save environment info
    env_info = save_environment_info(output_dir)
    print(f"✓ PyTorch: {env_info['pytorch_version']}")
    print(f"✓ CUDA: {env_info['cuda_version']}")
    if env_info['cuda_available']:
        print(f"✓ GPU: {env_info['gpu_name']}")

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Config saved: {config_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = TBX11KDataset(
        json_path=args.train_json,
        image_root=args.image_root,
        transforms=get_transform(train=True)
    )

    val_dataset = TBX11KDataset(
        json_path=args.val_json,
        image_root=args.image_root,
        transforms=get_transform(train=False)
    )

    # Limit dataset size for smoke testing
    if args.max_samples is not None:
        train_dataset.image_ids = train_dataset.image_ids[:args.max_samples]
        val_dataset.image_ids = val_dataset.image_ids[:min(args.max_samples // 4, len(val_dataset.image_ids))]
        print(f"  ⚠ Smoke test mode: limited to {args.max_samples} train samples")

    print(f"  ✓ Train: {len(train_dataset)} images")
    print(f"  ✓ Val: {len(val_dataset)} images")
    print(f"  ✓ Num classes: {train_dataset.num_classes}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed)  # Deterministic shuffling
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Model
    print(f"\nCreating model...")
    model = get_maskrcnn_model(
        num_classes=train_dataset.num_classes,
        pretrained_backbone=args.pretrained_backbone
    )
    model.to(device)

    # Optimizer
    optimizer = get_optimizer(
        model,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # LR Scheduler
    lr_scheduler = get_lr_scheduler(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 80)

    training_log = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'best_epoch': -1,
        'best_val_loss': float('inf')
    }

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_loss_history = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )

        # Validate
        val_loss = evaluate(model, val_loader, device)

        # LR step
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        # Log
        training_log['epochs'].append(epoch)
        training_log['train_losses'].append(train_loss)
        training_log['val_losses'].append(val_loss)
        training_log['learning_rates'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch [{epoch}/{args.num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save checkpoint
        if val_loss < training_log['best_val_loss']:
            training_log['best_val_loss'] = val_loss
            training_log['best_epoch'] = epoch

            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

        # Save latest
        latest_path = checkpoint_dir / 'latest_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss
        }, latest_path)

        print("-" * 80)

    # Save training log
    log_path = logs_dir / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch: {training_log['best_epoch']}")
    print(f"Best val loss: {training_log['best_val_loss']:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on TBX11K')

    # Data
    parser.add_argument('--train-json', type=str, required=True)
    parser.add_argument('--val-json', type=str, required=True)
    parser.add_argument('--image-root', type=str, required=True)

    # Model
    parser.add_argument('--pretrained-backbone', action='store_true', default=False,
                        help='Use ImageNet-pretrained backbone')

    # Training
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--lr-step-size', type=int, default=5)
    parser.add_argument('--lr-gamma', type=float, default=0.1)

    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit dataset to N samples for smoke testing (default: use full dataset)')

    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        # Log errors
        error_log = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        error_path = Path(args.output_dir) / 'error.log'
        error_path.parent.mkdir(parents=True, exist_ok=True)

        with open(error_path, 'w') as f:
            json.dump(error_log, f, indent=2)

        print(f"\n❌ ERROR: {e}")
        print(f"Full traceback saved to: {error_path}")
        raise
