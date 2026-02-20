#!/usr/bin/env python3
"""
Master experiment runner for TBX11K × Mask R-CNN experiments.

Executes:
  1. 10 independent scratch training runs
  2. 10 independent finetuned training runs
  3. FROC evaluation for each run
  4. Statistical analysis comparing scratch vs. finetuned

Usage:
    python run_experiments.py --config config/experiment_config.yaml
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json
import time
import traceback


def run_command(cmd, log_file=None):
    """
    Run a shell command and stream output.

    Args:
        cmd: Command to run (list or string)
        log_file: Optional file to save output

    Returns:
        success: Boolean indicating success
    """
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )
        else:
            result = subprocess.run(
                cmd,
                check=True,
                text=True
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}")
        if log_file:
            print(f"   See log file: {log_file}")
        return False


def run_single_experiment(
    experiment_name,
    run_id,
    pretrained_backbone,
    train_json,
    val_json,
    image_root,
    output_base_dir,
    base_seed,
    num_epochs,
    batch_size,
    learning_rate
):
    """
    Run a single training + evaluation experiment.

    Returns:
        success: Boolean indicating success
    """
    # Compute seed
    seed = base_seed + run_id

    # Output directory
    run_dir = Path(output_base_dir) / f'run_{run_id:02d}'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT: {experiment_name} - Run {run_id:02d}")
    print(f"Seed: {seed}")
    print(f"Output: {run_dir}")
    print(f"{'=' * 80}")

    # Training
    print(f"\n[1/2] Training...")

    train_cmd = [
        'python', 'src/train.py',
        '--train-json', train_json,
        '--val-json', val_json,
        '--image-root', image_root,
        '--experiment-name', experiment_name,
        '--run-id', str(run_id),
        '--output-dir', str(run_dir),
        '--seed', str(seed),
        '--num-epochs', str(num_epochs),
        '--batch-size', str(batch_size),
        '--learning-rate', str(learning_rate)
    ]

    if pretrained_backbone:
        train_cmd.append('--pretrained-backbone')

    train_log = run_dir / 'training_output.log'
    train_success = run_command(train_cmd, log_file=train_log)

    if not train_success:
        print(f"❌ Training failed for {experiment_name} run {run_id}")
        return False

    print(f"✓ Training completed")

    # Evaluation
    print(f"\n[2/2] FROC Evaluation...")

    checkpoint_path = run_dir / 'checkpoints' / 'best_model.pth'

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    eval_dir = run_dir / 'eval'
    eval_dir.mkdir(exist_ok=True)

    eval_cmd = [
        'python', 'src/evaluate_froc.py',
        '--eval-json', val_json,
        '--image-root', image_root,
        '--checkpoint', str(checkpoint_path),
        '--output-dir', str(eval_dir)
    ]

    eval_log = eval_dir / 'evaluation_output.log'
    eval_success = run_command(eval_cmd, log_file=eval_log)

    if not eval_success:
        print(f"❌ Evaluation failed for {experiment_name} run {run_id}")
        return False

    print(f"✓ Evaluation completed")

    return True


def run_experiment_suite(
    experiment_name,
    pretrained_backbone,
    n_runs,
    train_json,
    val_json,
    image_root,
    experiments_dir,
    base_seed,
    num_epochs,
    batch_size,
    learning_rate
):
    """
    Run a full suite of experiments (10 runs).

    Returns:
        successes: Number of successful runs
    """
    print(f"\n{'#' * 80}")
    print(f"EXPERIMENT SUITE: {experiment_name}")
    print(f"Pretrained backbone: {pretrained_backbone}")
    print(f"Number of runs: {n_runs}")
    print(f"{'#' * 80}")

    output_base_dir = Path(experiments_dir) / experiment_name

    successes = 0

    for run_id in range(n_runs):
        try:
            success = run_single_experiment(
                experiment_name=experiment_name,
                run_id=run_id,
                pretrained_backbone=pretrained_backbone,
                train_json=train_json,
                val_json=val_json,
                image_root=image_root,
                output_base_dir=output_base_dir,
                base_seed=base_seed,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

            if success:
                successes += 1
                print(f"\n✅ Run {run_id:02d} completed successfully")
            else:
                print(f"\n❌ Run {run_id:02d} failed")

        except Exception as e:
            print(f"\n❌ Run {run_id:02d} crashed with exception:")
            print(traceback.format_exc())

            # Save error
            error_path = output_base_dir / f'run_{run_id:02d}' / 'error.log'
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, 'w') as f:
                f.write(traceback.format_exc())

    print(f"\n{'#' * 80}")
    print(f"EXPERIMENT SUITE COMPLETE: {experiment_name}")
    print(f"Successful runs: {successes} / {n_runs}")
    print(f"{'#' * 80}")

    return successes


def run_statistical_analysis(
    scratch_dir,
    finetune_dir,
    n_runs,
    output_dir
):
    """Run statistical analysis comparing scratch vs. finetuned."""

    print(f"\n{'#' * 80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'#' * 80}")

    cmd = [
        'python', 'src/statistical_analysis.py',
        '--scratch-dir', scratch_dir,
        '--finetune-dir', finetune_dir,
        '--n-runs', str(n_runs),
        '--output-dir', output_dir
    ]

    log_file = Path(output_dir) / 'analysis_output.log'
    success = run_command(cmd, log_file=log_file)

    if success:
        print(f"\n✅ Statistical analysis completed")
    else:
        print(f"\n❌ Statistical analysis failed")

    return success


def main(args):
    """Main experiment orchestration."""

    start_time = time.time()

    print("\n" + "=" * 80)
    print("TBX11K × MASK R-CNN EXPERIMENT PIPELINE")
    print("=" * 80)

    # Paths
    train_json = args.train_json or './data/processed/train.json'
    val_json = args.val_json or './data/processed/val.json'
    image_root = args.image_root or './data/TBX11K/imgs'
    experiments_dir = args.experiments_dir or './experiments'
    outputs_dir = args.outputs_dir or './outputs'

    # Parameters
    n_runs = args.n_runs
    base_seed = args.base_seed
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # Create directories
    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Train JSON: {train_json}")
    print(f"  Val JSON: {val_json}")
    print(f"  Image root: {image_root}")
    print(f"  Experiments dir: {experiments_dir}")
    print(f"  Outputs dir: {outputs_dir}")
    print(f"  Number of runs: {n_runs}")
    print(f"  Base seed: {base_seed}")
    print(f"  Epochs per run: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    # Phase 1: Scratch experiments
    if not args.skip_scratch:
        scratch_successes = run_experiment_suite(
            experiment_name='maskrcnn_scratch',
            pretrained_backbone=False,
            n_runs=n_runs,
            train_json=train_json,
            val_json=val_json,
            image_root=image_root,
            experiments_dir=experiments_dir,
            base_seed=base_seed,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    else:
        print("\n⚠ Skipping scratch experiments (--skip-scratch)")
        scratch_successes = n_runs

    # Phase 2: Finetuned experiments
    if not args.skip_finetune:
        finetune_successes = run_experiment_suite(
            experiment_name='maskrcnn_finetune',
            pretrained_backbone=True,
            n_runs=n_runs,
            train_json=train_json,
            val_json=val_json,
            image_root=image_root,
            experiments_dir=experiments_dir,
            base_seed=base_seed,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    else:
        print("\n⚠ Skipping finetuned experiments (--skip-finetune)")
        finetune_successes = n_runs

    # Phase 3: Statistical analysis
    if not args.skip_analysis:
        if scratch_successes >= 3 and finetune_successes >= 3:  # Need at least 3 runs for stats
            analysis_success = run_statistical_analysis(
                scratch_dir=f'{experiments_dir}/maskrcnn_scratch',
                finetune_dir=f'{experiments_dir}/maskrcnn_finetune',
                n_runs=n_runs,
                output_dir=outputs_dir
            )
        else:
            print("\n⚠ Skipping statistical analysis (insufficient successful runs)")
            print(f"  Scratch: {scratch_successes}/{n_runs}")
            print(f"  Finetune: {finetune_successes}/{n_runs}")
            analysis_success = False
    else:
        print("\n⚠ Skipping statistical analysis (--skip-analysis)")
        analysis_success = True

    # Final summary
    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time / 3600

    print("\n" + "=" * 80)
    print("EXPERIMENT PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed_hours:.2f} hours")
    print(f"\nResults:")
    print(f"  Scratch experiments: {scratch_successes}/{n_runs} successful")
    print(f"  Finetuned experiments: {finetune_successes}/{n_runs} successful")
    print(f"  Statistical analysis: {'✓' if analysis_success else '✗'}")
    print(f"\nOutputs saved to:")
    print(f"  Experiments: {experiments_dir}")
    print(f"  Analysis: {outputs_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run full TBX11K × Mask R-CNN experiment pipeline'
    )

    # Data paths
    parser.add_argument('--train-json', type=str, default='./data/processed/train.json')
    parser.add_argument('--val-json', type=str, default='./data/processed/val.json')
    parser.add_argument('--image-root', type=str, default='./data/TBX11K/imgs')

    # Output paths
    parser.add_argument('--experiments-dir', type=str, default='./experiments')
    parser.add_argument('--outputs-dir', type=str, default='./outputs')

    # Experiment parameters
    parser.add_argument('--n-runs', type=int, default=10,
                        help='Number of runs per experiment')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='Base random seed (run i uses base_seed + i)')

    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='Initial learning rate')

    # Control flags
    parser.add_argument('--skip-scratch', action='store_true',
                        help='Skip scratch experiments')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Skip finetuned experiments')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip statistical analysis')

    args = parser.parse_args()

    main(args)
