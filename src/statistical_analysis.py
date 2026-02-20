"""
PHASE 7: Statistical Analysis
Compare scratch vs. finetuned models using paired statistical tests.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_froc_results(experiment_dir, n_runs):
    """
    Load FROC interpolated results from all runs.

    Args:
        experiment_dir: Path to experiment directory
        n_runs: Number of runs

    Returns:
        results: Dict mapping FP rate to list of sensitivities
    """
    experiment_dir = Path(experiment_dir)

    results = {}
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4]

    for fp_rate in fp_rates:
        results[fp_rate] = []

    for run_id in range(n_runs):
        run_dir = experiment_dir / f'run_{run_id:02d}' / 'eval'
        json_path = run_dir / 'froc_interpolated.json'

        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping run {run_id}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for fp_rate in fp_rates:
            sensitivity = data.get(str(fp_rate), 0.0)  # JSON keys are strings
            results[fp_rate].append(sensitivity)

    return results


def compute_statistics(scratch_values, finetune_values):
    """
    Compute paired statistical tests.

    Args:
        scratch_values: List of values from scratch training
        finetune_values: List of values from finetuned training

    Returns:
        stats_dict: Dictionary with statistical test results
    """
    scratch_values = np.array(scratch_values)
    finetune_values = np.array(finetune_values)

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(finetune_values, scratch_values)

    # Wilcoxon signed-rank test
    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(finetune_values, scratch_values)

    # Cohen's d (effect size)
    differences = finetune_values - scratch_values
    cohen_d = np.mean(differences) / np.std(differences, ddof=1) if len(differences) > 1 else 0

    # Summary statistics
    results = {
        'scratch': {
            'mean': float(np.mean(scratch_values)),
            'std': float(np.std(scratch_values, ddof=1)),
            'median': float(np.median(scratch_values)),
            'min': float(np.min(scratch_values)),
            'max': float(np.max(scratch_values)),
            'values': scratch_values.tolist()
        },
        'finetune': {
            'mean': float(np.mean(finetune_values)),
            'std': float(np.std(finetune_values, ddof=1)),
            'median': float(np.median(finetune_values)),
            'min': float(np.min(finetune_values)),
            'max': float(np.max(finetune_values)),
            'values': finetune_values.tolist()
        },
        'paired_ttest': {
            't_statistic': float(t_stat),
            'p_value': float(t_pvalue)
        },
        'wilcoxon_test': {
            'statistic': float(wilcoxon_stat),
            'p_value': float(wilcoxon_pvalue)
        },
        'effect_size': {
            'cohen_d': float(cohen_d)
        },
        'difference': {
            'mean': float(np.mean(differences)),
            'std': float(np.std(differences, ddof=1))
        }
    }

    return results


def plot_comparison(scratch_results, finetune_results, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp_rates = sorted(scratch_results.keys())

    # 1. FROC mean curves
    plt.figure(figsize=(10, 6))

    scratch_means = [np.mean(scratch_results[fp]) for fp in fp_rates]
    scratch_stds = [np.std(scratch_results[fp], ddof=1) for fp in fp_rates]

    finetune_means = [np.mean(finetune_results[fp]) for fp in fp_rates]
    finetune_stds = [np.std(finetune_results[fp], ddof=1) for fp in fp_rates]

    plt.plot(fp_rates, scratch_means, 'o-', label='Scratch', linewidth=2)
    plt.fill_between(fp_rates,
                     np.array(scratch_means) - np.array(scratch_stds),
                     np.array(scratch_means) + np.array(scratch_stds),
                     alpha=0.2)

    plt.plot(fp_rates, finetune_means, 's-', label='Finetuned', linewidth=2)
    plt.fill_between(fp_rates,
                     np.array(finetune_means) - np.array(finetune_stds),
                     np.array(finetune_means) + np.array(finetune_stds),
                     alpha=0.2)

    plt.xlabel('False Positives per Image', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('FROC Comparison: Scratch vs. Finetuned', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / 'froc_mean_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ FROC mean plot saved to: {plot_path}")

    # 2. Boxplot comparison at FP=1
    fp_rate_compare = 1.0
    scratch_vals = scratch_results[fp_rate_compare]
    finetune_vals = finetune_results[fp_rate_compare]

    plt.figure(figsize=(8, 6))

    data_to_plot = [scratch_vals, finetune_vals]
    labels = ['Scratch', 'Finetuned']

    box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)

    for patch, color in zip(box['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    plt.ylabel('Sensitivity', fontsize=12)
    plt.title(f'Sensitivity at FP/Image = {fp_rate_compare}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    boxplot_path = output_dir / 'boxplot_comparison.png'
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Boxplot saved to: {boxplot_path}")

    # 3. Detailed comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    table_data = []
    table_data.append(['FP/Image', 'Scratch Mean±Std', 'Finetuned Mean±Std', 'Difference'])

    for fp_rate in fp_rates:
        scratch_mean = np.mean(scratch_results[fp_rate])
        scratch_std = np.std(scratch_results[fp_rate], ddof=1)

        finetune_mean = np.mean(finetune_results[fp_rate])
        finetune_std = np.std(finetune_results[fp_rate], ddof=1)

        diff = finetune_mean - scratch_mean

        table_data.append([
            f'{fp_rate}',
            f'{scratch_mean:.4f}±{scratch_std:.4f}',
            f'{finetune_mean:.4f}±{finetune_std:.4f}',
            f'{diff:+.4f}'
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Detailed FROC Comparison', fontsize=14, pad=20)
    plt.tight_layout()

    table_path = output_dir / 'comparison_table.png'
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Comparison table saved to: {table_path}")


def main(args):
    """Main analysis function."""

    print("=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Load results
    print(f"\nLoading scratch results from: {args.scratch_dir}")
    scratch_results = load_froc_results(args.scratch_dir, args.n_runs)

    print(f"Loading finetuned results from: {args.finetune_dir}")
    finetune_results = load_froc_results(args.finetune_dir, args.n_runs)

    # Check we have data
    if not scratch_results or not finetune_results:
        print("\n❌ ERROR: No results found!")
        return

    # Primary comparison at FP=1
    fp_rate_primary = 1.0
    scratch_vals = scratch_results[fp_rate_primary]
    finetune_vals = finetune_results[fp_rate_primary]

    print(f"\n{'=' * 80}")
    print(f"PRIMARY COMPARISON (FP/Image = {fp_rate_primary})")
    print(f"{'=' * 80}")

    if len(scratch_vals) != len(finetune_vals):
        print(f"Warning: Unequal number of runs (scratch={len(scratch_vals)}, finetune={len(finetune_vals)})")

    # Compute statistics
    print(f"\nComputing statistics...")
    stats_results = compute_statistics(scratch_vals, finetune_vals)

    # Print results
    print(f"\nScratch:")
    print(f"  Mean ± Std: {stats_results['scratch']['mean']:.4f} ± {stats_results['scratch']['std']:.4f}")
    print(f"  Median: {stats_results['scratch']['median']:.4f}")
    print(f"  Range: [{stats_results['scratch']['min']:.4f}, {stats_results['scratch']['max']:.4f}]")

    print(f"\nFinetuned:")
    print(f"  Mean ± Std: {stats_results['finetune']['mean']:.4f} ± {stats_results['finetune']['std']:.4f}")
    print(f"  Median: {stats_results['finetune']['median']:.4f}")
    print(f"  Range: [{stats_results['finetune']['min']:.4f}, {stats_results['finetune']['max']:.4f}]")

    print(f"\nStatistical Tests:")
    print(f"  Paired t-test:")
    print(f"    t = {stats_results['paired_ttest']['t_statistic']:.4f}")
    print(f"    p-value = {stats_results['paired_ttest']['p_value']:.6f}")

    print(f"  Wilcoxon signed-rank test:")
    print(f"    statistic = {stats_results['wilcoxon_test']['statistic']:.4f}")
    print(f"    p-value = {stats_results['wilcoxon_test']['p_value']:.6f}")

    print(f"\nEffect Size:")
    print(f"  Cohen's d = {stats_results['effect_size']['cohen_d']:.4f}")

    # Significance
    alpha = 0.05
    is_significant = stats_results['paired_ttest']['p_value'] < alpha

    print(f"\nConclusion (α = {alpha}):")
    if is_significant:
        winner = "Finetuned" if stats_results['difference']['mean'] > 0 else "Scratch"
        print(f"  ✓ Statistically significant difference detected!")
        print(f"  → {winner} performs better")
    else:
        print(f"  ✗ No statistically significant difference")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed statistics
    final_stats_path = output_dir / 'final_statistics.json'
    with open(final_stats_path, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"\n✓ Statistics saved to: {final_stats_path}")

    # Save aggregated results
    aggregated = {
        'primary_fp_rate': fp_rate_primary,
        'n_runs': args.n_runs,
        'scratch_results': {str(k): v for k, v in scratch_results.items()},
        'finetune_results': {str(k): v for k, v in finetune_results.items()},
        'statistics': stats_results
    }

    agg_path = output_dir / 'aggregated_results.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"✓ Aggregated results saved to: {agg_path}")

    # Create plots
    print(f"\nGenerating plots...")
    plot_comparison(scratch_results, finetune_results, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical analysis of experiments')

    parser.add_argument('--scratch-dir', type=str, required=True,
                        help='Directory containing scratch experiment runs')
    parser.add_argument('--finetune-dir', type=str, required=True,
                        help='Directory containing finetuned experiment runs')
    parser.add_argument('--n-runs', type=int, default=10,
                        help='Number of runs per experiment')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for results')

    args = parser.parse_args()
    main(args)
