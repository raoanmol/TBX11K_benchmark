"""
FROC (Free-Response ROC) computation for object detection.
Evaluates detection performance at various false positive rates.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from dataset import TBX11KDataset, collate_fn, get_transform
from model import get_maskrcnn_model


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        iou: float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def match_detections_to_gt(detections, ground_truths, iou_threshold=0.5):
    """
    Match detections to ground truth boxes using greedy matching.

    Args:
        detections: List of (box, score) tuples, sorted by score descending
        ground_truths: List of GT boxes
        iou_threshold: IoU threshold for matching

    Returns:
        matches: List of (is_true_positive, score) for each detection
    """
    gt_matched = [False] * len(ground_truths)
    matches = []

    for det_box, score in detections:
        is_tp = False

        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue

            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if match
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            is_tp = True
            gt_matched[best_gt_idx] = True

        matches.append((is_tp, score))

    return matches


@torch.no_grad()
def get_predictions(model, data_loader, device, score_threshold=0.0):
    """
    Get all predictions from the model.

    Returns:
        all_detections: List of (image_id, box, score) for all detections
        all_ground_truths: Dict mapping image_id to list of GT boxes
        num_images: Total number of images
    """
    model.eval()

    all_detections = []
    all_ground_truths = {}

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)

        # Get predictions
        predictions = model(images)

        # Process each image
        for pred, target in zip(predictions, targets):
            image_id = target['image_id'].item()

            # Store ground truths
            gt_boxes = target['boxes'].cpu().numpy()
            all_ground_truths[image_id] = [box for box in gt_boxes]

            # Store detections
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            for box, score in zip(boxes, scores):
                if score >= score_threshold:
                    all_detections.append((image_id, box, score))

    num_images = len(all_ground_truths)

    return all_detections, all_ground_truths, num_images


def compute_froc(all_detections, all_ground_truths, num_images, iou_threshold=0.5):
    """
    Compute FROC curve.

    Args:
        all_detections: List of (image_id, box, score)
        all_ground_truths: Dict mapping image_id to list of GT boxes
        num_images: Total number of images
        iou_threshold: IoU threshold for matching

    Returns:
        froc_data: DataFrame with columns [threshold, tp, fp, sensitivity, fp_per_image]
    """

    # Total GT boxes
    total_gt = sum(len(gts) for gts in all_ground_truths.values())

    if total_gt == 0:
        print("Warning: No ground truth boxes found!")
        return pd.DataFrame(columns=['threshold', 'tp', 'fp', 'sensitivity', 'fp_per_image'])

    # Sort detections by score (descending)
    all_detections.sort(key=lambda x: x[2], reverse=True)

    # Group detections by image
    detections_by_image = {}
    for image_id, box, score in all_detections:
        if image_id not in detections_by_image:
            detections_by_image[image_id] = []
        detections_by_image[image_id].append((box, score))

    # Match detections for each image
    all_matches = []
    for image_id in all_ground_truths.keys():
        detections = detections_by_image.get(image_id, [])
        ground_truths = all_ground_truths[image_id]

        matches = match_detections_to_gt(detections, ground_truths, iou_threshold)
        all_matches.extend(matches)

    # Sort by score
    all_matches.sort(key=lambda x: x[1], reverse=True)

    # Compute cumulative TP/FP
    thresholds = []
    tps = []
    fps = []

    cumulative_tp = 0
    cumulative_fp = 0

    for is_tp, score in all_matches:
        if is_tp:
            cumulative_tp += 1
        else:
            cumulative_fp += 1

        thresholds.append(score)
        tps.append(cumulative_tp)
        fps.append(cumulative_fp)

    # Compute sensitivity and FP per image
    sensitivities = [tp / total_gt for tp in tps]
    fp_per_images = [fp / num_images for fp in fps]

    froc_data = pd.DataFrame({
        'threshold': thresholds,
        'tp': tps,
        'fp': fps,
        'sensitivity': sensitivities,
        'fp_per_image': fp_per_images
    })

    return froc_data


def interpolate_froc(froc_data, fp_rates=[0.125, 0.25, 0.5, 1, 2, 4]):
    """
    Interpolate sensitivity at specific FP per image rates.

    Args:
        froc_data: DataFrame from compute_froc
        fp_rates: List of FP per image rates to interpolate at

    Returns:
        interpolated: Dict mapping FP rate to sensitivity
    """
    if len(froc_data) == 0:
        return {fp: 0.0 for fp in fp_rates}

    interpolated = {}

    for fp_rate in fp_rates:
        # Find closest FP rates
        valid_points = froc_data[froc_data['fp_per_image'] >= fp_rate]

        if len(valid_points) == 0:
            # No detections reached this FP rate
            sensitivity = 0.0
        else:
            # Take the sensitivity at the first point that exceeds this FP rate
            sensitivity = valid_points.iloc[0]['sensitivity']

        interpolated[fp_rate] = sensitivity

    return interpolated


def plot_froc(froc_data, output_path):
    """Plot FROC curve."""
    plt.figure(figsize=(10, 6))

    plt.plot(froc_data['fp_per_image'], froc_data['sensitivity'], 'b-', linewidth=2)
    plt.xlabel('False Positives per Image', fontsize=12)
    plt.ylabel('Sensitivity (Recall)', fontsize=12)
    plt.title('FROC Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 5])
    plt.ylim([0, 1])

    # Mark standard FP rates
    standard_fps = [0.125, 0.25, 0.5, 1, 2, 4]
    for fp_rate in standard_fps:
        plt.axvline(x=fp_rate, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    """Main evaluation function."""

    print("=" * 80)
    print("FROC EVALUATION")
    print("=" * 80)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = TBX11KDataset(
        json_path=args.eval_json,
        image_root=args.image_root,
        transforms=get_transform(train=False)
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f"  ✓ {len(dataset)} images")
    print(f"  ✓ {dataset.num_classes} classes")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = get_maskrcnn_model(num_classes=dataset.num_classes, pretrained_backbone=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Get predictions
    print(f"\nGenerating predictions...")
    all_detections, all_ground_truths, num_images = get_predictions(
        model, data_loader, device, score_threshold=0.0
    )

    total_gt = sum(len(gts) for gts in all_ground_truths.values())
    print(f"  ✓ {len(all_detections)} detections")
    print(f"  ✓ {total_gt} ground truth boxes")
    print(f"  ✓ {num_images} images")

    # Compute FROC
    print(f"\nComputing FROC...")
    froc_data = compute_froc(
        all_detections,
        all_ground_truths,
        num_images,
        iou_threshold=args.iou_threshold
    )

    print(f"  ✓ FROC curve computed ({len(froc_data)} points)")

    # Interpolate at standard FP rates
    fp_rates = [0.125, 0.25, 0.5, 1, 2, 4]
    interpolated = interpolate_froc(froc_data, fp_rates)

    print(f"\nSensitivity at standard FP rates:")
    for fp_rate, sensitivity in interpolated.items():
        print(f"  FP/img = {fp_rate}: {sensitivity:.4f}")

    # Save FROC data
    froc_csv_path = output_dir / 'froc.csv'
    froc_data.to_csv(froc_csv_path, index=False)
    print(f"\n✓ FROC data saved to: {froc_csv_path}")

    # Save interpolated values
    interp_json_path = output_dir / 'froc_interpolated.json'
    with open(interp_json_path, 'w') as f:
        json.dump(interpolated, f, indent=2)
    print(f"✓ Interpolated values saved to: {interp_json_path}")

    # Plot FROC
    plot_path = output_dir / 'froc_curve.png'
    plot_froc(froc_data, plot_path)
    print(f"✓ FROC plot saved to: {plot_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FROC for Mask R-CNN')

    # Data
    parser.add_argument('--eval-json', type=str, required=True)
    parser.add_argument('--image-root', type=str, required=True)

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    # Evaluation
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)

    # Output
    parser.add_argument('--output-dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
