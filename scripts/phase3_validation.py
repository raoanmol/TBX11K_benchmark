"""
PHASE 3: Dataset Validation with Visualizations
Validate dataset integrity and create debug visualizations.
"""

import os
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def load_coco_data(json_path, image_root):
    """Load COCO JSON and create lookup dicts."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create image lookup
    images = {img['id']: img for img in data['images']}

    # Create category lookup
    categories = {cat['id']: cat for cat in data['categories']}

    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    return {
        'images': images,
        'categories': categories,
        'annotations_by_image': annotations_by_image,
        'image_root': Path(image_root)
    }


def visualize_sample(image_info, annotations, categories, image_root, ax):
    """Visualize a single image with bounding boxes."""
    # Load image
    img_path = image_root / image_info['file_name']
    img = Image.open(img_path)

    # Display image
    ax.imshow(img, cmap='gray')

    # Draw bounding boxes
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        category_name = categories[category_id]['name']

        # Create rectangle
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(
            bbox[0],
            bbox[1] - 5,
            category_name,
            color='red',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    ax.axis('off')
    ax.set_title(f"Image ID: {image_info['id']} | {len(annotations)} annotations", fontsize=10)


def validate_dataset(split_name, json_path, image_root, output_dir, n_samples=10):
    """
    Validate dataset and create visualizations.

    Args:
        split_name: Name of the split (train/val/test)
        json_path: Path to COCO JSON
        image_root: Root directory of images
        output_dir: Output directory for visualizations
        n_samples: Number of random samples to visualize
    """
    print(f"\n{'=' * 80}")
    print(f"Validating {split_name.upper()} split")
    print(f"{'=' * 80}")

    # Load data
    print(f"\n[1/4] Loading COCO data from {json_path}...")
    data = load_coco_data(json_path, image_root)

    print(f"  ✓ {len(data['images'])} images")
    print(f"  ✓ {len(data['categories'])} categories")
    print(f"  ✓ {sum(len(anns) for anns in data['annotations_by_image'].values())} annotations")

    # Validation checks
    print(f"\n[2/4] Running validation checks...")

    validation_report = {
        'split': split_name,
        'total_images': len(data['images']),
        'total_annotations': sum(len(anns) for anns in data['annotations_by_image'].values()),
        'images_with_annotations': len(data['annotations_by_image']),
        'images_without_annotations': len(data['images']) - len(data['annotations_by_image']),
        'categories': list(data['categories'].values()),
        'errors': []
    }

    # Check 1: Missing images
    missing_count = 0
    for img_id, img_info in data['images'].items():
        img_path = data['image_root'] / img_info['file_name']
        if not img_path.exists():
            missing_count += 1
            if missing_count <= 5:
                validation_report['errors'].append(f"Missing image: {img_info['file_name']}")

    if missing_count == 0:
        print("  ✓ All images found")
    else:
        print(f"  ✗ {missing_count} missing images")

    # Check 2: Invalid bboxes
    invalid_bbox_count = 0
    for anns in data['annotations_by_image'].values():
        for ann in anns:
            bbox = ann['bbox']
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                invalid_bbox_count += 1

    if invalid_bbox_count == 0:
        print("  ✓ All bounding boxes valid")
    else:
        print(f"  ✗ {invalid_bbox_count} invalid bounding boxes")

    # Check 3: Class consistency
    category_distribution = {}
    for anns in data['annotations_by_image'].values():
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in category_distribution:
                category_distribution[cat_id] = 0
            category_distribution[cat_id] += 1

    print(f"\n  Class distribution:")
    for cat_id, count in category_distribution.items():
        cat_name = data['categories'][cat_id]['name']
        print(f"    {cat_name}: {count}")

    validation_report['class_distribution'] = {
        data['categories'][cat_id]['name']: count
        for cat_id, count in category_distribution.items()
    }

    # Visualizations
    print(f"\n[3/4] Creating visualizations...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select random samples with annotations
    images_with_anns = list(data['annotations_by_image'].keys())

    if len(images_with_anns) == 0:
        print(f"  ⚠ No annotated images found in {split_name} split")
        n_samples = 0
    else:
        n_samples = min(n_samples, len(images_with_anns))
        sample_ids = random.sample(images_with_anns, n_samples)

        # Create visualization grid
        cols = 3
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, img_id in enumerate(sample_ids):
            img_info = data['images'][img_id]
            annotations = data['annotations_by_image'][img_id]

            visualize_sample(img_info, annotations, data['categories'], data['image_root'], axes[idx])

        # Hide empty subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f'{split_name}_samples.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualization to: {output_path}")

    # Save validation report
    print(f"\n[4/4] Saving validation report...")
    report_path = output_dir / f'{split_name}_validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)

    print(f"  ✓ Report saved to: {report_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"VALIDATION SUMMARY - {split_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Total images: {validation_report['total_images']}")
    print(f"Total annotations: {validation_report['total_annotations']}")
    print(f"Images with annotations: {validation_report['images_with_annotations']}")
    print(f"Images without annotations: {validation_report['images_without_annotations']}")
    print(f"Errors found: {len(validation_report['errors'])}")

    if len(validation_report['errors']) > 0:
        print(f"\n⚠ VALIDATION FAILED")
        for error in validation_report['errors'][:10]:
            print(f"  - {error}")
        return False

    print(f"\n✅ VALIDATION PASSED")
    return True


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)

    # Paths
    processed_dir = Path('./data/processed')
    image_root = Path('./data/TBX11K/imgs')
    output_dir = Path('./data/processed/debug_samples')

    # Validate each split
    splits = ['train', 'val']  # Skip test since it has no annotations

    all_valid = True
    for split_name in splits:
        json_path = processed_dir / f'{split_name}.json'

        if not json_path.exists():
            print(f"\n⚠ Warning: {json_path} not found, skipping")
            continue

        valid = validate_dataset(split_name, json_path, image_root, output_dir, n_samples=9)
        all_valid = all_valid and valid

    # Final summary
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED - Dataset ready for training")
    else:
        print("❌ VALIDATION FAILED - Please fix errors before training")
    print("=" * 80)
