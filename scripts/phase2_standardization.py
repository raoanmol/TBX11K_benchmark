"""
PHASE 2: Data Standardization
Convert discovered dataset to standardized COCO format with validation.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


def compute_json_hash(json_path):
    """Compute MD5 hash of JSON file."""
    hasher = hashlib.md5()
    with open(json_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def validate_coco_json(json_path, image_root):
    """Validate COCO JSON structure and image paths."""
    print(f"\n  Validating: {json_path.name}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    issues = []
    stats = {
        'num_images': len(data.get('images', [])),
        'num_annotations': len(data.get('annotations', [])),
        'num_categories': len(data.get('categories', [])),
        'categories': data.get('categories', []),
        'missing_images': 0,
        'empty_bbox': 0,
        'negative_bbox': 0,
        'annotations_per_image': {},
        'class_distribution': Counter()
    }

    # Check required keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            issues.append(f"Missing required key: {key}")

    if issues:
        return {'valid': False, 'issues': issues, 'stats': stats}

    # Validate images
    image_ids = set()
    for img in data['images']:
        image_ids.add(img['id'])
        img_path = image_root / img['file_name']
        if not img_path.exists():
            stats['missing_images'] += 1
            if stats['missing_images'] <= 3:  # Only log first 3
                issues.append(f"Missing image: {img['file_name']}")

    # Validate annotations
    for ann in data['annotations']:
        # Check image_id exists
        if ann['image_id'] not in image_ids:
            issues.append(f"Annotation {ann['id']} references non-existent image {ann['image_id']}")

        # Check bbox
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            stats['empty_bbox'] += 1
        elif bbox[2] <= 0 or bbox[3] <= 0:
            stats['negative_bbox'] += 1

        # Track annotations per image
        img_id = ann['image_id']
        if img_id not in stats['annotations_per_image']:
            stats['annotations_per_image'][img_id] = 0
        stats['annotations_per_image'][img_id] += 1

        # Class distribution
        stats['class_distribution'][ann['category_id']] += 1

    # Convert class_distribution Counter to regular dict with int values
    stats['class_distribution'] = {int(k): int(v) for k, v in stats['class_distribution'].items()}

    # Compute statistics
    if stats['annotations_per_image']:
        ann_counts = list(stats['annotations_per_image'].values())
        stats['avg_annotations_per_image'] = float(np.mean(ann_counts))
        stats['max_annotations_per_image'] = int(np.max(ann_counts))
        stats['images_with_annotations'] = int(len(ann_counts))
        stats['images_without_annotations'] = int(stats['num_images'] - len(ann_counts))

    # Summary
    print(f"    Images: {stats['num_images']}")
    print(f"    Annotations: {stats['num_annotations']}")
    print(f"    Categories: {stats['num_categories']}")
    print(f"    Missing images: {stats['missing_images']}")
    print(f"    Invalid bbox: {stats['empty_bbox'] + stats['negative_bbox']}")

    # Remove large dict before saving
    if 'annotations_per_image' in stats:
        del stats['annotations_per_image']

    return {
        'valid': len(issues) == 0 or (stats['missing_images'] == 0 and stats['empty_bbox'] == 0 and stats['negative_bbox'] == 0),
        'issues': issues,
        'stats': stats
    }


def create_processed_dataset(source_root, output_root):
    """
    Standardize dataset to processed/ directory.

    Structure:
        processed/
            train.json
            val.json
            test.json (if available)
            images/ -> symlink to original
            split_metadata.json
    """

    source_root = Path(source_root)
    output_root = Path(output_root)

    print("=" * 80)
    print("PHASE 2: Data Standardization")
    print("=" * 80)

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # Image root
    image_root = source_root / 'imgs'

    # Define splits
    split_mapping = {
        'train': 'TBX11K_train.json',
        'val': 'TBX11K_val.json',
        'test': 'all_test.json'  # Test has no annotations, but we keep it for completeness
    }

    results = {
        'source_root': str(source_root.absolute()),
        'output_root': str(output_root.absolute()),
        'image_root': str(image_root.absolute()),
        'splits': {},
        'validation_results': {}
    }

    # Process each split
    for split_name, json_file in split_mapping.items():
        source_json = source_root / 'annotations' / 'json' / json_file
        output_json = output_root / f'{split_name}.json'

        if not source_json.exists():
            print(f"\n⚠ Warning: {json_file} not found, skipping {split_name} split")
            continue

        print(f"\n[{split_name.upper()}] Processing...")

        # Load original JSON
        with open(source_json, 'r') as f:
            data = json.load(f)

        # Update image paths to be absolute
        for img in data.get('images', []):
            # Keep relative path for portability, but validate absolute exists
            rel_path = img['file_name']
            abs_path = image_root / rel_path
            if abs_path.exists():
                # Store both for flexibility
                img['file_name'] = rel_path  # Keep relative
                img['absolute_path'] = str(abs_path)

        # Save standardized JSON
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)

        # Validate
        validation = validate_coco_json(output_json, image_root)
        results['validation_results'][split_name] = validation

        # Compute hash
        json_hash = compute_json_hash(output_json)

        # Store split info
        results['splits'][split_name] = {
            'source_file': str(source_json),
            'output_file': str(output_json),
            'num_images': len(data.get('images', [])),
            'num_annotations': len(data.get('annotations', [])),
            'json_hash': json_hash,
            'valid': validation['valid']
        }

        print(f"  ✓ Saved to: {output_json}")
        print(f"  ✓ Hash: {json_hash[:16]}...")

        if not validation['valid']:
            print(f"  ⚠ VALIDATION FAILED:")
            for issue in validation['issues'][:5]:
                print(f"    - {issue}")

    # Create symlink to images (for convenience)
    images_symlink = output_root / 'images'
    if not images_symlink.exists():
        try:
            images_symlink.symlink_to(image_root.absolute())
            print(f"\n  ✓ Created symlink: images/ -> {image_root}")
        except:
            print(f"\n  ⚠ Could not create symlink (may require admin privileges)")

    # Save metadata
    metadata_path = output_root / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Metadata saved to: {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("STANDARDIZATION SUMMARY")
    print("=" * 80)
    for split_name, info in results['splits'].items():
        status = "✓" if info['valid'] else "✗"
        print(f"{status} {split_name.upper()}: {info['num_images']} images, {info['num_annotations']} annotations")
    print("=" * 80)

    # Check for critical errors
    all_valid = all(info['valid'] for info in results['splits'].values())
    if not all_valid:
        print("\n⚠ WARNING: Some splits have validation errors!")
        print("Please review validation results before proceeding with training.")
        return False

    return True


if __name__ == '__main__':
    source_root = Path('./data/TBX11K')
    output_root = Path('./data/processed')

    success = create_processed_dataset(source_root, output_root)

    if success:
        print("\n✅ Data standardization completed successfully!")
    else:
        print("\n❌ Data standardization completed with warnings.")
