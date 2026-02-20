"""
PHASE 1: Dataset Directory Introspection
Reverse-engineer the TBX11K dataset structure without assumptions.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import hashlib


def compute_file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None


def scan_directory_tree(root_path):
    """Recursively scan directory and collect file statistics."""
    root = Path(root_path)

    stats = {
        'root_path': str(root.absolute()),
        'directory_tree': [],
        'file_counts_by_extension': defaultdict(int),
        'total_files': 0,
        'total_dirs': 0,
        'file_size_stats': defaultdict(list),
        'sample_files_by_extension': defaultdict(list)
    }

    # Scan recursively
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = Path(dirpath).relative_to(root)
        stats['directory_tree'].append(str(rel_path))
        stats['total_dirs'] += len(dirnames)

        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            # Count by extension
            stats['file_counts_by_extension'][ext] += 1
            stats['total_files'] += 1

            # Collect file sizes
            try:
                size = filepath.stat().st_size
                stats['file_size_stats'][ext].append(size)
            except:
                pass

            # Sample files (first 5 per extension)
            if len(stats['sample_files_by_extension'][ext]) < 5:
                stats['sample_files_by_extension'][ext].append(str(filepath.relative_to(root)))

    # Convert defaultdicts to regular dicts for JSON serialization
    stats['file_counts_by_extension'] = dict(stats['file_counts_by_extension'])
    stats['sample_files_by_extension'] = dict(stats['sample_files_by_extension'])

    # Compute size statistics
    size_summary = {}
    for ext, sizes in stats['file_size_stats'].items():
        if sizes:
            size_summary[ext] = {
                'count': len(sizes),
                'total_mb': sum(sizes) / (1024**2),
                'avg_kb': sum(sizes) / len(sizes) / 1024,
                'min_kb': min(sizes) / 1024,
                'max_kb': max(sizes) / 1024
            }
    stats['file_size_stats'] = size_summary

    return stats


def parse_sample_annotations(root_path):
    """Attempt to parse and understand annotation formats."""
    root = Path(root_path)

    results = {
        'xml_annotations': {},
        'json_annotations': {},
        'detected_formats': []
    }

    # Find XML annotations
    xml_dir = root / 'annotations' / 'xml'
    if xml_dir.exists():
        xml_files = list(xml_dir.glob('*.xml'))[:3]
        results['xml_annotations']['count'] = len(list(xml_dir.glob('*.xml')))
        results['xml_annotations']['samples'] = []

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root_elem = tree.getroot()

                sample = {
                    'file': xml_file.name,
                    'structure': {}
                }

                # Extract key elements
                for child in root_elem:
                    if child.tag in ['filename', 'size', 'object']:
                        if child.tag == 'size':
                            sample['structure']['size'] = {
                                'width': child.find('width').text if child.find('width') is not None else None,
                                'height': child.find('height').text if child.find('height') is not None else None
                            }
                        elif child.tag == 'object':
                            if 'objects' not in sample['structure']:
                                sample['structure']['objects'] = []
                            obj_info = {}
                            for obj_child in child:
                                if obj_child.tag in ['name', 'bndbox', 'difficult']:
                                    if obj_child.tag == 'bndbox':
                                        obj_info['bbox'] = {
                                            'xmin': obj_child.find('xmin').text,
                                            'ymin': obj_child.find('ymin').text,
                                            'xmax': obj_child.find('xmax').text,
                                            'ymax': obj_child.find('ymax').text
                                        }
                                    else:
                                        obj_info[obj_child.tag] = obj_child.text
                            sample['structure']['objects'].append(obj_info)
                        else:
                            sample['structure'][child.tag] = child.text

                results['xml_annotations']['samples'].append(sample)
            except Exception as e:
                results['xml_annotations']['parse_error'] = str(e)

        if results['xml_annotations']['samples']:
            results['detected_formats'].append('Pascal VOC (XML)')

    # Find JSON annotations
    json_dir = root / 'annotations' / 'json'
    if json_dir.exists():
        json_files = list(json_dir.glob('*.json'))
        results['json_annotations']['count'] = len(json_files)
        results['json_annotations']['files'] = [f.name for f in json_files]
        results['json_annotations']['samples'] = []

        # Parse first JSON file
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)

                sample = {
                    'file': json_files[0].name,
                    'keys': list(data.keys()),
                    'structure': {}
                }

                # Check for COCO format
                if 'images' in data and 'annotations' in data:
                    sample['structure']['format'] = 'COCO'
                    sample['structure']['num_images'] = len(data.get('images', []))
                    sample['structure']['num_annotations'] = len(data.get('annotations', []))
                    sample['structure']['num_categories'] = len(data.get('categories', []))

                    # Sample first image
                    if data.get('images'):
                        sample['structure']['sample_image'] = data['images'][0]

                    # Sample first annotation
                    if data.get('annotations'):
                        sample['structure']['sample_annotation'] = data['annotations'][0]

                    # Categories
                    if data.get('categories'):
                        sample['structure']['categories'] = data['categories']

                results['json_annotations']['samples'].append(sample)
                results['detected_formats'].append('COCO (JSON)')
            except Exception as e:
                results['json_annotations']['parse_error'] = str(e)

    return results


def check_train_val_splits(root_path):
    """Check for predefined train/val/test splits."""
    root = Path(root_path)
    lists_dir = root / 'lists'

    splits = {}

    if lists_dir.exists():
        for txt_file in lists_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            splits[txt_file.name] = {
                'count': len(lines),
                'samples': lines[:5]
            }

    return splits


def infer_dataset_structure(root_path):
    """Main function to infer complete dataset structure."""

    print("=" * 80)
    print("PHASE 1: Dataset Directory Introspection")
    print("=" * 80)

    report = {
        'phase': 'dataset_discovery',
        'dataset_root': str(Path(root_path).absolute())
    }

    # Step 1: Directory tree scan
    print("\n[1/4] Scanning directory tree...")
    report['directory_scan'] = scan_directory_tree(root_path)
    print(f"  ✓ Found {report['directory_scan']['total_files']} files")
    print(f"  ✓ Found {report['directory_scan']['total_dirs']} directories")

    # Step 2: Parse annotations
    print("\n[2/4] Parsing sample annotations...")
    report['annotation_analysis'] = parse_sample_annotations(root_path)
    print(f"  ✓ Detected formats: {', '.join(report['annotation_analysis']['detected_formats'])}")

    # Step 3: Check splits
    print("\n[3/4] Checking train/val/test splits...")
    report['predefined_splits'] = check_train_val_splits(root_path)
    print(f"  ✓ Found {len(report['predefined_splits'])} split files")

    # Step 4: Inference summary
    print("\n[4/4] Generating inference summary...")

    summary = {
        'dataset_name': 'TBX11K',
        'total_images': report['directory_scan']['file_counts_by_extension'].get('.png', 0) +
                       report['directory_scan']['file_counts_by_extension'].get('.jpg', 0),
        'annotation_format': report['annotation_analysis']['detected_formats'],
        'has_bounding_boxes': True,  # Will verify from samples
        'has_segmentation': False,  # Need to check
        'classes_discovered': [],
        'official_splits_available': len(report['predefined_splits']) > 0
    }

    # Extract classes from JSON annotations
    if report['annotation_analysis']['json_annotations'].get('samples'):
        for sample in report['annotation_analysis']['json_annotations']['samples']:
            if 'categories' in sample['structure']:
                summary['classes_discovered'] = [
                    cat['name'] for cat in sample['structure']['categories']
                ]
                summary['num_classes'] = len(summary['classes_discovered'])

    report['inference_summary'] = summary

    # Save report
    output_path = Path(root_path) / 'data_structure_report.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET INFERENCE SUMMARY")
    print("=" * 80)
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Annotation formats: {', '.join(summary['annotation_format'])}")
    print(f"Classes: {', '.join(summary['classes_discovered'])} ({summary.get('num_classes', 0)} total)")
    print(f"Official splits available: {summary['official_splits_available']}")
    print("=" * 80)

    return report


if __name__ == '__main__':
    dataset_root = './data/TBX11K'
    report = infer_dataset_structure(dataset_root)
