"""
TBX11K Dataset loader for PyTorch.
Supports COCO format annotations with bounding boxes.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class TBX11KDataset(Dataset):
    """
    TBX11K dataset in COCO format.

    Args:
        json_path: Path to COCO JSON annotation file
        image_root: Root directory containing images
        transforms: Optional transforms to apply
    """

    def __init__(self, json_path, image_root, transforms=None):
        self.image_root = Path(image_root)
        self.transforms = transforms

        # Load COCO annotations
        with open(json_path, 'r') as f:
            self.coco = json.load(f)

        # Create lookups
        self.images = {img['id']: img for img in self.coco['images']}
        self.categories = {cat['id']: cat for cat in self.coco['categories']}

        # Group annotations by image
        self.annotations_by_image = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

        # Image IDs list
        self.image_ids = list(self.images.keys())

        # Category mapping (COCO categories to contiguous 1-indexed)
        self.category_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(self.categories.keys()), start=1)}
        self.num_classes = len(self.categories)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image
            target: dict with keys:
                - boxes: FloatTensor[N, 4] in [x1, y1, x2, y2] format
                - labels: Int64Tensor[N]
                - image_id: Int64Tensor[1]
                - area: FloatTensor[N]
                - iscrowd: UInt8Tensor[N]
        """
        # Get image info
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.image_root / img_info['file_name']
        image = Image.open(img_path).convert('RGB')

        # Get annotations
        annotations = self.annotations_by_image.get(img_id, [])

        # Prepare target
        boxes = []
        labels = []
        areas = []
        iscrowds = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox

            # Convert to [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Filter invalid boxes
            if w > 0 and h > 0:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.category_id_to_idx[ann['category_id']])
                areas.append(ann.get('area', w * h))
                iscrowds.append(ann.get('iscrowd', 0))

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowds = torch.as_tensor(iscrowds, dtype=torch.uint8)
        else:
            # No annotations (background image)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowds = torch.zeros((0,), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id], dtype=torch.int64),
            'area': areas,
            'iscrowd': iscrowds
        }

        # Create dummy masks from bounding boxes (required by Mask R-CNN)
        img_h, img_w = image.size[1], image.size[0]
        if len(boxes) > 0:
            masks = torch.zeros((len(boxes), img_h, img_w), dtype=torch.uint8)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # Clamp to image boundaries
                x1, x2 = max(0, x1), min(img_w, x2)
                y1, y2 = max(0, y1), min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    masks[i, y1:y2, x1:x2] = 1
        else:
            masks = torch.zeros((0, img_h, img_w), dtype=torch.uint8)
        target['masks'] = masks

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_img_info(self, idx):
        """Get image metadata."""
        img_id = self.image_ids[idx]
        return self.images[img_id]


def collate_fn(batch):
    """
    Custom collate function for batching.
    Handles variable number of annotations per image.
    """
    return tuple(zip(*batch))


# Simple transforms
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        import torchvision.transforms.functional as F
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        import torchvision.transforms.functional as F
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(train=True):
    """Get default transforms."""
    transforms = [ToTensor()]

    if train:
        # Add augmentations if needed
        pass

    # ImageNet normalization
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)
