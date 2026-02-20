"""
Mask R-CNN model setup for TBX11K experiments.
Supports both scratch and ImageNet-pretrained initialization.
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_maskrcnn_model(num_classes, pretrained_backbone=True):
    """
    Create Mask R-CNN model with ResNet-50 FPN backbone.

    Args:
        num_classes: Number of object classes (excluding background)
        pretrained_backbone: If True, use ImageNet-pretrained backbone

    Returns:
        model: Mask R-CNN model
    """

    # Load model
    # Note: pretrained parameter loads full COCO-pretrained model (deprecated in newer versions)
    # We want pretrained_backbone for just the ResNet backbone weights
    model = maskrcnn_resnet50_fpn(
        weights=None,  # Don't load full model weights
        num_classes=91  # Default COCO classes, will modify below
    )

    # Load ImageNet backbone weights if requested
    if pretrained_backbone:
        # Get pretrained backbone
        from torchvision.models import resnet50, ResNet50_Weights
        pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Transfer backbone weights
        # The backbone in Mask R-CNN is model.backbone.body
        pretrained_dict = pretrained_resnet.state_dict()
        model_dict = model.backbone.body.state_dict()

        # Filter out mismatched keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.backbone.body.load_state_dict(model_dict)

        print(f"  ✓ Loaded ImageNet-pretrained backbone ({len(pretrained_dict)} params)")
    else:
        print(f"  ✓ Training backbone from scratch")

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 for background

    # Replace mask predictor head (if using Mask R-CNN)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes + 1  # +1 for background
    )

    return model


def get_optimizer(model, lr=0.005, momentum=0.9, weight_decay=0.0005):
    """Get SGD optimizer with default Mask R-CNN parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    return optimizer


def get_lr_scheduler(optimizer, step_size=3, gamma=0.1):
    """Get StepLR scheduler."""
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return lr_scheduler
