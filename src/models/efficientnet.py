import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet_b3(num_classes):
    """
    Returns a pretrained EfficientNet-B3 model adapted for the specified number of classes.
    Uses 'IMAGENET1K_V1' weights as the baseline.
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # EfficientNet-B3 classifier is a Sequential containing a Dropout and a Linear layer
    in_features = model.classifier[1].in_features

    # Replace the classification head
    # Adding Dropout(0.4) as per standard robust baseline practice
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def get_efficientnet_b5(num_classes):
    """
    Returns a pretrained EfficientNet-B5 model adapted for the specified number of classes.
    Uses 'IMAGENET1K_V1' weights as the baseline.
    
    B5 vs B3:
    - B3: 12M params, 1536 features, 1.8B FLOPs
    - B5: 30M params, 2048 features, 9.9B FLOPs (+2-3% accuracy in medical imaging)
    """
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)

    # EfficientNet-B5 classifier: Dropout + Linear(2048 → num_classes)
    in_features = model.classifier[1].in_features  # 2048

    # Replace the classification head
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model
