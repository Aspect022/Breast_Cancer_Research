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
