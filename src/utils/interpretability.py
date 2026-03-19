"""
Model Interpretability and Visualization Tools.

Provides:
- Grad-CAM for CNNs and Transformers
- Attention weight visualization
- Saliency maps
- Feature visualization
- Gate distribution analysis for fusion models

Reference: implementation_plan.md §4 - Model Interpretability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# ─────────────────────────────────────────────
# Grad-CAM Implementation
# ─────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing model attention.
    
    Works with:
    - CNNs (EfficientNet, ConvNeXt)
    - Vision Transformers (Swin, ViT, DeiT)
    - Fusion models
    
    Reference: "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model.
            target_layer: Name of target layer for gradient extraction.
                         If None, uses last conv/attention layer.
        """
        self.model = model
        self.target_layer_name = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        for name, module in self.model.named_modules():
            if self.target_layer_name and self.target_layer_name in name:
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)
                return
        
        # If no target specified, find suitable layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)
                self.target_layer_name = name
                return
    
    def _forward_hook(self, module, input, output):
        """Store forward activations."""
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Store backward gradients."""
        if isinstance(grad_output[0], tuple):
            grad_output = grad_output[0]
        self.gradients = grad_output.detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor (1, 3, H, W).
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            Heatmap (H, W) normalized to [0, 1].
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()
        activations = self.activations.cpu().numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        
        # Weighted combination of activations
        cam = np.sum(activations * weights, axis=1)[0]
        
        # ReLU to keep only positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = 'jet',
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) in RGB, range [0, 255].
            heatmap: Heatmap (H, W) in range [0, 1].
            alpha: Blending alpha.
            colormap: Matplotlib colormap name.
        
        Returns:
            Overlaid image (H, W, 3).
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(
            image.astype(np.uint8), 1 - alpha,
            heatmap_colored.astype(np.uint8), alpha,
            0
        )
        
        return overlay


# ─────────────────────────────────────────────
# Attention Visualization for Transformers
# ─────────────────────────────────────────────

class AttentionVisualizer:
    """
    Extract and visualize attention weights from transformer models.
    
    Supports:
    - Swin Transformer (windowed attention)
    - ViT (global attention)
    - DeiT (distilled attention)
    """
    
    def __init__(self, model: nn.Module, model_type: str = 'swin'):
        """
        Initialize attention visualizer.
        
        Args:
            model: Transformer model.
            model_type: 'swin', 'vit', 'deit', or 'custom'.
        """
        self.model = model
        self.model_type = model_type
        self.attention_weights = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'attn') and hasattr(module.attn, 'attn_drop'):
                # Found attention module
                module.attn.register_forward_hook(
                    lambda m, i, o, n=name: self._attention_hook(n, i, o)
                )
    
    def _attention_hook(self, name, input, output):
        """Store attention weights."""
        if isinstance(output, tuple):
            # Some implementations return (output, attention_weights)
            if len(output) > 1:
                self.attention_weights.append({
                    'layer': name,
                    'weights': output[1].detach(),
                })
    
    def extract_attention(
        self,
        input_image: torch.Tensor,
    ) -> List[Dict]:
        """
        Extract attention weights from all layers.
        
        Args:
            input_image: Input image tensor.
        
        Returns:
            List of dicts with layer names and attention weights.
        """
        self.attention_weights = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_image)
        
        return self.attention_weights
    
    def visualize_attention_rollout(
        self,
        input_image: torch.Tensor,
        resize: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Visualize attention using attention rollout.
        
        Args:
            input_image: Input image tensor.
            resize: Target size for visualization.
        
        Returns:
            Attention rollout heatmap.
        """
        self.extract_attention(input_image)
        
        if not self.attention_weights:
            # Fallback to Grad-CAM
            return None
        
        # Average attention across heads and layers
        all_weights = []
        for aw in self.attention_weights:
            weights = aw['weights']
            # weights shape: (batch, heads, tokens, tokens)
            if len(weights.shape) == 4:
                # Average across heads
                avg_weights = weights.mean(dim=1)[0]  # (tokens, tokens)
                all_weights.append(avg_weights)
        
        if not all_weights:
            return None
        
        # Attention rollout: multiply attention matrices
        rollout = all_weights[0]
        for weights in all_weights[1:]:
            rollout = rollout @ weights
        
        # Get attention to CLS token
        cls_attention = rollout[0, 1:]  # Attention from CLS to patches
        
        # Reshape to image grid
        n_patches = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(n_patches, n_patches).cpu().numpy()
        
        # Upsample to image size
        attention_map = cv2.resize(attention_map, resize, interpolation=cv2.INTER_CUBIC)
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map


# ─────────────────────────────────────────────
# Saliency Map Computation
# ─────────────────────────────────────────────

def compute_saliency_map(
    model: nn.Module,
    input_image: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Compute vanilla saliency map (gradient w.r.t. input).
    
    Args:
        model: PyTorch model.
        input_image: Input image tensor (1, 3, H, W).
        target_class: Target class. If None, uses predicted class.
    
    Returns:
        Saliency map (H, W) normalized to [0, 1].
    """
    model.eval()
    input_image.requires_grad_(True)
    
    output = model(input_image)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()
    
    # Get gradient w.r.t. input
    saliency = input_image.grad.abs()
    
    # Max across channels
    saliency, _ = saliency.max(dim=1)
    
    # Normalize
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency


# ─────────────────────────────────────────────
# Fusion Gate Visualization
# ─────────────────────────────────────────────

def visualize_gate_distribution(
    alpha_values: List[float],
    save_path: Optional[str] = None,
    title: str = "Gating Network Alpha Distribution",
) -> plt.Figure:
    """
    Visualize distribution of gating weights (alpha values).
    
    Args:
        alpha_values: List of alpha values from gating network.
        save_path: Path to save figure.
        title: Plot title.
    
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    alpha_values = np.array(alpha_values)
    
    # Histogram
    axes[0].hist(alpha_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0.5, color='red', linestyle='--', label='Balanced (0.5)')
    axes[0].set_xlabel('Alpha Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Gating Weights')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_alpha = np.sort(alpha_values)
    cumulative = np.arange(len(sorted_alpha)) / len(sorted_alpha)
    axes[1].plot(sorted_alpha, cumulative)
    axes[1].axvline(0.5, color='red', linestyle='--')
    axes[1].set_xlabel('Alpha Value')
    axes[1].set_ylabel('Cumulative Proportion')
    axes[1].set_title('Cumulative Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Branch preference
    convnext_bias = np.mean(alpha_values > 0.5)
    swin_bias = 1 - convnext_bias
    
    axes[2].bar(['ConvNeXt', 'Swin'], [convnext_bias, swin_bias], 
                color=['blue', 'green'], alpha=0.7)
    axes[2].set_ylabel('Proportion of Samples')
    axes[2].set_title('Branch Preference')
    axes[2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    for i, v in enumerate([convnext_bias, swin_bias]):
        axes[2].text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert image tensor to numpy array (H, W, 3) in [0, 255]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if normalized
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.min() < 0 or tensor.max() < 1:
        tensor = tensor * std + mean
    
    tensor = tensor.clamp(0, 1)
    image = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    return image


def save_interpretability_results(
    image: torch.Tensor,
    gradcam_heatmap: np.ndarray,
    saliency_map: np.ndarray,
    attention_map: Optional[np.ndarray],
    save_dir: str,
    model_name: str,
    fold_idx: int,
) -> Dict[str, str]:
    """
    Save all interpretability visualizations.
    
    Args:
        image: Input image tensor.
        gradcam_heatmap: Grad-CAM heatmap.
        saliency_map: Saliency map.
        attention_map: Attention rollout map (optional).
        save_dir: Directory to save figures.
        model_name: Model name.
        fold_idx: Fold index.
    
    Returns:
        Dict mapping visualization type to saved path.
    """
    import os
    from PIL import Image
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensor to image
    img_np = tensor_to_image(image)
    
    saved_paths = {}
    
    # Grad-CAM overlay
    gradcam_overlay = GradCAM(None).overlay_heatmap(img_np, gradcam_heatmap)
    gradcam_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx}_gradcam.png')
    Image.fromarray(gradcam_overlay).save(gradcam_path)
    saved_paths['gradcam'] = gradcam_path
    
    # Saliency overlay
    saliency_colored = plt.get_cmap('jet')(saliency_map)[:, :, :3]
    saliency_colored = (saliency_colored * 255).astype(np.uint8)
    saliency_overlay = cv2.addWeighted(img_np, 0.6, saliency_colored, 0.4, 0)
    saliency_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx}_saliency.png')
    Image.fromarray(saliency_overlay).save(saliency_path)
    saved_paths['saliency'] = saliency_path
    
    # Attention map (if available)
    if attention_map is not None:
        attention_colored = plt.get_cmap('jet')(attention_map)[:, :, :3]
        attention_colored = (attention_colored * 255).astype(np.uint8)
        attention_overlay = cv2.addWeighted(img_np, 0.6, attention_colored, 0.4, 0)
        attention_path = os.path.join(save_dir, f'{model_name}_fold{fold_idx}_attention.png')
        Image.fromarray(attention_overlay).save(attention_path)
        saved_paths['attention'] = attention_path
    
    return saved_paths
