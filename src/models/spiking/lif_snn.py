"""
Native Spiking CNN with LIF Neurons for Breast Cancer Histopathology Classification.

CRITICAL DESIGN: Images ≠ EEG temporal sequences.
- EEG: (batch, time, features)
- Images: (batch, channels, height, width)
- Rate encoding creates temporal dimension: (batch, T, C, H, W)

Architecture:
    Input (B, 3, 224, 224)
        → Rate Encoding → (B, T, 3, 224, 224) binary spike trains
        ↓ For each time step t:
            Conv2d(3→32)  → BN → MaxPool → LIF membrane update → spike
            Conv2d(32→64) → BN → MaxPool → LIF membrane update → spike
            Conv2d(64→128)→ BN → MaxPool → LIF membrane update → spike
            GlobalAvgPool → accumulate spike rates
        ↓
    Average spike rate over T steps
        → Dense(128→64) → ReLU → Dropout → Dense(64→num_classes)

Key Features:
    - Proper rate encoding: pixel intensity → spike probability per time step
    - Spatial LIF: membrane tensors match 4D Conv feature maps exactly
    - Surrogate gradient (fast sigmoid) for backprop through binary spikes
    - Tracks spike sparsity & firing rate (paradigm-specific metrics)

Reference: Adapted from EEG project, corrected for 2D spatial data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ─────────────────────────────────────────────
# Surrogate Gradient Function
# ─────────────────────────────────────────────

class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for non-differentiable Heaviside step function.
    Forward: Heaviside (membrane > threshold → 1, else 0)
    Backward: Fast sigmoid approximation:
        dσ/dx = β / (2 * (1 + |β(x-θ)|)²)
    """
    @staticmethod
    def forward(ctx, membrane, threshold):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, = ctx.saved_tensors
        threshold = ctx.threshold
        beta = 25.0  # Sharpness of surrogate gradient
        grad = beta / (2 * (1 + torch.abs(beta * (membrane - threshold))) ** 2)
        return grad_output * grad, None


# ─────────────────────────────────────────────
# Rate Encoding for Images
# ─────────────────────────────────────────────

class RateEncoder(nn.Module):
    """
    Converts pixel intensities to binary spike trains via rate encoding.

    Each pixel's normalized intensity [0, 1] becomes the probability
    of spiking at each time step. A Bernoulli sample generates
    binary spikes: spike if rand() < pixel_intensity.

    Input:  (B, C, H, W) — normalized image tensor
    Output: (B, T, C, H, W) — binary spike trains over T time steps

    Note: During eval, uses deterministic thresholding (0.5) instead
    of stochastic sampling for reproducibility.
    """
    def __init__(self, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        T = self.num_steps

        # Clamp to [0, 1] — images should already be normalized,
        # but pixel values may be outside [0,1] after ImageNet normalization.
        # We use sigmoid to map any value range to [0, 1] probabilities.
        x_prob = torch.sigmoid(x)  # (B, C, H, W) → [0, 1]

        # Expand to time dimension: (B, T, C, H, W)
        x_expanded = x_prob.unsqueeze(1).expand(B, T, C, H, W)

        if self.training:
            # Stochastic: Bernoulli sampling
            spikes = (torch.rand_like(x_expanded) < x_expanded).float()
        else:
            # Deterministic: threshold at 0.5
            spikes = (x_expanded > 0.5).float()

        return spikes


# ─────────────────────────────────────────────
# Spiking CNN Model
# ─────────────────────────────────────────────

class SpikingCNN(nn.Module):
    """
    Native Spiking CNN for 2D image classification using LIF neurons.

    The image is first rate-encoded into binary spike trains (B, T, C, H, W).
    At each time step, the spike frame passes through Conv→BN→Pool layers,
    and the output current charges the LIF membrane potential. When membrane
    exceeds threshold, a spike is emitted and membrane resets.

    Final classification uses the average spike rate across all time steps
    after global average pooling.

    Args:
        num_classes: Number of output classes (2 for binary, 3 for multi).
        num_steps: Number of SNN time steps for rate coding (10-15).
        beta: LIF membrane decay constant (0.8-0.95).
        threshold: LIF spike firing threshold.
        dropout: Dropout probability for classifier head.
    """
    def __init__(
        self,
        num_classes: int = 2,
        num_steps: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold

        # ── Rate Encoder ──
        self.encoder = RateEncoder(num_steps)

        # ── Block 1: 3 → 32 channels ──
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        # After pool: (B, 32, 112, 112)

        # ── Block 2: 32 → 64 channels ──
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        # After pool: (B, 64, 56, 56)

        # ── Block 3: 64 → 128 channels ──
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        # After pool: (B, 128, 28, 28)

        # ── Global Average Pooling ──
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # After GAP: (B, 128)

        # ── Classification Head ──
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spiking dynamics.

        Args:
            x: Input images, shape (B, 3, 224, 224).

        Returns:
            Logits, shape (B, num_classes).
        """
        B = x.size(0)
        device = x.device
        T = self.num_steps

        # ── Step 1: Rate encode image into spike trains ──
        # (B, 3, 224, 224) → (B, T, 3, 224, 224)
        spike_input = self.encoder(x)

        # ── Step 2: Initialize membrane potentials (spatial 4D) ──
        mem1 = torch.zeros(B, 32, 112, 112, device=device)
        mem2 = torch.zeros(B, 64, 56, 56, device=device)
        mem3 = torch.zeros(B, 128, 28, 28, device=device)

        # Accumulate spike rates after GAP
        spk_sum = torch.zeros(B, 128, device=device)

        # ── Step 3: Process each time step ──
        for t in range(T):
            # Get spike frame for this time step: (B, 3, 224, 224)
            inp = spike_input[:, t]

            # Block 1: Conv → BN → Pool → LIF
            cur1 = self.pool1(self.bn1(self.conv1(inp)))  # (B, 32, 112, 112)
            mem1 = self.beta * mem1 + cur1
            spk1 = SurrogateSpike.apply(mem1, self.threshold)
            mem1 = mem1 * (1.0 - spk1)  # Reset after spike

            # Block 2: Conv → BN → Pool → LIF
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))  # (B, 64, 56, 56)
            mem2 = self.beta * mem2 + cur2
            spk2 = SurrogateSpike.apply(mem2, self.threshold)
            mem2 = mem2 * (1.0 - spk2)

            # Block 3: Conv → BN → Pool → LIF
            cur3 = self.pool3(self.bn3(self.conv3(spk2)))  # (B, 128, 28, 28)
            mem3 = self.beta * mem3 + cur3
            spk3 = SurrogateSpike.apply(mem3, self.threshold)
            mem3 = mem3 * (1.0 - spk3)

            # Global average pool of spikes → accumulate
            spk_pooled = self.avgpool(spk3).squeeze(-1).squeeze(-1)  # (B, 128)
            spk_sum += spk_pooled

        # ── Step 4: Average spike rate over time steps ──
        spk_rate = spk_sum / T  # (B, 128)

        # ── Step 5: Classification head ──
        out = F.relu(self.fc1(spk_rate))
        out = self.dropout(out)
        out = self.fc2(out)  # Raw logits

        return out

    def get_spike_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-layer spike statistics for paradigm-specific metrics.

        Returns:
            Dict with firing_rate and sparsity per layer, plus averages.
        """
        B = x.size(0)
        device = x.device
        T = self.num_steps

        spike_input = self.encoder(x)

        mem1 = torch.zeros(B, 32, 112, 112, device=device)
        mem2 = torch.zeros(B, 64, 56, 56, device=device)
        mem3 = torch.zeros(B, 128, 28, 28, device=device)

        total_spikes = {1: 0.0, 2: 0.0, 3: 0.0}
        total_neurons = {1: 0.0, 2: 0.0, 3: 0.0}

        with torch.no_grad():
            for t in range(T):
                inp = spike_input[:, t]

                cur1 = self.pool1(self.bn1(self.conv1(inp)))
                mem1 = self.beta * mem1 + cur1
                spk1 = SurrogateSpike.apply(mem1, self.threshold)
                mem1 = mem1 * (1.0 - spk1)
                total_spikes[1] += spk1.sum().item()
                total_neurons[1] += spk1.numel()

                cur2 = self.pool2(self.bn2(self.conv2(spk1)))
                mem2 = self.beta * mem2 + cur2
                spk2 = SurrogateSpike.apply(mem2, self.threshold)
                mem2 = mem2 * (1.0 - spk2)
                total_spikes[2] += spk2.sum().item()
                total_neurons[2] += spk2.numel()

                cur3 = self.pool3(self.bn3(self.conv3(spk2)))
                mem3 = self.beta * mem3 + cur3
                spk3 = SurrogateSpike.apply(mem3, self.threshold)
                mem3 = mem3 * (1.0 - spk3)
                total_spikes[3] += spk3.sum().item()
                total_neurons[3] += spk3.numel()

        stats = {}
        for layer_id in [1, 2, 3]:
            fr = total_spikes[layer_id] / total_neurons[layer_id] if total_neurons[layer_id] > 0 else 0
            stats[f'layer{layer_id}_firing_rate'] = fr
            stats[f'layer{layer_id}_sparsity'] = 1.0 - fr

        total_s = sum(total_spikes.values())
        total_n = sum(total_neurons.values())
        avg_fr = total_s / total_n if total_n > 0 else 0
        stats['avg_firing_rate'] = avg_fr
        stats['avg_sparsity'] = 1.0 - avg_fr

        return stats


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def get_spiking_cnn(
    num_classes: int = 2,
    num_steps: int = 10,
    beta: float = 0.9,
    threshold: float = 1.0,
    dropout: float = 0.5,
) -> SpikingCNN:
    """
    Factory to create a SpikingCNN model for breast cancer classification.

    Args:
        num_classes: 2 for binary, 3 for multi-class.
        num_steps: SNN time steps (10-15 recommended for images).
        beta: LIF membrane decay (0.8-0.95).
        threshold: Spike firing threshold.
        dropout: Dropout for classification head.
    """
    return SpikingCNN(
        num_classes=num_classes,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        dropout=dropout,
    )
