"""
Native Spiking CNN with LIF Neurons for Breast Cancer Histopathology Classification.

Architecture:
    Input (batch, 3, 224, 224)
        ↓
    Conv2d(3→32) → BN → LIF → Pool
        ↓
    Conv2d(32→64) → BN → LIF → Pool  [+ Residual shortcut]
        ↓
    Conv2d(64→128) → BN → LIF → Pool [+ Residual shortcut]
        ↓
    Conv2d(128→256) → BN → LIF → Pool [+ Residual shortcut]
        ↓
    GlobalAvgPool → Dense(256→128) → Dropout → Dense(128→num_classes)

Key Features:
    - Surrogate gradient (fast sigmoid) for backprop through spikes
    - Rate coding: pixel intensity → spike probability over T time steps
    - Residual connections for stable gradient flow
    - Tracks spike sparsity and firing rate metrics

Reference: Adapted from EEG project (D:\\Projects\\EEG) LIF implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────
# Surrogate Gradient Function
# ─────────────────────────────────────────────

class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for non-differentiable Heaviside step function.
    Uses fast sigmoid approximation: dσ/dx = β / (2 * (1 + |β(x-θ)|)²)
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


class SpikingActivation(nn.Module):
    """Wraps SurrogateSpike for use as a module."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return SurrogateSpike.apply(x, self.threshold)


# ─────────────────────────────────────────────
# Spiking CNN Model
# ─────────────────────────────────────────────

class SpikingCNN(nn.Module):
    """
    Native Spiking CNN for 2D image classification using LIF neurons.

    Uses rate coding: each image is presented T times. At each step,
    membrane potentials leak and accumulate input. Spikes are emitted
    when membrane exceeds threshold, then membrane resets.

    The final classification uses the average spike rate across all
    time steps (global average pooled spatial features).

    Args:
        num_classes: Number of output classes (2 for binary, 3 for multi).
        num_steps: Number of SNN time steps for rate coding (default: 10).
        beta: LIF membrane decay constant (default: 0.9).
        threshold: LIF spike threshold (default: 1.0).
        dropout: Dropout probability for classifier head (default: 0.5).
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

        # ── Block 1: 3 → 32 channels ──
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.spike1 = SpikingActivation(threshold)

        # ── Block 2: 32 → 64 channels (with residual) ──
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.spike2 = SpikingActivation(threshold)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # ── Block 3: 64 → 128 channels (with residual) ──
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.spike3 = SpikingActivation(threshold)
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )

        # ── Block 4: 128 → 256 channels (with residual) ──
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.spike4 = SpikingActivation(threshold)
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )

        # ── Global Average Pooling ──
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ── Classification Head ──
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with spiking dynamics (rate coding).

        Args:
            x: Input images, shape (batch, 3, 224, 224).

        Returns:
            Logits, shape (batch, num_classes).
        """
        batch_size = x.size(0)
        device = x.device

        # ── Initialize membrane potentials ──
        # After conv1+pool: (B, 32, 112, 112)
        mem1 = torch.zeros(batch_size, 32, 112, 112, device=device)
        # After conv2+pool: (B, 64, 56, 56)
        mem2 = torch.zeros(batch_size, 64, 56, 56, device=device)
        # After conv3+pool: (B, 128, 28, 28)
        mem3 = torch.zeros(batch_size, 128, 28, 28, device=device)
        # After conv4: (B, 256, 28, 28)
        mem4 = torch.zeros(batch_size, 256, 28, 28, device=device)

        # Accumulate spike rates across time steps
        spk_sum = torch.zeros(batch_size, 256, device=device)

        for _step in range(self.num_steps):
            # ── Block 1 ──
            cur1 = self.bn1(self.conv1(x))
            cur1 = self.pool1(cur1)
            mem1 = self.beta * mem1 + cur1
            spk1 = self.spike1(mem1)
            mem1 = mem1 * (1 - spk1)  # Reset after spike

            # ── Block 2 (Residual) ──
            main2 = self.bn2_1(self.conv2_1(spk1))
            main2 = self.spike2(main2)
            main2 = self.bn2_2(self.conv2_2(main2))
            sc2 = self.shortcut2(spk1)
            cur2 = main2 + sc2
            cur2 = self.pool2(cur2)
            mem2 = self.beta * mem2 + cur2
            spk2 = self.spike2(mem2)
            mem2 = mem2 * (1 - spk2)

            # ── Block 3 (Residual) ──
            main3 = self.bn3_1(self.conv3_1(spk2))
            main3 = self.spike3(main3)
            main3 = self.bn3_2(self.conv3_2(main3))
            sc3 = self.shortcut3(spk2)
            cur3 = main3 + sc3
            cur3 = self.pool3(cur3)
            mem3 = self.beta * mem3 + cur3
            spk3 = self.spike3(mem3)
            mem3 = mem3 * (1 - spk3)

            # ── Block 4 (Residual) ──
            main4 = self.bn4_1(self.conv4_1(spk3))
            main4 = self.spike4(main4)
            main4 = self.bn4_2(self.conv4_2(main4))
            sc4 = self.shortcut4(spk3)
            cur4 = main4 + sc4
            mem4 = self.beta * mem4 + cur4
            spk4 = self.spike4(mem4)
            mem4 = mem4 * (1 - spk4)

            # ── Global Average Pool of spikes ──
            spk_avg = self.avgpool(spk4).squeeze(-1).squeeze(-1)  # (B, 256)
            spk_sum += spk_avg

        # ── Average spike rate over time steps ──
        spk_rate = spk_sum / self.num_steps

        # ── Classification head ──
        out = F.relu(self.fc1(spk_rate))
        out = self.dropout(out)
        out = self.fc2(out)  # Raw logits for CrossEntropyLoss

        return out

    def get_spike_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute spike statistics for analysis (sparsity, firing rate).
        Runs a forward pass and tracks per-layer spike counts.

        Args:
            x: Input images, shape (batch, 3, 224, 224).

        Returns:
            Dictionary with spike_sparsity and avg_firing_rate per layer.
        """
        batch_size = x.size(0)
        device = x.device

        mem1 = torch.zeros(batch_size, 32, 112, 112, device=device)
        mem2 = torch.zeros(batch_size, 64, 56, 56, device=device)
        mem3 = torch.zeros(batch_size, 128, 28, 28, device=device)
        mem4 = torch.zeros(batch_size, 256, 28, 28, device=device)

        total_spikes = {1: 0, 2: 0, 3: 0, 4: 0}
        total_neurons = {1: 0, 2: 0, 3: 0, 4: 0}

        with torch.no_grad():
            for _step in range(self.num_steps):
                cur1 = self.pool1(self.bn1(self.conv1(x)))
                mem1 = self.beta * mem1 + cur1
                spk1 = self.spike1(mem1)
                mem1 = mem1 * (1 - spk1)
                total_spikes[1] += spk1.sum().item()
                total_neurons[1] += spk1.numel()

                main2 = self.spike2(self.bn2_1(self.conv2_1(spk1)))
                main2 = self.bn2_2(self.conv2_2(main2))
                cur2 = self.pool2(main2 + self.shortcut2(spk1))
                mem2 = self.beta * mem2 + cur2
                spk2 = self.spike2(mem2)
                mem2 = mem2 * (1 - spk2)
                total_spikes[2] += spk2.sum().item()
                total_neurons[2] += spk2.numel()

                main3 = self.spike3(self.bn3_1(self.conv3_1(spk2)))
                main3 = self.bn3_2(self.conv3_2(main3))
                cur3 = self.pool3(main3 + self.shortcut3(spk2))
                mem3 = self.beta * mem3 + cur3
                spk3 = self.spike3(mem3)
                mem3 = mem3 * (1 - spk3)
                total_spikes[3] += spk3.sum().item()
                total_neurons[3] += spk3.numel()

                main4 = self.spike4(self.bn4_1(self.conv4_1(spk3)))
                main4 = self.bn4_2(self.conv4_2(main4))
                cur4 = main4 + self.shortcut4(spk3)
                mem4 = self.beta * mem4 + cur4
                spk4 = self.spike4(mem4)
                mem4 = mem4 * (1 - spk4)
                total_spikes[4] += spk4.sum().item()
                total_neurons[4] += spk4.numel()

        stats = {}
        for layer_id in [1, 2, 3, 4]:
            firing_rate = total_spikes[layer_id] / total_neurons[layer_id] if total_neurons[layer_id] > 0 else 0
            stats[f'layer{layer_id}_firing_rate'] = firing_rate
            stats[f'layer{layer_id}_sparsity'] = 1.0 - firing_rate

        avg_firing = sum(total_spikes.values()) / sum(total_neurons.values())
        stats['avg_firing_rate'] = avg_firing
        stats['avg_sparsity'] = 1.0 - avg_firing

        return stats


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def get_spiking_cnn(num_classes: int = 2, num_steps: int = 10,
                    beta: float = 0.9, threshold: float = 1.0,
                    dropout: float = 0.5) -> SpikingCNN:
    """
    Factory to create a SpikingCNN model for breast cancer classification.

    Args:
        num_classes: 2 for binary, 3 for multi-class.
        num_steps: SNN time steps (10-20 recommended).
        beta: LIF membrane decay (0.8-0.95 recommended).
        threshold: Spike firing threshold.
        dropout: Dropout probability for classification head.

    Returns:
        SpikingCNN model instance.
    """
    return SpikingCNN(
        num_classes=num_classes,
        num_steps=num_steps,
        beta=beta,
        threshold=threshold,
        dropout=dropout,
    )
