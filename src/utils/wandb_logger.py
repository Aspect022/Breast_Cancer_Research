"""
Weights & Biases Integration for Breast Cancer Classification Project.

Provides:
- Experiment tracking
- Model weight/bias logging
- Gradient monitoring
- Confusion matrix & ROC curve logging
- Real-time metric visualization

Usage:
    from src.utils.wandb_logger import WandBLogger
    
    logger = WandBLogger(project="breast-cancer", config=cfg)
    logger.log_metrics({"accuracy": 0.95, "loss": 0.05})
    logger.log_model(model, epoch=10)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Any
from pathlib import Path
import wandb
from datetime import datetime
import yaml


class WandBLogger:
    """
    Weights & Biases logger for breast cancer classification experiments.
    
    Features:
    - Automatic metric logging
    - Model weight/bias histograms
    - Gradient monitoring
    - Confusion matrix visualization
    - ROC curve logging
    - Model checkpoint saving
    """
    
    def __init__(
        self,
        project: str = "breast-cancer-transformers",
        config: Optional[Dict] = None,
        wandb_config_path: str = "wandb_config.yaml",
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_dir: str = "wandb_logs",
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name.
            config: Experiment configuration dict.
            wandb_config_path: Path to W&B config file.
            run_name: Custom run name.
            tags: Additional tags for the run.
            log_dir: Directory for local logs.
        """
        self.project = project
        self.config = config or {}
        self.log_dir = log_dir
        self.initialized = False
        self.current_run = None
        
        # Load W&B config from file
        self.wandb_cfg = self._load_wandb_config(wandb_config_path)
        
        # Set API key
        if self.wandb_cfg.get('api_key'):
            os.environ['WANDB_API_KEY'] = self.wandb_cfg['api_key']
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.get('models', {}).keys()
            run_name = f"breast_cancer_{timestamp}"
        
        self.run_name = run_name
        self.tags = tags or self.wandb_cfg.get('tags', [])
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _load_wandb_config(self, config_path: str) -> Dict:
        """Load W&B configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('wandb', {})
        return {}
    
    def init(
        self,
        model: Optional[nn.Module] = None,
        model_name: str = "model",
    ):
        """
        Initialize W&B run.
        
        Args:
            model: PyTorch model to log architecture.
            model_name: Name for the model in W&B.
        """
        if self.initialized:
            return
        
        # Prepare W&B config
        wandb_cfg = {
            'project': self.wandb_cfg.get('project', self.project),
            'name': self.run_name,
            'tags': self.tags,
            'config': self.config,
            'dir': self.log_dir,
            'save_code': self.wandb_cfg.get('save_code', True),
        }
        
        # Set entity if provided
        if self.wandb_cfg.get('entity'):
            wandb_cfg['entity'] = self.wandb_cfg['entity']
        
        # Initialize W&B
        wandb.init(**wandb_cfg)
        self.current_run = wandb.run
        self.initialized = True
        
        # Log model architecture
        if model is not None:
            self.log_model_architecture(model, model_name)
        
        print(f"✓ W&B initialized: {wandb.run.get_url()}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric name -> value.
            step: Training step/epoch.
            commit: Save metrics to history.
        """
        if not self.initialized:
            return
        
        wandb.log(metrics, step=step, commit=commit)
    
    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_auc: float,
        lr: float,
        additional_metrics: Optional[Dict] = None,
    ):
        """
        Log epoch-level training metrics.
        
        Args:
            epoch: Current epoch number.
            train_loss: Training loss.
            train_acc: Training accuracy.
            val_loss: Validation loss.
            val_acc: Validation accuracy.
            val_auc: Validation AUC.
            lr: Learning rate.
            additional_metrics: Additional metrics to log.
        """
        metrics = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/auc': val_auc,
            'optimizer/lr': lr,
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_metrics(metrics, step=epoch)
    
    def log_model_architecture(
        self,
        model: nn.Module,
        model_name: str = "model",
    ):
        """
        Log model architecture as text.
        
        Args:
            model: PyTorch model.
            model_name: Name for the model.
        """
        if not self.initialized:
            return
        
        # Log model structure
        model_str = str(model)
        wandb.log({f"{model_name}/architecture": wandb.Html(f"<pre>{model_str}</pre>")})
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            f"{model_name}/total_params": total_params,
            f"{model_name}/trainable_params": trainable_params,
        })
    
    def log_weights_and_biases(
        self,
        model: nn.Module,
        epoch: int,
        log_interval: int = 5,
    ):
        """
        Log model weights and biases as histograms.
        
        Args:
            model: PyTorch model.
            epoch: Current epoch.
            log_interval: Log every N epochs.
        """
        if not self.initialized:
            return
        
        if epoch % log_interval != 0:
            return
        
        # Log weights and biases for each layer
        for name, param in model.named_parameters():
            if param.dim() > 1:
                # Log weight histograms
                wandb.log({
                    f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                }, step=epoch)
                
                # Log gradient histograms if available
                if param.grad is not None:
                    wandb.log({
                        f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()),
                    }, step=epoch)
            
            elif param.dim() == 1:
                # Log biases
                wandb.log({
                    f"biases/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                }, step=epoch)
    
    def log_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        epoch: int,
        title: str = "Confusion Matrix",
    ):
        """
        Log confusion matrix as heatmap.
        
        Args:
            confusion_matrix: NxN confusion matrix.
            class_names: List of class names.
            epoch: Current epoch.
            title: Plot title.
        """
        if not self.initialized:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            wandb.log({
                "metrics/confusion_matrix": wandb.Image(fig),
            }, step=epoch)
            
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log confusion matrix: {e}")
    
    def log_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        epoch: int,
        title: str = "ROC Curve",
    ):
        """
        Log ROC curve.
        
        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            auc: Area under curve.
            epoch: Current epoch.
            title: Plot title.
        """
        if not self.initialized:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            wandb.log({
                "metrics/roc_curve": wandb.Image(fig),
            }, step=epoch)
            
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log ROC curve: {e}")
    
    def log_model_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        save_dir: str = "checkpoints",
    ):
        """
        Save and log model checkpoint.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch.
            metrics: Metrics to include in checkpoint name.
            save_dir: Directory for checkpoints.
        """
        if not self.initialized:
            return
        
        if not self.wandb_cfg.get('log_model', True):
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'config': self.config,
        }
        
        # Save checkpoint
        acc = metrics.get('val_acc', metrics.get('accuracy', 0))
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}_acc{acc:.4f}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Log to W&B
        wandb.save(checkpoint_path, base_path=save_dir)
    
    def log_images(
        self,
        images: torch.Tensor,
        predictions: np.ndarray,
        labels: np.ndarray,
        epoch: int,
        num_images: int = 8,
        title: str = "Sample Predictions",
    ):
        """
        Log sample images with predictions.
        
        Args:
            images: Image tensor (B, C, H, W).
            predictions: Predicted labels.
            labels: True labels.
            epoch: Current epoch.
            num_images: Number of images to log.
            title: Plot title.
        """
        if not self.initialized:
            return
        
        if not self.wandb_cfg.get('log_images', True):
            return
        
        try:
            import matplotlib.pyplot as plt
            
            images = images[:num_images].cpu()
            predictions = predictions[:num_images]
            labels = labels[:num_images]
            
            # Denormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            images = images * std + mean
            images = torch.clamp(images, 0, 1)
            
            fig, axes = plt.subplots(2, num_images // 2, figsize=(12, 6))
            
            for i, ax in enumerate(axes.flat):
                if i >= num_images:
                    break
                
                img = images[i].permute(1, 2, 0).numpy()
                pred = predictions[i]
                label = labels[i]
                
                ax.imshow(img)
                color = 'green' if pred == label else 'red'
                ax.set_title(f'P: {pred} | T: {label}', color=color)
                ax.axis('off')
            
            plt.suptitle(title)
            plt.tight_layout()
            
            wandb.log({
                "samples/images": wandb.Image(fig),
            }, step=epoch)
            
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not log images: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if self.initialized:
            wandb.finish()
            self.initialized = False
            print("✓ W&B run finished")


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def get_wandb_logger(
    config: Dict,
    model_name: str = "model",
    run_name: Optional[str] = None,
) -> WandBLogger:
    """
    Get W&B logger with standard configuration.
    
    Args:
        config: Experiment configuration.
        model_name: Model name for logging.
        run_name: Custom run name.
    
    Returns:
        Initialized WandBLogger.
    """
    logger = WandBLogger(
        config=config,
        run_name=run_name,
    )
    return logger
