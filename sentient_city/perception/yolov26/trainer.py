"""
YOLOv26 Training Engine.

Provides training infrastructure with experiment tracking integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any
from pathlib import Path
from loguru import logger

from sentient_city.core import get_logger, get_config
from sentient_city.core.experiment_tracking import get_experiment_tracker
from .model import YOLOv26


class YOLOv26Trainer:
    """
    Training engine for YOLOv26.
    
    Handles training loop, validation, checkpointing, and experiment tracking.
    """
    
    def __init__(
        self,
        model: YOLOv26,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: YOLOv26 model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            experiment_id: Optional experiment ID for tracking
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_id = experiment_id or model.experiment_id
        self.config = config or {}
        
        self.logger = get_logger(__name__)
        self.config_manager = get_config()
        
        # Training configuration
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.criterion = self._create_loss_functions()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized YOLOv26 trainer on {self.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config.get("optimizer", "SGD")
        lr = self.config.get("learning_rate", 0.01)
        momentum = self.config.get("momentum", 0.937)
        weight_decay = self.config.get("weight_decay", 0.0005)
        
        if optimizer_type == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "Adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "AdamW":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", "cosine")
        epochs = self.config.get("epochs", 300)
        
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        else:
            return None
    
    def _create_loss_functions(self) -> Dict[str, nn.Module]:
        """Create loss functions."""
        return {
            "detection": nn.MSELoss(),  # Placeholder - would use YOLO loss
            "crowd_density": nn.MSELoss(),
            "behavior_embedding": nn.CosineEmbeddingLoss()
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch["images"].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.get("targets", {}).items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            loss["total"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss["total"].item()
            num_batches += 1
            self.global_step += 1
            
            # Log to experiment tracker
            if self.experiment_id:
                self.model.log_training_step(
                    loss["total"].item(),
                    {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in loss.items() if k != "total"}
                )
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss['total'].item():.4f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Detection loss
        if "detections" in outputs and "detections" in targets:
            # Placeholder - would compute YOLO loss
            losses["detection"] = torch.tensor(0.0, device=self.device)
        
        # Crowd density loss
        if "crowd_density" in outputs and "crowd_density" in targets:
            density_pred = outputs["crowd_density"]["density_map"]
            density_target = targets["crowd_density"]
            losses["crowd_density"] = self.criterion["crowd_density"](
                density_pred, density_target
            )
        else:
            losses["crowd_density"] = torch.tensor(0.0, device=self.device)
        
        # Behavior embedding loss (placeholder)
        losses["behavior_embedding"] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.get("targets", {}).items()}
                
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss["total"].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"val_loss": avg_loss}
    
    def train(self, num_epochs: int) -> None:
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model.training_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            self.logger.info(f"Epoch {epoch} metrics: {metrics}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, metrics)
            
            # Save best model
            if val_metrics.get("val_loss", float("inf")) < self.best_metric:
                self.best_metric = val_metrics["val_loss"]
                self.save_checkpoint(epoch, metrics, is_best=True)
        
        self.logger.info("Training completed")
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
