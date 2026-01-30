"""
Training loop for the multimodal classifier.

Per PROJECT_PLAN §5.5:
- Optimizer: Adam (lr=1e-4)
- Loss: Binary cross-entropy
- Dropout: 0.5
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .evaluate import evaluate, compute_metrics


class Trainer:
    """
    Trainer for multimodal chest X-ray classifier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Multimodal classifier
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function (BCE for multi-label)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Tracking
        self.best_val_auroc = 0.0
        self.epochs_without_improvement = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if self.config.training.optimizer == "adam":
            return Adam(params, lr=lr, weight_decay=weight_decay)
        elif self.config.training.optimizer == "adamw":
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
            )
        elif self.config.training.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=self.config.training.scheduler_patience,
                factor=0.5,
            )
        else:
            return None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            image = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    logits = self.model(image, input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(image, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        return evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            class_names=self.config.disease_labels,
        )
    
    def train(self) -> Dict[str, float]:
        """
        Full training loop.
        
        Returns:
            Best validation metrics
        """
        best_metrics = {}
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch} - Val AUROC: {val_metrics['auroc_macro']:.4f}")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auroc_macro"])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            if val_metrics["auroc_macro"] > self.best_val_auroc:
                self.best_val_auroc = val_metrics["auroc_macro"]
                self.epochs_without_improvement = 0
                best_metrics = val_metrics
                
                # Save best model
                if self.config.training.save_best_only:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return best_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auroc": self.best_val_auroc,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_auroc = checkpoint.get("best_val_auroc", 0.0)
        return checkpoint.get("epoch", 0)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    criterion,
    device: str,
    scaler=None,
) -> float:
    """
    Train for one epoch (standalone function).
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                logits = model(image, input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
