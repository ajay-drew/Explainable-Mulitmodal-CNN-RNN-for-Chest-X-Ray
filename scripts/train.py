#!/usr/bin/env python
"""
Training script for the Explainable Multimodal CNN-RNN Classifier.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from config import Config, get_config, set_config
from src.data import get_dataloaders
from src.models import MultimodalClassifier
from src.training import Trainer
from src.utils import set_seed, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override config with CLI args
    if args.data_root:
        config.data.data_root = args.data_root
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device
    
    set_config(config)
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.training.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    dataloaders = get_dataloaders(
        data_root=config.data.data_root,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        image_size=config.data.image_size,
        max_text_length=config.data.max_text_length,
        text_model_name=config.model.text_encoder_name,
    )
    
    # Create model
    print("Creating model...")
    model = MultimodalClassifier(
        image_model_name=config.model.image_encoder_name,
        image_pretrained=config.model.image_encoder_pretrained,
        freeze_image_encoder=config.model.freeze_image_encoder,
        text_model_name=config.model.text_encoder_name,
        freeze_text_encoder=config.model.freeze_text_encoder,
        fusion_type=config.model.fusion_type,
        fusion_hidden_dim=config.model.fusion_hidden_dim,
        fusion_dropout=config.model.fusion_dropout,
        num_classes=config.model.num_classes,
        classifier_hidden_dims=config.model.classifier_hidden_dims,
        classifier_dropout=config.model.classifier_dropout,
    )
    
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config,
        device=str(device),
    )
    
    # Train
    print("Starting training...")
    best_metrics = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best validation AUROC: {best_metrics.get('auroc_macro', 0):.4f}")
    
    # Save final config
    config.save(Path(config.training.checkpoint_dir) / "config.yaml")


if __name__ == "__main__":
    main()
