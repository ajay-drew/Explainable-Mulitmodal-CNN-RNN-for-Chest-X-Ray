"""
Configuration for Explainable Multimodal CNN-RNN for Chest X-Ray Diagnosis.

See PROJECT_PLAN.md for full requirements and architecture.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""
    
    # Paths
    data_root: str = "data/mimic-cxr"
    images_dir: str = "files"
    reports_file: str = "mimic-cxr-reports.csv"
    labels_file: str = "mimic-cxr-2.0.0-chexpert.csv"
    splits_file: str = "mimic-cxr-2.0.0-split.csv"
    
    # Image preprocessing (per PROJECT_PLAN §3.1)
    image_size: int = 224
    image_mean: float = 0.485
    image_std: float = 0.229
    
    # Text preprocessing (per PROJECT_PLAN §3.1)
    max_text_length: int = 512
    
    # Splits (per PROJECT_PLAN: 80/10/10)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Exclude lateral views (per PROJECT_PLAN)
    exclude_lateral: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # === Image Encoder (CNN) ===
    # TorchXRayVision DenseNet121 MIMIC-CXR (per PROJECT_PLAN §8)
    image_encoder_name: str = "densenet121-res224-mimic_nb"
    image_encoder_pretrained: bool = True
    image_feature_dim: int = 1024  # DenseNet121 features before classifier
    freeze_image_encoder: bool = False  # Fine-tune or freeze
    
    # === Text Encoder (BERT) ===
    # RadBERT-RoBERTa-4m (per PROJECT_PLAN §9)
    text_encoder_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m"
    text_feature_dim: int = 768  # BERT/RoBERTa hidden size
    freeze_text_encoder: bool = False
    
    # === Fusion ===
    fusion_type: str = "attention"  # "concat", "attention", "cross_attention"
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.5
    
    # === Classifier ===
    num_classes: int = 14  # 13 diseases + No Findings
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    classifier_dropout: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Hyperparameters (per PROJECT_PLAN §5.5)
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    
    # Optimizer
    optimizer: str = "adam"  # adam, adamw, sgd
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # cosine, step, plateau
    scheduler_patience: int = 5
    
    # Loss (per PROJECT_PLAN: BCE for multi-label)
    loss_fn: str = "bce_with_logits"
    use_class_weights: bool = True  # Handle imbalance
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda"  # cuda, cpu, mps
    num_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision
    use_amp: bool = True


@dataclass
class XAIConfig:
    """Explainability configuration (per PROJECT_PLAN §10)."""
    
    # Image XAI
    image_xai_method: str = "gradcam"  # gradcam, gradcam++
    gradcam_target_layer: str = "features"  # Layer for Grad-CAM
    
    # Text XAI
    text_xai_method: str = "integrated_gradients"  # shap, integrated_gradients, lime
    shap_n_samples: int = 100  # For SHAP
    ig_n_steps: int = 50  # For Integrated Gradients
    
    # Unified XAI
    unified_method: str = "weighted"  # weighted, concat
    faithfulness_threshold: float = 0.92  # Per PROJECT_PLAN target


@dataclass
class Config:
    """Main configuration class."""
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    
    # Project paths
    project_root: str = field(default_factory=lambda: str(Path(__file__).parent.parent))
    output_dir: str = "outputs"
    log_dir: str = "logs"
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Disease labels (per PROJECT_PLAN §3.2)
    disease_labels: List[str] = field(default_factory=lambda: [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ])
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration."""
    global _config
    _config = config
