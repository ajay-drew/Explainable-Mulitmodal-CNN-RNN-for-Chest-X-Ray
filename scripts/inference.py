#!/usr/bin/env python
"""
Inference script for the Explainable Multimodal CNN-RNN Classifier.

Usage:
    python scripts/inference.py --image path/to/image.jpg --report "Patient report text"
    python scripts/inference.py --checkpoint checkpoints/best_model.pt --image img.jpg --report report.txt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image

from config import Config
from src.data import ImagePreprocessor, TextPreprocessor
from src.models import MultimodalClassifier
from src.xai import UnifiedExplainer, GradCAM
from src.utils import get_device, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with explainability")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to chest X-ray image",
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Radiology report text or path to text file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explanations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for explanations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
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
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load report text
    report_path = Path(args.report)
    if report_path.exists():
        with open(report_path, "r") as f:
            report_text = f.read()
    else:
        report_text = args.report
    
    # Create preprocessors
    image_preprocessor = ImagePreprocessor(
        image_size=config.data.image_size,
        augment=False,
    )
    text_preprocessor = TextPreprocessor(
        model_name=config.model.text_encoder_name,
        max_length=config.data.max_text_length,
    )
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("L")
    image_tensor = image_preprocessor(image).unsqueeze(0).to(device)
    
    # Preprocess text
    print("Processing report...")
    text_encoded = text_preprocessor(report_text)
    input_ids = text_encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = text_encoded["attention_mask"].unsqueeze(0).to(device)
    
    # Create model
    print("Loading model...")
    model = MultimodalClassifier(
        image_model_name=config.model.image_encoder_name,
        text_model_name=config.model.text_encoder_name,
        fusion_type=config.model.fusion_type,
        fusion_hidden_dim=config.model.fusion_hidden_dim,
        num_classes=config.model.num_classes,
        classifier_hidden_dims=config.model.classifier_hidden_dims,
    )
    
    # Load weights
    load_model(model, args.checkpoint, device=str(device))
    model.to(device)
    model.eval()
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        probs = torch.sigmoid(logits)[0]
    
    # Print predictions
    print("\n" + "=" * 50)
    print("PREDICTIONS")
    print("=" * 50)
    
    for i, (label, prob) in enumerate(zip(config.disease_labels, probs)):
        status = "POSITIVE" if prob >= args.threshold else ""
        print(f"{label:30s}: {prob:.4f} {status}")
    
    # Generate explanations if requested
    if args.explain:
        print("\n" + "=" * 50)
        print("GENERATING EXPLANATIONS")
        print("=" * 50)
        
        # Get target layer for Grad-CAM
        target_layer = model.image_encoder.backbone.features
        
        # Create explainer
        explainer = UnifiedExplainer(
            model=model,
            target_layer=target_layer,
            tokenizer=text_preprocessor.tokenizer,
            class_names=config.disease_labels,
        )
        
        # Get predicted class
        predicted_class = probs.argmax().item()
        
        # Generate explanation
        import numpy as np
        original_image = np.array(image)
        
        explanation = explainer.explain(
            image=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_class=predicted_class,
            original_image=original_image,
            original_text=report_text,
        )
        
        print(f"\nExplaining prediction: {explanation.predicted_class}")
        print(f"Probability: {explanation.prediction_probability:.4f}")
        print(f"Image contribution: {explanation.image_contribution:.2%}")
        print(f"Text contribution: {explanation.text_contribution:.2%}")
        
        if explanation.token_attributions:
            print("\nTop contributing tokens:")
            for token, score in explanation.token_attributions[:5]:
                print(f"  '{token}': {score:.4f}")
        
        if explanation.faithfulness_score is not None:
            print(f"\nFaithfulness score: {explanation.faithfulness_score:.2%}")
        
        # Save outputs
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save heatmap overlay
            if explanation.image_overlay is not None:
                overlay_path = output_dir / "heatmap_overlay.png"
                Image.fromarray(explanation.image_overlay).save(overlay_path)
                print(f"\nSaved heatmap to: {overlay_path}")
            
            # Save explanation report
            from src.xai.unified import create_explanation_report
            report_path = output_dir / "explanation_report.html"
            create_explanation_report(explanation, save_path=str(report_path))
            print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
