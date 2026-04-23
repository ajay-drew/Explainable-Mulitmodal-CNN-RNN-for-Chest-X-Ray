"""
Single-image or single-text inference script.

Usage:
    # Chest X-ray (Mode A)
    python scripts/inference.py --image xray.jpg --mode cnn

    # With radiology report
    python scripts/inference.py --image xray.jpg --mode cnn
                                --text "Bilateral infiltrates noted."

    # Sentiment analysis (Mode B)
    python scripts/inference.py --text "feeling great today" --mode rnn

    # With custom device
    python scripts/inference.py --image xray.jpg --mode cnn --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_cnn_inference(
    image_path: str,
    report_text: Optional[str],
    device: torch.device,
    threshold: float = 0.5,
    gradcam_method: str = "gradcam",
    output_dir: Optional[str] = None,
) -> None:
    """
    Run Mode A (CNN) inference on a chest X-ray.

    Args:
        image_path:    Path to the input chest X-ray image (JPG/PNG).
        report_text:   Optional radiology report text.
        device:        Torch device.
        threshold:     Confidence threshold for positive predictions.
        gradcam_method: CAM method ("gradcam", "gradcam++", "eigencam").
        output_dir:    If set, save heatmap and JSON to this directory.
    """
    from config.config import seed_everything
    from data.preprocessing import _pil_to_xrv_tensor
    from models.image_encoder import load_image_encoder
    from xai.unified import UnifiedExplainer

    seed_everything(42)

    # Load image
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    original_pil = Image.open(image_path)
    image_tensor = _pil_to_xrv_tensor(original_pil).unsqueeze(0).to(device)  # (1,1,224,224)

    # Load model
    logger.info("Loading TorchXRayVision DenseNet121 …")
    t0 = time.perf_counter()
    encoder = load_image_encoder(device=device)
    load_time = (time.perf_counter() - t0) * 1000
    logger.info(f"Model loaded in {load_time:.0f} ms")

    # Load text encoder if report provided
    text_encoder = None
    tokenizer = None
    if report_text:
        from models.text_encoder import load_radbert
        from data.preprocessing import get_radbert_tokenizer
        logger.info("Loading RadBERT for text attribution …")
        text_encoder = load_radbert(device=device)
        tokenizer = get_radbert_tokenizer()

    # Build unified explainer
    explainer = UnifiedExplainer(
        image_encoder=encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        gradcam_method=gradcam_method,
        labels=encoder.target_labels,
    )

    # Run inference + XAI
    logger.info("Running inference …")
    t0 = time.perf_counter()
    with torch.no_grad():
        result = explainer.explain(
            image_tensor=image_tensor,
            original_pil=original_pil,
            report_text=report_text,
            threshold=threshold,
        )
    inf_time = (time.perf_counter() - t0) * 1000

    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTION:  {result.disease_name}")
    print(f"CONFIDENCE:  {result.confidence*100:.1f}%")
    print(f"FAITHFULNESS:{result.faithfulness_score:.3f}")
    print(f"TIME:        {inf_time:.0f} ms")
    print(f"\nNL SUMMARY:\n{result.nl_summary}")
    print(f"\nTOP PREDICTIONS (threshold={threshold}):")
    for label, conf in sorted(result.all_predictions.items(), key=lambda x: x[1], reverse=True):
        marker = "*" if conf >= threshold else " "
        print(f"  {marker} {label:<25} {conf*100:>5.1f}%")

    if report_text and result.token_shap:
        print(f"\nTOP SHAP TOKENS:")
        for tok, score in result.token_shap[:5]:
            print(f"  {tok:<20} {score:+.4f}")

    print("=" * 60)

    # Save outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = Path(image_path).stem
        if result.heatmap_overlay is not None:
            overlay_path = os.path.join(output_dir, f"{stem}_gradcam.png")
            result.heatmap_overlay.save(overlay_path)
            logger.info(f"Saved Grad-CAM overlay: {overlay_path}")

        json_path = os.path.join(output_dir, f"{stem}_explanation.json")
        with open(json_path, "w") as fh:
            fh.write(result.as_json())
        logger.info(f"Saved explanation JSON: {json_path}")


def run_rnn_inference(
    text: str,
    device: torch.device,
) -> None:
    """
    Run Mode B (RNN/transformer) sentiment inference on text.

    Args:
        text:   Input tweet or sentence.
        device: Torch device.
    """
    from config.config import seed_everything
    from data.preprocessing import get_twitter_tokenizer, tokenize_tweet
    from models.text_encoder import load_twitter_roberta
    from xai.text_attribution import SHAPAttributor
    from xai.nlp_summary import NLExplainer

    seed_everything(42)

    tokenizer = get_twitter_tokenizer()
    encoded = tokenize_tweet(text, tokenizer)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

    logger.info("Loading TwitterRoBERTa …")
    t0 = time.perf_counter()
    model = load_twitter_roberta(device=device)
    load_time = (time.perf_counter() - t0) * 1000
    logger.info(f"Model loaded in {load_time:.0f} ms")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    inf_time = (time.perf_counter() - t0) * 1000

    binary_pred = int(out["binary_pred"][0])
    binary_conf = float(out["binary_conf"][0])
    label = "POSITIVE" if binary_pred == 1 else "NEGATIVE"
    probs = out["probs"][0].tolist()

    # SHAP attribution
    shap = SHAPAttributor(model, tokenizer, device=device)
    token_attrs = shap.explain(text, target_class=2 if binary_pred == 1 else 0)

    nl = NLExplainer()
    summary = nl.explain_text_prediction(label.lower(), binary_conf, token_attrs)

    print(f"\n{'='*60}")
    print(f"LABEL:      {label}")
    print(f"CONFIDENCE: {binary_conf*100:.1f}%")
    print(f"TIME:       {inf_time:.0f} ms")
    print(f"RAW PROBS:  neg={probs[0]:.3f}  neu={probs[1]:.3f}  pos={probs[2]:.3f}")
    print(f"\nNL SUMMARY:\n{summary}")
    if token_attrs:
        print(f"\nTOP SHAP TOKENS:")
        for tok, score in token_attrs[:5]:
            print(f"  {tok:<20} {score:+.4f}")
    print("=" * 60)


def main() -> None:
    """CLI entry point for single inference."""
    parser = argparse.ArgumentParser(
        description="Single inference script for CNN (chest X-ray) or RNN (sentiment)"
    )
    parser.add_argument("--mode", choices=["cnn", "rnn"], required=True,
                        help="Inference mode: 'cnn' for chest X-ray, 'rnn' for text")
    parser.add_argument("--image", default=None,
                        help="Path to chest X-ray image (Mode A)")
    parser.add_argument("--text", default=None,
                        help="Text string (Mode A: radiology report; Mode B: tweet)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for Mode A positive predictions")
    parser.add_argument("--gradcam-method", default="gradcam",
                        choices=["gradcam", "gradcam++", "eigencam"],
                        help="Grad-CAM variant (Mode A only)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save heatmap and JSON output (Mode A)")

    args = parser.parse_args()

    # Resolve device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if args.mode == "cnn":
        if args.image is None:
            parser.error("--image is required for --mode cnn")
        run_cnn_inference(
            image_path=args.image,
            report_text=args.text,
            device=device,
            threshold=args.threshold,
            gradcam_method=args.gradcam_method,
            output_dir=args.output_dir,
        )
    else:
        if args.text is None:
            parser.error("--text is required for --mode rnn")
        run_rnn_inference(text=args.text, device=device)


if __name__ == "__main__":
    main()
