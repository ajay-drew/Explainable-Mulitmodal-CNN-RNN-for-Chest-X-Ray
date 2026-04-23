"""
Streamlit interactive demo for the Multimodal XAI Framework.

Mode A — Medical Imaging (CNN):
    Upload a chest X-ray → get 15-class pathology predictions + Grad-CAM + XAI

Mode B — Sentiment Analysis (RNN):
    Enter tweet text → get binary sentiment + SHAP token attribution

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path so local modules resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import torch
from PIL import Image

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal XAI Framework",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: cached model loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading image encoder …")
def _load_image_encoder(device_str: str):
    """Load and cache XRVImageEncoder on the selected device."""
    from models.image_encoder import load_image_encoder
    device = torch.device(device_str)
    return load_image_encoder(device=device), device


@st.cache_resource(show_spinner="Loading text encoder …")
def _load_radbert(device_str: str):
    """Load and cache RadBERTEncoder + tokenizer."""
    from models.text_encoder import load_radbert
    from data.preprocessing import get_radbert_tokenizer
    device = torch.device(device_str)
    enc = load_radbert(device=device)
    tok = get_radbert_tokenizer()
    return enc, tok, device


@st.cache_resource(show_spinner="Loading sentiment model …")
def _load_sentiment(device_str: str):
    """Load and cache TwitterRoBERTaEncoder + tokenizer."""
    from models.text_encoder import load_twitter_roberta
    from data.preprocessing import get_twitter_tokenizer
    device = torch.device(device_str)
    enc = load_twitter_roberta(device=device)
    tok = get_twitter_tokenizer()
    return enc, tok, device


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🫁 XAI Framework")
    st.markdown("**Modular CNN+Transformer with Explainability**")
    st.markdown("---")

    mode = st.radio(
        "Select Mode",
        options=["Medical Imaging (CNN)", "Sentiment Analysis (RNN)"],
        index=0,
    )

    st.markdown("---")
    device_choice = st.selectbox("Device", ["cpu", "cuda", "mps"], index=0)

    # Validate device availability
    if device_choice == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available — using CPU.")
        device_choice = "cpu"
    if device_choice == "mps" and not torch.backends.mps.is_available():
        st.warning("MPS not available — using CPU.")
        device_choice = "cpu"

    if "Medical" in mode:
        st.markdown("---")
        threshold = st.slider("Disease Threshold", 0.1, 0.9, 0.5, 0.05)
        gradcam_method = st.selectbox(
            "Grad-CAM Method", ["gradcam", "gradcam++", "eigencam"]
        )

    st.markdown("---")
    st.markdown("**Model Info**")
    if "Medical" in mode:
        st.info("Image: TorchXRayVision DenseNet121\nText: StanfordAIMI/RadBERT")
    else:
        st.info("cardiffnlp/twitter-roberta-base-sentiment")
    st.markdown("*All weights pretrained — zero training required*")


# ─────────────────────────────────────────────────────────────────────────────
# Mode A: Medical Imaging
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "low": "#2ecc71",      # green  — confidence < 0.5
    "medium": "#f39c12",   # orange — 0.5 ≤ confidence < 0.75
    "high": "#e74c3c",     # red    — confidence ≥ 0.75
}


def _conf_color(conf: float) -> str:
    if conf < 0.5:
        return LABEL_COLORS["low"]
    elif conf < 0.75:
        return LABEL_COLORS["medium"]
    return LABEL_COLORS["high"]


def _run_cnn_pipeline(
    original_pil: Image.Image,
    report_text: Optional[str],
    device_str: str,
    threshold: float,
    gradcam_method: str,
):
    """
    Full CNN inference + XAI pipeline.

    Returns an ExplanationResult.
    """
    from data.preprocessing import _pil_to_xrv_tensor
    from xai.unified import UnifiedExplainer

    encoder, device = _load_image_encoder(device_str)
    text_encoder, tokenizer, _ = _load_radbert(device_str) if report_text else (None, None, None)

    image_tensor = _pil_to_xrv_tensor(original_pil).unsqueeze(0).to(device)

    explainer = UnifiedExplainer(
        image_encoder=encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
        gradcam_method=gradcam_method,
        labels=encoder.target_labels,
    )

    with torch.no_grad():
        result = explainer.explain(
            image_tensor=image_tensor,
            original_pil=original_pil,
            report_text=report_text or None,
            threshold=threshold,
        )
    return result


def render_mode_a():
    """Render the Medical Imaging (Mode A) interface."""
    st.title("Medical Imaging — Chest X-Ray Analysis")
    st.markdown(
        "Upload a chest X-ray to get multi-label pathology predictions, "
        "Grad-CAM heatmaps, and full XAI attribution."
    )

    col_upload, col_results = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray", type=["jpg", "jpeg", "png"],
            help="Upload a frontal (PA or AP) chest X-ray in JPG or PNG format."
        )
        report_text = st.text_area(
            "Radiology Report (optional)",
            placeholder="Paste the radiology report text here for text-level XAI attribution …",
            height=120,
        )

        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if uploaded_file is None:
        with col_results:
            st.info("Upload a chest X-ray and click **Analyze** to begin.")
        return

    original_pil = Image.open(uploaded_file).convert("RGB")

    with col_upload:
        st.image(original_pil, caption="Uploaded X-Ray", use_column_width=True)

    if not analyze_btn:
        return

    with st.spinner("Running AI analysis …"):
        t0 = time.perf_counter()
        try:
            result = _run_cnn_pipeline(
                original_pil=original_pil,
                report_text=report_text.strip() or None,
                device_str=device_choice,
                threshold=threshold,
                gradcam_method=gradcam_method,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logger.exception("CNN pipeline error")
            return
        elapsed = (time.perf_counter() - t0) * 1000

    st.success(f"Analysis complete in **{elapsed:.0f} ms**")

    # Tabs for results
    tab_results, tab_xai, tab_explanation = st.tabs(
        ["📊 Predictions", "🔥 XAI Heatmap", "📝 Explanation"]
    )

    # ── Predictions tab ────────────────────────────────────────────────────
    with tab_results:
        st.subheader(
            f"Top Prediction: **{result.disease_name}** "
            f"({result.confidence*100:.1f}%)"
        )

        # All class predictions as progress bars
        sorted_preds = sorted(
            result.all_predictions.items(), key=lambda x: x[1], reverse=True
        )
        for label, conf in sorted_preds:
            color = _conf_color(conf)
            marker = "✅" if conf >= threshold else "  "
            st.markdown(
                f"{marker} **{label}**",
                help=f"Confidence: {conf*100:.1f}%",
            )
            st.markdown(
                f"<div style='background:{color}; width:{conf*100:.1f}%; "
                f"height:10px; border-radius:5px; margin-bottom:6px;'></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{conf*100:.1f}%")

    # ── XAI tab ───────────────────────────────────────────────────────────
    with tab_xai:
        col_orig, col_cam, col_tokens = st.columns(3)

        with col_orig:
            st.subheader("Original X-Ray")
            st.image(original_pil.resize((224, 224)), use_column_width=True)

        with col_cam:
            st.subheader(f"Grad-CAM ({gradcam_method})")
            if result.heatmap_overlay is not None:
                st.image(result.heatmap_overlay, use_column_width=True)
                st.caption(
                    f"Faithfulness score: **{result.faithfulness_score:.3f}**"
                )
            else:
                st.warning("Grad-CAM heatmap not available.")

        with col_tokens:
            st.subheader("Token Attribution (SHAP)")
            if result.token_shap:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                tokens = [t for t, _ in result.token_shap[:12]]
                scores = [s for _, s in result.token_shap[:12]]
                colors = ["#e74c3c" if s > 0 else "#3498db" for s in scores]
                fig, ax = plt.subplots(figsize=(4, 5))
                y = range(len(tokens))
                ax.barh(list(y), scores, color=colors, alpha=0.85)
                ax.set_yticks(list(y))
                ax.set_yticklabels(tokens, fontsize=9)
                ax.axvline(0, color="black", lw=0.8)
                ax.set_xlabel("Attribution Score", fontsize=9)
                ax.set_title("Token Importance", fontsize=10)
                ax.grid(True, alpha=0.3, axis="x")
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                st.image(buf, use_column_width=True)
            else:
                st.info("Provide a radiology report to see token attribution.")

    # ── Explanation tab ───────────────────────────────────────────────────
    with tab_explanation:
        st.subheader("Natural Language Explanation")
        st.markdown(result.nl_summary)

        col_faith, col_modal = st.columns(2)
        with col_faith:
            st.metric("Faithfulness Score", f"{result.faithfulness_score:.3f}",
                      help="Measures how much model confidence drops when top Grad-CAM "
                           "regions are occluded. Higher = more faithful.")

        with col_modal:
            img_pct = result.modality_contributions.get("image_pct", 100)
            txt_pct = result.modality_contributions.get("text_pct", 0)
            st.metric("Image Contribution", f"{img_pct:.1f}%")
            st.metric("Text Contribution", f"{txt_pct:.1f}%")

        # Export JSON
        st.markdown("---")
        json_str = result.as_json()
        st.download_button(
            "⬇ Download Full Explanation (JSON)",
            data=json_str,
            file_name=f"explanation_{result.disease_name.replace(' ', '_')}.json",
            mime="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mode B: Sentiment Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _run_sentiment_pipeline(text: str, device_str: str):
    """Run Mode B sentiment inference + SHAP attribution."""
    from xai.text_attribution import SHAPAttributor
    from xai.nlp_summary import NLExplainer
    from data.preprocessing import tokenize_tweet

    model, tokenizer, device = _load_sentiment(device_str)

    encoded = tokenize_tweet(text, tokenizer)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    binary_pred = int(out["binary_pred"][0])
    binary_conf = float(out["binary_conf"][0])
    probs = out["probs"][0].tolist()
    label = "POSITIVE" if binary_pred == 1 else "NEGATIVE"

    shap = SHAPAttributor(model, tokenizer, device=device)
    target_class = 2 if binary_pred == 1 else 0
    token_attrs = shap.explain(text, target_class=target_class)

    nl = NLExplainer()
    summary = nl.explain_text_prediction(label.lower(), binary_conf, token_attrs)

    return {
        "label": label,
        "confidence": binary_conf,
        "probs": probs,
        "token_attrs": token_attrs,
        "nl_summary": summary,
    }


def render_mode_b():
    """Render the Sentiment Analysis (Mode B) interface."""
    st.title("Sentiment Analysis — Text Classification")
    st.markdown(
        "Enter any tweet or sentence to get binary sentiment classification "
        "(positive/negative) with token-level attribution."
    )

    text_input = st.text_area(
        "Enter Text",
        placeholder="e.g. 'I love this product, it works perfectly!'",
        height=100,
    )
    analyze_btn = st.button("🔍 Analyze Sentiment", type="primary",
                             use_container_width=True)

    if not analyze_btn:
        if not text_input:
            st.info("Enter text and click **Analyze Sentiment** to begin.")
        return

    if not text_input.strip():
        st.warning("Please enter some text.")
        return

    with st.spinner("Analyzing sentiment …"):
        t0 = time.perf_counter()
        try:
            result = _run_sentiment_pipeline(text_input.strip(), device_choice)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logger.exception("Sentiment pipeline error")
            return
        elapsed = (time.perf_counter() - t0) * 1000

    st.success(f"Analysis complete in **{elapsed:.0f} ms**")

    # Big colored label
    color = "#2ecc71" if result["label"] == "POSITIVE" else "#e74c3c"
    st.markdown(
        f"""
        <div style="padding:1.5rem; background:{color}; color:white;
                    border-radius:10px; text-align:center; margin:1rem 0;">
            <h1 style="margin:0;">{result['label']}</h1>
            <p style="margin:0; font-size:1.3rem;">
                Confidence: {result['confidence']*100:.1f}%
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Raw 3-class probabilities
    probs = result["probs"]
    col_neg, col_neu, col_pos = st.columns(3)
    col_neg.metric("Negative", f"{probs[0]*100:.1f}%")
    col_neu.metric("Neutral", f"{probs[1]*100:.1f}%")
    col_pos.metric("Positive", f"{probs[2]*100:.1f}%")

    tab_xai, tab_exp = st.tabs(["🔍 Token Attribution", "📝 Explanation"])

    with tab_xai:
        token_attrs = result["token_attrs"]
        if token_attrs:
            st.subheader("Token Attribution (SHAP)")
            st.caption("Green = supports prediction. Red = contradicts prediction.")

            # Colour-highlighted text
            html_tokens = []
            attr_dict = dict(token_attrs)
            max_score = max(abs(s) for _, s in token_attrs) if token_attrs else 1.0

            # Simple highlighted display
            for tok, score in token_attrs[:20]:
                norm = score / (max_score + 1e-9)
                if norm > 0:
                    bg = f"rgba(46,204,113,{min(abs(norm),1)*0.8:.2f})"
                else:
                    bg = f"rgba(231,76,60,{min(abs(norm),1)*0.8:.2f})"
                html_tokens.append(
                    f"<span style='background:{bg};padding:2px 4px;border-radius:3px;"
                    f"margin:2px;display:inline-block;'>{tok}</span>"
                )
            st.markdown(" ".join(html_tokens), unsafe_allow_html=True)

            # Bar chart of top 10
            st.markdown("---")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top = token_attrs[:10]
            tokens = [t for t, _ in top]
            scores = [s for _, s in top]
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores]
            ax.barh(range(len(tokens)), scores, color=colors, alpha=0.85)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=10)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP Score", fontsize=10)
            ax.set_title("Top 10 Token Attributions", fontsize=11)
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            st.image(buf, use_column_width=True)

    with tab_exp:
        st.subheader("Natural Language Explanation")
        st.markdown(result["nl_summary"])


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

if "Medical" in mode:
    render_mode_a()
else:
    render_mode_b()
