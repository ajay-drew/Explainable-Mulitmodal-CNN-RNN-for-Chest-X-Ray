"""
Streamlit Web UI for Explainable Multimodal Chest X-Ray Diagnosis.

Per PROJECT_PLAN §3.7:
- Simple web-based UI for uploads, predictions, and explanations

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide",
)


def load_model():
    """Load the multimodal classifier (cached)."""
    # TODO: Implement actual model loading
    # This is a placeholder
    st.warning("Model loading not implemented. Please train a model first.")
    return None


def main():
    st.title("🫁 Explainable Chest X-Ray Diagnosis")
    st.markdown("""
    Upload a chest X-ray image and radiology report for multi-label disease classification
    with explainable AI visualizations.
    
    **Features:**
    - Multi-label classification (14 conditions)
    - Grad-CAM heatmaps for image regions
    - Token attribution for report text
    - Unified multimodal explanations
    """)
    
    st.divider()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        
        show_explanations = st.checkbox("Show Explanations", value=True)
        
        st.divider()
        st.markdown("### Model Info")
        st.markdown("""
        - **Image Encoder:** DenseNet121 (MIMIC-CXR)
        - **Text Encoder:** RadBERT-RoBERTa
        - **Fusion:** Attention-based
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Inputs")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload Chest X-Ray",
            type=["jpg", "jpeg", "png", "dcm"],
            help="Upload a frontal chest X-ray image",
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("L")
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        # Report input
        report_text = st.text_area(
            "Radiology Report",
            height=200,
            placeholder="Enter the radiology report text here...\n\nExample: Patient presents with shortness of breath. Chest X-ray shows bilateral infiltrates and possible consolidation in the right lower lobe.",
        )
    
    with col2:
        st.header("🔬 Results")
        
        if uploaded_image is not None and report_text:
            # Run inference button
            if st.button("🚀 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    # TODO: Implement actual inference
                    # This is a placeholder
                    
                    st.success("Analysis complete!")
                    
                    # Placeholder predictions
                    st.subheader("Predictions")
                    
                    diseases = [
                        ("Atelectasis", 0.72),
                        ("Cardiomegaly", 0.15),
                        ("Consolidation", 0.68),
                        ("Edema", 0.23),
                        ("Pleural Effusion", 0.45),
                        ("Pneumonia", 0.81),
                        ("Pneumothorax", 0.08),
                        ("No Finding", 0.12),
                    ]
                    
                    for disease, prob in diseases:
                        if prob >= threshold:
                            st.markdown(f"**{disease}**: {prob:.2%} ✅")
                        else:
                            st.markdown(f"{disease}: {prob:.2%}")
                    
                    # Explanations
                    if show_explanations:
                        st.divider()
                        st.subheader("Explanations")
                        
                        exp_col1, exp_col2 = st.columns(2)
                        
                        with exp_col1:
                            st.markdown("**Image Heatmap (Grad-CAM)**")
                            st.info("Heatmap visualization would appear here")
                            # TODO: Display actual Grad-CAM overlay
                        
                        with exp_col2:
                            st.markdown("**Key Report Phrases**")
                            st.markdown("""
                            - "shortness of breath" → Pneumonia (+0.42)
                            - "consolidation" → Consolidation (+0.38)
                            - "infiltrates" → Atelectasis (+0.31)
                            """)
                        
                        st.markdown("**Modality Contributions**")
                        contrib_col1, contrib_col2 = st.columns(2)
                        with contrib_col1:
                            st.metric("Image", "62%")
                        with contrib_col2:
                            st.metric("Text", "38%")
        else:
            st.info("👆 Upload an image and enter a report to analyze")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray;">
        Explainable Multimodal CNN-RNN for Chest X-Ray Diagnosis<br>
        See <a href="#">PROJECT_PLAN.md</a> for details
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
