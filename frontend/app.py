import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import time

# Configuration
API_URL = "http://localhost:7779"
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4e8cff;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

# Sidebar - System Status
with st.sidebar:
    st.title("🫁 System Status")
    
    is_online, health_data = check_api_health()
    
    if is_online:
        st.success("✅ Backend Online")
        metrics = health_data.get("metrics", {})
        
        st.markdown("### Live Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", metrics.get("total_predictions", 0))
            st.metric("Avg Confidence", f"{metrics.get('average_confidence', 0)}%")
        with col2:
            st.metric("Avg Latency", f"{metrics.get('average_inference_time_ms', 0)}ms")
            st.metric("Error Rate", f"{metrics.get('error_rate_percent', 0)}%")
            
        st.markdown("---")
        st.markdown("### Model Info")
        st.text(f"Device: {health_data.get('device', 'unknown')}")
        st.text(f"Model Loaded: {'Yes' if health_data.get('model_loaded') else 'No'}")
    else:
        st.error("❌ Backend Offline")
        st.warning("Please start the backend server:\n`python main.py`")
        st.stop()

# Main Interface
st.title("Pneumonia Detection & Explainability Module")
st.markdown("Upload a chest X-ray to detect pneumonia presence and visualize affected areas using Grad-CAM.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload X-Ray")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        # Predict Button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Running AI Analysis..."):
                try:
                    # Reset file pointer for request
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['result'] = result
                        st.session_state['uploaded_image'] = image
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {str(e)}")

with col2:
    st.subheader("2. Analysis Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # 1. Prediction Label
        pred_class = result['prediction']
        confidence = result['confidence']
        
        color = "red" if pred_class == "PNEUMONIA" else "green"
        st.markdown(f"""
            <div style="padding: 1rem; background-color: {color}; color: white; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <h2 style="margin:0;">{pred_class}</h2>
                <p style="margin:0; font-size: 1.2rem;">Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. Probability Chart
        probs = result['probabilities']
        df_probs = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability'])
        
        fig = px.bar(df_probs, x='Probability', y='Class', orientation='h', 
                     text='Probability', color='Class',
                     color_discrete_map={'NORMAL': '#2ecc71', 'PNEUMONIA': '#e74c3c'})
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Explainability (Grad-CAM)
        st.subheader("3. Visual Explanation (Grad-CAM)")
        heatmap_id = result['heatmap_id']
        heatmap_url = f"{API_URL}/explain/{heatmap_id}"
        
        try:
            heatmap_response = requests.get(heatmap_url)
            if heatmap_response.status_code == 200:
                heatmap_image = Image.open(io.BytesIO(heatmap_response.content))
                
                # Comparison
                # st.image(heatmap_image, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
                
                # Side by side comparison in a tab or expander
                
                tab1, tab2 = st.tabs(["Overlay View", "Side-by-Side"])
                
                with tab1:
                    st.image(heatmap_image, caption="AI Attention Heatmap (Red = High Attention)", use_column_width=True)
                    
                with tab2:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(st.session_state['uploaded_image'], caption="Original", use_column_width=True)
                    with c2:
                        st.image(heatmap_image, caption="Heatmap", use_column_width=True)
                        
            else:
                st.warning("Could not load heatmap explanation.")
        except Exception as e:
            st.error("Error loading heatmap.")
            
    else:
        st.info("Upload an image and click 'Analyze' to see results.")
