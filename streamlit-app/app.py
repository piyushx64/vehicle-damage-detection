import streamlit as st
from model_helper import predict  # Your actual model
from PIL import Image
import os
import time
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Damaged Car Detection Using machine learning",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for stunning design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 20%, #1a1a2e, #16213e);
    }

    /* Animated Background */
    .background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
    }

    .gradient-sphere {
        position: absolute;
        width: 800px;
        height: 800px;
        background: radial-gradient(circle, rgba(255,71,87,0.2) 0%, rgba(255,71,87,0) 70%);
        border-radius: 50%;
        top: -400px;
        right: -200px;
        animation: float 20s ease-in-out infinite;
    }

    .gradient-sphere-2 {
        position: absolute;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(0,227,150,0.15) 0%, rgba(0,227,150,0) 70%);
        border-radius: 50%;
        bottom: -300px;
        left: -200px;
        animation: float 15s ease-in-out infinite reverse;
    }

    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, 30px) scale(1.1); }
    }

    /* Particles */
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: rgba(255,255,255,0.3);
        border-radius: 50%;
        animation: particleFloat 10s linear infinite;
    }

    @keyframes particleFloat {
        0% {
            transform: translateY(100vh) translateX(0);
            opacity: 0;
        }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% {
            transform: translateY(-100vh) translateX(100px);
            opacity: 0;
        }
    }

    /* Header Section */
    .header {
        padding: 4rem 2rem;
        text-align: center;
        position: relative;
    }

    .header h1 {
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff4757, #00e396, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: titleGlow 3s ease-in-out infinite;
    }

    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(255,71,87,0.3)); }
        50% { filter: drop-shadow(0 0 40px rgba(0,227,150,0.5)); }
    }

    .header p {
        font-size: 1.3rem;
        color: #a0a0a0;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Stats Container */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 3rem 0;
        flex-wrap: wrap;
    }

    .stat-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-10px);
        background: rgba(255,255,255,0.1);
        border-color: #ff4757;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff4757, #00e396);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: #a0a0a0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Main Container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Upload Section */
    .upload-section {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 30px;
        padding: 2rem;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .section-title::before {
        content: '';
        width: 5px;
        height: 30px;
        background: linear-gradient(180deg, #ff4757, #00e396);
        border-radius: 5px;
    }

    /* Upload Box */
    .upload-box {
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .upload-box:hover {
        border-color: #ff4757;
        background: rgba(255,71,87,0.05);
    }

    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s ease-in-out infinite;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }

    .upload-text {
        color: white;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    .upload-hint {
        color: #a0a0a0;
        font-size: 0.9rem;
    }

    /* Image Preview */
    .image-preview {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        margin-top: 2rem;
        position: relative;
    }

    .image-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
        padding: 1rem;
        color: white;
    }

    /* Results Section */
    .results-section {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 30px;
        padding: 2rem;
    }

    /* Damage Card */
    .damage-card {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin-bottom: 2rem;
        animation: slideIn 0.5s ease;
    }

    .normal-card {
        background: linear-gradient(135deg, #00e396, #00d4a0);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin-bottom: 2rem;
        animation: slideIn 0.5s ease;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .result-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: #ff4757;
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem;
        color: #a0a0a0;
        font-size: 0.9rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2.5rem;
        }

        .stats-container {
            gap: 1rem;
        }

        .stat-card {
            padding: 1rem 2rem;
        }
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff4757, #ff6b81) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 20px 40px rgba(255,71,87,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Add animated background
st.markdown("""
<div class="background">
    <div class="gradient-sphere"></div>
    <div class="gradient-sphere-2"></div>
""", unsafe_allow_html=True)

# Add particles
for i in range(20):
    left = np.random.randint(0, 100)
    delay = np.random.randint(0, 10)
    st.markdown(f"""
    <div class="particle" style="left: {left}%; animation-delay: {delay}s;"></div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header">
    <h1>🔍Damaged Car Detection Using machine learning</h1>
    <p>Our mentor, Dr. S. K. Pandey
Assistant professor of
Department of Computer Science and Engineering, L.N.J.P.I.T. Chapra</p>
</div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
<div class="stats-container">
    <div class="stat-card">
        <div class="stat-number">2300+</div>
        <div class="stat-label">Images</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">6</div>
        <div class="stat-label">Damage Categories</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">⚡</div>
        <div class="stat-label">Real-time Detection</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📸 Upload Vehicle Image</div>', unsafe_allow_html=True)

# File uploader with custom styling
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    key="file_uploader",
    label_visibility="collapsed"
)

if not uploaded_file:
    st.markdown("""
    <div class="upload-box">
        <div class="upload-icon">📤</div>
        <div class="upload-text">Drag and drop or click to upload</div>
        <div class="upload-hint">Supports JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    # Save uploaded file
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image preview
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div class="image-overlay">
            <span>📏 {image.size[0]} x {image.size[1]}px</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border-radius: 20px; padding: 2rem; height: 100%;">
            <h3 style="color: white; margin-bottom: 1.5rem;">🔍 Analysis</h3>
        """, unsafe_allow_html=True)

        # Analyze button - ONLY ONE BUTTON
        if st.button("🔬 Detect Damage", use_container_width=True):
            with st.spinner("🤖 Analyzing image with AI model..."):
                # Simple progress animation (just for visual)
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Use YOUR ACTUAL MODEL for prediction
                prediction = predict(image_path)

                progress_bar.empty()

                # Store in session state
                st.session_state['prediction'] = prediction
                st.session_state['analyzed'] = True

        st.markdown("</div>", unsafe_allow_html=True)

    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

st.markdown('</div>', unsafe_allow_html=True)

# Results Section - Only shows when analysis is done
if 'analyzed' in st.session_state and st.session_state['analyzed']:
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Analysis Result</div>', unsafe_allow_html=True)

    prediction = st.session_state['prediction']

    # Main result card - using REAL prediction
    if "Normal" in prediction:
        st.markdown(f"""
        <div class="normal-card">
            <div class="result-icon">✅</div>
            <div class="result-title">No Damage Detected</div>
            <div class="result-subtitle">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.info("✓ Vehicle appears to be in good condition")

    else:
        st.markdown(f"""
        <div class="damage-card">
            <div class="result-icon">⚠️</div>
            <div class="result-title">Damage Detected</div>
            <div class="result-subtitle">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        st.warning("! Professional inspection recommended")

    # New Analysis button
    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state['analyzed'] = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🚗 7th sem minor project</p>
</div>
""", unsafe_allow_html=True)