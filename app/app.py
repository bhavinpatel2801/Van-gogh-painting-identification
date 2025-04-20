import streamlit as st
import torch
from PIL import Image
import sys
sys.path.append('../src')  # Ensure access to custom modules

# === Import project modules ===
from predict import load_best_models, predict_from_ensemble

# --- Constants ---
CLASS_NAMES = ["Not Van Gogh ❌", "Van Gogh 🎨"]

# === Load models only once using Streamlit cache ===
@st.cache_resource
def load_models():
    # Load both full-image and patch-based models with their paths
    return load_best_models()

# === Streamlit UI ===
st.title("🎨 Van Gogh Painting Identifier (Ensemble-Based)")
st.write("Upload a painting to see if it's likely painted by Vincent van Gogh based on deep learning models!")

# === File upload component ===
uploaded_file = st.file_uploader("Upload an image of a painting", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Load models (cached)
        model_full, path_full, model_patch, path_patch = load_models()

        # Run ensemble prediction
        label, confidence = predict_from_ensemble(
            image_path=uploaded_file,  # works since uploaded_file behaves like a file-like object
            model_full=model_full,
            best_full_model_path=path_full,
            model_patch=model_patch,
            best_patch_model_path=path_patch
        )

        # Display results
        st.success(f"🎯 Prediction: **{label}** with {confidence*100:.2f}% confidence")
