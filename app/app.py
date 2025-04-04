import streamlit as st
import torch
from PIL import Image

import sys
sys.path.append('../src')

from trainer import get_model
from preprocessing_patches import  predict_image_patches


# --- Constants ---
MODEL_PATH = '../models/efficientnet_best_patch.pth'
CLASS_NAMES = ["Non-Van Gogh", "Van Gogh"]

# --- Load Model Once ---
@st.cache_resource
def load_model():
    model = get_model('efficientnet')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Streamlit UI ---
st.title("🎨 Van Gogh Painting Identifier (Patch-Based)")
st.write("Upload a painting and find out if it was painted by Vincent van Gogh using deep learning!")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        model = load_model()
        label, confidence = predict_image_patches(image, model)
        st.success(f"🎯 Prediction: **{CLASS_NAMES[label]}** with {confidence*100:.2f}% confidence")
