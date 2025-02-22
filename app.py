import streamlit as st
import urllib.request
import tensorflow as tf
from keras.models import load_model

@st.cache_resource
def load_trained_model():
    model_url = "https://raw.githubusercontent.com/rafaqatkhan-ai/fire-smoke/main/bilkent.weights.h5"
    model_path = "bilkent.weights.h5"
    
    # Download model if not already present
    urllib.request.urlretrieve(model_url, model_path)
    
    # Load the model
    model = load_model(model_path)
    return model

# Load the model
model = load_trained_model()

st.title("Fire and Smoke Detection")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded File", use_column_width=True)

    # Placeholder for processing (replace with actual classification logic)
    st.write("Processing... (Add model inference logic here)")
