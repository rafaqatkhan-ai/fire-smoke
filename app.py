import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import imghdr  # Detects image type
from keras.models import Model

# Load trained model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("mivia_full_model.h5", compile=False)  # Use proper model file
    return model

# Check if the uploaded file is an image
def is_image_file(uploaded_file):
    file_bytes = uploaded_file.read(32)  # Read first 32 bytes
    file_type = imghdr.what(None, h=file_bytes)  # Detect type
    uploaded_file.seek(0)  # Reset pointer after reading
    return file_type in ["jpeg", "png", "jpg"]

# Preprocess a single image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Resize to model input size
    image = image.astype('float32')

    # Normalize (same as training)
    image[..., 0] -= 99.9
    image[..., 1] -= 92.1
    image[..., 2] -= 82.6
    image[..., 0] /= 65.8
    image[..., 1] /= 62.3
    image[..., 2] /= 60.3

    return np.expand_dims(image, axis=0)  # Add batch dimension

# Extract 16 frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("🚨 Error: Unable to open the video file.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames is None or total_frames < 16:
        st.error("🚨 Error: Not enough frames in the video.")
        cap.release()
        return None

    frames = []
    for i in np.linspace(0, total_frames - 1, 16, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames if len(frames) == 16 else None

# Streamlit App
st.title("🔥 Fire and Smoke Detection AI 🔥")

st.sidebar.header("Upload Image or Video")
uploaded_file = st.sidebar.file_uploader("Upload an image or video file", type=["mp4", "avi", "mov", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Check if uploaded file is an image
    if is_image_file(uploaded_file):
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # ✅ Updated

        # Read and preprocess image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        processed_input = preprocess_image(image)

        # Load model and make prediction
        model = load_trained_model()
        prediction = model.predict(processed_input)[0][0]
        result = "🔥 Fire/Smoke Detected" if prediction > 0.5 else "✅ Normal"

        st.subheader("Prediction Result:")
        st.write(f"**{result}** (Confidence: {prediction:.2f})")

    else:  # Process video
        st.sidebar.video(uploaded_file)

        # Save uploaded video
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        frames = extract_frames(temp_video_path)
        if frames:
            st.write("✅ Extracted 16 frames for prediction.")

            # Load model and preprocess frames
            model = load_trained_model()
            processed_input = preprocess_video(frames)

            # Make prediction
            prediction = model.predict(processed_input)[0][0]
            result = "🔥 Fire/Smoke Detected" if prediction > 0.5 else "✅ Normal"

            st.subheader("Prediction Result:")
            st.write(f"**{result}** (Confidence: {prediction:.2f})")
        else:
            st.error("🚨 Not enough frames extracted from the video. Try another video.")
