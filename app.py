import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import imghdr
from keras.models import Model
from keras.layers import (
    Dense, Dropout, Conv3D, Input, MaxPool3D, Activation, GlobalAveragePooling3D,
    Add, BatchNormalization, LayerNormalization, MultiHeadAttention, Layer
)
from keras.regularizers import l2

# Define ResNet Block
def resnet_block(inputs, filters, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_decay=0.005):
    shortcut = inputs
    x = Conv3D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    
    if inputs.shape[1:] != x.shape[1:]:
        shortcut = Conv3D(filters, kernel_size=(1, 1, 1), strides=stride, padding='same', kernel_regularizer=l2(weight_decay))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Transformer Block
def transformer_block(inputs, num_heads=2, key_dim=32, ff_dim=128, dropout_rate=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x, x)
    x = Dropout(dropout_rate)(x)
    res = Add()([x, inputs])  # Residual connection
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])  # Residual connection

# Positional Gating Unit
class PositionalGatingUnit(Layer):
    def __init__(self, gating_type="spatio-temporal", **kwargs):
        super(PositionalGatingUnit, self).__init__(**kwargs)
        self.gating_type = gating_type

    def call(self, inputs):
        if self.gating_type == "temporal":
            return tf.reduce_mean(inputs, axis=1, keepdims=True)
        elif self.gating_type == "spatial":
            return tf.reduce_mean(inputs, axis=(2, 3), keepdims=True)
        elif self.gating_type == "spatio-temporal":
            return tf.reduce_mean(inputs, axis=(1, 2, 3), keepdims=True)
        else:
            raise ValueError("Invalid gating type")

# DAMR_3DNet_PosMLP Model
def damr_3dnet_posmlp(input_shape):
    weight_decay = 0.005
    inputs = Input(input_shape)

    x = Conv3D(32, (3, 3, 3), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resnet_block(x, 32)
    x = MaxPool3D((2, 2, 2))(x)
    x = resnet_block(x, 64)
    x = MaxPool3D((2, 2, 2))(x)
    x = transformer_block(x, num_heads=4, key_dim=64)

    x = PositionalGatingUnit(gating_type="spatio-temporal")(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(weight_decay))(x)

    return Model(inputs, outputs)

# Load trained model weights
def load_trained_model():
    input_shape = (16, 128, 128, 3)  # Shape used during training
    model = damr_3dnet_posmlp(input_shape)
    model.load_weights("mivia_full_model.h5")  # Ensure this file is present in the app's directory
    return model

# Preprocess function for video
def preprocess_video(video_frames):
    processed_frames = np.zeros((16, 128, 128, 3), dtype='float32')

    for i, frame in enumerate(video_frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        processed_frames[i] = frame

    processed_frames[..., 0] -= 99.9
    processed_frames[..., 1] -= 92.1
    processed_frames[..., 2] -= 82.6
    processed_frames[..., 0] /= 65.8
    processed_frames[..., 1] /= 62.3
    processed_frames[..., 2] /= 60.3

    return np.expand_dims(processed_frames, axis=0)  # (1, 16, 128, 128, 3)

# Preprocess function for image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))

    image[..., 0] -= 99.9
    image[..., 1] -= 92.1
    image[..., 2] -= 82.6
    image[..., 0] /= 65.8
    image[..., 1] /= 62.3
    image[..., 2] /= 60.3

    return np.expand_dims(image, axis=0)  # (1, 128, 128, 3)

# Extract 16 frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in np.linspace(0, total_frames - 1, 16, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if len(frames) < 16:
        return None
    return frames

# Streamlit App
st.title("🔥 Fire and Smoke Detection AI 🔥")

st.sidebar.header("Upload Image or Video")
uploaded_file = st.sidebar.file_uploader("Upload an image or video file", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_type = imghdr.what(uploaded_file)
    is_image = file_type in ["jpeg", "png"]

    model = load_trained_model()

    if is_image:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        processed_input = preprocess_image(image)
        processed_input = np.expand_dims(processed_input, axis=0)  # (1, 1, 128, 128, 3)
    else:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        frames = extract_frames(temp_video_path)
        processed_input = preprocess_video(frames) if frames else None

    if processed_input is not None:
        prediction = model.predict(processed_input)[0][0]
        result = "🔥 Fire Detected" if prediction > 0.5 else "✅ Normal"
        st.subheader("Prediction Result:")
        st.write(f"**{result}** (Confidence: {prediction:.2f})")
    else:
        st.error("🚨 Not enough frames extracted from the video.")
