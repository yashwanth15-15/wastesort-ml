# app_test_io.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
import io, os

st.set_page_config(page_title="IO Test", layout="centered")
ImageFile.LOAD_TRUNCATED_IMAGES = True

@st.cache_resource
def load_model():
    path = "models/best_model.keras"
    if not os.path.exists(path):
        st.error("❌ Model file not found at models/best_model.keras")
        st.stop()
    return tf.keras.models.load_model(path)

# load model once
model = load_model()
IMG_SIZE = (224, 224)
classes = ["Organic", "Recyclable"]

def preprocess(pil_img):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img) / 255.0
    return np.expand_dims(arr, 0)

def predict(pil_img):
    x = preprocess(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return classes[idx], float(preds[idx]), preds

st.title("🧪 WasteSort I/O Test")

tab1, tab2 = st.tabs(["📁 Upload", "📸 Webcam"])

with tab1:
    st.subheader("Upload a test image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.write(f"Uploaded: {uploaded_file.name}, {uploaded_file.type}, {uploaded_file.size} bytes")
        try:
            bytes_data = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)

            label, conf, preds = predict(pil_img)
            st.success(f"✅ Prediction: {label} ({conf*100:.2f}%)")

        except Exception as e:
            st.error(f"❌ Error (debug): {repr(e)}")

with tab2:
    st.subheader("Capture from webcam")
    picture = st.camera_input("Take a photo")
    if picture:
        st.write("Captured from webcam")
        try:
            pil_img = Image.open(picture).convert("RGB")
            st.image(pil_img, caption="Webcam Image", use_column_width=True)

            label, conf, preds = predict(pil_img)
            st.success(f"✅ Prediction: {label} ({conf*100:.2f}%)")
        except Exception as e:
            st.error(f"❌ Webcam Error (debug): {repr(e)}")
