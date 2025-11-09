import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/best_model.keras")
    return model

model = load_model()

# App title
st.title("♻️ Waste Sort Classifier")
st.write("Upload an image to classify as **Organic (O)** or **Recyclable (R)**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When user uploads an image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    classes = ["Organic", "Recyclable"]
    pred_idx = np.argmax(preds[0])
    conf = preds[0][pred_idx]

    # Show result
    st.subheader(f"✅ Predicted: **{classes[pred_idx]}** ({conf * 100:.2f}% confidence)")
