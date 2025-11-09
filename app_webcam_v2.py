# app_webcam_v2.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Waste Sort Live", layout="centered", initial_sidebar_state="collapsed")

# simple CSS
st.markdown("""
<style>
body { background: #071423; color: #e6eef6; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; box-shadow:0 8px 24px rgba(0,0,0,0.6); }
.small { color:#94a3b8; font-size:0.95rem; }
.center { display:flex; justify-content:center; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path="models/best_model.keras"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(path)

model = load_model()
classes = sorted([d for d in os.listdir("data/train") if os.path.isdir(os.path.join("data/train", d))])
if not classes:
    classes = ["Organic", "Recyclable"]
IMG_SIZE = (224,224)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("Waste Sort — Live Demo")
st.markdown('<div class="small">Use the webcam or upload an image. Webcam may need browser permission.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    cam = st.camera_input("Or take a photo")
with col2:
    preview = st.empty()
    info = st.empty()
    probs = st.empty()

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)/255.0
    return np.expand_dims(arr, 0)

def infer_and_show(pil_img):
    preview.image(pil_img, use_container_width=True)
    x = preprocess(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    info.markdown(f"### ✅ {classes[idx]}  •  Confidence: {conf*100:.2f}%")
    probs.table([{ "Class": classes[i], "Prob (%)": f"{preds[i]*100:.2f}" } for i in range(len(classes))])

if cam is not None:
    img = Image.open(cam)
    infer_and_show(img)
elif uploaded is not None:
    img = Image.open(uploaded)
    infer_and_show(img)
else:
    preview.info("No input yet — upload or use the camera.")
