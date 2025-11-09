# app_pretty_v3.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO
import os, base64

st.set_page_config(page_title="Waste Sort Classifier", layout="centered", initial_sidebar_state="collapsed")

# ---- CSS ----
st.markdown(
    """
    <style>
    :root { --bg: #0b0f14; --card: #0f1720; --muted: #94a3b8; --accent: #10b981; --panel: #0b1220; }
    .main > .block-container{padding-top:20px;padding-bottom:20px;}
    body { background: linear-gradient(180deg,#071018 0%, #071423 100%); color: #e6eef6; }
    .app-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:16px; box-shadow: 0 6px 20px rgba(2,6,23,0.6); }
    .header { display:flex; align-items:center; gap:12px; }
    @keyframes pulseGlow { 0% { box-shadow: 0 0 5px rgba(16,185,129,0.4); } 50% { box-shadow: 0 0 15px rgba(16,185,129,0.8);} 100% { box-shadow: 0 0 5px rgba(16,185,129,0.4);} }
    .logo { width:48px;height:48px;border-radius:8px; background: rgba(16,185,129,0.08); padding:6px; animation: pulseGlow 2s infinite; }
    .muted { color: var(--muted); font-size:0.9rem; }
    .pred { font-weight:700; font-size:1.05rem; color:#e6eef6; }
    .small { font-size:0.9rem; color:var(--muted); }
    .legend { background: rgba(16,185,129,0.06); border-radius: 8px; padding: 10px; margin-top: 10px; }
    .thumb { border-radius:8px; box-shadow: 0 4px 14px rgba(2,6,23,0.6); }
    .right-card { min-height:260px; display:flex; flex-direction:column; justify-content:start; gap:8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Pillow safety ----
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- Model loader ----
@st.cache_resource
def load_model(path="models/best_model.keras"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return tf.keras.models.load_model(path)

# attempt to load (will raise in UI if missing)
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Cannot load model: {e}")
    st.stop()

IMG_SIZE = (224, 224)

# classes inferred from data/train folder; fallback names if not found
train_dir = "data/train"
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]) if os.path.isdir(train_dir) else []
if not classes:
    classes = ["Organic", "Recyclable"]

# ---- logo loader (base64) ----
logo_path = os.path.join("assets", "logo.png")
def get_base64_logo(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

logo_b64 = get_base64_logo(logo_path)

# ---- Header ----
st.markdown('<div class="app-card">', unsafe_allow_html=True)
if logo_b64:
    st.markdown(
        f"""
        <div class="header">
            <img src="data:image/png;base64,{logo_b64}" alt="logo" class="logo"/>
            <div>
                <h1 style="margin:0;">Waste Sort Classifier</h1>
                <div class="muted" style="margin-top:-4px;">AI — Organic (O) vs Recyclable (R)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="header">
            <div class="logo"></div>
            <div>
                <h1 style="margin:0;">Waste Sort Classifier</h1>
                <div class="muted" style="margin-top:-4px;">AI — Organic (O) vs Recyclable (R)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ---- Legend about labels ----
st.markdown(
    """
    <div class="legend">
        <b>🧾 Label Guide</b><br>
        <b>O → Organic</b> — biodegradable items (food scraps, fruits, vegetables).<br>
        <b>R → Recyclable</b> — plastics, bottles, paper, metals, etc.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---- Layout ----
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("Upload an image")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a cropped image of the item (recommended).", key="upv3")
    st.markdown('<div class="small muted">Tip: crop to the object for best results. You can also try the webcam app.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # sample thumbnails
    st.write("")
    if os.path.isdir("data/test"):
        st.subheader("Sample test images")
        thumbs = []
        try:
            for cls in sorted(os.listdir("data/test")):
                folder = os.path.join("data/test", cls)
                if not os.path.isdir(folder): continue
                for fname in os.listdir(folder)[:2]:
                    thumbs.append((cls, os.path.join(folder, fname)))
            if thumbs:
                cols = st.columns(4)
                for i, (cls, p) in enumerate(thumbs[:8]):
                    try:
                        cols[i % 4].image(Image.open(p).convert("RGB").resize((200,140)), caption=f"{cls}/{os.path.basename(p)}", use_column_width=True)
                    except Exception:
                        pass
        except Exception:
            pass

with right:
    st.markdown('<div class="app-card right-card">', unsafe_allow_html=True)
    st.subheader("Prediction")
    placeholder_img = st.empty()
    pred_area = st.empty()
    bar = st.empty()
    probs = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# ---- helpers ----
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

def predict(pil_img: Image.Image):
    x = preprocess(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return idx, float(preds[idx]), preds

# robust loader for uploads
def load_image_from_upload(uploaded_file):
    st.write(f"Uploaded: filename={uploaded_file.name}, type={uploaded_file.type}, size={uploaded_file.size} bytes")
    data = uploaded_file.read()
    if not data:
        raise ValueError("Uploaded file is empty.")
    buf = BytesIO(data)
    img = Image.open(buf).convert("RGB")
    return img

def show_results(img: Image.Image):
    placeholder_img.image(img, use_column_width=True)
    idx, conf, preds = predict(img)
    pred_area.markdown(f"<div class='pred'>✅ Predicted: {classes[idx]}  <span style='color:#10B981'>{conf*100:.2f}%</span></div>", unsafe_allow_html=True)
    bar.progress(min(max(conf, 0.0), 1.0))
    rows = [{"Class": classes[i], "Probability (%)": f"{preds[i]*100:.2f}"} for i in range(len(classes))]
    probs.table(rows)

# ---- main action ----
if uploaded:
    try:
        img = load_image_from_upload(uploaded)
        show_results(img)
    except Exception as e:
        st.error("❌ Could not read the uploaded image.")
        st.markdown(f"**Error (debug):** `{type(e).__name__}: {str(e)}`")
        import traceback, sys
        traceback.print_exc(file=sys.stdout)
else:
    placeholder_img.info("No image selected — use the uploader on the left or try sample images above.")

st.markdown("")
st.markdown("---")
st.markdown('<div class="small muted">Want a live webcam demo? Run <code>streamlit run app_test_io.py</code>.</div>', unsafe_allow_html=True)
