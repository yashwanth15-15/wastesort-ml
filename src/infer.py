import argparse
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Path to the image for prediction")
args = parser.parse_args()

# --- Load model ---
model = tf.keras.models.load_model("models/best_model.keras")

# --- Load and preprocess image ---
img_path = Path(args.image_path)
if not img_path.exists():
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224)) / 255.0
img_array = np.expand_dims(img_resized, axis=0)

# --- Predict ---
pred = model.predict(img_array)
class_names = ["Organic", "Recyclable"]
pred_class = class_names[int(pred[0][0] > 0.5)]
confidence = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]

print(f"✅ Predicted: {pred_class} ({confidence * 100:.2f}% confidence)")
