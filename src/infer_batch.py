import os, csv, numpy as np, tensorflow as tf
from PIL import Image

MODEL_PATH = "models/best_model.keras"
DATA_DIR = "data/test"
IMG_SIZE = (224,224)

model = tf.keras.models.load_model(MODEL_PATH)
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])

rows = []
for cls in classes:
    folder = os.path.join(DATA_DIR, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg",".jpeg",".png")):
            continue
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        x = np.expand_dims(np.array(img)/255.0, 0)
        probs = model.predict(x)[0]
        pred_idx = int(np.argmax(probs))
        rows.append([path, cls, classes[pred_idx], float(probs[pred_idx])])

out_csv = "predictions_test.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path","true_label","pred_label","pred_confidence"])
    writer.writerows(rows)

print("Wrote", out_csv)
