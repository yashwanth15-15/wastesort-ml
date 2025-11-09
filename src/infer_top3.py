import numpy as np, tensorflow as tf, os
from PIL import Image

MODEL = "models/best_model.keras"
IMG = "data/test/R/R_9999.jpg"   # change path if needed
IMG_SIZE = (224,224)

model = tf.keras.models.load_model(MODEL)
classes = sorted([d for d in os.listdir("data/train") if os.path.isdir(os.path.join("data/train",d))])
img = Image.open(IMG).convert("RGB").resize(IMG_SIZE)
x = np.expand_dims(np.array(img)/255.0, 0)
probs = model.predict(x)[0]
top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
print("classes (sorted):", classes)
for idx, p in top3:
    print(f"{classes[idx]}: {p:.4f}")
