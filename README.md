# ♻️ WasteSort-ML — Smart Waste Classification using Deep Learning

WasteSort-ML is an end-to-end **Machine Learning + Deep Learning** project that classifies waste into **Organic (O)** and **Recyclable (R)** categories using **TensorFlow (MobileNetV2)**.  
It also includes a beautiful **Streamlit web app** that allows users to classify waste images through **upload or webcam** — perfect for sustainability demos and ML showcases.

---

## 🚀 Features

- ✅ Deep Learning model trained using **MobileNetV2 (Transfer Learning)**  
- ✅ **Streamlit Web App** (`app_pretty_v3.py`) with dark theme and intuitive design  
- ✅ Real-time webcam-based waste detection  
- ✅ Lightweight — runs smoothly on CPU (no GPU required)  
- ✅ Modular Python scripts for **training, inference, and batch evaluation**  
- ✅ 94–95% validation accuracy  

---

## 🗂️ Project Structure

# ♻️ WasteSort-ML — Smart Waste Classification using Deep Learning

WasteSort-ML is an end-to-end **Machine Learning + Deep Learning** project that classifies waste into **Organic (O)** and **Recyclable (R)** categories using **TensorFlow (MobileNetV2)**.  
It also includes a beautiful **Streamlit web app** that allows users to classify waste images through **upload or webcam** — perfect for sustainability demos and ML showcases.

---

## 🚀 Features

- ✅ Deep Learning model trained using **MobileNetV2 (Transfer Learning)**  
- ✅ **Streamlit Web App** (`app_pretty_v3.py`) with dark theme and intuitive design  
- ✅ Real-time webcam-based waste detection  
- ✅ Lightweight — runs smoothly on CPU (no GPU required)  
- ✅ Modular Python scripts for **training, inference, and batch evaluation**  
- ✅ 94–95% validation accuracy  

---

## 🗂️ Project Structure
wastesort-ml/
│
├── assets/
│ └── logo.png # App logo
│
├── models/
│ └── best_model.keras # Trained model file
│
├── src/
│ ├── train.py # Train model using MobileNetV2
│ ├── split_data.py # Split dataset into train/val/test
│ ├── infer.py # Predict a single image
│ ├── infer_batch.py # Predict batch of test images
│ ├── summarize_preds.py # Evaluate test accuracy
│ ├── show_classes.py # Show dataset class indices
│ └── collect_images.py # Optional helper script
│
├── app_pretty_v3.py # Streamlit UI (upload version)
├── app_webcam_v2.py # Streamlit UI (webcam version)
├── requirements.txt # Dependencies
├── README.md # Documentation (you are here)
└── .gitignore # Ignore unnecessary files


---

## 🧩 Dataset Details

The dataset is organized into two classes:

| Label | Meaning | Example |
|--------|----------|---------|
| `O` | **Organic Waste** (fruits, food leftovers, plants, etc.) | 🍌🍎🥬 |
| `R` | **Recyclable Waste** (plastic bottles, glass, metal cans, etc.) | ♻️🧴📦 |

Dataset directory structure:


data/
├── train/
│ ├── O/
│ └── R/
├── val/
│ ├── O/
│ └── R/
└── test/
├── O/
└── R/


---

## 🧠 Model Overview

| Property | Value |
|-----------|--------|
| **Base Model** | MobileNetV2 (Transfer Learning) |
| **Input Size** | 224×224×3 |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Accuracy (Validation)** | ≈94.5% |
| **Framework** | TensorFlow 2.12.0 (CPU) |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/yashwanth15-15/wastesort-ml.git
cd wastesort-ml

2️⃣ Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
🖼️ For Image Upload Mode:
streamlit run app_pretty_v3.py

🎥 For Webcam Mode:
streamlit run app_webcam_v2.py

🧪 Testing and Inference
🔹 Single Image Test
python src/infer.py "data/test/O/O_2.jpg"

🔹 Batch Prediction (All test images)
python src/infer_batch.py

🔹 Evaluation Report
python src/summarize_preds.py

📊 Training (Optional)

If you want to retrain from scratch:

python src/train.py --data_dir data --out_path models/best_model.keras --epochs 6

🖼️ App Preview
Upload Image	Get Prediction

	✅ Predicted: Recyclable (99.7% confidence)
📦 Model Download

Pretrained model available on GitHub releases:
📁 Download best_model.zip

🧑‍💻 Tech Stack

Python 3.10

TensorFlow 2.12.0 (CPU)

OpenCV

Pandas, NumPy, Matplotlib

Streamlit 1.30.0

Pillow (PIL)

🌱 Future Enhancements

🧾 Add more categories (metal, paper, glass, etc.)

🎥 Real-time classification dashboard

☁️ Deploy app to Streamlit Cloud or Hugging Face Spaces

📲 Mobile-friendly responsive design

🏆 Results Summary
Metric	Value
Train Accuracy	94.7%
Validation Accuracy	94.5%
Test Accuracy	92.2%
Model	MobileNetV2 (Transfer Learning)
🤝 Contributing

Want to contribute? Follow these steps:

git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature


Then open a Pull Request 🚀

🧾 License

Licensed under the MIT License.
Feel free to use, share, or modify this project with proper credit.

🙌 Acknowledgments

TensorFlow & Keras community for transfer learning resources

Public waste datasets on Kaggle and TensorFlow Datasets

Streamlit for providing such an easy way to build AI apps

✨ Author

👨‍💻 Bankapalli Yashwanth
🎓 B.Tech — Computer Science & Engineering
🏫 Acharya Nagarjuna University

