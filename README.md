# 🏍️ MotoLens – Optimized Motorcycle ANPR System (YOLOv10 + EasyOCR)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://motorcycle-anpr-app.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv10](https://img.shields.io/badge/YOLO-v10n-green)

## 🔗 Live Demo
> **Try the App Here:** [**Click to Launch ANPR System**](https://motorcycle-anpr-app-qd7jdt4yelfpaimng5dh7i.streamlit.app/)

---

## 📖 About The Project

This project is a specialized **Automatic Number Plate Recognition (ANPR)** system designed specifically for motorcycles. Motorcycles present unique challenges for computer vision due to smaller plate sizes, varied mounting angles, and diverse lighting conditions compared to cars.

This application utilizes **YOLOv10 (Nano)** for high-speed, real-time object detection to locate the license plate, and **EasyOCR** to extract the alphanumeric characters. The system is wrapped in a user-friendly **Streamlit** web interface, allowing users to upload images or videos for instant processing.

### 🎯 Key Objectives
* **High Precision:** Trained on high-resolution images to maximize detection accuracy for small plates.
* **Speed:** Utilizing the lightweight YOLOv10n architecture to balance performance with the computational cost of processing larger images.
* **Accessibility:** Simple web-based deployment for easy testing and usage.

---

## ✨ Features

* **📷 Image Inference:** Upload any image (`.jpg`, `.png`) to detect plates and read text with bounding box visualization.
* **🎥 Video Processing:** Support for video files (`.mp4`, `.mov`) with frame-by-frame detection and annotated video output.
* **🧠 Advanced OCR:** Uses EasyOCR with GPU acceleration support (where available) to read text from cropped plate regions.
* **🎛️ Adjustable Confidence:** Sidebar slider to tweak the model's confidence threshold, allowing users to filter out weak detections.
* **☁️ Cloud Ready:** Fully configured for deployment on Streamlit Community Cloud.

---

## 🛠️ Tech Stack

* **Detection Model:** [YOLOv10n](https://github.com/THU-MIG/yolov10) (Trained on custom dataset)
* **OCR Engine:** [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Image Processing:** OpenCV, PIL, NumPy
* **Environment:** Python 3.9+

---

## 📂 Repository Structure

```text
├── app.py                 # Main Streamlit application script
├── best.pt                # Trained YOLOv10n model weights
├── requirements.txt       # Python dependencies
├── packages.txt           # System-level dependencies (libgl1)
└── README.md              # Project documentation
```

---

## 🚀 How to Run Locally

If you want to run this app on your own machine instead of the cloud, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/motorcycle-anpr-app.git
    cd motorcycle-anpr-app
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Dataset & Training

The model was trained using a specialized dataset of motorcycle images annotated in YOLO format.

* **Resolution:** **Original High-Resolution**. Images were *not* downscaled during training.
    * *Benefit:* This allows the model to "see" smaller details, significantly improving accuracy for reading text on small or distant number plates.
    * *Trade-off:* Training occupied more GPU memory and took longer per epoch compared to resized datasets.
* **Training Platform:** Google Colab (T4 GPU).
* **Epochs:** 50
* **Model Architecture:** YOLOv10 Nano (`yolov10n`) - chosen for its efficiency.
