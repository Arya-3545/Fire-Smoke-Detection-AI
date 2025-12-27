# 🔥 FireGuard AI – Intelligent Fire & Smoke Detection System

## 📌 Project Overview

FireGuard AI is a deep learning–based fire and smoke detection system designed to identify potential fire hazards in real time using computer vision techniques. The system leverages a YOLO-based object detection model to accurately detect fire and smoke from images or video streams, enabling faster alerts and improved safety monitoring.

This project focuses on improving early detection reliability by combining high-quality datasets with optimized deep learning models suitable for real-world deployment.

---

## 🎯 Objectives

* Detect **fire and smoke** in images and videos with high accuracy
* Reduce false alarms compared to traditional sensor-based systems
* Enable real-time detection for safety monitoring applications
* Build a scalable and easy-to-deploy AI model

---

## 🧠 Technologies Used

* **Python 3**
* **YOLO (You Only Look Once)** – Object Detection Model
* **PyTorch**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **Kaggle Dataset (Fire & Smoke Detection)**

---

## 📂 Project Structure

```
Fire-Smoke-Detection-AI/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── training.ipynb
│
├── inference/
│   └── detect.py
│
├── README.md
├── requirements.txt
└── data.yaml
```

---

## 📊 Dataset Information

* **Dataset Name:** Smoke-Fire Detection Dataset
* **Classes:**

  * 0 → Smoke
  * 1 → Fire
* **Source:** Kaggle (Public Dataset)
* **Format:** YOLO annotation format

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Fire-Smoke-Detection-AI.git
cd Fire-Smoke-Detection-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

---

## 🔍 Running Inference

```bash
yolo predict model=best.pt source=your_image_or_video.mp4
```

---

## 📈 Output

* Bounding boxes for **Fire** and **Smoke**
* Confidence scores for each detection
* Real-time visualization (optional)

---

## 🧪 Evaluation Metrics

* Precision
* Recall
* mAP (mean Average Precision)

---

## 💡 Future Enhancements

* Real-time CCTV integration
* Alert system using SMS / Email
* Smoke density estimation
* Deployment on edge devices (Raspberry Pi, Jetson Nano)

---

## 👥 Team Members

| Name     | Role                              |
| -------- | --------------------------------- |
| Arsha S Pillai | Model Training & Dataset Handling |
| Arya Selvan | Model Optimization & Evaluation   |
| Disna Elcy | Documentation & Deployment        |
| Abhinaya Baiju | Testing & Visualization           |

---

## 📜 License

This project is for academic and educational purposes.

---

## 📬 Contact

For any queries or collaboration:
📧 Email: aryaselvan11@gmail.com

---

