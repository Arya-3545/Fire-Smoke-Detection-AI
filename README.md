

# 🔥 FireGuard AI – Intelligent Fire & Smoke Detection System

## 📌 Project Overview

FireGuard AI is a deep learning–based fire and smoke detection system designed to identify potential fire hazards in real time using computer vision techniques.

The system now leverages a **MobileNet-based Convolutional Neural Network (CNN)** as its backbone architecture for efficient and lightweight fire and smoke detection. MobileNet enables high accuracy while maintaining low computational cost, making the model suitable for edge devices and real-time deployment scenarios.

This project focuses on improving early detection reliability by combining a well-annotated dataset with an optimized lightweight deep learning model suitable for real-world safety monitoring systems.

---

## 🎯 Objectives

* Detect fire and smoke in images and videos with high accuracy
* Reduce false alarms compared to traditional sensor-based systems
* Enable real-time detection for safety monitoring applications
* Build a lightweight, scalable, and edge-device-friendly AI model

---

## 🧠 Technologies Used

* Python 3
* MobileNet (Lightweight CNN Architecture)
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Kaggle Fire & Smoke Dataset

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
│   └── best_model.pth
│
├── notebooks/
│   └── training.ipynb
│
├── inference/
│   └── detect.py
│
├── README.md
├── requirements.txt
└── config.yaml
```

---

## 📊 Dataset Information

**Dataset Name:** Smoke-Fire Detection Dataset
**Source:** Kaggle (Public Dataset)

**Classes:**

* 1 → Smoke
* 0 → Fire

**Format:** Image classification format (labeled images organized into class folders)

> Note: Since MobileNet is used as a CNN classifier backbone, the dataset is structured in image classification format instead of YOLO annotation format.

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

Example PyTorch-based training approach:

```bash
python training.py --epochs 30 --batch_size 32 --img_size 224
```

MobileNet (pretrained on ImageNet) is fine-tuned for binary classification (Fire / Smoke).

---

## 🔍 Running Inference

```bash
python detect.py --source your_image_or_video.mp4
```

### Output

* Class prediction (Fire / Smoke)
* Confidence score
* Real-time visualization using OpenCV

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 💡 Future Enhancements

* Real-time CCTV integration
* SMS / Email alert system
* Smoke intensity estimation
* Deployment on edge devices (Raspberry Pi, Jetson Nano)
* Integration with IoT-based alarm systems

---

## 👥 Team Members

| Name           | Role                              |
| -------------- | --------------------------------- |
| Arsha S Pillai | Model Training & Dataset Handling |
| Arya Selvan    | Model Optimization & Evaluation   |
| Disna Elcy     | Documentation & Deployment        |
| Abhinaya Baiju | Testing & Visualization           |

---

## 📜 License

This project is developed for academic and educational purposes.

---

## 📬 Contact

For queries or collaboration:
📧 Email: [aryaselvan11@gmail.com](mailto:aryaselvan11@gmail.com)

