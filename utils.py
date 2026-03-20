import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os

os.makedirs("alerts", exist_ok=True)

model = load_model("fire_smoke_model.keras")
classes = ['fire', 'normal', 'smoke']

last_saved = 0

def save_alert(frame, label):
    global last_saved

    if time.time() - last_saved < 5:
        return

    filename = f"alerts/{label}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print("🚨 Saved:", filename)

    last_saved = time.time()


def predict_frame(frame):
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img, verbose=0)
    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    if label in ["fire", "smoke"] and confidence > 0.6:
        save_alert(frame, label)

    return label, confidence