import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("fire_smoke_model.keras")

classes = ['fire', 'normal', 'smoke']

def predict_frame(frame):
    # Resize correctly (IMPORTANT)
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    # Prediction
    pred = model.predict(img, verbose=0)
    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    print(label, confidence)  # debug

    return label, confidence