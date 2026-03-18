import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("fire_smoke_model.keras")
classes = ['fire', 'normal', 'smoke']

cap = cv2.VideoCapture(0)

print("Camera starting... Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img, verbose=0)
    label = classes[np.argmax(pred)]
    confidence = np.max(pred)

    if label == 'fire':
        color = (0, 0, 255)
    elif label == 'smoke':
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    text = f"{label} ({confidence:.2f})"

    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Fire Detection", frame)

    # ALERT
    if label in ['fire', 'smoke'] and confidence > 0.85:
        print(f"🔥 ALERT: {label.upper()} detected!")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()