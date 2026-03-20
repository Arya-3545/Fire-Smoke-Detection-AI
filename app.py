from flask import Flask, render_template, request, jsonify
import cv2
import base64
import numpy as np
from utils import predict_frame

app = Flask(__name__)

# Store multiple cameras
cameras = {}

@app.route('/')
def index():
    return render_template('index.html')

# =========================
# ADD CAMERA
# =========================
@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.get_json()
    cam_id = data['id']
    url = data['url']

    cameras[cam_id] = cv2.VideoCapture(url)

    return jsonify({"status": "added"})


# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    mode = data.get("mode")
    cam_id = data.get("cam_id")

    frame = None

    # CCTV MODE
    if mode == "cctv":
        cap = cameras.get(cam_id)

        if cap is None:
            return jsonify({"label": "no camera", "confidence": 0})

        ret, frame = cap.read()
        if not ret:
            return jsonify({"label": "no signal", "confidence": 0})

    # WEBCAM MODE
    else:
        image_data = data.get("image")

        if not image_data or "," not in image_data:
            return jsonify({"label": "no image", "confidence": 0})

        image_data = image_data.split(',')[1]
        image = base64.b64decode(image_data)

        np_arr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    label, confidence = predict_frame(frame)

    return jsonify({
        "label": label,
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5008)