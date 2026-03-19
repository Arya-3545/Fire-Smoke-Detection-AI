from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from utils import predict_frame

app = Flask(__name__)

cap = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global cap
    data = request.get_json()
    rtsp = data['rtsp']
    cap = cv2.VideoCapture(rtsp)
    return jsonify({"status": "connected"})

@app.route('/predict', methods=['POST'])
def predict():
    global cap

    data = request.get_json()
    mode = data.get("mode", "webcam")

    # CCTV MODE
    if mode == "cctv" and cap is not None:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"label": "no signal", "confidence": 0})

    # WEBCAM MODE
    else:
        if 'image' not in data or not data['image']:
            return jsonify({"label": "no image", "confidence": 0})

        try:
            image_data = data['image'].split(',')[1]
            image = base64.b64decode(image_data)
            np_arr = np.frombuffer(image, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except:
            return jsonify({"label": "error", "confidence": 0})

    label, confidence = predict_frame(frame)

    return jsonify({
        "label": label,
        "confidence": float(round(confidence, 2))
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)