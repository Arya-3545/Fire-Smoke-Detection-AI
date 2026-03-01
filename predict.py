import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("fire_smoke_model.keras")

# Path to test image
img_path = "test.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)

# Print full probability array
print("Raw prediction values:", prediction)

# Get highest probability index
class_index = np.argmax(prediction)

# ⚠ IMPORTANT:
# These numbers MUST match what was printed in train_model.py
# Example:
# {'fire': 0, 'normal': 1, 'smoke': 2}

if class_index == 0:
    print("🔥 Fire Detected")
elif class_index == 1:
    print("✅ Normal (No Fire/Smoke)")
elif class_index == 2:
    print("💨 Smoke Detected")
