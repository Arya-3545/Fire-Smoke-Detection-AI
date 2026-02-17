import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("fire_smoke_model.keras")

# Path to test image (place test.jpg inside flareguard folder)
img_path = "test.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)

# Print raw value for verification
print("Raw prediction value:", prediction[0][0])

# Since:
# fire  = 0
# smoke = 1

if prediction[0][0] < 0.5:
    print("🔥 Fire Detected")
else:
    print("💨 Smoke Detected")
