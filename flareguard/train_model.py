import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Dataset Path
# =========================
dataset_path = "dataset/"

# Count total images
total_images = 0
for root, dirs, files in os.walk(dataset_path):
    total_images += len(files)

print(f"Total images found: {total_images}")

# =========================
# Data Generator
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=4,
    class_mode='binary',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

print("Class indices:", train_data.class_indices)

# =========================
# Handle Imbalanced Dataset
# =========================
classes = train_data.classes

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# =========================
# Load MobileNetV2
# =========================
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# =========================
# Build Model
# =========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Early Stopping
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# =========================
# Train Model
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    callbacks=[early_stop],
    class_weight=class_weights
)

# =========================
# Save Model
# =========================
model.save("fire_smoke_model.keras")
print("Model training complete and saved!")

# =========================
# Plot Accuracy
# =========================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()