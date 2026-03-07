import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("fire_smoke_model.keras")

# Dataset path
dataset_path = "dataset/"

# Only rescaling for evaluation
datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Get predictions
predictions = model.predict(val_data)
predicted_classes = (predictions > 0.5).astype(int)

true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Classification Report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()