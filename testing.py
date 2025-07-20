# test_holdout.py
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing_with_split import load_audio_files_multiclass  # Ensure this file is in your PYTHONPATH

# Global parameters (must match those used during training)
SR = 16000         # Sampling rate: 16 kHz
DURATION = 5.0     # Duration in seconds per audio file
N_MELS = 128       # Number of Mel bands

# -----------------------------------------------------------------------------
# Set Holdout Data Path
# -----------------------------------------------------------------------------
HOLDOUT_PATH = r"D:\realtest"  # Update with your holdout dataset path

# -----------------------------------------------------------------------------
# Load Holdout Data
# -----------------------------------------------------------------------------
print("Loading holdout test data...")
X_holdout, y_holdout, class_names = load_audio_files_multiclass(HOLDOUT_PATH, sr=SR, duration=DURATION, n_mels=N_MELS)
print("Holdout data shape:", X_holdout.shape)
print("Classes:", class_names)

# -----------------------------------------------------------------------------
# Load the Trained Model
# -----------------------------------------------------------------------------
model = tf.keras.models.load_model("best_enhanced_multiclass_model.h5")
print("Model loaded successfully.")

# -----------------------------------------------------------------------------
# Evaluate the Model on Holdout Data
# -----------------------------------------------------------------------------
loss, accuracy = model.evaluate(X_holdout, y_holdout, verbose=0)
print(f"Holdout Test Loss: {loss:.4f}")
print(f"Holdout Test Accuracy: {accuracy * 100:.2f}%")

# -----------------------------------------------------------------------------
# Generate Detailed Metrics (Confusion Matrix & Classification Report)
# -----------------------------------------------------------------------------
# Get predictions (as probabilities) and convert them to predicted class indices
y_pred_probs = model.predict(X_holdout)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_holdout, y_pred))
print("\nClassification Report:")
print(classification_report(y_holdout, y_pred, target_names=class_names))
