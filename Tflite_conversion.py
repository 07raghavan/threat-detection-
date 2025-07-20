# tflite_conversion.py
import tensorflow as tf

# Load the best saved Keras model (ensure the path matches your saved model)
model = tf.keras.models.load_model(r"model location")

# Convert the model to TensorFlow Lite format with optimization (e.g., quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open("multiclass_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved as 'multiclass_model.tflite'.")
