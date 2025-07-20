# tflite_inference_lora.py
import os
import datetime
import numpy as np
import librosa
import tensorflow as tf

# Function to send a LoRa alert 
def send_lora_alert(message):
    # Insert your LoRa library code here to send the message
    print("LoRa Alert Sent:", message)

def is_nighttime(start_hour=20, end_hour=6):
    """
    Returns True if the current time is within the nighttime period.
    """
    now = datetime.datetime.now().time()
    if start_hour > end_hour:  # e.g., 20:00 to 06:00 spans midnight
        return now.hour >= start_hour or now.hour < end_hour
    else:
        return start_hour <= now.hour < end_hour

# Preprocessing parameters (must match training)
SR = 16000
DURATION = 5.0
N_MELS = 128

def preprocess_audio(file_path, sr=SR, duration=DURATION, n_mels=N_MELS):
    """
    Loads an audio file, trims/pads it to the fixed duration, and converts it to a log-mel spectrogram.
    Returns the spectrogram with an added batch dimension.
    """
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    req_len = int(sr * duration)
    if len(audio) < req_len:
        audio = np.pad(audio, (0, req_len - len(audio)))
    else:
        audio = audio[:req_len]
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = log_mel_spec[..., np.newaxis]  # Add channel dimension
    return np.expand_dims(log_mel_spec, axis=0)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\rpheo\multiclass_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference(file_path):
    """
    Runs inference on the given audio file using the TFLite model.
    Returns the predicted class index and its confidence.
    """
    input_data = preprocess_audio(file_path)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]
    confidence = np.max(output_data)
    return predicted_class, confidence

# Define class names (the order must match training; here is an example)
class_names = ['Gunshot', 'NonThreat', 'Vehicle', 'Wood_Cutting']

# Example: Inference on a new audio file
test_audio_file = r""  # Update with your test file path
pred_class, conf = run_inference(test_audio_file)
pred_label = class_names[pred_class]
print(f"Predicted: {pred_label} with confidence {conf:.2f}")

# Alert logic:
# - For 'Vehicle': send alert only at night
# - For 'Gunshot' and 'Wood_Cutting': send alert immediately
if pred_label == "Vehicle":
    if is_nighttime():
        send_lora_alert("Alert: Vehicle movement detected at night!")
    else:
        print("Vehicle detected but it's daytime – no alert sent.")
elif pred_label in ["Gunshot", "Wood_Cutting"]:
    send_lora_alert(f"Alert: {pred_label} detected!")
else:
    print("No threat detected.")
