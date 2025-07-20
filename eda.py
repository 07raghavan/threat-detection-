# eda_analysis.py
import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Set your dataset path (the parent folder with subfolders for each class)
DATASET_PATH = r"D:\org_dataset"  # Update with your path

def dataset_overview(directory):
    """
    Lists all class folders in the dataset and counts the number of .wav files in each.
    """
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    print("Classes found:", classes)
    distribution = {}
    for cls in classes:swith('.wav')]
        distribution[cls] = len(files)
    return distribution

def display_random_spectrograms(directory, num_samples=4, sr=
        cls_path = os.path.join(directory, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().end16000, duration=5, n_mels=128):
    """
    Loads random audio files from random classes and displays their log-mel spectrograms.
    """
    plt.figure(figsize=(12, 8))
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for i in range(num_samples):
        cls = random.choice(classes)
        cls_path = os.path.join(directory, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith('.wav')]
        file_path = os.path.join(cls_path, random.choice(files))
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        plt.subplot(2, 2, i + 1)
        librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.title(f"{cls}")
        plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# Run EDA
dist = dataset_overview(DATASET_PATH)
print("Dataset Distribution:")
for cls, count in dist.items():
    print(f"  {cls}: {count} files")

display_random_spectrograms(DATASET_PATH)
