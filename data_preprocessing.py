# data_preprocessing_with_split.py
import os
import random
import shutil
import numpy as np
import librosa

# Global parameters for audio processing
SR = 16000         # Sampling rate: 16 kHz
DURATION = 5.0     # Duration in seconds for each audio clip
N_MELS = 128       # Number of Mel bands

def split_holdout_test(source_dir, holdout_dir, num_holdout=100):
    """
    For each class in source_dir, randomly select num_holdout files and copy them to holdout_dir.
    This ensures these files are held out for testing and not used during training/validation.
    """
    if not os.path.exists(holdout_dir):
        os.makedirs(holdout_dir)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    for cls in classes:
        src_class_dir = os.path.join(source_dir, cls)
        holdout_class_dir = os.path.join(holdout_dir, cls)
        if not os.path.exists(holdout_class_dir):
            os.makedirs(holdout_class_dir)
        files = [f for f in os.listdir(src_class_dir) if f.lower().endswith('.wav')]
        random.shuffle(files)
        holdout_files = files[:num_holdout]
        for f in holdout_files:
            src_file = os.path.join(src_class_dir, f)
            dst_file = os.path.join(holdout_class_dir, f)
            shutil.copy(src_file, dst_file)
            os.remove(src_file)

    print("Holdout test data prepared.")

def load_audio_files_multiclass(directory, sr=SR, duration=DURATION, n_mels=N_MELS):
    """
    Loads audio files from a dataset folder (each subfolder is a class),
    shuffles the files within each class, and converts each audio file
    into a log-mel spectrogram.
    
    Returns:
      X: numpy array of spectrograms with shape (num_samples, height, width, 1)
      y: numpy array of integer labels for the classes
      classes: list of class names (alphabetically sorted)
    """
    X, y = [], []
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    print("Classes found:", classes)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith('.wav')]
        random.shuffle(files)
        print(f"Loading {len(files)} files from class: {cls}")
        for file in files:
            file_path = os.path.join(cls_path, file)
            try:
                audio, _ = librosa.load(file_path, sr=sr, duration=duration)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            req_len = int(sr * duration)
            if len(audio) < req_len:
                audio = np.pad(audio, (0, req_len - len(audio)))
            else:
                audio = audio[:req_len]
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            X.append(log_mel_spec)
            y.append(class_to_index[cls])
    
    X = np.array(X)
    y = np.array(y)
    X = X[..., np.newaxis]  # Add channel dimension for CNN
    print("Data loaded. X shape:", X.shape, "y shape:", y.shape)
    return X, y, classes

if __name__ == '__main__':
    # Paths (update with your actual directories)
    DATASET_PATH = r"D:\org_dataset"      # Original dataset folder
    HOLDOUT_PATH = r"D:\realtest"         # Folder to store holdout test data
    
    # Split out holdout test data: 100 files per class
    split_holdout_test(DATASET_PATH, HOLDOUT_PATH, num_holdout=100)
    
    # Load remaining data (training/validation)
    X, y, classes = load_audio_files_multiclass(DATASET_PATH)
