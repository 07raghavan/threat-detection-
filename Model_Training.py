# model_training_enhanced.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from data_preprocessing_with_split import load_audio_files_multiclass  # Ensure this file is in your PYTHONPATH

# =============================================================================
# Parameters & Paths
# =============================================================================
DATASET_PATH = r"D:\org_dataset"  # Update this path (holdout test files already removed)
SR = 16000         # Sampling rate: 16 kHz
DURATION = 5.0     # Duration in seconds per audio file
N_MELS = 128       # Number of Mel bands

# =============================================================================
# Load Preprocessed Data
# =============================================================================
# This function loads your audio files from the dataset structure:
#   multiclass_dataset/
#     ├── Vehicle/
#     ├── Wood_Cutting/
#     ├── Gunshot/
#     └── NonThreat/
print("Loading preprocessed data...")
X, y, classes = load_audio_files_multiclass(DATASET_PATH, sr=SR, duration=DURATION, n_mels=N_MELS)
print("Data shape:", X.shape, "Labels shape:", y.shape)
print("Classes:", classes)

# =============================================================================
# Split Data into Training and Validation Sets
# =============================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# =============================================================================
# Define Data Augmentation Pipeline for Spectrogram Images
# =============================================================================
data_augmentation = tf.keras.Sequential([
    # Randomly translate the spectrogram in both time and frequency dimensions.
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='reflect'),
    # Random zoom to slightly scale the spectrogram image.
    layers.RandomZoom(height_factor=0.1, width_factor=0.1)
])

# =============================================================================
# Build an Enhanced Multi-Class Model
# =============================================================================
def create_enhanced_model(input_shape, num_classes):
    """
    Constructs a CNN model with data augmentation and regularization.
    The model uses:
      - Data augmentation layers (applied only during training)
      - Convolutional blocks with L2 regularization and batch normalization
      - GlobalAveragePooling2D to reduce overfitting and parameter count
      - Dropout for additional regularization
      - A final softmax layer for multi-class classification
    """
    model = models.Sequential([
        # Data augmentation (only active during training)
        data_augmentation,
        
        # First convolution block
        layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.001),
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolution block
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolution block
        layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling to reduce features and overfitting
        layers.GlobalAveragePooling2D(),
        
        # Fully connected layer with dropout
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        
        # Output layer for multi-class classification
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_shape = X_train[0].shape  # e.g., (height, width, 1)
num_classes = len(classes)
model = create_enhanced_model(input_shape, num_classes)
model.summary()

# =============================================================================
# Define Callbacks for Optimal Training
# =============================================================================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = callbacks.ModelCheckpoint("best_enhanced_multiclass_model.h5",
                                         monitor='val_accuracy',
                                         save_best_only=True,
                                         verbose=1)

# =============================================================================
# Train the Model
# =============================================================================
history = model.fit(
    X_train, y_train,
    epochs=100,                # Train for more epochs; early stopping will prevent overfitting.
    batch_size=32,
    validation_data=(X_val, y_val),
    shuffle=True,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Enhanced Model Validation Accuracy: {val_accuracy * 100:.2f}%")
