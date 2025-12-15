import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The dataset file produced by 'split_normalize_loso_v4.py'
# It contains the Train/Test split based on the Leave-One-Subject-Out strategy.
DATASET_FILE = "motion_dataset_loso_v4.npz"

# The filename for saving the best version of the trained model.
# .keras is the modern, recommended format for Keras models.
MODEL_SAVE_NAME = "best_motion_model_v4.keras"


# ==============================================================================
# 1. LOAD AND INSPECT DATA
# ==============================================================================
# Verify file exists to avoid confusing error messages later
assert os.path.exists(DATASET_FILE), f"Error: {DATASET_FILE} not found."

# Load the .npz archive
data = np.load(DATASET_FILE)

# Extract Training and Testing sets
# X = Input features (Sensor data windows)
# y = Output labels (One-hot encoded classes)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Auto-detect input dimensions from the loaded data
# n_timesteps: How long is one window? (e.g., 128 samples approx 2.56s)
n_timesteps = X_train.shape[1] 
# n_features: How many sensors? (e.g., 4: AccX, AccY, AccZ, Magnitude)
n_features = X_train.shape[2]  
# n_outputs: How many activities? (e.g., 4: Walk, Up, Down, Idle)
n_outputs = y_train.shape[1]   

print(f"Input Shape: {n_timesteps} steps, {n_features} features")
print(f"Classes found: {n_outputs}")
print(f"Training on {len(X_train)} samples")
print(f"Validating on {len(X_test)} samples (Unseen User)")


# ==============================================================================
# 2. BUILD THE ROBUST CNN MODEL
# ==============================================================================
# We use a 1D Convolutional Neural Network (CNN).
# CNNs are excellent for sensor data because they can learn "temporal patterns"
# (like the rhythmic rise and fall of a step) regardless of exactly when 
# it happens in the window.

model = models.Sequential([
    # Input Layer: Defines the shape of the data entering the model
    layers.Input(shape=(n_timesteps, n_features)),
    
    # --- BLOCK 1: Low-Level Feature Extraction ---
    # Conv1D: Scans the window looking for basic patterns (e.g., peaks, slopes).
    # filters=64: Learns 64 different types of basic patterns.
    # kernel_size=3: Looks at 3 time-steps at a time.
    layers.Conv1D(filters=64, kernel_size=3, padding='same'),
    
    # BatchNormalization: Normalizes internal inputs. stabilizing learning.
    # It allows the model to train faster and be less sensitive to initialization.
    layers.BatchNormalization(),
    
    # ReLU: "Rectified Linear Unit". Introduces non-linearity.
    # Converts negative values to 0. Essential for learning complex logic.
    layers.ReLU(),
    
    # MaxPooling: Reduces the data size by half (downsampling).
    # Keeps only the strongest signal in every group of 2.
    # This makes the model translation-invariant (doesn't matter if step starts at t=0 or t=5).
    layers.MaxPooling1D(pool_size=2),
    
    # Dropout: Randomly turns off 30% of neurons during training.
    # This forces the model to not rely on any single neuron, preventing
    # it from memorizing specific user quirks (Overfitting).
    layers.Dropout(0.3),

    # --- BLOCK 2: Mid-Level Feature Extraction ---
    # More filters (128) to combine basic patterns into complex ones 
    # (e.g., a full step cycle).
    layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # --- BLOCK 3: High-Level Feature Extraction ---
    layers.Conv1D(filters=64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    # GlobalAveragePooling1D: A smarter alternative to 'Flatten'.
    # Instead of flattening all time steps into a huge vector, it calculates 
    # the average activation of each filter across the entire time window.
    # drastically reduces parameter count and model size (great for mobile).
    layers.GlobalAveragePooling1D(), 

    # --- OUTPUT BLOCK: Classification ---
    # Dense: Fully connected layer to interpret the pooled features.
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    # Final Layer: Output probability for each class.
    # Softmax ensures all outputs sum to 1.0 (e.g., 0.8 Walk, 0.1 Up, 0.1 Down).
    layers.Dense(n_outputs, activation='softmax')
])

# Compile the model
# Optimizer 'adam': Adaptive Learning Rate. Good default for most problems.
# Loss 'categorical_crossentropy': Standard loss for multi-class classification.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ==============================================================================
# 3. CALLBACKS (TRAINING HELPERS)
# ==============================================================================
# Checkpoint: Saves the model every time the validation accuracy improves.
# This ensures we keep the "best" version, not just the "last" version.
checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_NAME, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# EarlyStopping: Stops training if the model stops improving.
# Prevents wasting time and over-training.
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15, # Wait 15 epochs for improvement before quitting
    restore_best_weights=True # Go back to the best weights found
)


# ==============================================================================
# 4. TRAIN THE MODEL
# ==============================================================================
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,      # Max epochs (EarlyStopping usually stops it sooner)
    batch_size=32,  # Process 32 windows at a time
    callbacks=[checkpoint, early_stopping]
)


# ==============================================================================
# 5. EVALUATE (FINAL TEST)
# ==============================================================================
# Crucial: Load the BEST saved model, not the one currently in memory.
# The one in memory might be from Epoch 60, but the best one might be Epoch 45.
print("\nLoading best model for final evaluation...")
best_model = models.load_model(MODEL_SAVE_NAME)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n" + "="*30)
print(f"FINAL TEST RESULT (UNSEEN USER)")
print(f"Accuracy: {test_acc:.4f}")
print(f"="*30 + "\n")


# ==============================================================================
# 6. VISUALIZE RESULTS
# ==============================================================================
plt.figure(figsize=(12, 4))

# Plot 1: Accuracy
# We want the 'Val' line to be high and close to the 'Train' line.
# If 'Train' is 99% and 'Val' is 60%, you are overfitting.
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Unseen User)')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot 2: Loss
# We want both lines to go down. 
# If 'Val' starts going UP while 'Train' goes DOWN, stop training!
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (Unseen User)')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Create a comprehensive archive containing all relevant model information

from datetime import datetime

# Prepare comprehensive metadata dictionary
metadata = {
    "model_architecture": {
        "input_shape": (n_timesteps, n_features),
        "n_timesteps": int(n_timesteps),
        "n_features": int(n_features),
        "n_outputs": int(n_outputs),
        "model_type": "1D CNN Sequential"
    },
    "training_config": {
        "max_epochs": 60,
        "batch_size": 32,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "metrics": ["accuracy"]
    },
    "training_results": {
        "final_test_accuracy": float(test_acc),
        "final_test_loss": float(test_loss),
        "actual_epochs_trained": len(history.history['loss']),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1])
    },
    "dataset_info": {
        "dataset_file": DATASET_FILE,
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "training_strategy": "Leave-One-Subject-Out (LOSO)"
    }
}

# Save a summary report as text
report_filename = "model_v4_report.txt"
with open(report_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL V4 - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Input Shape: {n_timesteps} timesteps × {n_features} features\n")
    f.write(f"Output Classes: {n_outputs}\n")
    f.write(f"Model Type: {metadata['model_architecture']['model_type']}\n\n")
    
    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Max Epochs: {metadata['training_config']['max_epochs']}\n")
    f.write(f"Batch Size: {metadata['training_config']['batch_size']}\n")
    f.write(f"Optimizer: {metadata['training_config']['optimizer']}\n")
    f.write(f"Loss Function: {metadata['training_config']['loss_function']}\n\n")
    
    f.write("DATASET\n")
    f.write("-" * 70 + "\n")
    f.write(f"Dataset File: {DATASET_FILE}\n")
    f.write(f"Training Samples: {metadata['dataset_info']['n_training_samples']}\n")
    f.write(f"Test Samples: {metadata['dataset_info']['n_test_samples']}\n")
    f.write(f"Strategy: {metadata['dataset_info']['training_strategy']}\n\n")
    
    f.write("TRAINING RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Epochs Trained: {metadata['training_results']['actual_epochs_trained']}\n")
    f.write(f"Final Train Accuracy: {metadata['training_results']['final_train_accuracy']:.4f}\n")
    f.write(f"Final Train Loss: {metadata['training_results']['final_train_loss']:.4f}\n")
    f.write(f"Final Validation Accuracy: {metadata['training_results']['final_val_accuracy']:.4f}\n")
    f.write(f"Final Validation Loss: {metadata['training_results']['final_val_loss']:.4f}\n\n")
    
    f.write("FINAL TEST EVALUATION (UNSEEN USER)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Test Accuracy: {metadata['training_results']['final_test_accuracy']:.4f}\n")
    f.write(f"Test Loss: {metadata['training_results']['final_test_loss']:.4f}\n\n")
    
    f.write("=" * 70 + "\n")
print(f"✓ Summary report saved to {report_filename}")