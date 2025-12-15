import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# ==============================================================================
# CONFIGURATION (from config.py)
# ==============================================================================
DATASET_FILE = "motion_dataset_loso_v5.npz"
MODEL_SAVE_NAME = "best_motion_model_v5.keras"

# Architecture
CONV_FILTERS_BLOCK1 = 64
CONV_FILTERS_BLOCK2 = 128
CONV_FILTERS_BLOCK3 = 64
KERNEL_SIZE = 3
POOL_SIZE = 2
DROPOUT_RATE = 0.5             # Increased: Stronger regularization against overfitting
DENSE_UNITS = 64

# Training hyperparameters
MAX_EPOCHS = 60
BATCH_SIZE = 64                # Increased: Larger batches reduce validation noise
LEARNING_RATE = 0.0005         # Reduced: Lower rate for more stable convergence
LR_DECAY_FACTOR = 0.5          # Reduce LR by this factor
LR_DECAY_PATIENCE = 5          # Epochs without improvement before decay
EARLY_STOPPING_PATIENCE = 20   # Increased: Allow more time for stability
L2_REGULARIZATION = 0.001      # NEW: L2 penalty to reduce overfitting

MODEL_REPORT_FILE = "model_v5_report.txt"


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
    # filters: Learns different types of basic patterns.
    # kernel_size: Looks at multiple time-steps at a time.
    layers.Conv1D(filters=CONV_FILTERS_BLOCK1, kernel_size=KERNEL_SIZE, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    
    # BatchNormalization: Normalizes internal inputs, stabilizing learning.
    # It allows the model to train faster and be less sensitive to initialization.
    layers.BatchNormalization(),
    
    # ReLU: "Rectified Linear Unit". Introduces non-linearity.
    # Converts negative values to 0. Essential for learning complex logic.
    layers.ReLU(),
    
    # MaxPooling: Reduces the data size by downsampling.
    # Keeps only the strongest signal in every group.
    # This makes the model translation-invariant (doesn't matter if step starts at t=0 or t=5).
    layers.MaxPooling1D(pool_size=POOL_SIZE),
    
    # Dropout: Randomly turns off neurons during training.
    # This forces the model to not rely on any single neuron, preventing
    # it from memorizing specific user quirks (Overfitting).
    layers.Dropout(DROPOUT_RATE),

    # --- BLOCK 2: Mid-Level Feature Extraction ---
    # More filters to combine basic patterns into complex ones 
    # (e.g., a full step cycle).
    layers.Conv1D(filters=CONV_FILTERS_BLOCK2, kernel_size=KERNEL_SIZE, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=POOL_SIZE),
    layers.Dropout(DROPOUT_RATE),
    
    # --- BLOCK 3: High-Level Feature Extraction ---
    layers.Conv1D(filters=CONV_FILTERS_BLOCK3, kernel_size=KERNEL_SIZE, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    # GlobalAveragePooling1D: A smarter alternative to 'Flatten'.
    # Instead of flattening all time steps into a huge vector, it calculates 
    # the average activation of each filter across the entire time window.
    # drastically reduces parameter count and model size (great for mobile).
    layers.GlobalAveragePooling1D(), 

    # --- OUTPUT BLOCK: Classification ---
    # Dense: Fully connected layer to interpret the pooled features.
    layers.Dense(DENSE_UNITS, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.Dropout(DROPOUT_RATE),
    
    # Final Layer: Output probability for each class.
    # Softmax ensures all outputs sum to 1.0 (e.g., 0.8 Walk, 0.1 Up, 0.1 Down).
    layers.Dense(n_outputs, activation='softmax')
])

# Compile the model with learning rate scheduler
# Optimizer 'adam': Adaptive Learning Rate. Good default for most problems.
# Loss 'categorical_crossentropy': Standard loss for multi-class classification.
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
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

# ReduceLROnPlateau: Reduces learning rate if validation loss plateaus.
# Allows the model to fine-tune when learning slows down.
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=LR_DECAY_FACTOR,
    patience=LR_DECAY_PATIENCE,
    min_lr=1e-6,
    verbose=1
)

# EarlyStopping: Stops training if the model stops improving.
# Prevents wasting time and over-training.
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True # Go back to the best weights found
)


# ==============================================================================
# 4. TRAIN THE MODEL
# ==============================================================================
# Compute class weights to help distinguish difficult pairs (Walking vs Upstairs)
from sklearn.utils.class_weight import compute_class_weight

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print("\nClass Weights (to balance training):")
class_names_list = ['Walking', 'Upstairs', 'Downstairs', 'Idle']
for i, (name, weight) in enumerate(zip(class_names_list, class_weights)):
    print(f"  {name:12}: {weight:.4f}")

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, lr_scheduler, early_stopping],
    class_weight=class_weight_dict,  # NEW: Apply class weights
    verbose=1
)


# ==============================================================================
# 5. EVALUATE (FINAL TEST)
# ==============================================================================
# Crucial: Load the BEST saved model, not the one currently in memory.
# The one in memory might be from Epoch 60, but the best one might be Epoch 45.
print("\nLoading best model for final evaluation...")
best_model = models.load_model(MODEL_SAVE_NAME)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n" + "="*50)
print(f"FINAL TEST RESULT (UNSEEN USER)")
print(f"Accuracy: {test_acc:.4f}")
print(f"Loss: {test_loss:.4f}")
print(f"="*50 + "\n")

# Get predictions for confusion matrix and per-class metrics
y_pred_prob = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
print("Confusion Matrix:")
print(cm)

# Class names
class_names = ['Walking', 'Upstairs', 'Downstairs', 'Idle']
print("\nPer-Class Performance Metrics:")
print(classification_report(y_test_labels, y_pred, target_names=class_names, digits=4))


# ==============================================================================
# 6. VISUALIZE RESULTS
# ==============================================================================
plt.figure(figsize=(16, 5))

# Plot 1: Accuracy
# We want the 'Val' line to be high and close to the 'Train' line.
# If 'Train' is 99% and 'Val' is 60%, you are overfitting.
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Unseen User)')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss
# We want both lines to go down. 
# If 'Val' starts going UP while 'Train' goes DOWN, stop training!
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (Unseen User)')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix Heatmap
plt.subplot(1, 3, 3)
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(range(4), class_names, rotation=45)
plt.yticks(range(4), class_names)

# Add text annotations
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()


# ==============================================================================
# 7. SAVE ALL MODEL DATA AND METADATA
# ==============================================================================
# Create a comprehensive archive containing all relevant model information

# Prepare comprehensive metadata dictionary
metadata = {
    "timestamp": datetime.now().isoformat(),
    "model_version": "v5",
    "model_architecture": {
        "input_shape": (int(n_timesteps), int(n_features)),
        "n_timesteps": int(n_timesteps),
        "n_features": int(n_features),
        "n_outputs": int(n_outputs),
        "model_type": "1D CNN Sequential",
        "conv_filters": [CONV_FILTERS_BLOCK1, CONV_FILTERS_BLOCK2, CONV_FILTERS_BLOCK3],
        "kernel_size": KERNEL_SIZE,
        "pool_size": POOL_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "dense_units": DENSE_UNITS
    },
    "training_config": {
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lr_decay_factor": LR_DECAY_FACTOR,
        "lr_decay_patience": LR_DECAY_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "l2_regularization": L2_REGULARIZATION,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "class_weights": {int(k): float(v) for k, v in class_weight_dict.items()}
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
    "confusion_matrix": cm.tolist(),
    "dataset_info": {
        "dataset_file": DATASET_FILE,
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "training_strategy": "Leave-One-Subject-Out (LOSO)"
    }
}

# Save a summary report as text
with open(MODEL_REPORT_FILE, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL V5 - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {metadata['timestamp']}\n\n")
    
    f.write("ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Input Shape: {n_timesteps} timesteps × {n_features} features\n")
    f.write(f"Output Classes: {n_outputs}\n")
    f.write(f"Model Type: {metadata['model_architecture']['model_type']}\n")
    f.write(f"Conv Filters: {metadata['model_architecture']['conv_filters']}\n")
    f.write(f"Kernel Size: {KERNEL_SIZE}, Pool Size: {POOL_SIZE}\n")
    f.write(f"Dropout Rate: {DROPOUT_RATE}\n\n")
    
    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Max Epochs: {MAX_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Initial Learning Rate: {LEARNING_RATE}\n")
    f.write(f"LR Decay: factor={LR_DECAY_FACTOR}, patience={LR_DECAY_PATIENCE}\n")
    f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
    f.write(f"L2 Regularization: {L2_REGULARIZATION}\n")
    f.write(f"Optimizer: {metadata['training_config']['optimizer']}\n")
    f.write(f"Loss Function: {metadata['training_config']['loss_function']}\n")
    f.write(f"Class Weights: {metadata['training_config']['class_weights']}\n\n")
    
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
    
    f.write("CONFUSION MATRIX\n")
    f.write("-" * 70 + "\n")
    f.write(str(cm) + "\n\n")
    
print("\n" + "=" * 70)
print("ALL MODEL DATA SAVED SUCCESSFULLY")
print("=" * 70)