import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# ==============================================================================
# CONFIGURATION (V6_64 — 64-sample window variant)
# ==============================================================================
DATASET_FILE = "motion_dataset_loso_v6_64.npz"
MODEL_SAVE_NAME = "best_motion_model_v6_64.keras"

# Architecture — tuned for 64-sample (1.28s) window
# Key difference vs v6 (96): Block 2 MaxPool removed to preserve temporal resolution
# Sequence flow: 64 → Pool → 32 → (no pool) → 32 → GAP
CONV_FILTERS_BLOCK1 = 64
CONV_FILTERS_BLOCK2 = 128
CONV_FILTERS_BLOCK3 = 192       # Increased: compensate for less temporal info
KERNEL_SIZE_BLOCK1 = 7           # Wider: capture ~140ms context (almost a half-step at 50Hz)
KERNEL_SIZE_BLOCK2 = 5           # Wider than v6: extract more from each position
KERNEL_SIZE_BLOCK3 = 3
POOL_SIZE = 2
DROPOUT_CONV = 0.3
DROPOUT_DENSE = 0.45             # Slightly higher: more augmented data = risk of memorizing noise
DENSE_UNITS_1 = 192              # Larger head: compensate for richer Block 3 output
DENSE_UNITS_2 = 96

# Training hyperparameters
MAX_EPOCHS = 120                 # More epochs: 4x augmented data needs more time
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 30     # More patience: cosine annealing can recover late
L2_REGULARIZATION = 0.0005

MODEL_REPORT_FILE = "model_v6_64_report.txt"


# ==============================================================================
# 1. LOAD AND INSPECT DATA
# ==============================================================================
assert os.path.exists(DATASET_FILE), f"Error: {DATASET_FILE} not found."

data = np.load(DATASET_FILE)

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]
n_outputs = y_train.shape[1]

print(f"Input Shape: {n_timesteps} steps, {n_features} features")
print(f"Classes found: {n_outputs}")
print(f"Training on {len(X_train)} samples (with augmentation)")
print(f"Validating on {len(X_test)} samples (Unseen User)")


# ==============================================================================
# 2. BUILD THE V6_64 CNN MODEL
# ==============================================================================
# Key adaptation for 64-sample window:
# - Only 1 MaxPool (Block 1): preserves 32 timesteps for Blocks 2 & 3
# - Wider kernels (7, 5, 3): captures more context per convolution
# - 192 filters in Block 3: richer high-level features
# - Larger dense head (192 → 96): more classification capacity

model = models.Sequential([
    layers.Input(shape=(n_timesteps, n_features)),
    
    # --- BLOCK 1: Low-Level Feature Extraction ---
    # Kernel=7 captures ~140ms at 50Hz — nearly a half-step.
    # MaxPool here: 64 → 32 timesteps.
    layers.Conv1D(filters=CONV_FILTERS_BLOCK1, kernel_size=KERNEL_SIZE_BLOCK1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=POOL_SIZE),
    layers.Dropout(DROPOUT_CONV),

    # --- BLOCK 2: Mid-Level Feature Extraction ---
    # NO MaxPool here (unlike v6): preserves temporal resolution at 32 timesteps.
    # Wider kernel (5) compensates by covering more context per position.
    layers.Conv1D(filters=CONV_FILTERS_BLOCK2, kernel_size=KERNEL_SIZE_BLOCK2, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(DROPOUT_CONV),
    
    # --- BLOCK 3: High-Level Feature Extraction ---
    # 192 filters to capture rich discriminative patterns from 32 timesteps.
    layers.Conv1D(filters=CONV_FILTERS_BLOCK3, kernel_size=KERNEL_SIZE_BLOCK3, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.ReLU(),
    
    # GlobalAveragePooling: 32 timesteps × 192 filters → 192-dim vector.
    layers.GlobalAveragePooling1D(), 

    # --- OUTPUT BLOCK: Two-Layer Classification Head ---
    layers.Dense(DENSE_UNITS_1, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.Dropout(DROPOUT_DENSE),
    
    layers.Dense(DENSE_UNITS_2, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    layers.Dropout(DROPOUT_CONV),
    
    # Softmax output
    layers.Dense(n_outputs, activation='softmax')
])

# Compile with label smoothing loss
# Label smoothing: Instead of [1, 0, 0, 0], we use [0.925, 0.025, 0.025, 0.025]
# This prevents the model from becoming overconfident and improves generalization.
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()


# ==============================================================================
# 3. CALLBACKS (TRAINING HELPERS)
# ==============================================================================
checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_NAME,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Cosine Annealing LR: Smoothly decays LR following a cosine curve.
# Better than step decay — avoids sudden jumps and explores loss landscape more thoroughly.
cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=MAX_EPOCHS * (len(X_train) // BATCH_SIZE + 1),
    alpha=1e-6  # Minimum LR
)

# We use a LearningRateScheduler callback to apply cosine decay
def cosine_lr_schedule(epoch, lr):
    """Compute LR at given epoch using cosine annealing."""
    total_epochs = MAX_EPOCHS
    min_lr = 1e-6
    new_lr = min_lr + 0.5 * (LEARNING_RATE - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))
    return float(new_lr)

lr_scheduler = callbacks.LearningRateScheduler(cosine_lr_schedule, verbose=1)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)


# ==============================================================================
# 4. TRAIN THE MODEL
# ==============================================================================
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
    class_weight=class_weight_dict,
    verbose=1
)


# ==============================================================================
# 5. EVALUATE (FINAL TEST)
# ==============================================================================
print("\nLoading best model for final evaluation...")
best_model = models.load_model(MODEL_SAVE_NAME)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n" + "="*50)
print(f"FINAL TEST RESULT (UNSEEN USER)")
print(f"Accuracy: {test_acc:.4f}")
print(f"Loss: {test_loss:.4f}")
print(f"="*50 + "\n")

y_pred_prob = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test_labels, y_pred)
print("Confusion Matrix:")
print(cm)

class_names = ['Walking', 'Upstairs', 'Downstairs', 'Idle']
print("\nPer-Class Performance Metrics:")
print(classification_report(y_test_labels, y_pred, target_names=class_names, digits=4))


# ==============================================================================
# 6. VISUALIZE RESULTS
# ==============================================================================
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Unseen User)')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (Unseen User)')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(range(4), class_names, rotation=45)
plt.yticks(range(4), class_names)

for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()


# ==============================================================================
# 7. SAVE ALL MODEL DATA AND METADATA
# ==============================================================================
metadata = {
    "timestamp": datetime.now().isoformat(),
    "model_version": "v6_64",
    "model_architecture": {
        "input_shape": (int(n_timesteps), int(n_features)),
        "n_timesteps": int(n_timesteps),
        "n_features": int(n_features),
        "n_outputs": int(n_outputs),
        "model_type": "1D CNN Sequential",
        "conv_filters": [CONV_FILTERS_BLOCK1, CONV_FILTERS_BLOCK2, CONV_FILTERS_BLOCK3],
        "kernel_sizes": [KERNEL_SIZE_BLOCK1, KERNEL_SIZE_BLOCK2, KERNEL_SIZE_BLOCK3],
        "pool_size": POOL_SIZE,
        "dropout_conv": DROPOUT_CONV,
        "dropout_dense": DROPOUT_DENSE,
        "dense_units": [DENSE_UNITS_1, DENSE_UNITS_2]
    },
    "training_config": {
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lr_schedule": "cosine_annealing",
        "label_smoothing": LABEL_SMOOTHING,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "l2_regularization": L2_REGULARIZATION,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy + label_smoothing",
        "metrics": ["accuracy"],
        "class_weights": {int(k): float(v) for k, v in class_weight_dict.items()},
        "data_augmentation": ["jitter", "scaling", "time_shift"]
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
        "training_strategy": "Leave-One-Subject-Out (LOSO)",
        "augmentation": "jitter + scaling + time_shift (3 copies)"
    }
}

with open(MODEL_REPORT_FILE, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL V6_64 - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {metadata['timestamp']}\n\n")
    
    f.write("ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Input Shape: {n_timesteps} timesteps x {n_features} features\n")
    f.write(f"Output Classes: {n_outputs}\n")
    f.write(f"Model Type: {metadata['model_architecture']['model_type']}\n")
    f.write(f"Conv Filters: {metadata['model_architecture']['conv_filters']}\n")
    f.write(f"Kernel Sizes: {metadata['model_architecture']['kernel_sizes']}\n")
    f.write(f"Pool Size: {POOL_SIZE}\n")
    f.write(f"Dropout (conv/dense): {DROPOUT_CONV}/{DROPOUT_DENSE}\n")
    f.write(f"Dense Units: {metadata['model_architecture']['dense_units']}\n\n")
    
    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Max Epochs: {MAX_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Initial Learning Rate: {LEARNING_RATE}\n")
    f.write(f"LR Schedule: Cosine Annealing\n")
    f.write(f"Label Smoothing: {LABEL_SMOOTHING}\n")
    f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
    f.write(f"L2 Regularization: {L2_REGULARIZATION}\n")
    f.write(f"Optimizer: {metadata['training_config']['optimizer']}\n")
    f.write(f"Loss Function: {metadata['training_config']['loss_function']}\n")
    f.write(f"Class Weights: {metadata['training_config']['class_weights']}\n")
    f.write(f"Data Augmentation: {metadata['training_config']['data_augmentation']}\n\n")
    
    f.write("DATASET\n")
    f.write("-" * 70 + "\n")
    f.write(f"Dataset File: {DATASET_FILE}\n")
    f.write(f"Training Samples: {metadata['dataset_info']['n_training_samples']}\n")
    f.write(f"Test Samples: {metadata['dataset_info']['n_test_samples']}\n")
    f.write(f"Strategy: {metadata['dataset_info']['training_strategy']}\n")
    f.write(f"Augmentation: {metadata['dataset_info']['augmentation']}\n\n")
    
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
    f.write(f"Classes: {class_names}\n")
    f.write(str(cm) + "\n\n")
    
    f.write("PER-CLASS METRICS\n")
    f.write("-" * 70 + "\n")
    f.write(classification_report(y_test_labels, y_pred, target_names=class_names, digits=4))
    f.write("\n")

print("\n" + "=" * 70)
print("ALL MODEL DATA SAVED SUCCESSFULLY")
print("=" * 70)
