import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_FILE = "motion_dataset_loso_v7.npz"
MODEL_SAVE_NAME = "best_motion_model_v7.keras"

# Architecture
CONV_FILTERS_BLOCK1 = 64
CONV_FILTERS_BLOCK2 = 128
CONV_FILTERS_BLOCK3 = 128
KERNEL_SIZE_BLOCK1 = 5           # Wider kernel: captures more context per step
KERNEL_SIZE_BLOCK2 = 3
KERNEL_SIZE_BLOCK3 = 3
POOL_SIZE = 2
DROPOUT_CONV = 0.3
DROPOUT_DENSE = 0.4
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64

# SE (Squeeze-and-Excitation) attention block config
# See build section for explanation
SE_REDUCTION_RATIO = 8

# Training hyperparameters
MAX_EPOCHS = 120                 # Slightly more: focal loss converges slower
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30    # More patience for focal loss convergence
L2_REGULARIZATION = 0.0005

# Focal Loss hyperparameters (replaces label smoothing)
# See section 2 for detailed explanation
FOCAL_ALPHA = 0.25               # Base class weight factor
FOCAL_GAMMA = 1.5                # Focus factor (reduced from 2.0 — gentler focus on hard examples)

# Walking class weight boost
# V7.2: Set to 1.0 (disabled). The bidirectional soft-label mixup now handles
# the walking/upstairs boundary symmetrically. Asymmetric weight boosting was
# causing the model to swing between over-predicting walking or upstairs.
WALKING_WEIGHT_BOOST = 1.0

MODEL_REPORT_FILE = "model_v7_report.txt"


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
print(f"  Features: AccX, AccY, AccZ, Magnitude, JerkZ, AccZ_detrended")
print(f"Classes found: {n_outputs}")
print(f"Training on {len(X_train)} samples (with augmentation)")
print(f"Validating on {len(X_test)} samples (Unseen User)")


# ==============================================================================
# 2. FOCAL LOSS FUNCTION
# ==============================================================================
# WHY FOCAL LOSS INSTEAD OF LABEL-SMOOTHED CROSSENTROPY?
#
# Label smoothing (v6) uniformly softens ALL labels. This is a general
# regularizer but doesn't help with the specific walking/upstairs problem.
#
# Focal Loss was designed for exactly this scenario — class confusion where
# some examples are "easy" (idle, downstairs) and some are "hard" (walking
# that looks like upstairs).
#
# Standard cross-entropy:  CE(p) = -log(p)
# Focal loss:              FL(p) = -alpha * (1-p)^gamma * log(p)
#
# The (1-p)^gamma factor DOWNWEIGHTS easy examples (high p → near 0) and
# UPWEIGHTS hard examples (low p → near 1). With gamma=2:
#   - A sample predicted with 95% confidence contributes 0.25% of normal loss
#   - A sample predicted with 50% confidence contributes 25% of normal loss
#   - A sample predicted with 20% confidence contributes 64% of normal loss
#
# This means the model stops wasting capacity on "obviously idle" and
# focuses on "is this walking or upstairs?".
# ==============================================================================

def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Focal loss for multi-class classification.
    
    Args:
        gamma: Focus parameter. Higher = more focus on hard examples.
               0 = standard cross-entropy. 2 is the paper's recommended value.
        alpha: Base weight factor. Applied uniformly.
    """
    def focal_loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Standard cross-entropy per sample
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Focal weight: (1 - p_t)^gamma
        # p_t is the predicted probability for the TRUE class
        focal_weight = tf.pow(1 - y_pred, gamma)
        
        # Combine: focal_loss = alpha * focal_weight * cross_entropy
        focal_loss_value = alpha * focal_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss_value, axis=-1))
    
    return focal_loss_fn


# ==============================================================================
# 3. BUILD THE V7 CNN MODEL WITH SE ATTENTION
# ==============================================================================
# Key changes from V6:
# 
# 1. SE (Squeeze-and-Excitation) attention after each conv block
#    - WHY: Standard convolutions treat all feature channels equally. But for
#      our walking/upstairs problem, JerkZ and AccZ_detrended are MORE important
#      than, say, AccX. SE blocks let the model LEARN which channels matter.
#    - HOW: GlobalAvgPool → Dense(filters/ratio) → ReLU → Dense(filters) → Sigmoid
#      This produces a per-channel weight vector (e.g., [0.3, 0.9, 0.1, ...])
#      that scales each channel's importance.
#    - COST: Negligible — SE adds ~2% parameters. No latency impact on mobile.
#
# 2. Input shape: (96, 6) instead of (96, 4) due to new features
#
# 3. Focal loss instead of label-smoothed crossentropy (see section 2)
#
# 4. Boosted walking class weight (see section 4)
# ==============================================================================

def se_block(input_tensor, ratio=SE_REDUCTION_RATIO):
    """
    Squeeze-and-Excitation attention block.
    
    This learns to weight feature channels by importance. Think of it as
    the model saying "for distinguishing walking from upstairs, pay 90%
    attention to JerkZ and AccZ_detrended, and only 30% to AccX".
    
    Architecture:
    1. Squeeze: GlobalAvgPool reduces (batch, timesteps, channels) → (batch, channels)
       This captures the "average signal" in each channel.
    2. Excitation: Two Dense layers learn channel importance weights.
       - Dense(channels/ratio, ReLU): bottleneck to prevent overfitting
       - Dense(channels, Sigmoid): output weights between 0 and 1
    3. Scale: Multiply original features by learned weights.
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze: global average of each channel
    se = layers.GlobalAveragePooling1D()(input_tensor)
    
    # Excitation: learn channel importance
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    
    # Reshape for broadcasting: (batch, channels) → (batch, 1, channels)
    se = layers.Reshape((1, channels))(se)
    
    # Scale: multiply each channel by its learned importance weight
    return layers.Multiply()([input_tensor, se])


# Build model using Functional API (needed for SE blocks which have skip-like connections)
inputs = layers.Input(shape=(n_timesteps, n_features))

# --- BLOCK 1: Low-Level Feature Extraction ---
# Wider kernel (5) captures ~100ms of context per step at 50Hz.
# With 6 features, this first layer learns low-level patterns in ALL channels
# including the new JerkZ and AccZ_detrended.
x = layers.Conv1D(filters=CONV_FILTERS_BLOCK1, kernel_size=KERNEL_SIZE_BLOCK1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = se_block(x)                    # ← NEW: SE attention on block 1 features
x = layers.MaxPooling1D(pool_size=POOL_SIZE)(x)
x = layers.Dropout(DROPOUT_CONV)(x)

# --- BLOCK 2: Mid-Level Feature Extraction ---
# Combines block 1 features into step-level patterns.
x = layers.Conv1D(filters=CONV_FILTERS_BLOCK2, kernel_size=KERNEL_SIZE_BLOCK2, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = se_block(x)                    # ← NEW: SE attention on block 2 features
x = layers.MaxPooling1D(pool_size=POOL_SIZE)(x)
x = layers.Dropout(DROPOUT_CONV)(x)

# --- BLOCK 3: High-Level Feature Extraction ---
# 128 filters to capture rich high-level patterns for activity discrimination.
x = layers.Conv1D(filters=CONV_FILTERS_BLOCK3, kernel_size=KERNEL_SIZE_BLOCK3, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = se_block(x)                    # ← NEW: SE attention on block 3 features

# GlobalAveragePooling: Reduces to (n_filters,) — lightweight for mobile.
x = layers.GlobalAveragePooling1D()(x)

# --- OUTPUT BLOCK: Two-Layer Classification Head ---
x = layers.Dense(DENSE_UNITS_1, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))(x)
x = layers.Dropout(DROPOUT_DENSE)(x)

x = layers.Dense(DENSE_UNITS_2, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION))(x)
x = layers.Dropout(DROPOUT_CONV)(x)  # Lighter dropout before final layer

# Softmax output
outputs = layers.Dense(n_outputs, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Compile with Focal Loss
loss_fn = focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()


# ==============================================================================
# 4. CALLBACKS (TRAINING HELPERS)
# ==============================================================================
checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_NAME,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Cosine Annealing LR: Smoothly decays LR following a cosine curve.
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
# 5. CLASS WEIGHTS — WITH WALKING BOOST
# ==============================================================================
# sklearn's compute_class_weight('balanced') gives weights inversely proportional
# to class frequency. But we want walking to be penalized EVEN MORE because:
#
# 1. Walking→upstairs misclassification is our specific problem
# 2. Focal loss alone focuses on hard examples, but doesn't know which 
#    misclassification direction is worse
# 3. By boosting walking's weight by 1.5x, we tell the model:
#    "Getting walking wrong costs 50% more than getting other classes wrong"
#
# This is asymmetric: we DON'T boost upstairs because upstairs→walking is
# already rare (only 4 out of 106 in v6).
# ==============================================================================
from sklearn.utils.class_weight import compute_class_weight

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Boost walking class weight
class_weight_dict[0] *= WALKING_WEIGHT_BOOST

print("\nClass Weights (balanced + walking boost):")
class_names_list = ['Walking', 'Upstairs', 'Downstairs', 'Idle']
for i, name in enumerate(class_names_list):
    boost_note = f" (boosted {WALKING_WEIGHT_BOOST}x)" if i == 0 else ""
    print(f"  {name:12}: {class_weight_dict[i]:.4f}{boost_note}")

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
# 6. EVALUATE (FINAL TEST)
# ==============================================================================
print("\nLoading best model for final evaluation...")
best_model = models.load_model(MODEL_SAVE_NAME, custom_objects={'focal_loss_fn': focal_loss()})

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

# --- Walking-specific analysis ---
# Since walking→upstairs is our target problem, print extra diagnostics
walking_idx = 0
upstairs_idx = 1
walking_total = np.sum(y_test_labels == walking_idx)
walking_as_upstairs = cm[walking_idx][upstairs_idx]
walking_correct = cm[walking_idx][walking_idx]

print(f"\n{'='*50}")
print(f"WALKING -> UPSTAIRS CONFUSION ANALYSIS")
print(f"{'='*50}")
print(f"Total walking test samples: {walking_total}")
print(f"Correctly predicted walking: {walking_correct} ({walking_correct/walking_total*100:.1f}%)")
print(f"Mislabeled as upstairs:      {walking_as_upstairs} ({walking_as_upstairs/walking_total*100:.1f}%)")
if walking_total > 0:
    improvement = ((walking_correct/walking_total) - 0.7907) * 100  # vs v6 baseline
    print(f"Recall change vs V6:         {improvement:+.1f} percentage points")


# ==============================================================================
# 7. VISUALIZE RESULTS
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
# 8. SAVE ALL MODEL DATA AND METADATA
# ==============================================================================
metadata = {
    "timestamp": datetime.now().isoformat(),
    "model_version": "v7",
    "model_architecture": {
        "input_shape": (int(n_timesteps), int(n_features)),
        "n_timesteps": int(n_timesteps),
        "n_features": int(n_features),
        "n_outputs": int(n_outputs),
        "model_type": "1D CNN with SE Attention (Functional API)",
        "conv_filters": [CONV_FILTERS_BLOCK1, CONV_FILTERS_BLOCK2, CONV_FILTERS_BLOCK3],
        "kernel_sizes": [KERNEL_SIZE_BLOCK1, KERNEL_SIZE_BLOCK2, KERNEL_SIZE_BLOCK3],
        "pool_size": POOL_SIZE,
        "dropout_conv": DROPOUT_CONV,
        "dropout_dense": DROPOUT_DENSE,
        "dense_units": [DENSE_UNITS_1, DENSE_UNITS_2],
        "se_reduction_ratio": SE_REDUCTION_RATIO
    },
    "training_config": {
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lr_schedule": "cosine_annealing",
        "loss_function": f"focal_loss(gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA})",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "l2_regularization": L2_REGULARIZATION,
        "optimizer": "adam",
        "metrics": ["accuracy"],
        "class_weights": {int(k): float(v) for k, v in class_weight_dict.items()},
        "walking_weight_boost": WALKING_WEIGHT_BOOST,
        "data_augmentation": ["jitter", "scaling", "time_shift", "walking_extra_copies", "walking_upstairs_mixup"]
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
        "augmentation": "jitter + scaling + time_shift (2 copies) + walking extra (1 copy) + walking-upstairs mixup"
    },
    "v7_changes": {
        "new_features": ["JerkZ (vertical jerk)", "AccZ_detrended (gravity-removed Z)"],
        "feature_count": "6 (was 4 in v6)",
        "architecture": "SE attention blocks after each conv layer",
        "loss": "Focal loss (was label-smoothed CE in v6)",
        "walking_boost": f"{WALKING_WEIGHT_BOOST}x class weight boost for walking"
    }
}

with open(MODEL_REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL V7 - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {metadata['timestamp']}\n\n")
    
    f.write("V7 CHANGES (vs V6)\n")
    f.write("-" * 70 + "\n")
    f.write(f"New Features: JerkZ (vertical jerk), AccZ_detrended (gravity-removed Z)\n")
    f.write(f"Feature Count: 6 (was 4 in v6)\n")
    f.write(f"Architecture: SE attention blocks after each conv layer\n")
    f.write(f"Loss: Focal loss gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA} (was label-smoothed CE)\n")
    f.write(f"Walking Boost: {WALKING_WEIGHT_BOOST}x class weight multiplier\n")
    f.write(f"Augmentation: Actually implemented (was config-only in v6)\n\n")
    
    f.write("ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Input Shape: {n_timesteps} timesteps x {n_features} features\n")
    f.write(f"Output Classes: {n_outputs}\n")
    f.write(f"Model Type: {metadata['model_architecture']['model_type']}\n")
    f.write(f"Conv Filters: {metadata['model_architecture']['conv_filters']}\n")
    f.write(f"Kernel Sizes: {metadata['model_architecture']['kernel_sizes']}\n")
    f.write(f"Pool Size: {POOL_SIZE}\n")
    f.write(f"Dropout (conv/dense): {DROPOUT_CONV}/{DROPOUT_DENSE}\n")
    f.write(f"Dense Units: {metadata['model_architecture']['dense_units']}\n")
    f.write(f"SE Reduction Ratio: {SE_REDUCTION_RATIO}\n\n")
    
    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Max Epochs: {MAX_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Initial Learning Rate: {LEARNING_RATE}\n")
    f.write(f"LR Schedule: Cosine Annealing\n")
    f.write(f"Loss Function: {metadata['training_config']['loss_function']}\n")
    f.write(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}\n")
    f.write(f"L2 Regularization: {L2_REGULARIZATION}\n")
    f.write(f"Optimizer: {metadata['training_config']['optimizer']}\n")
    f.write(f"Class Weights: {metadata['training_config']['class_weights']}\n")
    f.write(f"Walking Weight Boost: {WALKING_WEIGHT_BOOST}x\n")
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

    f.write("WALKING -> UPSTAIRS CONFUSION ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Walking recall: {walking_correct/walking_total*100:.1f}% (V6: 79.1%)\n")
    f.write(f"Walking->Upstairs misclassification: {walking_as_upstairs/walking_total*100:.1f}% (V6: 18.6%)\n\n")
    
    f.write("PER-CLASS METRICS\n")
    f.write("-" * 70 + "\n")
    f.write(classification_report(y_test_labels, y_pred, target_names=class_names, digits=4))
    f.write("\n")

print("\n" + "=" * 70)
print("ALL MODEL DATA SAVED SUCCESSFULLY")
print("=" * 70)
