import numpy as np
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# ========== CONFIG ==========
# V3 UPDATE: Make sure this matches the output of split_normalize_loso.py
DATASET_FILE = "motion_dataset_loso.npz"
MODEL_SAVE_NAME = "best_motion_model_v3.keras"
# ============================

# 1. Load Data
assert os.path.exists(DATASET_FILE), f"Error: {DATASET_FILE} not found. Did you run split_normalize_loso.py?"
data = np.load(DATASET_FILE)

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Check input shape
n_timesteps = X_train.shape[1]
n_features = X_train.shape[2] # Should be 4 (AccX, AccY, AccZ, Mag)
n_outputs = y_train.shape[1]  # Should be 3 (Walk, Up, Down)

print(f"Loaded: {DATASET_FILE}")
print(f"Input Shape: {n_timesteps} steps, {n_features} features")
print(f"Training on {len(X_train)} samples")
print(f"Validating on {len(X_test)} samples (Unseen User)")

# 2. Build Robust CNN Model
# This architecture is excellent for LOSO because the Dropout layers
# prevent it from memorizing specific user 'quirks'.
model = models.Sequential([
    # First Conv Block
    layers.Input(shape=(n_timesteps, n_features)),
    layers.Conv1D(filters=64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3), # Essential for generalizing to new users

    # Second Conv Block
    layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # Third Conv Block
    layers.Conv1D(filters=64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.GlobalAveragePooling1D(), 

    # Output Block
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_outputs, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Callbacks
checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_NAME, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15, # Increased patience slightly for LOSO as it can be noisier
    restore_best_weights=True
)

# 4. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60, 
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# 5. Evaluate
# Load the best saved model to ensure we test the optimal version
best_model = models.load_model(MODEL_SAVE_NAME)
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n--- FINAL RESULTS ---")
print(f"Test Accuracy on UNSEEN USER: {test_acc:.4f}")

if test_acc < 0.85:
    print("Warning: Accuracy is below 85%. You may need more diverse training subjects.")
else:
    print("Success: Model generalizes well to new users!")

# 6. Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val (Unseen User)')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val (Unseen User)')
plt.title('Loss')
plt.legend()
plt.show()