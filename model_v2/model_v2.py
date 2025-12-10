import numpy as np
from keras import layers, models, callbacks
import matplotlib.pyplot as plt

# 1. Load Data
data = np.load("motion_dataset_splits.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Check input shape (should be window_size x 4)
n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
n_outputs = y_train.shape[1]

print(f"Input Shape: {n_timesteps} steps, {n_features} features")

# 2. Build Robust CNN Model
model = models.Sequential([
    # First Conv Block
    layers.Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(n_timesteps, n_features)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3), # Drops 30% of connections to prevent overfitting

    # Second Conv Block
    layers.Conv1D(filters=128, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # Third Conv Block (Captures higher level patterns)
    layers.Conv1D(filters=64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.GlobalAveragePooling1D(), # Better than Flatten for time series

    # Output Block
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_outputs, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Callbacks (Save only the best model)
checkpoint = callbacks.ModelCheckpoint(
    'best_motion_model.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10, # Stop if no improvement for 10 epochs
    restore_best_weights=True
)

# 4. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,          # Increased epochs (early stopping will handle it)
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# 5. Evaluate
# Load the best saved model first
best_model = models.load_model('best_motion_model.keras')
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 6. Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()