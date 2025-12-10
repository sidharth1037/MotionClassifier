import numpy as np
from keras import layers, models
import matplotlib.pyplot as plt

# load prepared dataset
data = np.load("motion_dataset_splits.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# build CNN model
model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=(100, 6)),
    layers.Conv1D(128, kernel_size=3, activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(y_train.shape[1], activation="softmax")  # 3 classes
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# save trained model
model.save("motion_model_1.test.h5")


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
