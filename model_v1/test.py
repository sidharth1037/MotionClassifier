# import numpy as np

# data = np.load("motion_windows.npz")
# X, y = data["X"], data["y"]

# print("X.shape:", X.shape, "y.shape:", y.shape)

# # 👇 Check first few timesteps of first window
# print("First 5 timesteps of first window:\n", X[0, :5])

# print("\nGlobal stats per feature (before normalization):")
# for i in range(X.shape[2]):
#     col = X[:, :, i].ravel()   # flatten all windows
#     print(f"  Feature[{i}] min={np.min(col):.4f}, max={np.max(col):.4f}, mean={np.mean(col):.4f}, std={np.std(col):.4f}")

import numpy as np
from sklearn.model_selection import train_test_split

# === Load windowed data ===
data = np.load("C:\\Users\\sidha\\Desktop\\Main Project\\raw_data\\motion_windows.npz")
X, y = data["X"], data["y"]

print("Loaded:", X.shape, y.shape)

# === Sanity check ===
print("NaN count before normalization:", np.isnan(X).sum())

# === Normalize per-feature ===
# Compute stats across entire dataset (all windows, all timesteps)
X_reshaped = X.reshape(-1, X.shape[-1])   # (num_windows*window_size, num_features)
mean = np.mean(X_reshaped, axis=0)
std = np.std(X_reshaped, axis=0)

# Avoid divide by zero
std[std == 0] = 1.0

X_norm = (X - mean) / std

print("NaN count after normalization:", np.isnan(X_norm).sum())
print("Global stats after normalization:")
for i in range(X.shape[-1]):
    print(f"  Feature[{i}] mean={X_norm[..., i].mean():.3f}, std={X_norm[..., i].std():.3f}")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y
)

print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
