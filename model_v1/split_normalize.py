# save this as prepare_train_test.py and run: python prepare_train_test.py
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ========== CONFIG ==========
DATA_FILE = "motion_windows.npz"     # your single npz
OUT_FILE = "motion_dataset_splits.npz"
TEST_SIZE = 0.20
RANDOM_STATE = 42
# ============================

# 1) load
assert os.path.exists(DATA_FILE), f"{DATA_FILE} not found"
data = np.load(DATA_FILE)
X = data["X"]    # shape (N_windows, window_size, n_features)
y = data["y"]    # shape (N_windows,)

print("Loaded:", DATA_FILE)
print("  X.shape =", X.shape)
print("  y.shape =", y.shape)
print("  dtype:", X.dtype, y.dtype)

# 2) inspect classes
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique.tolist(), counts.tolist()))
print("Class distribution (label:count):", class_counts)

# 3) stratified split (keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("\nAfter split:")
print("  X_train.shape =", X_train.shape)
print("  X_test.shape  =", X_test.shape)
print("  y_train counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("  y_test  counts:", dict(zip(*np.unique(y_test, return_counts=True))))

# 4) compute normalization stats on TRAIN set only
#    compute mean/std per feature across all windows and time steps
n_train, win_len, n_features = X_train.shape
train_flat = X_train.reshape(-1, n_features)   # shape (n_train * win_len, n_features)
mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
# avoid divide-by-zero
std[std == 0] = 1.0

print("\nNormalization stats (per feature):")
for i, (m,s) in enumerate(zip(mean, std)):
    print(f"  feat[{i}] mean={m:.4f}, std={s:.4f}")

# 5) normalize function (broadcasting)
def normalize(X, mean, std):
    # X shape: (N, window_len, n_features)
    return (X - mean[None, None, :]) / std[None, None, :]

X_train_n = normalize(X_train, mean, std)
X_test_n  = normalize(X_test, mean, std)

print("\nAfter normalization sample stats:")
print("  X_train_n mean (approx) per feature:", X_train_n.reshape(-1, n_features).mean(axis=0))
print("  X_train_n std  (approx) per feature:", X_train_n.reshape(-1, n_features).std(axis=0))

# 6) one-hot encode labels (numpy only)
classes = np.unique(y)
num_classes = len(classes)
# build map from label -> index (in case labels are 0,1,2 already it's fine)
label_to_index = {lab: idx for idx, lab in enumerate(classes)}
y_train_idx = np.array([label_to_index[v] for v in y_train])
y_test_idx  = np.array([label_to_index[v] for v in y_test])

y_train_oh = np.eye(num_classes)[y_train_idx]
y_test_oh  = np.eye(num_classes)[y_test_idx]

print(f"\nNumber of classes: {num_classes}. One-hot shapes: {y_train_oh.shape}, {y_test_oh.shape}")

# 7) save prepared dataset (ready for model training)
np.savez_compressed(
    OUT_FILE,
    X_train=X_train_n,
    X_test=X_test_n,
    y_train=y_train_oh,
    y_test=y_test_oh,
    y_train_idx=y_train_idx,
    y_test_idx=y_test_idx,
    mean=mean,
    std=std,
    classes=classes
)
print(f"\nSaved prepared train/test dataset to: {OUT_FILE}")
