# save this as split_normalize_loso.py and run: python split_normalize_loso.py
import numpy as np
import os

# ========== CONFIG ==========
# V3 UPDATE: Loading the file that contains User IDs
DATA_FILE = "motion_windows_with_users.npz" 
OUT_FILE = "motion_dataset_loso.npz"
# ============================

# 1) load
assert os.path.exists(DATA_FILE), f"{DATA_FILE} not found"
data = np.load(DATA_FILE)
X = data["X"]       # shape (N_windows, window_size, n_features)
y = data["y"]       # shape (N_windows,)
users = data["users"] # V3 UPDATE: Load user IDs

print("Loaded:", DATA_FILE)
print("  X.shape =", X.shape)
print("  y.shape =", y.shape)
print("  users found:", np.unique(users))

# 2) inspect classes
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique.tolist(), counts.tolist()))
print("Class distribution (label:count):", class_counts)

# 3) LEAVE-ONE-SUBJECT-OUT SPLIT (V3 Improvement)
# Instead of random shuffling, we split by User ID.
unique_users = np.unique(users)

if len(unique_users) > 1:
    # STRATEGY: Pick the highest User ID as the "Unseen Test Subject"
    test_user_id = np.max(unique_users)
    print(f"\n--- LOSO MODE ACTIVATED ---")
    print(f"Training on Users: {unique_users[unique_users != test_user_id]}")
    print(f"Testing on User  : {test_user_id} (The model has NEVER seen this person)")

    # Create masks
    train_mask = users != test_user_id
    test_mask  = users == test_user_id

    # Apply masks
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
else:
    # Fallback if you only have 1 user (e.g., old dataset)
    print(f"\n[WARNING] Only 1 user found ({unique_users[0]}). Cannot perform LOSO.")
    print("Falling back to random train_test_split (V2 legacy mode).")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

print("\nAfter split:")
print("  X_train.shape =", X_train.shape)
print("  X_test.shape  =", X_test.shape)
# Use np.unique safely for counts
train_labels, train_counts = np.unique(y_train, return_counts=True)
test_labels, test_counts = np.unique(y_test, return_counts=True)
print("  y_train counts:", dict(zip(train_labels, train_counts)))
print("  y_test  counts:", dict(zip(test_labels, test_counts)))

# 4) compute normalization stats on TRAIN set only (CRITICAL for LOSO)
#    We must not let the Test User's walking style influence the mean/std
n_train, win_len, n_features = X_train.shape
train_flat = X_train.reshape(-1, n_features)   # shape (n_train * win_len, n_features)

mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
# avoid divide-by-zero
std[std == 0] = 1.0

print("\nNormalization stats (calculated on Training Users only):")
for i, (m,s) in enumerate(zip(mean, std)):
    print(f"  feat[{i}] mean={m:.4f}, std={s:.4f}")

# 5) normalize function
def normalize(X, mean, std):
    return (X - mean[None, None, :]) / std[None, None, :]

X_train_n = normalize(X_train, mean, std)
X_test_n  = normalize(X_test, mean, std)

# 6) one-hot encode labels
classes = np.unique(y) # Should be [0, 1, 2]
num_classes = len(classes)

# Safety: ensure 0,1,2 map to 0,1,2 even if some are missing
y_train_idx = y_train.astype(int)
y_test_idx  = y_test.astype(int)

y_train_oh = np.eye(num_classes)[y_train_idx]
y_test_oh  = np.eye(num_classes)[y_test_idx]

print(f"\nNumber of classes: {num_classes}. One-hot shapes: {y_train_oh.shape}, {y_test_oh.shape}")

# 7) save prepared dataset
np.savez_compressed(
    OUT_FILE,
    X_train=X_train_n,
    X_test=X_test_n,
    y_train=y_train_oh,
    y_test=y_test_oh,
    mean=mean,
    std=std,
    classes=classes
)
print(f"\nSaved LOSO dataset to: {OUT_FILE}")