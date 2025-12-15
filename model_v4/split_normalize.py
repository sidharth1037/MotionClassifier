import numpy as np
import os

# ========== CONFIG ==========
DATA_FILE = "motion_windows_v4.npz" 
OUT_FILE = "motion_dataset_loso_v4.npz"
# ============================

# 1) load
assert os.path.exists(DATA_FILE), f"{DATA_FILE} not found"
data = np.load(DATA_FILE)
X = data["X"]       
y = data["y"]       
users = data["users"] 

print("Loaded:", DATA_FILE)
print("  X.shape =", X.shape)
print("  users found:", np.unique(users))

# 2) inspect classes
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique.tolist(), counts.tolist()))
print("Class distribution (label:count):", class_counts)

# 3) LEAVE-ONE-SUBJECT-OUT SPLIT (SMART SELECTION)
unique_users = np.unique(users)

if len(unique_users) > 1:
    print(f"\n--- LOSO MODE ACTIVATED ---")
    
    # --- SMART LOGIC START ---
    # We want a Test User who has performed ALL activities (or as many as possible).
    # We search backwards from the newest users.
    best_user = -1
    max_classes_found = 0
    
    # Check every user to see how "complete" their data is
    for uid in reversed(unique_users):
        # Get all labels belonging to this user
        user_labels = y[users == uid]
        unique_labels_for_user = np.unique(user_labels)
        count = len(unique_labels_for_user)
        
        # If we find a user with all 4 classes, stop immediately. They are perfect.
        if count == 4:
            best_user = uid
            max_classes_found = count
            break
        
        # Otherwise, keep track of the best one we've seen so far
        if count > max_classes_found:
            max_classes_found = count
            best_user = uid

    test_user_id = best_user
    print(f"Selected Best Test Candidate: User {test_user_id}")
    print(f"  - This user has data for {max_classes_found} out of 4 classes.")
    
    if max_classes_found < 4:
        print("  - [WARNING] This user is missing some activities! Test results will be partial.")

    # Apply masks based on the smart selection
    train_mask = users != test_user_id
    test_mask  = users == test_user_id

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    # --- SMART LOGIC END ---

else:
    print(f"\n[WARNING] Only 1 user found. Fallback to random split.")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

# --- Detailed Split Inspection ---
print("\nAfter split:")
print("  X_train.shape =", X_train.shape)
print("  X_test.shape  =", X_test.shape)

def print_counts(label_arr, name):
    u, c = np.unique(label_arr, return_counts=True)
    print(f"  {name} counts:", dict(zip(u, c)))

print_counts(y_train, "y_train")
print_counts(y_test, "y_test") # Verify this now has multiple classes!
# ---------------------------------------------

# 4) Normalization (Train set only)
n_features = X_train.shape[2]
train_flat = X_train.reshape(-1, n_features)

mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
std[std == 0] = 1.0

print("\nNormalization stats:")
for i, (m,s) in enumerate(zip(mean, std)):
    print(f"  feat[{i}] mean={m:.4f}, std={s:.4f}")

X_train_n = (X_train - mean) / std
X_test_n  = (X_test - mean) / std

# 5) One-hot encode
num_classes = 4 
y_train_idx = y_train.astype(int)
y_test_idx  = y_test.astype(int)

y_train_oh = np.eye(num_classes)[y_train_idx]
y_test_oh  = np.eye(num_classes)[y_test_idx]

print(f"\nOne-hot shapes: {y_train_oh.shape}, {y_test_oh.shape}")

# 6) Save
np.savez_compressed(
    OUT_FILE,
    X_train=X_train_n,
    X_test=X_test_n,
    y_train=y_train_oh,
    y_test=y_test_oh,
    mean=mean,
    std=std,
    classes=np.unique(y)
)
print(f"Saved LOSO dataset to: {OUT_FILE}")