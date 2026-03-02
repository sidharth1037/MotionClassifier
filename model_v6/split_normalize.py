import numpy as np
import os

# ========== CONFIG ==========
DATA_FILE = "motion_windows_v6.npz"
OUT_FILE = "motion_dataset_loso_v6.npz"
NUM_CLASSES = 4
MIN_TEST_SAMPLES = 50
MIN_CLASSES_IN_TEST = 3

# Augmentation config
AUGMENT_JITTER_STD = 0.05     # Gaussian noise std
AUGMENT_SCALE_RANGE = (0.9, 1.1)  # Amplitude scaling range
AUGMENT_TIME_SHIFT_MAX = 10   # Max samples to shift
AUGMENT_COPIES = 2            # Number of augmented copies per original sample
# ============================

print("=" * 70)
print("DATASET SPLITTING, AUGMENTATION & NORMALIZATION - LOSO Validation (V6)")
print("=" * 70)

# 1) Load data
assert os.path.exists(DATA_FILE), f"ERROR: {DATA_FILE} not found"
data = np.load(DATA_FILE)
X = data["X"]       
y = data["y"]       
users = data["users"] 

print("\n1. DATA LOADING")
print(f"   Loaded: {DATA_FILE}")
print(f"   X.shape: {X.shape}")
print(f"   User IDs: {sorted(np.unique(users).astype(int))}")

# 2) Inspect class distribution
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique.astype(int).tolist(), counts.tolist()))
class_names = ['Walking', 'Upstairs', 'Downstairs', 'Idle']

print("\n2. CLASS DISTRIBUTION (before split)")
for class_id, count in sorted(class_counts.items()):
    pct = (count / len(y) * 100)
    print(f"   {class_names[class_id]:12} (label {class_id}): {count:6} windows ({pct:5.1f}%)")

# 3) LEAVE-ONE-SUBJECT-OUT SPLIT (SMART SELECTION)
unique_users = np.unique(users)

if len(unique_users) > 1:
    print(f"\n3. LOSO SPLIT - Selecting test user...")
    
    # Find the best test candidate (user with most diverse activities)
    best_user = -1
    max_classes_found = 0
    
    for uid in reversed(unique_users):
        user_labels = y[users == uid]
        unique_labels_for_user = np.unique(user_labels)
        count = len(unique_labels_for_user)
        
        if count == NUM_CLASSES:
            best_user = uid
            max_classes_found = count
            break
        
        if count > max_classes_found:
            max_classes_found = count
            best_user = uid

    test_user_id = best_user
    print(f"   Selected User {int(test_user_id)}")
    print(f"   - Has {max_classes_found} out of {NUM_CLASSES} activities")
    
    # Quality checks
    if max_classes_found < NUM_CLASSES:
        print(f"   ⚠ WARNING: Test user missing {NUM_CLASSES - max_classes_found} activities!")
    
    # Apply masks
    train_mask = users != test_user_id
    test_mask  = users == test_user_id

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

else:
    print(f"\n3. INSUFFICIENT USERS - Using random 80/20 split")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

# Quality checks on split
print("\n4. SPLIT RESULTS")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

if len(X_test) < MIN_TEST_SAMPLES:
    print(f"   ⚠ WARNING: Test set very small ({len(X_test)} < {MIN_TEST_SAMPLES})")

# Check test set class distribution
test_unique, test_counts = np.unique(y_test, return_counts=True)
test_class_dist = dict(zip(test_unique.astype(int).tolist(), test_counts.tolist()))
classes_in_test = len(test_class_dist)

print(f"   Classes in test set: {classes_in_test}/{NUM_CLASSES}")
if classes_in_test < MIN_CLASSES_IN_TEST:
    print(f"   ⚠ WARNING: Test set missing {NUM_CLASSES - classes_in_test} classes!")

print("\n   Training distribution:")
for class_id in range(NUM_CLASSES):
    count = np.sum(y_train == class_id)
    pct = (count / len(y_train) * 100) if len(y_train) > 0 else 0
    print(f"     {class_names[class_id]:12}: {count:6} ({pct:5.1f}%)")

print("\n   Test distribution:")
for class_id in range(NUM_CLASSES):
    count = np.sum(y_test == class_id)
    pct = (count / len(y_test) * 100) if len(y_test) > 0 else 0
    status = "✓" if count > 0 else "✗"
    print(f"     {class_names[class_id]:12}: {count:6} ({pct:5.1f}%) {status}")

# 4) Normalization (Train set statistics only — computed AFTER augmentation)
print("\n6. NORMALIZATION")
n_features = X_train.shape[2]
train_flat = X_train.reshape(-1, n_features)

mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
std[std == 0] = 1.0

print(f"   Computing stats from training set...")
print(f"   Feature statistics:")
feature_names = ['AccX', 'AccY', 'AccZ', 'Magnitude']
for i, (m, s) in enumerate(zip(mean, std)):
    print(f"     {feature_names[i]:10}: mean={m:8.4f}, std={s:8.4f}")

X_train_n = (X_train - mean) / std
X_test_n  = (X_test - mean) / std

# 5) One-hot encode
print("\n7. ONE-HOT ENCODING")
y_train_idx = y_train.astype(int)
y_test_idx  = y_test.astype(int)

y_train_oh = np.eye(NUM_CLASSES)[y_train_idx]
y_test_oh  = np.eye(NUM_CLASSES)[y_test_idx]

print(f"   Train shape: {y_train_oh.shape}")
print(f"   Test shape: {y_test_oh.shape}")

# 6) Save
print("\n8. SAVING")
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
print(f"   ✓ Saved to: {OUT_FILE}")

print("\n" + "=" * 70)
print("NEXT STEP: Run 'python model_v6.py' to train the model")
print("=" * 70)