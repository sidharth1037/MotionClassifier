import numpy as np
import os

# ========== CONFIG ==========
DATA_FILE = "motion_windows_v7.npz"
OUT_FILE = "motion_dataset_loso_v7.npz"
NUM_CLASSES = 4
MIN_TEST_SAMPLES = 50
MIN_CLASSES_IN_TEST = 3

# ==============================================================================
# AUGMENTATION CONFIG
# ==============================================================================
# V6 had augmentation config vars but NO actual augmentation code.
# V7 implements it properly + adds walking-specific augmentation.
#
# Standard augmentation (applied to ALL classes):
AUGMENT_JITTER_STD = 0.03        # Gaussian noise std (slightly lower — new features are sensitive)
AUGMENT_SCALE_RANGE = (0.9, 1.1) # Amplitude scaling range
AUGMENT_TIME_SHIFT_MAX = 8       # Max samples to shift (reduced from 10 — shorter jerk features)
AUGMENT_COPIES = 2               # Number of augmented copies per original sample

# Walking-specific augmentation:
# V7.0 had extra walking copies + aggressive mixup + 1.5x weight boost, which
# overcorrected: walking recall hit 93% but upstairs recall dropped to 73%.
# V7.1 fix: Remove extra copies (the mixup + weight boost alone are enough).
# The feature engineering (JerkZ, AccZ_detrended) does the heavy lifting.
WALKING_EXTRA_COPIES = 0         # Removed: was causing walking to dominate training
WALKING_CLASS_ID = 0
UPSTAIRS_CLASS_ID = 1

# Inter-class mixup between walking and upstairs:
# Creates synthetic "hard" walking examples by blending walking & upstairs at a
# high walking ratio. This pushes the decision boundary AWAY from walking and
# toward upstairs, teaching the model "this is still walking even though it
# looks a bit like upstairs".
# V7.1: Softened from 85/15 to 90/10 — the 85/15 blend was too aggressive and
# made the model think real upstairs samples were walking.
MIXUP_ALPHA = 0.90               # 90% walking, 10% upstairs (was 85/15 — too aggressive)
MIXUP_COPIES_PER_WALKING = 1     # 1 mixup sample per walking sample
# ============================

print("=" * 70)
print("DATASET SPLITTING, AUGMENTATION & NORMALIZATION - LOSO Validation (V7)")
print("=" * 70)

# ==============================================================================
# 1) LOAD DATA
# ==============================================================================
assert os.path.exists(DATA_FILE), f"ERROR: {DATA_FILE} not found"
data = np.load(DATA_FILE)
X = data["X"]       
y = data["y"]       
users = data["users"] 

print("\n1. DATA LOADING")
print(f"   Loaded: {DATA_FILE}")
print(f"   X.shape: {X.shape}")
print(f"   Features per window: {X.shape[2]} (AccX, AccY, AccZ, Mag, JerkZ, AccZ_detrended)")
print(f"   User IDs: {sorted(np.unique(users).astype(int))}")

# ==============================================================================
# 2) CLASS DISTRIBUTION (before split)
# ==============================================================================
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique.astype(int).tolist(), counts.tolist()))
class_names = ['Walking', 'Upstairs', 'Downstairs', 'Idle']

print("\n2. CLASS DISTRIBUTION (before split)")
for class_id, count in sorted(class_counts.items()):
    pct = (count / len(y) * 100)
    print(f"   {class_names[class_id]:12} (label {class_id}): {count:6} windows ({pct:5.1f}%)")

# ==============================================================================
# 3) LEAVE-ONE-SUBJECT-OUT SPLIT
# ==============================================================================
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
print("\n4. SPLIT RESULTS (before augmentation)")
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


# ==============================================================================
# 5) DATA AUGMENTATION
# ==============================================================================
# V7.2 Strategy:
#   A) Standard augmentation (jitter, scale, time_shift) for ALL classes
#   B) NO walking extra copies (removed — was biasing the class distribution)
#   C) Bidirectional SOFT-LABEL mixup between walking & upstairs (see step 7b)
#
# The soft-label mixup is applied AFTER one-hot encoding because the mixed
# samples need soft labels like [0.9, 0.1, 0, 0] instead of hard [1, 0, 0, 0].
# ==============================================================================

print("\n5. DATA AUGMENTATION")

def augment_jitter(X, sigma=AUGMENT_JITTER_STD):
    """
    Add random Gaussian noise to the signal.
    Simulates sensor noise and slight variations in phone placement.
    """
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def augment_scale(X, scale_range=AUGMENT_SCALE_RANGE):
    """
    Randomly scale the amplitude of the signal.
    Simulates different step intensities (heavy vs light walker).
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return X * scale

def augment_time_shift(X, max_shift=AUGMENT_TIME_SHIFT_MAX):
    """
    Randomly shift the signal forward or backward in time (circular shift).
    Simulates different phase alignments within the window.
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(X, shift, axis=0)

def augment_sample(x):
    """Apply all three augmentations to a single sample."""
    x_aug = augment_jitter(x)
    x_aug = augment_scale(x_aug)
    x_aug = augment_time_shift(x_aug)
    return x_aug

# --- Standard augmentation for ALL classes (equal treatment) ---
print(f"   Standard augmentation: {AUGMENT_COPIES} copies per sample (all classes)")
print(f"      - Jitter (std={AUGMENT_JITTER_STD})")
print(f"      - Scaling ({AUGMENT_SCALE_RANGE})")
print(f"      - Time shift (+/-{AUGMENT_TIME_SHIFT_MAX} samples)")

aug_X_list = [X_train.copy()]   # Start with original data
aug_y_list = [y_train.copy()]

for copy_i in range(AUGMENT_COPIES):
    X_aug = np.array([augment_sample(x) for x in X_train])
    aug_X_list.append(X_aug)
    aug_y_list.append(y_train.copy())
    print(f"      Copy {copy_i + 1}: {len(X_aug)} augmented samples")

# Combine standard augmentation
X_train = np.concatenate(aug_X_list, axis=0)
y_train = np.concatenate(aug_y_list, axis=0)

print(f"\n   After standard augmentation: {len(X_train)} samples")
print(f"   Class distribution (equal augmentation, no bias):")
for class_id in range(NUM_CLASSES):
    count = np.sum(y_train == class_id)
    pct = (count / len(y_train) * 100) if len(y_train) > 0 else 0
    print(f"     {class_names[class_id]:12}: {count:6} ({pct:5.1f}%)")


# ==============================================================================
# 6) NORMALIZATION (computed from training set only, BEFORE mixup)
# ==============================================================================
# We normalize BEFORE mixup so that the blended samples are in the same
# normalized space. Also, normalization stats should come from real data only.
print("\n6. NORMALIZATION")
n_features = X_train.shape[2]
train_flat = X_train.reshape(-1, n_features)

mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
std[std == 0] = 1.0  # Prevent division by zero

print(f"   Computing stats from training set (real + augmented data)...")
print(f"   Feature statistics:")
feature_names = ['AccX', 'AccY', 'AccZ', 'Magnitude', 'JerkZ', 'AccZ_detrend']
for i, (m, s) in enumerate(zip(mean, std)):
    print(f"     {feature_names[i]:14}: mean={m:8.4f}, std={s:8.4f}")

X_train_n = (X_train - mean) / std
X_test_n  = (X_test - mean) / std


# ==============================================================================
# 7) ONE-HOT ENCODE + BIDIRECTIONAL SOFT-LABEL MIXUP
# ==============================================================================
# 7a) Standard one-hot encoding for all existing samples
print("\n7. ONE-HOT ENCODING + SOFT-LABEL MIXUP")
y_train_idx = y_train.astype(int)
y_test_idx  = y_test.astype(int)

y_train_oh = np.eye(NUM_CLASSES)[y_train_idx]
y_test_oh  = np.eye(NUM_CLASSES)[y_test_idx]

print(f"   7a) One-hot encoded: train={y_train_oh.shape}, test={y_test_oh.shape}")

# --------------------------------------------------------------------------
# 7b) BIDIRECTIONAL SOFT-LABEL MIXUP (walking <-> upstairs)
# --------------------------------------------------------------------------
# WHY THIS IS DIFFERENT FROM V7.0/V7.1:
#
# Previous approach: One-directional hard-label mixup
#   - Blended walking + upstairs samples, ALL labeled as walking
#   - Problem: This told the model "anything near the boundary is walking",
#     which caused upstairs samples to be misclassified as walking
#
# New approach: Bidirectional SOFT-label mixup
#   - For each walking sample, blend with random upstairs:
#     x_mix = 0.9*walking + 0.1*upstairs, label = [0.9, 0.1, 0, 0]
#   - For each upstairs sample, blend with random walking:
#     x_mix = 0.9*upstairs + 0.1*walking, label = [0.1, 0.9, 0, 0]
#
# This teaches the model:
#   1. The BOUNDARY between walking and upstairs (what do ambiguous samples look like?)
#   2. From BOTH sides equally (no bias toward either class)
#   3. With UNCERTAINTY (soft labels say "this is 90% walking, 10% upstairs"
#      instead of "this is definitely walking")
#
# The soft labels are crucial: they prevent the model from becoming overconfident
# about boundary samples, which is what was causing the misclassification.
# --------------------------------------------------------------------------

walking_mask_n = y_train_idx == WALKING_CLASS_ID
upstairs_mask_n = y_train_idx == UPSTAIRS_CLASS_ID
X_walking_n = X_train_n[walking_mask_n]
X_upstairs_n = X_train_n[upstairs_mask_n]

print(f"\n   7b) Bidirectional soft-label mixup (walking <-> upstairs)")
print(f"       Alpha: {MIXUP_ALPHA} (dominant class proportion)")
print(f"       Walking samples: {len(X_walking_n)}")
print(f"       Upstairs samples: {len(X_upstairs_n)}")

mixup_X_list = []
mixup_y_list = []  # These will be SOFT labels (not integer indices)

if len(X_walking_n) > 0 and len(X_upstairs_n) > 0:
    
    # Direction 1: Walking-dominant blends
    # x = 0.9*walking + 0.1*upstairs → label = [0.9, 0.1, 0, 0]
    walking_soft_label = np.zeros(NUM_CLASSES)
    walking_soft_label[WALKING_CLASS_ID] = MIXUP_ALPHA         # 0.9
    walking_soft_label[UPSTAIRS_CLASS_ID] = 1 - MIXUP_ALPHA    # 0.1
    
    for x_walk in X_walking_n:
        x_up = X_upstairs_n[np.random.randint(len(X_upstairs_n))]
        x_mixed = MIXUP_ALPHA * x_walk + (1 - MIXUP_ALPHA) * x_up
        mixup_X_list.append(x_mixed)
        mixup_y_list.append(walking_soft_label.copy())
    
    print(f"       Direction 1 (walking-dominant): {len(X_walking_n)} samples")
    print(f"         Label: {walking_soft_label}")
    
    # Direction 2: Upstairs-dominant blends
    # x = 0.9*upstairs + 0.1*walking → label = [0.1, 0.9, 0, 0]
    upstairs_soft_label = np.zeros(NUM_CLASSES)
    upstairs_soft_label[UPSTAIRS_CLASS_ID] = MIXUP_ALPHA       # 0.9
    upstairs_soft_label[WALKING_CLASS_ID] = 1 - MIXUP_ALPHA    # 0.1
    
    for x_up in X_upstairs_n:
        x_walk = X_walking_n[np.random.randint(len(X_walking_n))]
        x_mixed = MIXUP_ALPHA * x_up + (1 - MIXUP_ALPHA) * x_walk
        mixup_X_list.append(x_mixed)
        mixup_y_list.append(upstairs_soft_label.copy())
    
    print(f"       Direction 2 (upstairs-dominant): {len(X_upstairs_n)} samples")
    print(f"         Label: {upstairs_soft_label}")
    
    # Append mixup samples to training data
    X_mixup = np.array(mixup_X_list)
    y_mixup = np.array(mixup_y_list)
    
    X_train_n = np.concatenate([X_train_n, X_mixup], axis=0)
    y_train_oh = np.concatenate([y_train_oh, y_mixup], axis=0)
    
    total_mixup = len(X_walking_n) + len(X_upstairs_n)
    print(f"\n       Total mixup samples added: {total_mixup}")
else:
    print(f"       Skipped: insufficient walking or upstairs samples")

# Shuffle the final training set
shuffle_idx = np.random.permutation(len(X_train_n))
X_train_n = X_train_n[shuffle_idx]
y_train_oh = y_train_oh[shuffle_idx]

print(f"\n   FINAL TRAINING SET: {len(X_train_n)} samples")
print(f"   (includes {len(mixup_X_list)} soft-labeled boundary samples)")


# ==============================================================================
# 8) SAVE
# ==============================================================================
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
print(f"   Saved to: {OUT_FILE}")

print("\n" + "=" * 70)
print("NEXT STEP: Run 'python model_v7.py' to train the model")
print("=" * 70)
