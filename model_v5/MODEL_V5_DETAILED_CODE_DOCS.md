# model_v5.py - Detailed Code Documentation

## Complete Walkthrough with Inline Explanations

---

## Section 0: Imports

```python
import numpy as np                          # Numerical operations on arrays
import tensorflow as tf                     # Deep learning framework
from keras import layers, models, callbacks # Neural network components
import matplotlib.pyplot as plt              # Data visualization
import os                                    # File system operations
from sklearn.metrics import confusion_matrix, classification_report  # ML metrics
from datetime import datetime                # Timestamp generation
```

**What Each Import Does:**
- `numpy`: Matrix operations, array manipulation, mathematical functions
- `tensorflow`: GPU-accelerated deep learning, auto-differentiation
- `keras.layers`: Building blocks (Conv1D, Dense, Dropout, etc.)
- `keras.models`: Container for connecting layers (Sequential, Functional)
- `keras.callbacks`: Training utilities (checkpointing, early stopping)
- `matplotlib.pyplot`: Generate plots for visualization
- `os.path.exists()`: Check if files exist before loading
- `confusion_matrix, classification_report`: Performance evaluation metrics
- `datetime.now()`: Generate timestamps for metadata

---

## Section 1: Configuration Block

```python
# ==============================================================================
# CONFIGURATION (from config.py)
# ==============================================================================
DATASET_FILE = "motion_dataset_loso_v5.npz"
MODEL_SAVE_NAME = "best_motion_model_v5.keras"

# Architecture
CONV_FILTERS_BLOCK1 = 64
CONV_FILTERS_BLOCK2 = 128
CONV_FILTERS_BLOCK3 = 64
KERNEL_SIZE = 3
POOL_SIZE = 2
DROPOUT_RATE = 0.5             
DENSE_UNITS = 64

# Training hyperparameters
MAX_EPOCHS = 60
BATCH_SIZE = 64                
LEARNING_RATE = 0.0005         
LR_DECAY_FACTOR = 0.5          
LR_DECAY_PATIENCE = 5          
EARLY_STOPPING_PATIENCE = 20   
L2_REGULARIZATION = 0.001      

MODEL_REPORT_FILE = "model_v5_report.txt"
```

### Parameter Explanations

#### File Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `DATASET_FILE` | `"motion_dataset_loso_v5.npz"` | Input data file from split_normalize.py; contains X_train, X_test, y_train, y_test |
| `MODEL_SAVE_NAME` | `"best_motion_model_v5.keras"` | Checkpoint file to save best model weights during training |
| `MODEL_REPORT_FILE` | `"model_v5_report.txt"` | Text file for detailed training report and metadata |

#### Architecture Parameters

**Convolutional Filters:**
```
CONV_FILTERS_BLOCK1 = 64   # Block 1 learns low-level patterns (peaks, slopes)
CONV_FILTERS_BLOCK2 = 128  # Block 2 learns mid-level patterns (full cycles)
CONV_FILTERS_BLOCK3 = 64   # Block 3 learns high-level patterns (activity signatures)
```

**Why these values?**
- 64 → 128 → 64: Pyramid shape compresses information progressively
- 128 in middle: Captures maximum feature diversity
- 64 at end: Reduces back down before dense layers
- Range 32-256 typical; these are well-balanced

**Other Architecture Parameters:**
```
KERNEL_SIZE = 3             # Conv filter looks at 3 consecutive timesteps
                            # Larger (5-7) = more context, fewer outputs
                            # Smaller (2) = less context, more positions
                            
POOL_SIZE = 2               # MaxPooling reduces length by 2x after each conv
                            # After 3 blocks: 128 → 64 → 32 → 16 timesteps
                            
DROPOUT_RATE = 0.5          # During training, 50% of neurons randomly disabled
                            # Prevents overfitting to specific users
                            # Range: 0.3-0.7 typical (higher = stronger regularization)
                            
DENSE_UNITS = 64            # Hidden layer before final output
                            # Smaller = less capacity, faster training
                            # Larger = more capacity, better for complex patterns
```

#### Training Hyperparameters

**Epochs and Batch Size:**
```
MAX_EPOCHS = 60             # Maximum iterations through entire dataset
                            # Usually stops earlier due to early stopping
                            
BATCH_SIZE = 64             # Samples processed before updating weights
                            # Larger (64) = less noisy gradients, faster training
                            # Smaller (16) = more updates per epoch, more memory efficient
```

**Learning Rate Schedule:**
```
LEARNING_RATE = 0.0005      # Initial step size for weight updates
                            # Too high (0.01) = unstable training, overshoots
                            # Too low (1e-6) = very slow convergence
                            # 0.0005 = conservative, stable
                            
LR_DECAY_FACTOR = 0.5       # Multiply learning rate by 0.5 when loss plateaus
                            # New LR = old_LR * 0.5
                            
LR_DECAY_PATIENCE = 5       # Wait 5 epochs without improvement before decaying LR
                            # Allows model time to escape local minima
```

**Early Stopping:**
```
EARLY_STOPPING_PATIENCE = 20  # Stop training if validation loss doesn't improve for 20 epochs
                              # Prevents wasting compute and overfitting
                              # Larger patience = wait longer for improvement
```

**Regularization:**
```
L2_REGULARIZATION = 0.001    # Weight penalty: Loss += 0.001 * sum(weights^2)
                             # Forces model to prefer smaller weights
                             # Prevents overfitting to noise
                             # Range: 0.0001-0.01 typical
```

---

## Section 2: Load and Inspect Data

```python
# ==============================================================================
# 1. LOAD AND INSPECT DATA
# ==============================================================================
assert os.path.exists(DATASET_FILE), f"Error: {DATASET_FILE} not found."

# Load the .npz archive
data = np.load(DATASET_FILE)

# Extract Training and Testing sets
# X = Input features (Sensor data windows)
# y = Output labels (One-hot encoded classes)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Auto-detect input dimensions from the loaded data
# n_timesteps: How long is one window? (e.g., 128 samples approx 2.56s)
n_timesteps = X_train.shape[1] 
# n_features: How many sensors? (e.g., 4: AccX, AccY, AccZ, Magnitude)
n_features = X_train.shape[2]  
# n_outputs: How many activities? (e.g., 4: Walk, Up, Down, Idle)
n_outputs = y_train.shape[1]   

print(f"Input Shape: {n_timesteps} steps, {n_features} features")
print(f"Classes found: {n_outputs}")
print(f"Training on {len(X_train)} samples")
print(f"Validating on {len(X_test)} samples (Unseen User)")
```

### What This Block Does:

1. **Assert File Exists**: 
   - Prevents confusing "file not found" error later during `np.load()`
   - Fails fast with clear error message

2. **Load NPZ Archive**:
   - `.npz` is compressed NumPy format (like ZIP for arrays)
   - Contains multiple arrays: X_train, X_test, y_train, y_test, mean, std, classes

3. **Extract Arrays**:
   ```
   X_train.shape = (n_train_samples, 128, 4)   # All training windows
   X_test.shape = (n_test_samples, 128, 4)     # All test windows (unseen user)
   y_train.shape = (n_train_samples, 4)        # One-hot encoded (e.g., [0,1,0,0])
   y_test.shape = (n_test_samples, 4)          # One-hot encoded
   ```

4. **Auto-Detect Dimensions**:
   ```
   n_timesteps = 128         # Window length (samples per window)
   n_features = 4            # AccX, AccY, AccZ, Magnitude
   n_outputs = 4             # Walking, Upstairs, Downstairs, Idle
   ```

### Why Auto-Detection?
- Makes code flexible: works with any window size or number of classes
- Prevents hardcoding (if you change process_raw_data.py, this still works)
- Enables reuse for different datasets

---

## Section 3: Build the CNN Model

```python
# ==============================================================================
# 2. BUILD THE ROBUST CNN MODEL
# ==============================================================================
# OVERVIEW: 1D Convolutional Neural Network with 3 progressive blocks
# Each block: Conv → BatchNorm → ReLU → MaxPool → Dropout
# Pattern: Learn patterns at different scales
# Input: (batch, 128 timesteps, 4 features)
# Output: (batch, 4 probabilities for each activity)

model = models.Sequential([
    # ============================================================================
    # INPUT LAYER
    # ============================================================================
    # Shape: (128, 4) = 128 timesteps × 4 acceleration features
    # No learnable parameters, just defines expected data shape
    # Example batch: (64, 128, 4) = 64 samples, 128 timesteps, 4 features
    layers.Input(shape=(n_timesteps, n_features)),
    
    # ============================================================================
    # BLOCK 1: LOW-LEVEL FEATURE EXTRACTION (64 Filters)
    # ============================================================================
    # PURPOSE: Detect basic temporal patterns in raw sensor data
    # PATTERNS LEARNED: peaks, valleys, slopes, step frequency components
    # INPUT:  (batch, 128, 4)
    # OUTPUT: (batch, 64, 64) after entire block
    
    # CONV1D LAYER - Scans data with 64 pattern detectors
    # - filters=64: Creates 64 independent pattern detectors
    # - kernel_size=3: Each detector looks at 3 consecutive timesteps
    # - padding='same': Keep temporal length at 128 (not downsampled here)
    # - kernel_regularizer: L2 penalty (loss += 0.001 × sum(weights²))
    #   Effect: Prevents large weights, keeps filters simple
    # - Output shape after Conv: (batch, 128, 64)
    # - Parameters: 64 filters × 3 timesteps × 4 input_features = ~800 params
    # - Interpretation: 64 different ways to summarize 3-timestep windows
    layers.Conv1D(filters=CONV_FILTERS_BLOCK1, kernel_size=KERNEL_SIZE, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    
    # BATCH NORMALIZATION - Stabilizes learning
    # - Normalizes each filter's output to mean=0, std=1
    # - Computed per batch (statistics change per batch)
    # - Effect: Reduces internal covariate shift
    # - Benefit: Allows higher learning rates, faster convergence
    # - At inference: Uses running average from training
    # - Output shape: (batch, 128, 64) [same shape, normalized values]
    layers.BatchNormalization(),
    
    # RELU ACTIVATION - Introduces non-linearity
    # - Formula: f(x) = max(0, x)
    # - Negative values → 0 (killed), Positive values → unchanged
    # - Why needed: Without activation, stacked layers = just matrix multiplication (linear)
    # - Benefit: Enables learning complex, non-linear patterns
    # - Sparsity: ~50% of activations become 0 (efficient computation)
    # - Output shape: (batch, 128, 64) [same shape, transformed values]
    layers.ReLU(),
    
    # MAXPOOLING1D - Downsamples temporal dimension
    # - pool_size=2: Takes maximum value in each 2-sample window
    # - Reduces length: 128 → 64
    # - Effect: Keeps strongest signals, discards weak noise
    # - Benefit: Translation invariant (pattern at t=10 or t=12 similar after pooling)
    # - Reduces computation by 50%
    # - Output shape: (batch, 64, 64) [64 timesteps, 64 feature maps]
    layers.MaxPooling1D(pool_size=POOL_SIZE),
    
    # DROPOUT - Regularization against overfitting
    # - rate=0.5: Randomly disables 50% of neurons during training
    # - Training: 50% neurons set to 0, remaining scaled by 2x to maintain expected value
    # - Inference: All neurons active, scaled by (1-rate)=0.5
    # - Effect: Forces model to learn distributed representations
    # - Prevents co-adaptation (neurons don't become dependent on neighbors)
    # - Benefit: Model generalizes better to unseen users with different movement styles
    # - Output shape: (batch, 64, 64) [shape unchanged, values randomly zeroed]
    layers.Dropout(DROPOUT_RATE),

    # ============================================================================
    # BLOCK 2: MID-LEVEL FEATURE EXTRACTION (128 Filters)
    # ============================================================================
    # PURPOSE: Combine Block 1 outputs into more complex patterns
    # PATTERNS LEARNED: step cycles, rhythmic patterns, acceleration sequences
    # INPUT:  (batch, 64, 64) from Block 1
    # OUTPUT: (batch, 32, 128) after entire block
    # PYRAMID ARCHITECTURE: 64 filters → 128 filters (increase) → 64 filters (compress)
    #   Reason: More filters in middle to learn diverse combinations
    #           Reduced after to compress information
    
    # CONV1D LAYER - Combines patterns from Block 1
    # - Input: 64 feature maps from Block 1
    # - filters=128: More filters than Block 1 (captures more combinations)
    # - kernel_size=3: Still looks at 3 consecutive timesteps
    # - Output shape after Conv: (batch, 64, 128)
    # - Parameters: 128 filters × 3 timesteps × 64 input_features = ~24,576 params
    # - Much larger than Block 1! Learning more complex patterns
    layers.Conv1D(filters=CONV_FILTERS_BLOCK2, kernel_size=KERNEL_SIZE, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    
    # BATCH NORMALIZATION - Same as Block 1
    # - Normalizes 128 output channels
    # - Stabilizes training of deeper network
    # - Output shape: (batch, 64, 128)
    layers.BatchNormalization(),
    
    # RELU ACTIVATION - Non-linearity for Block 2 combinations
    # - Essential for learning complex mid-level patterns
    # - Output shape: (batch, 64, 128)
    layers.ReLU(),
    
    # MAXPOOLING1D - Downsample again
    # - 64 timesteps → 32 timesteps
    # - Reduces resolution but increases robustness
    # - Now operating at coarser temporal scale
    # - Output shape: (batch, 32, 128)
    layers.MaxPooling1D(pool_size=POOL_SIZE),
    
    # DROPOUT - Prevent Block 2 overfitting
    # - Same rate (0.5) as Block 1
    # - Output shape: (batch, 32, 128)
    layers.Dropout(DROPOUT_RATE),
    
    # ============================================================================
    # BLOCK 3: HIGH-LEVEL FEATURE EXTRACTION (64 Filters)
    # ============================================================================
    # PURPOSE: Learn activity-discriminative features
    # PATTERNS LEARNED: Walking signature, Upstairs signature, etc.
    # INPUT:  (batch, 32, 128) from Block 2
    # OUTPUT: (batch, 64) after GlobalAveragePooling
    # NOTE: No MaxPooling after this block (goes straight to pooling)
    
    # CONV1D LAYER - Learn discriminative patterns
    # - Input: 128 feature maps from Block 2
    # - filters=64: Reduce back to 64 (pyramid shape: 64→128→64)
    # - Output shape after Conv: (batch, 32, 64)
    # - Parameters: 64 filters × 3 timesteps × 128 input_features = ~24,576 params
    # - These 64 filters now encode: "Is this Walking? Upstairs? etc."
    layers.Conv1D(filters=CONV_FILTERS_BLOCK3, kernel_size=KERNEL_SIZE, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    
    # BATCH NORMALIZATION - Stabilize final features
    # - Normalizes the discriminative features
    # - Output shape: (batch, 32, 64)
    layers.BatchNormalization(),
    
    # RELU ACTIVATION - Final non-linearity in conv layers
    # - Output shape: (batch, 32, 64)
    layers.ReLU(),
    
    # GLOBAL AVERAGE POOLING1D - Collapse temporal dimension
    # - Computes average of each feature map across all 32 timesteps
    # - (batch, 32, 64) → (batch, 64)
    # - For each feature map k: output[k] = sum(all_timesteps) / 32
    # - Interpretation: "How active is this discriminative feature across window?"
    # - Benefit: Drastically reduces parameters (compared to Flatten)
    #   Alternative Flatten: 32×64 = 2048 params to Dense layer
    #   GlobalAveragePooling: 64 params to Dense layer (30× smaller!)
    # - Mobile-friendly: Smaller model for edge deployment
    layers.GlobalAveragePooling1D(), 

    # ============================================================================
    # OUTPUT BLOCK: CLASSIFICATION
    # ============================================================================
    # PURPOSE: Convert 64 abstract features into 4 activity probabilities
    # Connects high-level features to activity decisions
    
    # DENSE HIDDEN LAYER - Learn decision logic
    # - Input: 64 features from GlobalAveragePooling
    # - Units: 64 hidden neurons
    # - activation='relu': Non-linear decision logic
    # - kernel_regularizer: L2 penalty on weights
    # - Output shape: (batch, 64)
    # - Parameters: 64 × 64 weights + 64 biases = 4,160 params
    # - Operation: output = ReLU(input × W + b)
    # - Interpretation: Combines features in complex ways
    #   "When features A, B, C are high together → likely Walking"
    layers.Dense(DENSE_UNITS, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)),
    
    # DROPOUT - Final regularization
    # - Prevents hidden layer from overfitting to training users
    # - Output shape: (batch, 64)
    layers.Dropout(DROPOUT_RATE),
    
    # DENSE OUTPUT LAYER + SOFTMAX - Final classification
    # - Input: 64 hidden features
    # - Units: 4 (one per activity: Walking, Upstairs, Downstairs, Idle)
    # - activation='softmax': Converts to probabilities
    # - Parameters: 64 × 4 weights + 4 biases = 260 params
    # - Output shape: (batch, 4)
    # - Softmax operation: exp(logit_i) / sum(exp(all_logits))
    #   Example: logits=[2.3, -0.5, 1.2, 0.1]
    #   softmax=[0.664, 0.041, 0.221, 0.074]
    # - Interpretation: Probability distribution over 4 activities
    #   Sum of probabilities = 1.0, all values between 0-1
    # - Prediction: argmax of 4 probabilities → predicted activity
    layers.Dense(n_outputs, activation='softmax')
])
```

### Architecture Breakdown with Data Flow

#### Input Layer Details
```python
layers.Input(shape=(128, 4))

SHAPE SPECIFICATION:
- 128: Timesteps per window (~2.56 seconds at 50Hz sampling)
- 4: Features (AccX, AccY, AccZ, Magnitude)

EXAMPLE DATA STRUCTURE:
Sample at timestep 0: [0.12, -0.05, 9.81, 9.82]  ← 4 features
Sample at timestep 1: [0.15, -0.03, 9.80, 9.81]
...
Sample at timestep 127: [0.08, -0.02, 9.82, 9.83]

BATCH DIMENSION:
During training batch: (64, 128, 4)
- 64 samples processed simultaneously
- Each sample: 128 timesteps × 4 features
- This parallelization speeds up training
```

#### Block 1: Low-Level Features (64 Filters)

```python
# STEP 1: CONVOLUTION
layers.Conv1D(filters=64, kernel_size=3, padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))

FILTER MECHANICS:
├── 64 independent filters created
├── Each filter: 3×4 sliding window (3 timesteps, 4 features)
│   └── Example filter: [0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.08, -0.05, 0.12, 0.03, -0.1, 0.06]
│       └── 3 timesteps × 4 features = 12 weights
├── Filters slide across 128 timesteps
└── Each position: compute dot_product(window, filter)

INPUT →  [0.1, -0.05, 9.8] → [0.2, -0.03, 9.8] → [0.15, 0.01, 9.81] → ...
          ↓       ↓        ↓         ↓        ↓
CONV1D → Filter1_output, Filter2_output, ..., Filter64_output

OUTPUT SHAPE: (batch, 128, 64)
- Same 128 timesteps (padding='same' preserves length)
- 64 feature maps (one per filter)
- Each value: "Strength of pattern at this timestep"

PATTERN EXAMPLES:
Filter 1: Learns to detect acceleration peaks
Filter 2: Learns to detect acceleration valleys  
Filter 3: Learns to detect rising acceleration
...
Filter 64: Learns to detect rapid oscillations

PARAMETERS: 64 filters × (3 timesteps × 4 features + 1 bias) = ~832

L2 REGULARIZATION:
Loss = CrossEntropy + 0.001 × sum(all_weights²)
Effect: Large weights penalized
        Model prefers simple filters over memorization
```

```python
# STEP 2: BATCH NORMALIZATION
layers.BatchNormalization()

NORMALIZATION PROCESS:
For each of 64 feature maps:
1. Calculate batch statistics
   ├── mean_k = average value across batch
   └── std_k = standard deviation across batch

2. Normalize
   ├── normalized = (value - mean_k) / std_k
   └── All values now roughly in [-1, 1] range

3. Scale & shift (learned during training)
   ├── gamma and beta are trainable parameters
   └── output = gamma × normalized + beta

EFFECT ON TRAINING:
Without BatchNorm: Internal values drift → Gradient vanishing
With BatchNorm: Stable internal statistics → Stable gradients → Faster training

OUTPUT SHAPE: (batch, 128, 64) [same, normalized values]
ADDITIONAL PARAMETERS: 2 × 64 = 128 (gamma and beta per feature map)
```

```python
# STEP 3: RELU ACTIVATION
layers.ReLU()

OPERATION:
f(x) = max(0, x)

Input signal: [-0.5, -0.2, 0.0, 0.3, 0.8, 1.2, -0.1, 0.5]
Output:       [ 0,    0,   0.0, 0.3, 0.8, 1.2,   0,   0.5]
              ↑ killed     ↑            ↑       ↑ killed

BENEFITS:
├── Non-linearity: Without ReLU, deep network = linear transformation
├── Sparsity: ~50% become 0 (efficient computation)
└── Selective: Only strong signals pass through

WHY NOT LINEAR?
Linear chain: x → W1×x → W2×(W1×x) = (W2×W1)×x = W3×x (still linear!)
ReLU chain: x → ReLU(W1×x) → ReLU(W2×ReLU(W1×x)) (non-linear!)

OUTPUT SHAPE: (batch, 128, 64) [same shape, transformed values]
PARAMETERS: 0 (no learnable parameters in ReLU)
```

```python
# STEP 4: MAXPOOLING
layers.MaxPooling1D(pool_size=2)

POOLING PROCESS:
Input: [0.1, 0.8, 0.3, 0.5, 0.2, 0.9, 0.4, 0.6]
       └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
         max(0.1,0.8)  max(0.3,0.5)  max(0.2,0.9)  max(0.4,0.6)
           0.8            0.5           0.9           0.6
Output: [0.8, 0.5, 0.9, 0.6]

EFFECT ON TEMPORAL DIMENSION:
Before: (batch, 128, 64)
After:  (batch, 64, 64)  ← Length halved!

BENEFITS:
├── Dimensionality reduction: 50% smaller
├── Strongest signal preserved: Ignores weak noise
├── Translation invariance: Pattern at t=10 or t=12 similar after pooling
└── Reduced computation: Half the parameters downstream

PARAMETERS: 0 (just max operation, no learning)
```

```python
# STEP 5: DROPOUT
layers.Dropout(0.5)

TRAINING BEHAVIOR:
Original: [0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6]
Mask:     [1,   0,   1,   0,   1,   0,   1,   0]   ← Random 50%
Dropped:  [0.5, 0,   0.2, 0,   0.1, 0,   0.3, 0]
Rescaled: [1.0, 0,   0.4, 0,   0.2, 0,   0.6, 0]   ← Scaled by 1/(1-0.5)=2

INFERENCE BEHAVIOR:
All neurons active but scaled: [0.25, 0.4, 0.1, 0.45, 0.05, 0.35, 0.15, 0.3]
                               ← Each multiplied by (1-rate)=0.5

EFFECT:
Training: Forces distributed representations
          Neurons can't co-depend
          "Backup neurons" always ready
Inference: Integrated predictions from ensemble of sub-networks

PARAMETERS: 0 (just random masking, no learning)
```

**Block 1 Complete Output Shape:** `(batch, 64, 64)`
- 64 timesteps (pooled from 128)
- 64 feature maps (low-level patterns)
- Example values: Pattern strengths at each time point

#### Block 2: Mid-Level Features (128 Filters)

```python
# INPUT FROM BLOCK 1
(batch, 64, 64)
├─ 64 timesteps (coarser resolution than original 128)
└─ 64 feature maps (low-level patterns)

# CONV1D WITH 128 FILTERS
layers.Conv1D(filters=128, kernel_size=3, padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))

FILTER MECHANICS (DIFFERENT FROM BLOCK 1):
Now filters take 64-dimensional input!
├── Input channels: 64 (from Block 1)
├── Kernel size: 3 (still looks at 3 timesteps)
└── Weights per filter: 3 × 64 = 192 (much larger!)

PARAMETERS: 128 × (3 × 64 + 1 bias) = ~24,576
            (Much larger than Block 1's 832!)

PATTERN LEARNING (HIGHER ORDER):
Block 1 filters learned: "Peak", "Valley", "Slope"
Block 2 filters learn: "Peak + Valley sequence" = Step cycle
                       "Multiple peaks" = Repetition
                       "All high together" = Fast motion

OUTPUT SHAPE: (batch, 64, 128)
- Same 64 timesteps
- 128 feature maps (combinations of Block 1 patterns)

PYRAMID ARCHITECTURE RATIONALE:
64 filters → 128 filters (expand to learn diversity) → 64 filters (compress info)
Why expand? Need many combinations of 64 inputs
Why compress? Information flows forward, redundant filters eliminated
```

```python
# COMPLETE BLOCK 2: BatchNorm → ReLU → MaxPool → Dropout
layers.BatchNormalization()       # Normalize 128 channels
layers.ReLU()                     # Non-linearity on combinations
layers.MaxPooling1D(pool_size=2)  # 64 → 32 timesteps
layers.Dropout(0.5)               # Regularize Block 2
```

**Block 2 Complete Output Shape:** `(batch, 32, 128)`
- 32 timesteps (now very coarse, ~1.3 second resolution)
- 128 feature maps (mid-level discriminative patterns)

#### Block 3: High-Level Features (64 Filters)

```python
# INPUT FROM BLOCK 2
(batch, 32, 128)
├─ 32 timesteps (very coarse temporal resolution)
└─ 128 feature maps (complex patterns)

# CONV1D WITH 64 FILTERS (Back to 64!)
layers.Conv1D(filters=64, kernel_size=3, padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))

FILTER MECHANICS:
├── Input channels: 128 (from Block 2)
├── Output channels: 64 (reduce back down)
└── Weights per filter: 3 × 128 = 384

PARAMETERS: 64 × (3 × 128 + 1 bias) = ~24,576
            (Same as Block 2, even though filters reduced)

PYRAMID PATTERN: 64 → 128 → 64
Why squeeze at end? Forces information compression
             → Model learns only essential features
             → Prevents overfitting
             → Enables mobile deployment

PATTERN LEARNING (ACTIVITY-LEVEL):
These 64 filters now learn: "This is Walking"
                            "This is Upstairs"
                            "This is Downstairs"  
                            "This is Idle"

Each filter activates strongly for one activity type

OUTPUT SHAPE: (batch, 32, 64)
- 32 timesteps
- 64 discriminative feature maps
- IMPORTANT: No pooling after this! Goes to GlobalAveragePooling
```

```python
# GLOBAL AVERAGE POOLING (TEMPORAL COLLAPSE)
layers.GlobalAveragePooling1D()

OPERATION:
For each of 64 feature maps:
Feature_map_0 values: [0.5, 0.6, 0.7, 0.8, 0.4, ...]  (32 timesteps)
Average: (0.5 + 0.6 + 0.7 + 0.8 + 0.4 + ...) / 32 = 0.58

Result: Single value representing activity of this feature throughout window

INPUT:  (batch, 32, 64)
OUTPUT: (batch, 64)

INTERPRETATION:
64 summarized values = "Activity summary of the window"
├─ Feature 0 avg = 0.75 (Step detector strongly active)
├─ Feature 1 avg = 0.20 (Arm detector weakly active)
├─ Feature 2 avg = 0.85 (Upward acceleration detector very active)
└─ ... (61 more features)

WHY GLOBAL AVERAGE POOLING?
Alternative 1 - Flatten:
  (batch, 32, 64) → (batch, 2048)
  Dense(2048, 64): 2048 × 64 = 131,072 parameters!
  Risk: Overfitting, slow, huge model

Alternative 2 - GlobalAveragePooling:
  (batch, 32, 64) → (batch, 64)
  Dense(64, 64): 64 × 64 = 4,096 parameters
  Benefit: 30× smaller, faster, generalizes better!

PARAMETERS: 0 (just averaging, no learning)
```

#### Output Block: Classification

```python
# HIDDEN DENSE LAYER
layers.Dense(DENSE_UNITS=64, activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(0.001))

OPERATION:
Input: 64 values from GlobalAveragePooling
       [0.75, 0.20, 0.85, ..., 0.42]  (feature strengths)

Dense layer: output = ReLU(input × W + b)

WEIGHT MATRIX: 64 × 64 = 4,096 weights
BIAS VECTOR: 64 biases
TOTAL PARAMETERS: 4,160

INTERPRETATION:
Each of 64 hidden neurons learns:
  "When features A, B, C are high together AND feature D is low → activate"
  
Example neuron activation:
  h1 = ReLU(0.1×feat_0 + 0.3×feat_1 - 0.2×feat_2 + ... + bias)
       = max(0, -0.15) = 0  (This neuron inactive)
       
  h2 = ReLU(0.8×feat_0 + 0.7×feat_1 + 0.5×feat_2 + ... + bias)
       = max(0, 2.34) = 2.34  (This neuron very active!)

RELU IN DENSE: Non-linearity for final decision boundaries
OUTPUT SHAPE: (batch, 64)
```

```python
# FINAL DROPOUT
layers.Dropout(0.5)

Final regularization before output
Prevents hidden neurons from memorizing user-specific quirks
OUTPUT SHAPE: (batch, 64)
```

```python
# OUTPUT DENSE LAYER + SOFTMAX
layers.Dense(n_outputs=4, activation='softmax')

OPERATION:
Input: 64 hidden features
       [h1, h2, h3, ..., h64]

Output computation:
logits = hidden × W + b
logits shape: (batch, 4)
logits example: [2.3, -0.5, 1.2, 0.1]

softmax normalization:
z_i = e^logit_i / Σ(e^logits)
z = [e^2.3, e^-0.5, e^1.2, e^0.1] / sum
  = [9.97, 0.61, 3.32, 1.11] / 15.01
  = [0.664, 0.041, 0.221, 0.074]
     ↑        ↑       ↑       ↑
  Walking Upstairs Down  Idle

PARAMETERS: 64 × 4 weights + 4 biases = 260

OUTPUT SHAPE: (batch, 4)
MEANING:
  prob_walking = 0.664 (66.4% confident)
  prob_upstairs = 0.041 (4.1%)
  prob_downstairs = 0.221 (22.1%)
  prob_idle = 0.074 (7.4%)
  
  Sum = 1.0 ✓ (valid probability distribution)
  
PREDICTION:
  argmax(0.664, 0.041, 0.221, 0.074) = 0 → "Walking"
```

---

## Section 4: Model Compilation

```python
# Compile the model with learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

### Optimizer

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
```

**Adam (Adaptive Moment Estimation):**
- Combines momentum (velocity) and RMSprop (adaptive learning rates)
- Each parameter gets its own learning rate
- Works well across many problems (robust default)
- Converges faster than plain SGD

**Learning Rate 0.0005:**
- Controls step size during weight updates
- `new_weight = old_weight - 0.0005 × gradient`
- Conservative: slow but stable
- Later reduced by LR scheduler if loss plateaus

### Loss Function

```python
loss='categorical_crossentropy'
```

**Categorical Crossentropy:**
- Standard for multi-class classification
- Formula: `Loss = -Σ(y_true × log(y_pred))`
- Penalizes wrong predictions heavily
- Requires one-hot encoded labels (which we have)

**Example:**
```
True: [0, 1, 0, 0] (Upstairs)
Pred: [0.1, 0.8, 0.05, 0.05]
Loss = -(0×log(0.1) + 1×log(0.8) + 0×log(0.05) + 0×log(0.05))
     = -log(0.8) ≈ 0.22  (small loss, prediction correct)

Pred: [0.3, 0.2, 0.4, 0.1]  (wrong prediction)
Loss = -log(0.2) ≈ 1.6  (large loss, penalizes)
```

### Metrics

```python
metrics=['accuracy']
```

- Reported at each epoch
- Not used for optimization (loss is optimized)
- Human-readable: % of samples correctly predicted

### Model Summary

```python
model.summary()
```

- Prints architecture: layer names, output shapes, parameters
- Useful for debugging: ensure shapes match expectations
- Identifies layers with millions of parameters (potential bottlenecks)

---

## Section 5: Callbacks

```python
# ==============================================================================
# 3. CALLBACKS (TRAINING HELPERS)
# ==============================================================================

# Checkpoint: Saves the model every time the validation accuracy improves.
checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_NAME, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)
```

**ModelCheckpoint Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_SAVE_NAME` | `"best_motion_model_v5.keras"` | File to save weights |
| `monitor` | `'val_accuracy'` | Track validation accuracy |
| `save_best_only` | `True` | Only save if accuracy improves |
| `mode` | `'max'` | Save when accuracy goes UP (not down) |
| `verbose` | `1` | Print when checkpoint saves |

**Why Checkpointing?**
- Epoch 45: 90% accuracy → Save
- Epoch 46-60: 88% accuracy → Don't save
- At the end, use saved epoch 45 weights
- Without this: you'd use final epoch's overfit weights

```python
# ReduceLROnPlateau: Reduces learning rate if validation loss plateaus.
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=LR_DECAY_FACTOR,
    patience=LR_DECAY_PATIENCE,
    min_lr=1e-6,
    verbose=1
)
```

**ReduceLROnPlateau Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `monitor` | `'val_loss'` | Track validation loss |
| `factor` | `0.5` | Multiply LR by 0.5 |
| `patience` | `5` | Wait 5 epochs without improvement |
| `min_lr` | `1e-6` | Don't reduce below 0.000001 |
| `verbose` | `1` | Print when LR reduces |

**How It Works:**
```
Epoch 1-5: LR = 0.0005, loss improving → Keep LR
Epoch 6-10: LR = 0.0005, loss not improving → Reduce LR
Epoch 11: LR = 0.0005 × 0.5 = 0.00025
Epoch 12-16: LR = 0.00025, loss improving → Keep LR
```

**Why Reduce Learning Rate?**
- Early training: large LR needed to escape initialization
- Late training: small LR needed to fine-tune local optimum
- Adaptive: reduces only when learning plateau detected

```python
# EarlyStopping: Stops training if the model stops improving.
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)
```

**EarlyStopping Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `monitor` | `'val_loss'` | Track validation loss |
| `patience` | `20` | Stop if no improvement for 20 epochs |
| `restore_best_weights` | `True` | Revert to best epoch's weights |

**Timeline Example:**
```
Epoch 40: val_loss = 0.45 → Best so far, save weights
Epoch 41-60: val_loss stays ≥ 0.45 → No improvement
Epoch 60: 20 epochs without improvement → Stop training
Restore weights from epoch 40
```

**Why Early Stopping?**
- Saves compute: don't train unnecessary epochs
- Prevents overfitting: stops before memorization starts
- Automatic: no manual epoch tuning needed

---

## Section 6: Train the Model

```python
# ==============================================================================
# 4. TRAIN THE MODEL
# ==============================================================================
from sklearn.utils.class_weight import compute_class_weight

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print("\nClass Weights (to balance training):")
class_names_list = ['Walking', 'Upstairs', 'Downstairs', 'Idle']
for i, (name, weight) in enumerate(zip(class_names_list, class_weights)):
    print(f"  {name:12}: {weight:.4f}")

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
```

### Compute Class Weights

```python
y_train_labels = np.argmax(y_train, axis=1)
```
- Convert one-hot `[0, 1, 0, 0]` to class index `1`
- Creates simple array: `[0, 2, 1, 3, 0, 1, ...]`

```python
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train_labels),
                                     y=y_train_labels)
```

**What It Does:**
- Count samples per class
- Calculate: `weight_i = total_samples / (n_classes × samples_i)`

**Example:**
```
Walking: 1000 samples → weight = 4000 / (4 × 1000) = 1.0
Upstairs: 500 samples  → weight = 4000 / (4 × 500) = 2.0
Downstairs: 500 samples → weight = 4000 / (4 × 500) = 2.0
Idle: 100 samples → weight = 4000 / (4 × 100) = 10.0
```

**Why?**
- Without weights: model ignores rare classes (Idle)
- With weights: Idle samples count 10x more in loss
- Balances training for all activities equally

```python
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
```
- Creates: `{0: 1.0, 1: 2.0, 2: 2.0, 3: 10.0}`
- Maps class index to weight

### Model.fit() - The Training Loop

```python
history = model.fit(
    X_train, y_train,                           # Training data
    validation_data=(X_test, y_test),           # Test data (unseen user)
    epochs=MAX_EPOCHS,                          # Max 60 iterations
    batch_size=BATCH_SIZE,                      # Process 64 samples at a time
    callbacks=[checkpoint, lr_scheduler, early_stopping],  # Training helpers
    class_weight=class_weight_dict,             # Balance classes
    verbose=1                                   # Print progress
)
```

**Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `X_train, y_train` | Training data | Learn from this |
| `validation_data` | Test set (new user) | Evaluate during training |
| `epochs` | 60 | Max iterations through data |
| `batch_size` | 64 | Update weights every 64 samples |
| `callbacks` | [checkpoint, lr_scheduler, early_stopping] | Save best, adapt LR, stop early |
| `class_weight` | `{0: 1.0, 1: 2.0, ...}` | Upweight rare classes |
| `verbose` | 1 | Print epoch progress |

**What Happens Each Epoch:**
```
1. Shuffle training data
2. Loop through batches (64 samples each):
   a. Forward pass: predict on batch
   b. Calculate loss (including class weights)
   c. Backward pass: compute gradients
   d. Update weights using optimizer
3. After epoch:
   a. Validate on test set
   b. Check callbacks (save checkpoint, decay LR, early stop)
   c. Print: accuracy, loss, val_accuracy, val_loss
```

**Output Example:**
```
Epoch 1/60
100/100 [==============================] - 2s 20ms/step - loss: 1.234 - accuracy: 0.65 - val_loss: 1.156 - val_accuracy: 0.68
Epoch 2/60
100/100 [==============================] - 2s 19ms/step - loss: 0.890 - accuracy: 0.75 - val_loss: 0.920 - val_accuracy: 0.73
...
```

**History Object:**
```python
history.history = {
    'accuracy': [0.65, 0.75, 0.82, ...],          # Train accuracy per epoch
    'loss': [1.234, 0.890, 0.567, ...],           # Train loss per epoch
    'val_accuracy': [0.68, 0.73, 0.79, ...],      # Val accuracy per epoch
    'val_loss': [1.156, 0.920, 0.612, ...]        # Val loss per epoch
}
```

---

## Section 7: Evaluate the Model

```python
# ==============================================================================
# 5. EVALUATE (FINAL TEST)
# ==============================================================================
print("\nLoading best model for final evaluation...")
best_model = models.load_model(MODEL_SAVE_NAME)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n" + "="*50)
print(f"FINAL TEST RESULT (UNSEEN USER)")
print(f"Accuracy: {test_acc:.4f}")
print(f"Loss: {test_loss:.4f}")
print(f"="*50 + "\n")
```

### Load Best Model

```python
best_model = models.load_model(MODEL_SAVE_NAME)
```
- Loads from checkpoint (best epoch, not final epoch)
- Critical: final model might be overfit, checkpoint has better generalization

### Evaluate

```python
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
```

**What It Does:**
- Forward pass on entire test set (unseen user)
- Computes loss and accuracy
- Returns two values: loss and accuracy

**Why Test on Unseen User?**
- Simulates real deployment
- Proves model generalizes to new people
- Single number: "Model accuracy on unseen person = 87%"

```python
# Get predictions for confusion matrix and per-class metrics
y_pred_prob = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
print("Confusion Matrix:")
print(cm)

# Class names
class_names = ['Walking', 'Upstairs', 'Downstairs', 'Idle']
print("\nPer-Class Performance Metrics:")
print(classification_report(y_test_labels, y_pred, target_names=class_names, digits=4))
```

**Get Predictions:**
```python
y_pred_prob = best_model.predict(X_test, verbose=0)
```
- Output shape: `(n_test, 4)` with probabilities
- Example: `[[0.1, 0.85, 0.02, 0.03], ...]`

```python
y_pred = np.argmax(y_pred_prob, axis=1)
```
- Take index of highest probability
- Example: `[1, 0, 2, 3, ...]` (class indices)

```python
y_test_labels = np.argmax(y_test, axis=1)
```
- Convert one-hot to indices for comparison
- Example: `[1, 0, 2, 3, ...]` (true classes)

**Confusion Matrix:**
```
         Predicted
         0    1    2    3
Actual 0 85   5    5    5    (Walking)
       1 10   70   15   5    (Upstairs)
       2 5    20   70   5    (Downstairs)
       3 10   5    5    80   (Idle)
```

- Diagonal: correct predictions
- Off-diagonal: misclassifications

**Classification Report:**
```
             Precision  Recall  F1-Score  Support
Walking      0.85       0.85    0.85      100
Upstairs     0.70       0.70    0.70      100
Downstairs   0.70       0.70    0.70      100
Idle         0.80       0.80    0.80      100
```

- **Precision**: Of predicted X, how many correct?
- **Recall**: Of actual X, how many found?
- **F1**: Harmonic mean of precision and recall

---

## Section 8: Visualize Results

```python
# ==============================================================================
# 6. VISUALIZE RESULTS
# ==============================================================================
plt.figure(figsize=(16, 5))

# Plot 1: Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Unseen User)')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (Unseen User)')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix Heatmap
plt.subplot(1, 3, 3)
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(range(4), class_names, rotation=45)
plt.yticks(range(4), class_names)

# Add text annotations
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.tight_layout()
plt.show()
```

### Figure Setup

```python
plt.figure(figsize=(16, 5))
```
- Creates figure 16 inches wide, 5 inches tall
- Room for 3 side-by-side plots

### Subplot 1: Accuracy

```python
plt.subplot(1, 3, 1)  # 1 row, 3 columns, plot 1
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Unseen User)')
```

**What to Look For:**
```
Healthy:
- Train and val lines track together
- Both increasing toward ~90%
- Small gap (≤5%)

Overfitting:
- Train: 98%, Val: 80% (large gap)
- Val plateaus or decreases while train increases

Underfitting:
- Both: low (60%) and barely increasing
```

### Subplot 2: Loss

```python
plt.subplot(1, 3, 2)  # Plot 2
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (Unseen User)')
```

**What to Look For:**
```
Healthy:
- Both lines decreasing
- Converging to ~0.3
- Similar values

Problem:
- Val loss increases while train decreases (overfitting)
- Both stuck high (underfitting)
```

### Subplot 3: Confusion Matrix

```python
plt.subplot(1, 3, 3)  # Plot 3
plt.imshow(cm, cmap='Blues', interpolation='nearest')
```

- `cmap='Blues'`: color intensity = count
- Darker blue = more samples

```python
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
```

- Adds counts as text in each cell
- Makes matrix readable

**What to Look For:**
```
Diagonal dominant (good):
[[85, 5, 5, 5],
 [10, 70, 15, 5],
 [5, 20, 70, 5],
 [10, 5, 5, 80]]

Off-diagonal patterns (problem):
- Row 1 & 2 similar: confusing Upstairs/Downstairs
  Solution: collect more diverse data
```

---

## Section 9: Save Model Metadata

```python
# ==============================================================================
# 7. SAVE ALL MODEL DATA AND METADATA
# ==============================================================================

metadata = {
    "timestamp": datetime.now().isoformat(),
    "model_version": "v5",
    "model_architecture": {
        "input_shape": (int(n_timesteps), int(n_features)),
        "n_timesteps": int(n_timesteps),
        "n_features": int(n_features),
        "n_outputs": int(n_outputs),
        "model_type": "1D CNN Sequential",
        "conv_filters": [CONV_FILTERS_BLOCK1, CONV_FILTERS_BLOCK2, CONV_FILTERS_BLOCK3],
        "kernel_size": KERNEL_SIZE,
        "pool_size": POOL_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "dense_units": DENSE_UNITS
    },
    "training_config": {
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lr_decay_factor": LR_DECAY_FACTOR,
        "lr_decay_patience": LR_DECAY_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "l2_regularization": L2_REGULARIZATION,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "class_weights": {int(k): float(v) for k, v in class_weight_dict.items()}
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
        "training_strategy": "Leave-One-Subject-Out (LOSO)"
    }
}
```

### Metadata Dictionary

**Structure:** Nested dictionaries containing all model information

| Section | Purpose | Example |
|---------|---------|---------|
| `timestamp` | When model trained | `"2026-01-08T14:30:45.123456"` |
| `model_architecture` | Model structure | `"conv_filters": [64, 128, 64]` |
| `training_config` | All hyperparameters | `"learning_rate": 0.0005` |
| `training_results` | Performance metrics | `"final_test_accuracy": 0.87` |
| `confusion_matrix` | Predictions vs true | `[[85, 5, ...], ...]` |
| `dataset_info` | Data details | `"n_training_samples": 5000` |

**Why Save Metadata?**
- Reproducibility: exact configuration to recreate results
- Comparison: compare multiple model versions
- Documentation: understand what this model does
- Debugging: if accuracy drops, check if parameters changed

### Save Report as Text

```python
with open(MODEL_REPORT_FILE, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL V5 - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated: {metadata['timestamp']}\n\n")
    
    f.write("ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Input Shape: {n_timesteps} timesteps × {n_features} features\n")
    # ... more sections
```

**Output File: model_v5_report.txt**
```
======================================================================
MODEL V5 - TRAINING REPORT
======================================================================

Generated: 2026-01-08T14:30:45.123456

ARCHITECTURE
----------------------------------------------------------------------
Input Shape: 128 timesteps × 4 features
Output Classes: 4
Model Type: 1D CNN Sequential
Conv Filters: [64, 128, 64]
Kernel Size: 3, Pool Size: 2
Dropout Rate: 0.5

TRAINING CONFIGURATION
----------------------------------------------------------------------
Max Epochs: 60
Batch Size: 64
Initial Learning Rate: 0.0005
LR Decay: factor=0.5, patience=5
Early Stopping Patience: 20
L2 Regularization: 0.001
Optimizer: adam
Loss Function: categorical_crossentropy
Class Weights: {0: 1.0, 1: 2.0, 2: 2.0, 3: 10.0}

DATASET
----------------------------------------------------------------------
Dataset File: motion_dataset_loso_v5.npz
Training Samples: 4800
Test Samples: 450
Strategy: Leave-One-Subject-Out (LOSO)

TRAINING RESULTS
----------------------------------------------------------------------
Epochs Trained: 42
Final Train Accuracy: 0.9234
Final Train Loss: 0.2156
Final Validation Accuracy: 0.8967
Final Validation Loss: 0.3123

FINAL TEST EVALUATION (UNSEEN USER)
----------------------------------------------------------------------
Test Accuracy: 0.8756
Test Loss: 0.3456

CONFUSION MATRIX
----------------------------------------------------------------------
[[85, 5, 5, 5],
 [10, 70, 15, 5],
 [5, 20, 70, 5],
 [10, 5, 5, 80]]
```

**Why Text Report?**
- Human readable (no Python needed to view)
- Easy to share in emails/reports
- Searchable documentation
- Permanent record of model version

---

## Complete Flow Diagram

```
Start
  ↓
1. Load data (DATASET_FILE)
   - Extract X_train, X_test, y_train, y_test
   - Auto-detect n_timesteps, n_features, n_outputs
   ↓
2. Build CNN model
   - Input layer (128, 4)
   - Conv Block 1 (64 filters)
   - Conv Block 2 (128 filters)
   - Conv Block 3 (64 filters)
   - GlobalAveragePooling
   - Dense layer (64 units)
   - Output layer (4 units, softmax)
   ↓
3. Compile model
   - Optimizer: Adam
   - Loss: categorical_crossentropy
   - Metrics: accuracy
   ↓
4. Setup callbacks
   - ModelCheckpoint (save best)
   - ReduceLROnPlateau (adapt learning rate)
   - EarlyStopping (stop when plateaus)
   ↓
5. Compute class weights
   - Upweight rare classes (Idle)
   - Balance training
   ↓
6. Train model
   - Fit on X_train, y_train for max 60 epochs
   - Validate on X_test, y_test each epoch
   - Callbacks manage training
   ↓
7. Evaluate
   - Load best checkpoint
   - Test on unseen user (X_test, y_test)
   - Generate confusion matrix
   - Print per-class metrics
   ↓
8. Visualize
   - Plot train/val accuracy
   - Plot train/val loss
   - Plot confusion matrix heatmap
   ↓
9. Save metadata
   - Architecture (filters, sizes, etc.)
   - Training config (epochs, learning rate, etc.)
   - Results (accuracy, loss, confusion matrix)
   - Save to text report
   ↓
Done
```

---

## Key Takeaways

| Concept | Why It Matters | Implementation |
|---------|----------------|-----------------|
| **1D CNN** | Learn temporal patterns in sensor data | Conv1D layers with kernel_size=3 |
| **Dropout** | Prevent overfitting to specific users | Disable 50% of neurons during training |
| **Batch Norm** | Stabilize training, faster convergence | Applied after each Conv1D |
| **Class Weights** | Balance rare activities | Upweight Idle samples 10x |
| **Learning Rate Decay** | Escape plateaus, fine-tune later | Reduce by 0.5x if loss stalls 5 epochs |
| **Early Stopping** | Prevent wasting compute and overfitting | Stop if validation loss doesn't improve 20 epochs |
| **Model Checkpointing** | Keep best version, not last version | Save when validation accuracy improves |
| **LOSO Validation** | Prove generalization to unseen users | Test on completely different person |
| **GlobalAveragePooling** | Reduce parameters, enable mobile deployment | Average across time instead of flatten |
| **Comprehensive Metadata** | Reproducibility and comparison | Save all params and results to text report |

