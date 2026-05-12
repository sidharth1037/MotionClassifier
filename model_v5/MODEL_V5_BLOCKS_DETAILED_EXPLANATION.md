# MODEL_V5 - Deep Dive into Each Block

## Block-by-Block Analysis with Visualizations and Examples

---

## BLOCK 1: Input Layer

### Code
```python
layers.Input(shape=(n_timesteps, n_features))
# Input shape: (128, 4)
```

### What It Does

Defines the expected shape of incoming data. No learnable parameters, just metadata that tells TensorFlow what to expect.

### Data Flow Visualization

```
Input Data Shape:
┌─────────────────────────────────────────────────────────────┐
│ Batch of Samples                                            │
│ ┌───────────────────────────────────────────────────────┐  │
│ │ Sample 1                                              │  │
│ │ ┌─────────────────────────────────────────────────┐  │  │
│ │ │ Timestep 0: [AccX, AccY, AccZ, Magnitude]      │  │  │
│ │ │ Timestep 1: [AccX, AccY, AccZ, Magnitude]      │  │  │
│ │ │ ...                                             │  │  │
│ │ │ Timestep 127: [AccX, AccY, AccZ, Magnitude]    │  │  │
│ │ └─────────────────────────────────────────────────┘  │  │
│ │ Shape: (128, 4)                                       │  │
│ └───────────────────────────────────────────────────────┘  │
│ ┌───────────────────────────────────────────────────────┐  │
│ │ Sample 2                                              │  │
│ │ ... (same structure)                                  │  │
│ └───────────────────────────────────────────────────────┘  │
│ ...                                                         │
└─────────────────────────────────────────────────────────────┘
Final Input Shape: (batch_size, 128, 4)
Example with batch_size=64: (64, 128, 4)
```

### Numerical Example

```python
# A single sample (remove batch dimension temporarily)
sample = np.random.randn(128, 4)  # 128 timesteps, 4 features

# Breakdown:
sample.shape[0] = 128   # n_timesteps
sample.shape[1] = 4     # n_features (AccX, AccY, AccZ, Mag)

# Example values:
sample[0] = [0.12, -0.05, 9.81, 9.82]   # Timestep 0 (first acceleration reading)
sample[1] = [0.15, -0.03, 9.80, 9.81]   # Timestep 1
sample[127] = [0.08, -0.02, 9.82, 9.83] # Timestep 127 (last reading, ~2.56 seconds later)
```

### Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_timesteps` | 128 | Duration of window in samples (~2.56 seconds at 50Hz) |
| `n_features` | 4 | Number of acceleration features |
| Batch dimension | Variable | Changes per epoch (typically 1-64) |

### Why This Shape?

**Why 128 timesteps?**
- At 50 Hz sampling rate: 128 samples ≈ 2.56 seconds
- Long enough to capture full activity pattern
- Short enough for quick computation
- Can be changed in `process_raw_data.py` (WINDOW_SIZE parameter)

**Why 4 features?**
- AccX, AccY, AccZ: individual axis accelerations
- Magnitude: combined acceleration magnitude = √(X² + Y² + Z²)
- 4th feature helps model capture total motion intensity
- Can be changed in `process_raw_data.py` (feature engineering)

### Common Issues

**Issue: Shape mismatch error**
```
ValueError: Input 0 is incompatible with layer conv1d_1: expected shape=(None, 128, 4), received shape=(None, 256, 4)
```
→ Your data has 256 timesteps, but model expects 128
→ Fix: Change WINDOW_SIZE in process_raw_data.py

**Issue: 3D array expected**
```
Error: Expected 3D tensor, got 1D or 2D
```
→ Forgot batch dimension or forgot time dimension
→ Data should be: `(batch, timesteps, features)` not `(timesteps, features)`

### Modifications

**Use different window size:**
```python
n_timesteps = 256  # Longer window, more context
# or
n_timesteps = 64   # Shorter window, more samples per epoch
```

**Add more features:**
```python
# In process_raw_data.py, add more computed features:
n_features = 6  # Add velocity and acceleration rate
data = np.hstack((data, velocity, accel_rate))
```

---

## BLOCK 2: Conv1D Layer (Block 1, Conv)

### Code
```python
layers.Conv1D(filters=64, kernel_size=3, padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))
```

### What It Does

Scans the time series with 64 different pattern detectors (filters). Each filter is a small window (3 timesteps) that slides across the entire sequence, learning to recognize basic patterns like peaks, valleys, and slopes.

### Visualization: How Conv1D Works

```
Example: Finding a Peak Pattern

Filter (kernel) to detect peaks:
┌─────────────────────┐
│ [-0.5, +1.0, -0.5] │  ← Learns this pattern recognizes peaks
└─────────────────────┘

Input signal (1 feature, simplified):
Timestep: 0    1    2    3    4    5    6
Value:    0.1  0.3  0.8  0.5  0.2  0.1  0.3
              ↑    ↑    ↑
           Peak at timestep 2!

Convolution Process:
Position 0: [0.1, 0.3, 0.8] × [-0.5, +1.0, -0.5] = -0.05 + 0.30 - 0.40 = -0.15
Position 1: [0.3, 0.8, 0.5] × [-0.5, +1.0, -0.5] = -0.15 + 0.80 - 0.25 = +0.40 ← HIGH! Peak detected!
Position 2: [0.8, 0.5, 0.2] × [-0.5, +1.0, -0.5] = -0.40 + 0.50 - 0.10 = 0.00
Position 3: [0.5, 0.2, 0.1] × [-0.5, +1.0, -0.5] = -0.25 + 0.20 - 0.05 = -0.10

Output: [-0.15, +0.40, 0.00, -0.10]  ← High value at position 1!
```

### Data Transformation

```
Input to Conv1D Block 1:
Shape: (batch, 128, 4)
Example: (64, 128, 4)

Inside Conv1D:
- 64 filters applied independently
- Each filter: kernel_size=3, input_channels=4
- Parameters per filter: 3 × 4 = 12 weights + 1 bias = 13
- Total parameters: 64 filters × 13 = 832 parameters

Output of Conv1D:
Shape: (batch, 128, 64)
Example: (64, 128, 64)

Interpretation:
- Same 128 timesteps (padding='same' preserves length)
- 64 feature maps (one output per filter)
- Each value represents: "How well does this filter match at this timestep?"
```

### Visual Layer Transformation

```
INPUT (64, 128, 4):
┌──────────────────────────────────────────┐
│ 4 Features across 128 timesteps         │
│ [F1]────────────────────────────────────│
│ [F2]────────────────────────────────────│
│ [F3]────────────────────────────────────│
│ [F4]────────────────────────────────────│
└──────────────────────────────────────────┘

CONV1D (64 filters applied):
┌──────────────────────────────────────────┐
│ 64 Feature Maps across 128 timesteps    │
│ [FM1]──────────────────────────────────│
│ [FM2]──────────────────────────────────│
│ [FM3]──────────────────────────────────│
│ ... (59 more feature maps)              │
│ [FM64]──────────────────────────────────│
└──────────────────────────────────────────┘

OUTPUT (64, 128, 64):
```

### Parameter Count

```
Conv1D(filters=64, kernel_size=3, input_channels=4)

Weights: 64 × 3 × 4 = 768
Biases:  64 × 1 = 64
Total:   768 + 64 = 832 parameters

Formula: filters × kernel_size × input_channels + filters
```

### L2 Regularization Impact

```python
kernel_regularizer=tf.keras.regularizers.l2(0.001)
```

**What it does:**
```
Original Loss = CrossEntropy Loss
New Loss = CrossEntropy Loss + 0.001 × sum(weights²)

Effect:
- Large weights penalized more
- Forces weights toward 0
- Prevents memorizing noise

Example:
Weight 1: 0.5  →  Loss penalty = 0.001 × 0.5² = 0.00025
Weight 2: 0.1  →  Loss penalty = 0.001 × 0.1² = 0.00001
Weight 3: 0.01 →  Loss penalty = 0.001 × 0.01² = 0.0000001
```

**Why important:**
- Without L2: Model learns filter weights [10.0, 20.0, -15.0]
- With L2: Model learns filter weights [0.5, 0.6, -0.4] (smaller, simpler)
- Simpler = generalizes better to unseen users

### Visualizing 64 Filters

```
Each filter learns different patterns:

Filter 1:  Detects peaks
Filter 2:  Detects valleys
Filter 3:  Detects slopes
Filter 4:  Detects plateaus
Filter 5:  Detects oscillations
...
Filter 64: Detects rapid changes

All 64 filters applied simultaneously!
```

### What Each Filter Output Means

```
For one sample at one timestep:
feature_map[k][t] = how strongly does filter k activate at timestep t?

High value (e.g., 0.8): Pattern strongly detected
Zero value (0.0): Pattern not present
Negative value (e.g., -0.5): Opposite pattern detected

Result: 64 different perspectives on the same window
```

### Parameter Tuning

| Change | Effect | When to Use |
|--------|--------|------------|
| filters: 64 → 32 | Fewer patterns, faster, less capacity | Overfitting, memory constraints |
| filters: 64 → 128 | More patterns, slower, more capacity | Underfitting, plenty of data |
| kernel_size: 3 → 5 | Larger context window | Need longer patterns |
| kernel_size: 3 → 2 | Shorter context window | Fine-grained patterns |
| L2: 0.001 → 0.01 | Stronger regularization | Overfitting |
| L2: 0.001 → 0.0001 | Weaker regularization | Underfitting |

---

## BLOCK 2b: BatchNormalization

### Code
```python
layers.BatchNormalization()
```

### What It Does

Normalizes the output of Conv1D to have mean ≈ 0 and std ≈ 1 for each feature map. This stabilizes training and allows higher learning rates.

### Visualization

```
Before BatchNorm:
Filter outputs (raw): [5.2, -3.1, 8.4, -1.2, 12.3, ...]
Mean: 4.2, Std: 6.8

After BatchNorm:
Normalized: [-0.15, -1.06, 0.62, -0.78, 1.21, ...]
Mean: 0.0, Std: 1.0

Effect: Stabilizes gradient flow in backprop!
```

### Mathematical Process

```
For each feature map k across batch:

Step 1: Calculate batch statistics
mean_k = (sum of all feature_map[k] values in batch) / batch_size
std_k = standard deviation of feature_map[k] in batch

Step 2: Normalize
normalized_k = (feature_map[k] - mean_k) / (std_k + epsilon)

Step 3: Scale and shift (learnable parameters)
output_k = gamma * normalized_k + beta

Where gamma and beta are learned during training
```

### Training vs Inference

```
Training Mode:
- Use current batch statistics
- Continuous updates to running mean/variance

Inference Mode:
- Use exponential moving average of training batches
- Consistent predictions across different batch sizes
```

### Why It Helps

```
Without BatchNorm:
Epoch 1: Learning rate 0.0005 works fine
Epoch 10: Internal activations shifted → gradient vanishing
         Need to reduce learning rate to 0.0001

With BatchNorm:
Epoch 1-50: Can maintain learning rate 0.0005
            Internal statistics stay stable
            Faster convergence!
```

### Impact on Training

```
Without BatchNorm:
Loss: 1.5 → 1.2 → 0.9 → 0.7 → 0.5 → 0.45 (slow, needs careful tuning)

With BatchNorm:
Loss: 1.5 → 0.8 → 0.4 → 0.2 → 0.15 → 0.10 (fast, fewer iterations)
```

---

## BLOCK 2c: ReLU Activation

### Code
```python
layers.ReLU()
```

### What It Does

Applies activation function: f(x) = max(0, x)

Negative values → 0
Positive values → unchanged

### Visualization

```
Input signal (after BatchNorm):
[-0.5, -0.2, 0.0, 0.3, 0.8, 1.2, -0.1, 0.5]

After ReLU:
[  0,    0,  0.0, 0.3, 0.8, 1.2,   0,  0.5]
    ↑         ↑
Killed!      Preserved!
```

### Graphically

```
      Output
        ↑
        │       /
        │      /
      1 ├─────/───────
        │    /
        │   /
      0 ├──┼────────
        │ /
        │/
       -1─────────→ Input
       -2 -1  0  1  2

y = max(0, x)

All negatives → 0
Positives → straight through
```

### Why ReLU?

**Without activation (linear model):**
```python
x = input_data
y = W1 × x
z = W2 × y  # This is just: (W2 × W1) × x = W3 × x
# Still linear! Can't learn complex patterns!
```

**With ReLU (non-linear model):**
```python
x = input_data
y = max(0, W1 × x)  # Non-linearity!
z = max(0, W2 × y)  # More non-linearity!
# Now can learn complex, non-linear patterns!
```

### Sparsity Benefit

```
Input: [-0.5, 0.2, -0.1, 0.8, -0.3, 0.5]

After ReLU: [0, 0.2, 0, 0.8, 0, 0.5]
            ↓
         50% zeros!

Benefit: 
- Sparse activations (many zeros)
- Faster computation (skip zero calculations)
- Reduced overfitting (forces model to be selective)
```

### Alternatives to ReLU

```python
# Leaky ReLU: Allows small negative values through
layers.LeakyReLU(alpha=0.1)  # f(x) = x if x > 0, else 0.1 × x

# ELU: Smooth for negative values
layers.ELU(alpha=1.0)  # f(x) = x if x > 0, else α(e^x - 1)

# GELU: Modern, used in transformers
layers.Activation('gelu')

# When to use:
# - ReLU: Standard, works well (default choice)
# - LeakyReLU: If dying ReLU problem (many zeros)
# - ELU: Smooth gradients, slower training
# - GELU: State-of-the-art, more compute
```

---

## BLOCK 2d: MaxPooling1D

### Code
```python
layers.MaxPooling1D(pool_size=2)
```

### What It Does

Reduces temporal dimension by taking maximum value in each 2-sample window.

### Visualization

```
Input (1 feature, simplified):
Timestep: 0    1    2    3    4    5    6    7
Value:    0.1  0.8  0.3  0.5  0.2  0.9  0.4  0.6

MaxPooling with pool_size=2:
Window [0, 1]: max(0.1, 0.8) = 0.8
Window [2, 3]: max(0.3, 0.5) = 0.5
Window [4, 5]: max(0.2, 0.9) = 0.9
Window [6, 7]: max(0.4, 0.6) = 0.6

Output: [0.8, 0.5, 0.9, 0.6]

Length: 8 → 4 (50% reduction)
```

### Full Block 1 Pooling Effect

```
Before MaxPooling:
Shape: (batch, 128, 64)

After MaxPooling (pool_size=2):
Shape: (batch, 64, 64)  ← Length halved!

Each of 64 feature maps gets downsampled 128→64
```

### Translation Invariance

```
Why MaxPooling helps generalize:

Scenario: Step pattern starts at different times

User 1 - Step pattern:
Timestep: [10, 11, 12, 13, 14] → acceleration peaks
         [15, 16, 17, 18, 19] → returns to normal

User 2 - Same step pattern:
Timestep: [12, 13, 14, 15, 16] → acceleration peaks
         [17, 18, 19, 20, 21] → returns to normal
           (shifted by 2 timesteps)

MaxPooling reduces sensitivity to exact timing:
- Both users' peaks will activate same feature maps
- Nearby timesteps collapsed to single output
- Model doesn't memorize exact timing
```

### Data Reduction Through Blocks

```
Block 1:
Input:    (batch, 128, 4)
Conv:     (batch, 128, 64)  ← Same length, 64 features
Pooling:  (batch, 64, 64)   ← Half length

Block 2:
Input:    (batch, 64, 64)
Conv:     (batch, 64, 128)  ← Same length, 128 features
Pooling:  (batch, 32, 128)  ← Half length

Block 3:
Input:    (batch, 32, 128)
Conv:     (batch, 32, 64)   ← Same length, 64 features
(No pooling)

GlobalAveragePooling:
Input:    (batch, 32, 64)
Output:   (batch, 64)       ← Compressed to single vector!
```

### Alternative: Average Pooling

```python
layers.AveragePooling1D(pool_size=2)  # Takes average instead of max

Input: [0.1, 0.8, 0.3, 0.5, 0.2, 0.9, 0.4, 0.6]

MaxPooling:
[0.8, 0.5, 0.9, 0.6]  ← Keeps strongest signals

AveragePooling:
[(0.1+0.8)/2, (0.3+0.5)/2, (0.2+0.9)/2, (0.4+0.6)/2]
= [0.45, 0.4, 0.55, 0.5]  ← Smooths everything

When to use:
- MaxPooling: Detect presence of patterns (recommended)
- AveragePooling: Smooth signals, less aggressive
```

---

## BLOCK 2e: Dropout

### Code
```python
layers.Dropout(0.5)
```

### What It Does

During training: randomly disables 50% of neurons.
During inference: all neurons active but scaled by 0.5.

### Visualization

```
Before Dropout (training):
Activations: [0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6]

Dropout mask (random 50%): [1, 0, 1, 0, 1, 0, 1, 0]
                           (1=keep, 0=kill)

After Dropout: [0.5, 0, 0.2, 0, 0.1, 0, 0.3, 0]
                ↑      ↑      ↑      ↑
           Random killing of neurons!

Inverse scaling (important!):
Rescale to maintain expected value:
[0.5×2, 0, 0.2×2, 0, 0.1×2, 0, 0.3×2, 0]
= [1.0, 0, 0.4, 0, 0.2, 0, 0.6, 0]

At inference: All neurons active, each scaled by (1-rate)=0.5
```

### Why 50% Dropout?

```
rate = 0.5 means:
- 50% of neurons disabled each training step
- Network must learn redundant representations
- Equivalent to training 2^n ensemble models

Effect on Walking classification:
Without dropout: "If neuron_x activates → Walking"
With dropout: neuron_x drops randomly
            Network learns: "If neuron_x OR neuron_y OR neuron_z → Walking"
            More robust!
```

### Co-adaptation Problem

```
Without Dropout:
Neuron 1: "I detect acceleration peak"
Neuron 2: "I detect when Neuron 1 fires" ← Co-adapted!
Neuron 3: "I detect when Neuron 2 fires" ← Chain of dependency!

Problem: Works on training set, fails on unseen users

With Dropout:
Neuron 1 disappears 50% of the time
Neurons 2 and 3 must learn independently
Each neuron becomes self-sufficient
Better generalization!
```

### Effect on Overfitting

```
Without Dropout:
Train Acc: 98% → 99% → 99.5%
Val Acc:   85% → 83% → 80% ← Large gap! Overfitting!

With Dropout (0.5):
Train Acc: 92% → 93% → 94%
Val Acc:   90% → 91% → 91% ← Small gap! Good generalization!
```

### Dropout Rate Tuning

```
Too low (0.1):
- Minimal regularization
- Still memorizes user patterns
- Limited benefit

Optimal (0.3-0.7):
- Balanced learning and regularization
- Good for activity recognition

Too high (0.9):
- Kills too many neurons
- Model can't learn anything
- Severe underfitting
```

---

## BLOCK 3: Conv Block 2 (128 filters)

### Code
```python
layers.Conv1D(filters=128, kernel_size=3, padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))
layers.BatchNormalization()
layers.ReLU()
layers.MaxPooling1D(pool_size=2)
layers.Dropout(0.5)
```

### What's Different From Block 1?

```
Block 1:
- Input: 4 features (raw accelerometer)
- Filters: 64
- Output: 64 feature maps
- Purpose: Learn low-level patterns (peaks, slopes)

Block 2:
- Input: 64 feature maps (from Block 1)
- Filters: 128
- Output: 128 feature maps
- Purpose: Combine low-level into mid-level patterns
```

### Pattern Learning Progression

```
Block 1 - Low-level patterns (learned by 64 filters):
├── Peak detector
├── Valley detector
├── Slope detector
├── Plateau detector
├── Oscillation detector
└── ... (59 more)

Block 2 - Mid-level patterns (learned by 128 filters, using Block 1 outputs):
├── Peak + Valley = Step cycle detector
├── Multiple peaks = Repetition detector
├── Slow slope = Gradual change detector
├── Rapid oscillation = Tremor detector
├── Peak pattern shifts = Speed detector
└── ... (123 more combinations)
```

### Shape Transformation Through Block 2

```
Input to Block 2: (batch, 64, 64)
                  64 timesteps, 64 feature maps from Block 1

Conv1D (128 filters):
(batch, 64, 128)
                  Same 64 timesteps, now 128 different combinations

MaxPooling:
(batch, 32, 128)
                  32 timesteps (halved), 128 feature maps
```

### Why 128 Filters?

```
Pyramid architecture: 64 → 128 → 64

Why pyramid?
- Increase filters to learn more combinations
- Then decrease to prevent explosion of parameters
- Forces compression of information
- Reduces data as it goes deeper

Total parameters through blocks:
Block 1 Conv: 64 × 3 × 4 = 768
Block 2 Conv: 128 × 3 × 64 = 24,576  ← Many more parameters!
Block 3 Conv: 64 × 3 × 128 = 24,576  ← Still high
```

---

## BLOCK 4: GlobalAveragePooling1D

### Code
```python
layers.GlobalAveragePooling1D()
```

### What It Does

Collapses temporal dimension by averaging across all timesteps.

```
Input: (batch, 32, 64)
       32 timesteps × 64 feature maps

For each feature map, compute:
average = sum(all 32 timestep values) / 32

Output: (batch, 64)
        One averaged value per feature map
```

### Detailed Example

```
Input feature map 1:
Timesteps: [0.5, 0.6, 0.8, 0.7, 0.9, ..., 0.4]  (32 values)

GlobalAveragePooling:
Average = (0.5 + 0.6 + 0.8 + 0.7 + 0.9 + ... + 0.4) / 32
        = 15.2 / 32
        = 0.475

Output: 0.475

This done for all 64 feature maps!
Final output shape: (batch, 64)
```

### Why Global Average Pooling?

**Alternative 1: Flatten**
```python
layers.Flatten()

Input: (batch, 32, 64)
Output: (batch, 32 × 64) = (batch, 2048)

Parameters to Dense layer: 2048 → Dense(64) = 131,072 parameters!
Risk: Overfitting, slow, parameter explosion
```

**Alternative 2: GlobalAveragePooling**
```python
layers.GlobalAveragePooling1D()

Input: (batch, 32, 64)
Output: (batch, 64)

Parameters to Dense layer: 64 → Dense(64) = 4,096 parameters
Benefit: 30× smaller, faster, generalizes better!
```

### Semantic Meaning

```
Each feature map output = "How active is this pattern throughout the window?"

Feature map 0: Average activation = 0.75 → Strong pattern throughout
Feature map 1: Average activation = 0.20 → Weak pattern
Feature map 2: Average activation = 0.85 → Very strong pattern

These 64 averaged values = "Summary of what happened in the window"

Example interpretation for Walking detection:
High average values in:
- Step cycle detectors
- Rhythmic pattern detectors
- Leg acceleration detectors

Low average values in:
- Static pattern detectors
- Arm-only detectors
```

### Pooling Strategies Comparison

```
Input: (batch, 32, 64)

1. GlobalAveragePooling:
   Output: (batch, 64)
   Meaning: Average strength of each pattern
   
2. GlobalMaxPooling:
   Output: (batch, 64)
   Meaning: Peak strength of each pattern
   
3. Flatten:
   Output: (batch, 2048)
   Meaning: Raw timestamp and feature details
   
Use GlobalAveragePooling: Reduces parameters, captures essence
```

---

## BLOCK 5: Dense Output Layer

### Code
```python
layers.Dense(DENSE_UNITS=64, activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(0.001))
layers.Dropout(0.5)
layers.Dense(n_outputs=4, activation='softmax')
```

### Part A: Hidden Dense Layer

```python
layers.Dense(64, activation='relu', kernel_regularizer=...)
```

### What It Does

Fully connected layer: every input connects to every neuron.

```
Input: (batch, 64)

Dense(64):
- 64 input values
- 64 output neurons
- Weight matrix: 64 × 64 = 4,096 weights
- Bias vector: 64 biases
- Total: 4,160 parameters

Output: (batch, 64)

Operation: output = ReLU(input × W + b)
           (dot product + activation)
```

### Interpretation

```
Input to Dense (64 values):
[avg_pattern_0, avg_pattern_1, ..., avg_pattern_63]
↓
These 64 pattern strengths are processed by 64 neurons

Each neuron learns:
neuron_k = "When patterns A, B, C are strong together → high activation"

Output (64 hidden values):
[combined_feature_0, combined_feature_1, ..., combined_feature_63]
↓
These combined features fed to final output layer
```

### Why ReLU in Dense Layer?

```
Without ReLU (linear):
output = input × W + b
Stacking multiple dense layers = one big matrix multiplication
= just a linear transformation (no non-linearity!)

With ReLU (non-linear):
output = ReLU(input × W + b)
Stacking enables learning complex decision boundaries
Can separate Walking, Upstairs, Downstairs, Idle in 64D space
```

### Part B: Output Dense Layer

```python
layers.Dense(n_outputs=4, activation='softmax')
```

### What It Does

Final classification: outputs probability for each activity.

```
Input: (batch, 64)

Dense(4):
- 64 input values (hidden layer)
- 4 output neurons (one per activity)
- Weight matrix: 64 × 4 = 256 weights
- Bias vector: 4 biases
- Total: 260 parameters

Output before softmax: (batch, 4)
Raw logits: [2.3, -0.5, 1.2, 0.1]

Softmax applied:
exp(2.3) = 9.97
exp(-0.5) = 0.61
exp(1.2) = 3.32
exp(0.1) = 1.11
Sum = 15.01

Probabilities:
[9.97/15.01, 0.61/15.01, 3.32/15.01, 1.11/15.01]
= [0.664, 0.041, 0.221, 0.074]
         ↑        ↑        ↑        ↑
      Walking Upstairs Down Idle

Prediction: Walking (highest probability)
```

### Softmax Importance

```
Why softmax instead of ReLU?

ReLU output: [2.3, -0.5, 1.2, 0.1]
- Raw values, hard to interpret
- No probability meaning
- Could be any range

Softmax output: [0.664, 0.041, 0.221, 0.074]
- All between 0-1
- Sum to 1.0 (valid probability distribution)
- Can be interpreted as "confidence"
- Used with CrossEntropy loss
```

### Decision Boundary Visualization

```
2D simplified view (64D reality is higher):

    Upstairs
        ^
        |   /──────────────
    Idle|  /
        | /
        |/────────────── Walking
        └────────────────────→ Downstairs

Dense layer learns decision boundaries separating:
- Walking (high step frequency)
- Upstairs (sustained upward acceleration)
- Downstairs (sustained downward acceleration)  
- Idle (minimal movement)
```

---

## Complete Block Summary Table

| Block | Operation | Input Shape | Output Shape | Parameters | Purpose |
|-------|-----------|-------------|--------------|------------|---------|
| Input | Metadata | - | (batch, 128, 4) | 0 | Define expected shape |
| Block 1 Conv | Conv1D(64) | (batch, 128, 4) | (batch, 128, 64) | 832 | Low-level patterns |
| Block 1 BN | BatchNorm | (batch, 128, 64) | (batch, 128, 64) | 256 | Stabilize training |
| Block 1 ReLU | Activation | (batch, 128, 64) | (batch, 128, 64) | 0 | Non-linearity |
| Block 1 Pool | MaxPool(2) | (batch, 128, 64) | (batch, 64, 64) | 0 | Downsample time |
| Block 1 Drop | Dropout(0.5) | (batch, 64, 64) | (batch, 64, 64) | 0 | Regularization |
| Block 2 Conv | Conv1D(128) | (batch, 64, 64) | (batch, 64, 128) | 24,576 | Mid-level patterns |
| Block 2 BN | BatchNorm | (batch, 64, 128) | (batch, 64, 128) | 512 | Stabilize |
| Block 2 ReLU | Activation | (batch, 64, 128) | (batch, 64, 128) | 0 | Non-linearity |
| Block 2 Pool | MaxPool(2) | (batch, 64, 128) | (batch, 32, 128) | 0 | Downsample time |
| Block 2 Drop | Dropout(0.5) | (batch, 32, 128) | (batch, 32, 128) | 0 | Regularization |
| Block 3 Conv | Conv1D(64) | (batch, 32, 128) | (batch, 32, 64) | 24,576 | High-level patterns |
| Block 3 BN | BatchNorm | (batch, 32, 64) | (batch, 32, 64) | 256 | Stabilize |
| Block 3 ReLU | Activation | (batch, 32, 64) | (batch, 32, 64) | 0 | Non-linearity |
| GAP | GlobalAvgPool | (batch, 32, 64) | (batch, 64) | 0 | Collapse temporal |
| Dense Hidden | Dense(64) | (batch, 64) | (batch, 64) | 4,160 | Combine patterns |
| Drop Final | Dropout(0.5) | (batch, 64) | (batch, 64) | 0 | Regularization |
| Dense Out | Dense(4) + Softmax | (batch, 64) | (batch, 4) | 260 | Output probabilities |

**Total Parameters: 832 + 24,576 + 24,576 + 4,160 + 260 + 768 (biases) ≈ 56,000**

---

## Data Flow Through Entire Model

```
Raw accelerometer window:
(64, 128, 4)
├─ 64 samples in batch
├─ 128 timesteps per sample
└─ 4 features: AccX, AccY, AccZ, Magnitude

↓ Conv Block 1 (64 filters)
(64, 64, 64)
├─ 64 feature maps detected
├─ 64 timesteps remaining
└─ Information condensed but preserved

↓ Conv Block 2 (128 filters)
(64, 32, 128)
├─ 128 combinations of patterns
├─ 32 timesteps (coarse level)
└─ High-level features

↓ Conv Block 3 (64 filters)
(64, 32, 64)
├─ 64 discriminative features
├─ Ready for classification
└─ Final convolution layer

↓ GlobalAveragePooling
(64, 64)
├─ One value per feature
├─ Summarizes entire window
└─ Compact representation

↓ Dense Hidden (64 neurons)
(64, 64)
├─ Combines features non-linearly
├─ Learns decision logic
└─ Hidden representation

↓ Output Dense (4 neurons) + Softmax
(64, 4)
├─ Walking:    [probability]
├─ Upstairs:   [probability]
├─ Downstairs: [probability]
└─ Idle:       [probability]

Final prediction: argmax of 4 probabilities
```

---

## Debugging Each Block

### Block Issues and Solutions

**Issue: Poor accuracy after Conv Block 1**
```
Symptom: Accuracy stuck at 25% (random guessing)
Likely cause: Conv filters not learning patterns
Solutions:
1. Increase filters: 64 → 128
2. Decrease L2 regularization: 0.001 → 0.0001
3. Check data preprocessing (normalization)
4. Increase training data
```

**Issue: Training very slow**
```
Symptom: 30+ minutes per epoch
Likely cause: Model too large or batch size too small
Solutions:
1. Reduce filters: 128 → 64
2. Reduce kernel size: 3 → 2
3. Increase batch size: 64 → 128
4. Use GPU (if not already)
```

**Issue: Overfitting detected**
```
Symptom: Train acc 98%, val acc 75%
Likely cause: Model too complex, insufficient regularization
Solutions:
1. Increase dropout: 0.5 → 0.7
2. Increase L2: 0.001 → 0.01
3. Reduce filters: 128 → 64
4. Early stopping patience: 20 → 10
```

**Issue: Underfitting detected**
```
Symptom: Train acc 60%, val acc 60% (both low and flat)
Likely cause: Model too simple, too much regularization
Solutions:
1. Decrease dropout: 0.5 → 0.3
2. Decrease L2: 0.001 → 0.0001
3. Increase filters: 64 → 128
4. More training data
```

---

## Modifications and Experiments

### Experiment 1: Increase Capacity

```python
# Original
CONV_FILTERS_BLOCK1 = 64
CONV_FILTERS_BLOCK2 = 128
CONV_FILTERS_BLOCK3 = 64
DENSE_UNITS = 64

# Increased
CONV_FILTERS_BLOCK1 = 128
CONV_FILTERS_BLOCK2 = 256
CONV_FILTERS_BLOCK3 = 128
DENSE_UNITS = 128

Expected result: Higher accuracy if underfitting
Risk: Slower training, risk of overfitting
```

### Experiment 2: Add Residual Connection

```python
# Create functional API instead of Sequential
def build_model_with_residual():
    input_layer = layers.Input(shape=(128, 4))
    
    # Block 1
    x = layers.Conv1D(64, 3, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.5)(x)
    
    # Block 2
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x_skip = x  # Save for skip connection
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.5)(x)
    
    # Block 3 with skip
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, x_skip])  # Add skip connection
    x = layers.ReLU()(x)
    
    # Rest...
    model = models.Model(inputs=input_layer, outputs=...)
    return model
```

### Experiment 3: Different Pooling Strategy

```python
# Original: MaxPooling
layers.MaxPooling1D(pool_size=2)

# Alternative 1: No pooling, stride instead
layers.Conv1D(64, 3, strides=2, padding='same')  # Reduces by 2

# Alternative 2: Adaptive pooling (dynamic)
layers.Conv1D(64, 3, padding='same')
layers.GlobalAveragePooling1D()  # Applied earlier

# Comparison:
MaxPooling: Good for capturing peaks, translation invariant
Striding: Integrated into convolution, slight parameter reduction
GlobalAveragePooling: Too aggressive if used mid-network
```

This detailed breakdown should make each block's operation crystal clear!
