# model_v5.py - Documentation

## Overview
This script builds, trains, and evaluates a 1D Convolutional Neural Network (1D CNN) for human activity recognition. It loads preprocessed sensor data, trains the model with advanced regularization techniques, saves the best checkpoint, and generates comprehensive evaluation metrics and visualizations.

## Why This Architecture?

### 1. **1D Convolutional Neural Network (1D CNN)**
- **Why CNNs?**: Excellent at learning temporal patterns in sequential data
- **How It Works**: 
  - Conv1D filters scan across time, learning local patterns
  - Example: Filter learns "peak followed by valley" pattern of a footstep
  - Pooling reduces data size, making model translation-invariant
- **Advantage over RNN/LSTM**: 
  - Faster training (parallelizable convolutions)
  - Better for periodic/rhythmic patterns (activity recognition)
  - Requires less memory
- **Why 1D?**: Sensor data is a 1D time series, not 2D images

### 2. **Three-Block Architecture**
```
Input → Conv Block 1 (low-level features) 
      → Conv Block 2 (mid-level features) 
      → Conv Block 3 (high-level features)
      → Global Average Pooling
      → Dense Classification Layer
      → Output (4 activities)
```

**Why Multiple Blocks?**
- Block 1: Learns basic patterns (peaks, slopes)
- Block 2: Combines basic patterns into complex ones (full step cycle)
- Block 3: Refines high-level features for classification

### 3. **Regularization Techniques**

| Technique | Why | Effect |
|-----------|-----|--------|
| **Dropout (0.5)** | Randomly disables neurons during training | Prevents memorizing user quirks |
| **L2 Regularization (0.001)** | Penalizes large weights | Encourages simpler models |
| **Batch Normalization** | Normalizes internal layer inputs | Stabilizes training, faster convergence |
| **Class Weights** | Gives higher weight to underrepresented classes | Balances imbalanced dataset |

**Combined Effect**: Robust model that generalizes to unseen users

### 4. **Learning Rate Scheduling**
- **Initial**: 0.0005 (small steps = stable learning)
- **Reduction**: If validation loss plateaus, reduce by 0.5x
- **Purpose**: Fine-tune loss landscape when learning slows

### 5. **Early Stopping**
- **Why**: Prevent overfitting by stopping when validation loss stops improving
- **Patience**: 20 epochs (allow time for fluctuations)
- **Benefit**: Saves training time, ensures best generalization

## Key Parameters

### Architecture Configuration
```python
CONV_FILTERS_BLOCK1 = 64        # Low-level feature detectors
CONV_FILTERS_BLOCK2 = 128       # Mid-level feature detectors
CONV_FILTERS_BLOCK3 = 64        # High-level feature detectors
KERNEL_SIZE = 3                 # Look at 3 timesteps at a time
POOL_SIZE = 2                   # Downsample by 2x after each conv block
DROPOUT_RATE = 0.5              # Disable 50% of neurons during training
DENSE_UNITS = 64                # Hidden layer before output
L2_REGULARIZATION = 0.001       # Weight penalty strength
```

**What You Can Change:**

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| `CONV_FILTERS_*` | 32-256 | More filters = more patterns, but slower + more parameters |
| `KERNEL_SIZE` | 2-7 | Larger = more temporal context, fewer output steps |
| `DROPOUT_RATE` | 0.3-0.7 | Higher = stronger regularization, risk of underfitting |
| `L2_REGULARIZATION` | 0.0001-0.01 | Higher = more aggressive weight penalty |
| `DENSE_UNITS` | 32-512 | Larger = more capacity, but slower |

### Training Configuration
```python
MAX_EPOCHS = 60                 # Maximum training duration
BATCH_SIZE = 64                 # Samples per gradient update
LEARNING_RATE = 0.0005          # Initial step size for optimizer
LR_DECAY_FACTOR = 0.5           # Reduce learning rate by this factor
LR_DECAY_PATIENCE = 5           # Epochs without improvement before decay
EARLY_STOPPING_PATIENCE = 20    # Epochs without improvement before stop
```

**What You Can Change:**

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| `BATCH_SIZE` | 16-128 | Larger = faster training, less gradient noise |
| `LEARNING_RATE` | 0.0001-0.001 | Higher = faster learning, risk of instability |
| `MAX_EPOCHS` | 30-200 | Upper limit; early stopping usually stops earlier |
| `EARLY_STOPPING_PATIENCE` | 10-30 | Higher = wait longer for improvement |
| `LR_DECAY_PATIENCE` | 3-10 | Lower = reduce LR more aggressively |

### Data Configuration
```python
DATASET_FILE = "motion_dataset_loso_v5.npz"  # From split_normalize.py
MODEL_SAVE_NAME = "best_motion_model_v5.keras"
MODEL_REPORT_FILE = "model_v5_report.txt"
```

## General Structure

### Input Format
**File**: `motion_dataset_loso_v5.npz` (output from split_normalize.py)
```python
X_train.shape = (n_train, 128, 4)      # Training features
X_test.shape = (n_test, 128, 4)        # Test features (unseen user)
y_train.shape = (n_train, 4)           # One-hot encoded labels
y_test.shape = (n_test, 4)             # One-hot encoded labels
mean, std                              # Normalization statistics
```

### Training Pipeline

```
1. LOAD DATA
   - Load preprocessed & normalized dataset
   - Auto-detect shapes: timesteps, features, classes
   
2. BUILD MODEL
   - Create Sequential CNN with 3 conv blocks
   - Add regularization (dropout, L2, batch norm)
   - Compile with Adam optimizer & categorical crossentropy
   
3. COMPUTE CLASS WEIGHTS
   - Identify imbalanced classes
   - Upweight minority classes during training
   
4. TRAIN
   - Use callbacks: checkpoint, LR scheduling, early stopping
   - Training continues until plateau or max epochs
   - Best weights saved automatically
   
5. EVALUATE
   - Load best saved model
   - Compute accuracy/loss on test set
   - Generate confusion matrix & per-class metrics
   
6. VISUALIZE
   - Plot training curves (accuracy, loss)
   - Visualize confusion matrix heatmap
   
7. SAVE METADATA
   - Architecture details
   - Training configuration
   - Results & metrics
   - Save to model_v5_report.txt
```

### Output Format

**Model Files**:
- `best_motion_model_v5.keras` - Trained model weights and architecture
- `model_v5_report.txt` - Comprehensive text report with all metadata

**Metrics Computed**:
- Test Accuracy / Loss
- Confusion Matrix (4×4)
- Per-class Precision, Recall, F1-Score
- Training curves (accuracy, loss over epochs)

## Important Considerations

### Class Imbalance Handling
```python
class_weights = compute_class_weight('balanced', ...)
# Automatically upweights minority activities
# Example: If Walking=1000 samples, Idle=100 samples
#          Idle gets weight 10x higher in loss calculation
```

**Why Important**: Without this, model ignores Idle (too rare to impact overall accuracy)

### Global Average Pooling
```python
layers.GlobalAveragePooling1D()  # Replace Flatten
```
- **Alternative**: Flatten all conv outputs into 1D vector
- **Why GAP is better**:
  - Drastically fewer parameters (128×64 = 8K vs flatten = 8M+)
  - More mobile-friendly (critical for deployment)
  - More robust to small shifts in time

### Model Checkpointing
```python
checkpoint = callbacks.ModelCheckpoint(
    'best_motion_model_v5.keras',
    monitor='val_accuracy',
    save_best_only=True
)
```

**Why**: Ensures you keep the best version, not the final version
- Example: Best at epoch 45 (90% acc), but epoch 60 overfits (87% acc)
- Without checkpoint, you'd use epoch 60

### Data Leakage Prevention
```python
# ✓ CORRECT: Use training statistics only
mean = X_train.mean()
X_test_norm = (X_test - mean) / std_train

# ✗ WRONG: Using combined statistics
mean = np.concatenate([X_train, X_test]).mean()
```
The script correctly uses training statistics (already done in split_normalize.py).

## Debugging & Optimization

### Training Curves Analysis

**Healthy Training**:
```
Train Accuracy: 95% → 98%
Val Accuracy: 92% → 93% (small gap)
→ Model is learning without severe overfitting
```

**Overfitting**:
```
Train Accuracy: 99% → 99.5%
Val Accuracy: 85% → 80% (large gap, decreasing)
→ Increase DROPOUT_RATE or L2_REGULARIZATION
```

**Underfitting**:
```
Train Accuracy: 60% → 65%
Val Accuracy: 62% → 63% (tracking together)
→ Increase model capacity or decrease regularization
```

### Common Issues

**"Early stopping triggered at epoch 20"**
- Model plateau early; insufficient improvement
- Try: Reduce LEARNING_RATE, increase MAX_EPOCHS
- Or: Increase CONV_FILTERS_* for more capacity

**Very low accuracy (< 60%)**
- Data preprocessing issue (verify split_normalize.py output)
- Or: Insufficient training data (need >100 samples per class)
- Or: Feature engineering problem (check raw CSV data quality)

**Memory error during training**
- Reduce BATCH_SIZE (e.g., 64 → 32)
- Or: Reduce CONV_FILTERS_BLOCK2 (e.g., 128 → 64)

**Large gap between train and validation accuracy**
- Increase DROPOUT_RATE (0.5 → 0.6 or 0.7)
- Increase L2_REGULARIZATION (0.001 → 0.005)
- Reduce model capacity (fewer filters)

### Hyperparameter Tuning Strategy

**Step 1**: Get baseline with default parameters
**Step 2**: If overfitting:
- Increase `DROPOUT_RATE` to 0.6-0.7
- Increase `L2_REGULARIZATION` to 0.005-0.01

**Step 3**: If underfitting:
- Decrease `DROPOUT_RATE` to 0.3
- Increase `CONV_FILTERS_BLOCK2` to 256
- Decrease `LEARNING_RATE` to 0.0001 (slower learning)

**Step 4**: Fine-tune learning rate schedule:
- Reduce `EARLY_STOPPING_PATIENCE` if converging slowly
- Reduce `LR_DECAY_PATIENCE` to be more aggressive with LR reduction

## Model Performance Interpretation

### Confusion Matrix Example
```
         Predicted
         Walk  Up  Down Idle
Actual   Walk  85   5    5    5
         Up    10  70   15    5
         Down  5   20   70    5
         Idle  10   5    5   80
```

**Insights**:
- Walking: Well-predicted (85% diagonal)
- Upstairs: Confused with Downstairs (20% off-diagonal)
- Idle: Good performance (80%)

**Actions**:
- Upstairs/Downstairs confusion → Might need more training data
- Or: Consider collecting more fine-grained features

### Per-Class Metrics
```
             Precision  Recall  F1-Score
Walking        0.89     0.85     0.87
Upstairs       0.78     0.70     0.74
Downstairs     0.82     0.70     0.75
Idle           0.94     0.80     0.87
```

- **Precision**: Of predicted X, how many correct?
- **Recall**: Of actual X, how many found?
- **F1**: Harmonic mean; single score balancing both

## Advanced Customization

### Using Different Optimizers
```python
# Default: Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Alternative: SGD with momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# Alternative: RMSprop
optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
```

### Adding Skip Connections (ResNet-style)
```python
# Advanced: For deeper networks (not in current v5)
x = layers.Conv1D(...)(input_layer)
x = layers.ReLU()(x)
x = layers.Conv1D(...)(x)
x = layers.Add()([x, input_layer])  # Skip connection
```

### Data Augmentation
```python
# Before training, could add:
# - Time warping (stretch/compress windows)
# - Noise injection (add Gaussian noise)
# - Rotation (different phone orientations)
```

## Next Steps

1. **Monitor Training**: Watch console output for accuracy/loss trends
2. **Check Report**: After training, review `model_v5_report.txt`
3. **Convert to Mobile**: Use `converter.py` to create `.tflite` model
4. **Deploy**: Use `.tflite` on edge devices (phones, smartwatches)

## Reproduction

To reproduce exact results:
```python
# Ensure all random seeds are set
np.random.seed(42)
tf.random.set_seed(42)

# Use exact parameters from this file
# Use exact training data from split_normalize.py
```

Results may vary slightly due to:
- GPU/CPU numerical differences
- Random weight initialization (set seed to fix)
- Different TensorFlow/Keras versions
