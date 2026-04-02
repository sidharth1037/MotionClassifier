import tensorflow as tf

# 1. Load the model
model_path = 'best_motion_model_v7.keras' 
print(f"Loading {model_path}...")

# ==============================================================================
# V7 NOTE: We need to register the custom focal loss function when loading.
# Without this, Keras won't know how to deserialize the model because it uses
# a custom loss (focal_loss_fn) that isn't a built-in Keras loss.
# ==============================================================================

# Focal loss parameters (must match training)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """Focal loss for multi-class classification (must match training definition)."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal_loss_value = alpha * focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss_value, axis=-1))
    return focal_loss_fn

model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fn': focal_loss()})

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIONAL: Optimization (makes the app smaller/faster)
# This quantizes weights from Float32 to Int8/Float16 where possible
#converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. Save
tflite_filename = 'model_v7.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Saved to {tflite_filename}")
