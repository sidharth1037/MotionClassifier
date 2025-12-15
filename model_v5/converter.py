import tensorflow as tf

# 1. Load the model (Update filename if you switched to .keras)
model_path = 'best_motion_model_v5.keras' 
print(f"Loading {model_path}...")
model = tf.keras.models.load_model(model_path)

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIONAL: Optimization (makes the app smaller/faster)
# This quantizes weights from Float32 to Int8/Float16 where possible
#converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. Save
tflite_filename = 'model_v5.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Saved to {tflite_filename}")