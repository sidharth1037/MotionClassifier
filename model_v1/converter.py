import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model('motion_model_1.0.h5')

# Convert to .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
