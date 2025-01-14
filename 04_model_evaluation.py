import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('best_model.keras')

# Load test data
X_test = np.load('./processed_data/X_test.npy')
y_test = np.load('./processed_data/y_test.npy')

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

