import numpy as np
import tensorflow as tf
import os
import shutil

# Load model robustly
model_path = 'best_model.keras'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    h5_path = 'best_model.h5'
    if not os.path.exists(h5_path):
        shutil.copy(model_path, h5_path)
    model = tf.keras.models.load_model(h5_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load test data
X_test = np.load('./test_data/X_test.npy')
y_test = np.load('./test_data/y_test.npy')

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

