import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Paths
PROCESSED_DATA_PATH = "./processed_data"
TEST_DATA_PATH = "./test_data"

# Load data
try:
    with open(f"{PROCESSED_DATA_PATH}/features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(f"{PROCESSED_DATA_PATH}/labels.pkl", "rb") as f:
        labels = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure you have run the data preparation script correctly.")
    exit()

# Check if data is non-empty
if not features or not labels:
    print("Error: Loaded features or labels are empty. Check your data preparation step.")
    exit()

# Display data shape for debugging
print(f"Number of features: {len(features)}, Number of labels: {len(labels)}")

# Convert labels to numerical format
encoder = LabelEncoder()
try:
    labels_encoded = encoder.fit_transform(labels)
    print(f"Encoded labels: {set(labels_encoded)}")
    # Save the LabelEncoder
    with open(f"{PROCESSED_DATA_PATH}/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

except ValueError as e:
    print(f"Error encoding labels: {e}")
    exit()

# One-hot encode the labels
labels_categorical = to_categorical(labels_encoded)

# Check categorical labels for debugging
print(f"Categorical labels shape: {labels_categorical.shape}")

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels_categorical, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save splits
os.makedirs(TEST_DATA_PATH, exist_ok=True)
np.save(f"{PROCESSED_DATA_PATH}/X_train.npy", X_train)
np.save(f"{PROCESSED_DATA_PATH}/X_val.npy", X_val)
np.save(f"{TEST_DATA_PATH}/X_test.npy", X_test)
np.save(f"{PROCESSED_DATA_PATH}/y_train.npy", y_train)
np.save(f"{PROCESSED_DATA_PATH}/y_val.npy", y_val)
np.save(f"{TEST_DATA_PATH}/y_test.npy", y_test)

print("Data splitting complete!")
