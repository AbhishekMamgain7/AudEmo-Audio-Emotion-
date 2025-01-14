import numpy as np
import tensorflow as tf
import pickle

# Paths
MODEL_PATH = "./best_model.keras"
PROCESSED_DATA_PATH = "./processed_data"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data
X_test = np.load(f"{PROCESSED_DATA_PATH}/X_test.npy")
y_test = np.load(f"{PROCESSED_DATA_PATH}/y_test.npy")

# Load label encoder for decoding predictions
LABEL_ENCODER_PATH = f"{PROCESSED_DATA_PATH}/label_encoder.pkl"
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Decode class indices to emotion labels
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
y_true_labels = label_encoder.inverse_transform(y_true_classes)


# print(type(y_pred_labels), type(y_true_labels))
while True:
    for i in range(10):
        print(f"{i+1}. sample{i+1}.wav")
    user_input = input("Enter file number to predict emotion (type 'exit' for exit): ")
    if user_input.lower() == 'exit':
        break
    else:
        if int(user_input)<=10 and int(user_input)>=0:
            print(f"{user_input}. sample{user_input}.wav  ->  predicted value : {y_pred_labels[int(user_input)-1]} | true value : {y_true_labels[int(user_input)-1]}")
        else:
            print("Enter valid file number")

# Optional: Save results to a file
# with open(f"{PROCESSED_DATA_PATH}/test_predictions.txt", "w") as f:
#     f.write("Sample,Predicted,True\n")
#     for i in range(len(y_pred_labels)):
#         f.write(f"sample{i + 1}.wav,{y_pred_labels[i]},{y_true_labels[i]}\n")

# print("\nPredictions saved to 'test_predictions.txt'")