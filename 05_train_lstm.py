import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Configurable Hyperparameters
LSTM_UNITS = 64
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Paths
PROCESSED_DATA_PATH = "./processed_data"
RESULTS_DIR = "./results/cnn_lstm"
MODEL_PATH = "./results/cnn_lstm/best_lstm_model.keras"

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=== Training CNN + LSTM Model ===")
    
    # Load dataset
    X_train = np.load(f"{PROCESSED_DATA_PATH}/X_train_seq.npy")
    X_val = np.load(f"{PROCESSED_DATA_PATH}/X_val_seq.npy")
    X_test = np.load(f"{PROCESSED_DATA_PATH}/X_test_seq.npy")
    
    y_train = np.load(f"{PROCESSED_DATA_PATH}/y_train_seq.npy")
    y_val = np.load(f"{PROCESSED_DATA_PATH}/y_val_seq.npy")
    y_test = np.load(f"{PROCESSED_DATA_PATH}/y_test_seq.npy")
    
    # Load metadata and label encoder
    with open(f"{PROCESSED_DATA_PATH}/sequence_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    with open(f"{PROCESSED_DATA_PATH}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    num_classes = metadata["num_classes"]
    max_sequence_length = metadata["max_sequence_length"]
    embedding_dim = metadata["embedding_dim"]
    
    print(f"Data Loaded:")
    print(f"  - Train shape: {X_train.shape}")
    print(f"  - Val shape: {X_val.shape}")
    print(f"  - Test shape: {X_test.shape}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Max sequence length: {max_sequence_length}")
    print(f"  - Embedding dim: {embedding_dim}")
    
    # One-hot encode targets
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    # Define LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_sequence_length, embedding_dim)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(units=LSTM_UNITS, return_sequences=False),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model summary to text file
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    with open(f"{RESULTS_DIR}/model_summary.txt", "w") as f:
        f.write("\n".join(summary_list))
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    # Train Model
    import time
    start_time = time.time()
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Save history
    history_dict = history.history
    with open(f"{RESULTS_DIR}/training_history.json", "w") as f:
        json.dump(history_dict, f)
        
    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title('LSTM Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Train Acc')
    plt.plot(history_dict['val_accuracy'], label='Val Acc')
    plt.title('LSTM Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/loss_accuracy_curves.png")
    plt.close()
    
    # Load best model for evaluation
    print(f"Loading best model for evaluation from {MODEL_PATH}...")
    best_model = tf.keras.models.load_model(MODEL_PATH)
    best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Evaluate
    test_loss, test_acc = best_model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "training_time_seconds": training_time,
        "trainable_parameters": best_model.count_params()
    }
    
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Classification Report
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write(class_report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('LSTM Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.close()
    
    # Save Config
    config = {
        "lstm_units": LSTM_UNITS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_sequence_length": max_sequence_length,
        "embedding_dim": embedding_dim
    }
    with open(f"{RESULTS_DIR}/lstm_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("LSTM Training and Evaluation completed successfully!")

if __name__ == "__main__":
    main()
