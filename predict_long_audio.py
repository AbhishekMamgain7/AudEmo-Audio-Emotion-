import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
import librosa

# Paths
PROCESSED_DATA_PATH = "./processed_data"
LSTM_MODEL_PATH = "./results/cnn_lstm/best_lstm_model.keras"
GRU_MODEL_PATH = "./results/cnn_gru/best_gru_model.keras"
CNN_MODEL_PATH = "./best_model.keras"

def load_recurrent_model(model_path):
    # Load and compile model robustly
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        h5_path = model_path.replace('.keras', '.h5')
        if not os.path.exists(h5_path):
            import shutil
            shutil.copy(model_path, h5_path)
        model = tf.keras.models.load_model(h5_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_cnn_extractor(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        h5_path = model_path.replace('.keras', '.h5')
        if not os.path.exists(h5_path):
            import shutil
            shutil.copy(model_path, h5_path)
        model = tf.keras.models.load_model(h5_path)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Find penultimate dense layer
    dense_layer = None
    for layer in reversed(model.layers[:-1]):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 128:
            dense_layer = layer
            break
    if dense_layer is None:
        dense_layer = model.layers[-3]
        
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=dense_layer.output)
    return feature_extractor, model

def process_audio(file_path, chunk_duration, overlap, sr):
    y, _ = librosa.load(file_path, sr=sr)
    chunk_size = int(chunk_duration * sr)
    hop_size = int((chunk_duration - overlap) * sr)
    if hop_size <= 0:
        hop_size = chunk_size
        
    chunks = []
    start = 0
    while start < len(y):
        end = start + chunk_size
        chunk = y[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        chunks.append(chunk)
        start += hop_size
    return chunks

def extract_chunk_features(chunks, sr):
    features_list = []
    for chunk in chunks:
        spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        if log_spectrogram.shape[1] < 128:
            pad_width = 128 - log_spectrogram.shape[1]
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_spectrogram = log_spectrogram[:, :128]
        features_list.append(log_spectrogram)
    return np.array(features_list)

def predict(audio_path, model_type="lstm", show_chunks=True):
    # 1. Load configuration and encoders
    with open(os.path.join(PROCESSED_DATA_PATH, "sequence_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
        
    chunk_duration = metadata["chunk_duration"]
    overlap = metadata["overlap"]
    sr = metadata["sample_rate"]
    max_sequence_length = metadata["max_sequence_length"]
    embedding_dim = metadata["embedding_dim"]
    
    print(f"Loading models...")
    # 2. Load feature extractor
    feature_extractor, cnn_model = load_cnn_extractor(CNN_MODEL_PATH)
    
    # 3. Load recurrent model
    if model_type.lower() == "lstm":
        model_path = LSTM_MODEL_PATH
        model_name = "CNN + LSTM"
    elif model_type.lower() == "gru":
        model_path = GRU_MODEL_PATH
        model_name = "CNN + GRU"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Select 'lstm' or 'gru'.")
        
    rec_model = load_recurrent_model(model_path)
    
    # 4. Load & Chunk audio
    print(f"Processing audio: {audio_path}")
    chunks = process_audio(audio_path, chunk_duration, overlap, sr)
    if not chunks:
        print("Error: Could not process audio.")
        return
        
    print(f"Divided audio into {len(chunks)} chunks.")
    
    # 5. Extract features & embeddings
    chunk_feats = extract_chunk_features(chunks, sr)
    embeddings = feature_extractor(chunk_feats, training=False).numpy()
    cnn_probs = cnn_model(chunk_feats, training=False).numpy()
    
    # Expose chunk level predictions
    if show_chunks:
        print("\n--- Chunk-Level Predictions (CNN) ---")
        for idx, probs in enumerate(cnn_probs):
            pred_idx = np.argmax(probs)
            label = label_encoder.inverse_transform([pred_idx])[0]
            conf = probs[pred_idx] * 100
            print(f"Chunk {idx+1}: {label:<10} | Confidence: {conf:.1f}%")
        print("--------------------------------------\n")
        
    # 6. Sequence construction (Padding / Truncating)
    actual_len = min(len(embeddings), max_sequence_length)
    padded_emb = np.zeros((1, max_sequence_length, embedding_dim))
    padded_emb[0, :actual_len] = embeddings[:actual_len]
    
    # 7. Recurrent Model Inference
    y_pred_probs = rec_model(padded_emb, training=False).numpy()[0]
    pred_class_idx = np.argmax(y_pred_probs)
    predicted_emotion = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = y_pred_probs[pred_class_idx] * 100
    
    print("=== FINAL PREDICTION ===")
    print(f"Predicted Emotion: {predicted_emotion.capitalize()}")
    print(f"Confidence:        {confidence:.1f}%")
    print(f"Model used:        {model_name}")
    print("========================\n")
    
    return {
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "model_used": model_name,
        "chunk_predictions": [
            {
                "chunk_id": i + 1,
                "predicted_emotion": label_encoder.inverse_transform([np.argmax(p)])[0],
                "confidence": float(np.max(p))
            } for i, p in enumerate(cnn_probs)
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion of a long audio file using sequential models.")
    parser.add_argument("--file", type=str, required=True, help="Path to audio (.wav) file.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"], help="Recurrent model to use (lstm or gru).")
    parser.add_argument("--hide_chunks", action="store_true", help="Hide chunk-level predictions.")
    
    args = parser.parse_args()
    predict(args.file, args.model, not args.hide_chunks)
