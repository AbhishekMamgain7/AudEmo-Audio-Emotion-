import os
import pickle
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration block
CHUNK_DURATION = 1.0       # Chunk duration in seconds
OVERLAP = 0.5              # Overlap between consecutive chunks in seconds
SAMPLE_RATE = 16000        # Resampling rate
MAX_SEQUENCE_LENGTH = 10   # Max sequence length (number of chunks)
DATASET_PATH = "./TESS Toronto emotional speech set data"
PROCESSED_DATA_PATH = "./processed_data"
MODEL_PATH = "./best_model.keras"

def build_feature_extractor(model_path):
    print(f"Loading CNN model from {model_path}...")
    # Load CNN robustly (handling Keras 3 HDF5 issue)
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        h5_path = model_path.replace('.keras', '.h5')
        if not os.path.exists(h5_path):
            import shutil
            shutil.copy(model_path, h5_path)
        model = tf.keras.models.load_model(h5_path)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Find penultimate dense layer (128 units)
    dense_layer = None
    for layer in reversed(model.layers[:-1]):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 128:
            dense_layer = layer
            break
    if dense_layer is None:
        dense_layer = model.layers[-3]  # Fallback to index
    
    print(f"Using layer '{dense_layer.name}' for extracting embeddings.")
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=dense_layer.output)
    
    # We also return the original model for baseline probability predictions
    return feature_extractor, model

def process_file_into_chunks(file_path, chunk_duration, overlap, sr):
    try:
        y, _ = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Warning: Failed to load {file_path}. Skipping. Error: {e}")
        return None
    
    chunk_size = int(chunk_duration * sr)
    hop_size = int((chunk_duration - overlap) * sr)
    if hop_size <= 0:
        hop_size = chunk_size  # No overlap fallback
    
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
        # Pad or crop to 128 frames
        if log_spectrogram.shape[1] < 128:
            pad_width = 128 - log_spectrogram.shape[1]
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_spectrogram = log_spectrogram[:, :128]
        features_list.append(log_spectrogram)
    return np.array(features_list)

def main():
    print("=== Phase 1: Dataset Inspection ===")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} does not exist.")
        return
    
    all_files = []
    all_labels = []
    
    # Discover dataset
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".wav") and not file.startswith("._"):
                file_path = os.path.join(root, file)
                parts = file.split('_')
                if len(parts) >= 3:
                    label = parts[2].split('.')[0].lower()
                    all_files.append(file_path)
                    all_labels.append(label)
    
    print(f"Found {len(all_files)} audio files.")
    unique_labels = sorted(list(set(all_labels)))
    print(f"Labels discovered: {unique_labels}")
    
    # Load/Create LabelEncoder
    encoder_path = os.path.join(PROCESSED_DATA_PATH, "label_encoder.pkl")
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print("Loaded existing label encoder.")
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        print("Created and saved new label encoder.")
        
    num_classes = len(label_encoder.classes_)
    
    print("\n=== Phase 2: Split Dataset (Audio-File Level) ===")
    # Perform split on files directly to prevent leakage
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Assert disjoint splits
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    assert len(train_set.intersection(val_set)) == 0, "Leakage between train and validation!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage between validation and test!"
    print(f"Split completed successfully:")
    print(f"  - Train: {len(train_files)} files")
    print(f"  - Validation: {len(val_files)} files")
    print(f"  - Test: {len(test_files)} files")
    print("Verification: Train, validation, and test splits are completely disjoint. No data leakage.")
    
    print("\n=== Phase 3: CNN Feature & Probability Extraction ===")
    feature_extractor, cnn_model = build_feature_extractor(MODEL_PATH)
    
    def process_split(files, labels, name):
        print(f"Processing {name} split ({len(files)} files)...")
        embedding_dim = feature_extractor.output_shape[-1]
        seq_embeddings = []
        seq_probs = []
        seq_lengths = []
        final_labels = []
        
        for idx, (file_path, label) in enumerate(zip(files, labels)):
            chunks = process_file_into_chunks(file_path, CHUNK_DURATION, OVERLAP, SAMPLE_RATE)
            if chunks is None or len(chunks) == 0:
                continue
                
            # Extract features for chunks
            chunk_feats = extract_chunk_features(chunks, SAMPLE_RATE)
            
            # Predict embeddings and probabilities
            embeddings = feature_extractor(chunk_feats, training=False).numpy()
            probs = cnn_model(chunk_feats, training=False).numpy()
            
            # Sequence padding / truncating
            actual_len = min(len(embeddings), MAX_SEQUENCE_LENGTH)
            
            padded_emb = np.zeros((MAX_SEQUENCE_LENGTH, embedding_dim))
            padded_emb[:actual_len] = embeddings[:actual_len]
            
            padded_prob = np.zeros((MAX_SEQUENCE_LENGTH, num_classes))
            padded_prob[:actual_len] = probs[:actual_len]
            
            seq_embeddings.append(padded_emb)
            seq_probs.append(padded_prob)
            seq_lengths.append(actual_len)
            
            # Target label
            encoded_label = label_encoder.transform([label])[0]
            final_labels.append(encoded_label)
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(files)} files...")
                
        return np.array(seq_embeddings), np.array(seq_probs), np.array(seq_lengths), np.array(final_labels)
    
    X_train_seq, X_train_probs, train_lengths, y_train_seq = process_split(train_files, train_labels, "train")
    X_val_seq, X_val_probs, val_lengths, y_val_seq = process_split(val_files, val_labels, "validation")
    X_test_seq, X_test_probs, test_lengths, y_test_seq = process_split(test_files, test_labels, "test")
    
    # Save splits
    print("\nSaving sequences...")
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_train_seq.npy"), X_train_seq)
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_val_seq.npy"), X_val_seq)
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_test_seq.npy"), X_test_seq)
    
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_train_seq.npy"), y_train_seq)
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_val_seq.npy"), y_val_seq)
    np.save(os.path.join(PROCESSED_DATA_PATH, "y_test_seq.npy"), y_test_seq)
    
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_train_probs.npy"), X_train_probs)
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_val_probs.npy"), X_val_probs)
    np.save(os.path.join(PROCESSED_DATA_PATH, "X_test_probs.npy"), X_test_probs)
    
    np.save(os.path.join(PROCESSED_DATA_PATH, "train_lengths.npy"), train_lengths)
    np.save(os.path.join(PROCESSED_DATA_PATH, "val_lengths.npy"), val_lengths)
    np.save(os.path.join(PROCESSED_DATA_PATH, "test_lengths.npy"), test_lengths)
    
    # Save preprocessing metadata
    metadata = {
        "chunk_duration": CHUNK_DURATION,
        "overlap": OVERLAP,
        "sample_rate": SAMPLE_RATE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embedding_dim": feature_extractor.output_shape[-1],
        "num_classes": num_classes
    }
    with open(os.path.join(PROCESSED_DATA_PATH, "sequence_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print("Sequence generation completed!")
    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"X_val_seq shape: {X_val_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}")

if __name__ == "__main__":
    main()
