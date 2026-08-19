import os
import time
import pickle
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
UPLOAD_FOLDER = os.path.abspath('./uploads')
PROCESSED_DATA_PATH = "./processed_data"
GRU_MODEL_PATH = "./results/cnn_gru/best_gru_model.keras"
CNN_MODEL_PATH = "./best_model.keras"
SAMPLE_RATE = 16000

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models and encoder
cnn_extractor = None
cnn_model = None
gru_model = None
label_encoder = None
metadata = None

def load_models():
    global cnn_extractor, cnn_model, gru_model, label_encoder, metadata
    print("=== Loading models and metadata... ===")
    
    # Load metadata and label encoder
    with open(os.path.join(PROCESSED_DATA_PATH, "sequence_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    with open(os.path.join(PROCESSED_DATA_PATH, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
        
    # Load CNN
    try:
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        h5_path = CNN_MODEL_PATH.replace('.keras', '.h5')
        if not os.path.exists(h5_path):
            import shutil
            shutil.copy(CNN_MODEL_PATH, h5_path)
        model = tf.keras.models.load_model(h5_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model = model

    # Find penultimate layer for extractor
    dense_layer = None
    for layer in reversed(model.layers[:-1]):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 128:
            dense_layer = layer
            break
    if dense_layer is None:
        dense_layer = model.layers[-3]
    cnn_extractor = tf.keras.Model(inputs=model.inputs, outputs=dense_layer.output)
    
    # Load GRU
    try:
        gru = tf.keras.models.load_model(GRU_MODEL_PATH)
    except Exception as e:
        h5_path = GRU_MODEL_PATH.replace('.keras', '.h5')
        if not os.path.exists(h5_path):
            import shutil
            shutil.copy(GRU_MODEL_PATH, h5_path)
        gru = tf.keras.models.load_model(h5_path)
    gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    gru_model = gru
    
    print("=== Models loaded successfully! ===")

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    start_time = time.time()
    
    # Save the file
    filename = f"uploaded_{int(time.time())}_{audio_file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)
    
    try:
        # Load parameters from metadata
        chunk_duration = metadata.get("chunk_duration", 1.0)
        overlap = metadata.get("overlap", 0.5)
        max_sequence_length = metadata.get("max_sequence_length", 10)
        embedding_dim = metadata.get("embedding_dim", 128)
        
        # 1. Preprocessing (Load audio)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        # Compute downsampled waveform envelope (amplitude) for rendering
        step = max(1, len(y) // 500)
        waveform = [float(np.max(np.abs(y[i:i+step]))) for i in range(0, len(y), step)]
        # Normalize waveform envelope
        max_wave = np.max(waveform) if len(waveform) > 0 else 1.0
        if max_wave > 1e-5:
            waveform = [w / max_wave for w in waveform]
        waveform = waveform[:500]
        
        # 2. Chunking
        chunk_size = int(chunk_duration * sr)
        hop_size = int((chunk_duration - overlap) * sr)
        if hop_size <= 0:
            hop_size = chunk_size
            
        chunks = []
        chunk_start_indices = []
        start = 0
        while start < len(y):
            end = start + chunk_size
            chunk = y[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            chunks.append(chunk)
            chunk_start_indices.append(start)
            start += hop_size
            
        total_chunks = len(chunks)
        if total_chunks == 0:
            return jsonify({"error": "Audio file is too short to extract chunks"}), 400
            
        # 3. Feature & Spectrogram extraction
        features_list = []
        spectrograms_list = []
        for chunk in chunks:
            spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Pad/truncate spectrogram to (128, 128)
            if log_spectrogram.shape[1] < 128:
                pad_width = 128 - log_spectrogram.shape[1]
                log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
            else:
                log_spectrogram = log_spectrogram[:, :128]
                
            features_list.append(log_spectrogram)
            
            # Downsample to 32x32 for lightweight transmission
            downsampled = log_spectrogram[::4, ::4]
            # Normalize downsampled spectrogram to [0, 1] range for frontend display
            min_val = np.min(downsampled)
            max_val = np.max(downsampled)
            if max_val - min_val > 1e-5:
                norm_downsampled = (downsampled - min_val) / (max_val - min_val)
            else:
                norm_downsampled = np.zeros_like(downsampled)
            spectrograms_list.append(norm_downsampled.tolist())
            
        chunk_feats = np.array(features_list)
        
        # 4. CNN Embedding and Probability Extraction
        embeddings = cnn_extractor(chunk_feats, training=False).numpy()
        cnn_probs = cnn_model(chunk_feats, training=False).numpy()
        
        # 5. GRU Sequence Inference
        actual_len = min(len(embeddings), max_sequence_length)
        padded_emb = np.zeros((1, max_sequence_length, embedding_dim))
        padded_emb[0, :actual_len] = embeddings[:actual_len]
        
        gru_probs = gru_model(padded_emb, training=False).numpy()[0]
        pred_class_idx = np.argmax(gru_probs)
        overall_emotion = label_encoder.inverse_transform([pred_class_idx])[0]
        overall_confidence = float(gru_probs[pred_class_idx])
        
        # 6. Create Segment-Level details
        segments = []
        for i in range(total_chunks):
            chunk_start_sec = float(chunk_start_indices[i]) / sr
            chunk_end_sec = chunk_start_sec + chunk_duration
            
            chunk_prob = cnn_probs[i]
            chunk_class_idx = np.argmax(chunk_prob)
            chunk_emotion = label_encoder.inverse_transform([chunk_class_idx])[0]
            chunk_conf = float(chunk_prob[chunk_class_idx])
            
            segments.append({
                "chunk_id": i + 1,
                "start": chunk_start_sec,
                "end": chunk_end_sec,
                "emotion": chunk_emotion,
                "confidence": chunk_conf,
                "spectrogram": spectrograms_list[i]
            })
            
        prediction_time = time.time() - start_time
        
        # Response Dictionary
        response = {
            "overall_emotion": overall_emotion.capitalize(),
            "overall_confidence": overall_confidence,
            "model": "CNN + GRU",
            "duration": duration,
            "chunk_duration": chunk_duration,
            "overlap": overlap,
            "waveform": waveform,
            "segments": segments,
            "audio_url": f"/uploads/{filename}",
            "technical_details": {
                "model_name": "CNN + GRU",
                "chunk_size_seconds": chunk_duration,
                "overlap_seconds": overlap,
                "total_chunks": total_chunks,
                "sampling_rate_hz": sr,
                "feature_dim": embedding_dim,
                "prediction_time_seconds": round(prediction_time, 3)
            }
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to analyze audio: {str(e)}"}), 500

if __name__ == '__main__':
    load_models()
    app.run(host='127.0.0.1', port=5000, debug=True)
