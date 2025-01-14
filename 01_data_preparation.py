import os
import librosa
import numpy as np
import pickle

# Path to the TESS dataset
DATASET_PATH = "./TESS Toronto emotional speech set data"
OUTPUT_PATH = "./processed_data"

# Emotions in TESS dataset
EMOTIONS = {
    "angry": "angry",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "disgust": "disgust",
    "pleasant_surprise": "ps"
}

# Function to extract log-mel spectrogram
def extract_features(file_path, n_mels=128, max_length=128):
    y, sr = librosa.load(file_path, sr=16000)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    if log_spectrogram.shape[1] < max_length:
        pad_width = max_length - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_spectrogram = log_spectrogram[:, :max_length]
    return log_spectrogram

# Load data and extract features
data = []
labels = []

for dirname, _, filenames in os.walk(DATASET_PATH):
    for filename in filenames:
        if filename.endswith(".wav"):
            file_path = os.path.join(dirname, filename)
            features = extract_features(file_path)
            label = filename.split('_')[2].split('.')[0]  # Adjust for TESS filename structure
            data.append(features)
            labels.append(label.lower())

# Save processed data
os.makedirs(OUTPUT_PATH, exist_ok=True)
with open(os.path.join(OUTPUT_PATH, "features.pkl"), "wb") as f:
    pickle.dump(data, f)
with open(os.path.join(OUTPUT_PATH, "labels.pkl"), "wb") as f:
    pickle.dump(labels, f)

print("Data preparation complete!")