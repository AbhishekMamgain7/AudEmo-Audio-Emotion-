# **AudEmo (Audio Emotion)**  

## **Introduction**  
AudEmo is a deep learning-based project designed to detect emotions from speech signals. The project processes audio data into log-mel spectrograms and uses a Convolutional Neural Network (CNN) to classify emotions such as:  
- **Anger**  
- **Happiness**  
- **Sadness**  
- **Fear**  
- **Disgust**  
- **Neutral**  
- **Pleasant Surprise**  

With its high classification accuracy, AudEmo is ideal for real-world applications like:  
- Enhancing customer support systems.  
- Supporting mental health analysis.  
- Improving user experience in human-computer interaction.

---

## **Features**  
- Converts audio (.wav) files into log-mel spectrograms.  
- Splits dataset into training, validation, and test sets for robust evaluation.  
- Trains a CNN model with real-time saving of the best model.  
- Evaluates model performance using metrics like accuracy, precision, recall, and F1-score.  
- Predicts emotions from unseen audio data.  

---

## **Repository Structure**  

```plaintext
AudEmo/
├── processed_data/         # Folder for processed features and labels
├── test_data/              # Folder for storing unseen test audio files
├── results/                # Model predictions and evaluation reports
├── 01_data_preparation.py  # Script for dataset preparation
├── 02_data_splitting.py    # Script for splitting data into train/val/test
├── 03_model_training.py    # Script for training the model
├── test_prediction.py      # Script for testing and evaluating unseen data
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## **Installation**

### Prerequisites
1. Python 3.8 or higher.
2. Required Python libraries:
   - TensorFlow
   - NumPy
   - Librosa
   - Scikit-learn
   - Matplotlib
3. [Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) Dataset.
### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/AudEmo.git
   cd AudEmo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is placed in the correct path (e.g., `./TESS Toronto emotional speech set data`).

---

## **Usage**

### 1. **Data Preparation**
Prepare the dataset by extracting log-mel spectrogram features:
```bash
python 01_data_preparation.py
```

### 2. **Data Splitting**
Split the data into training, validation, and test sets:
```bash
python 02_data_splitting.py
```

### 3. **Model Training**
Train the model on the prepared dataset:
```bash
python 03_model_training.py
```

### 4. **Testing**
Run predictions on the test dataset:
```bash
python test_prediction.py
```

---

## **Results**

- **Model Accuracy:** 99%

**Macro Averages**
- **Precision:** 99.31%
- **Recall:** 99.29%
- **F1-Score:** 99.29%
---

## **Future Work**

1. Real-time emotion detection.
2. Support for multilingual datasets.
3. Mobile and edge device optimization.
4. Explore advanced architectures like CRNNs or Transformers.
