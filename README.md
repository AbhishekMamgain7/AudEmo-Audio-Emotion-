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
