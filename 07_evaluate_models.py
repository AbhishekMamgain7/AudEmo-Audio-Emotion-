import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Paths
PROCESSED_DATA_PATH = "./processed_data"
RESULTS_DIR = "./results/cnn_baseline"

def get_majority_vote(probs, length):
    # Get prediction for each active chunk
    chunk_preds = [np.argmax(probs[j]) for j in range(length)]
    # Count frequencies
    counts = Counter(chunk_preds)
    # Return the most common prediction
    return counts.most_common(1)[0][0]

def get_mean_prob(probs, length):
    # Get active chunks
    active_probs = probs[:length]
    # Mean across the active chunks
    mean_probs = np.mean(active_probs, axis=0)
    return np.argmax(mean_probs)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=== Evaluating CNN Baseline Models ===")
    
    # Load test labels, prediction probabilities, and lengths
    y_test = np.load(f"{PROCESSED_DATA_PATH}/y_test_seq.npy")
    X_test_probs = np.load(f"{PROCESSED_DATA_PATH}/X_test_probs.npy")
    test_lengths = np.load(f"{PROCESSED_DATA_PATH}/test_lengths.npy")
    
    with open(f"{PROCESSED_DATA_PATH}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    print(f"Test sequences: {len(y_test)}")
    
    # 1. Baseline A: Mean Probability
    mean_prob_preds = []
    for i in range(len(y_test)):
        pred = get_mean_prob(X_test_probs[i], test_lengths[i])
        mean_prob_preds.append(pred)
    mean_prob_preds = np.array(mean_prob_preds)
    
    # Calculate Mean Probability Metrics
    acc_mean = accuracy_score(y_test, mean_prob_preds)
    p_mean, r_mean, f1_mean, _ = precision_recall_fscore_support(y_test, mean_prob_preds, average='macro')
    pw_mean, rw_mean, f1w_mean, _ = precision_recall_fscore_support(y_test, mean_prob_preds, average='weighted')
    
    mean_metrics = {
        "accuracy": acc_mean,
        "precision_macro": p_mean,
        "recall_macro": r_mean,
        "f1_macro": f1_mean,
        "precision_weighted": pw_mean,
        "recall_weighted": rw_mean,
        "f1_weighted": f1w_mean
    }
    
    with open(f"{RESULTS_DIR}/mean_prob_metrics.json", "w") as f:
        json.dump(mean_metrics, f, indent=4)
        
    # Mean Prob Classification Report
    mean_report = classification_report(y_test, mean_prob_preds, target_names=label_encoder.classes_)
    with open(f"{RESULTS_DIR}/mean_prob_classification_report.txt", "w") as f:
        f.write(mean_report)
        
    # Mean Prob Confusion Matrix
    cm_mean = confusion_matrix(y_test, mean_prob_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_mean, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('CNN Mean Probability Baseline Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/mean_prob_confusion_matrix.png")
    plt.close()
    
    # 2. Baseline B: Majority Voting
    maj_vote_preds = []
    for i in range(len(y_test)):
        pred = get_majority_vote(X_test_probs[i], test_lengths[i])
        maj_vote_preds.append(pred)
    maj_vote_preds = np.array(maj_vote_preds)
    
    # Calculate Majority Voting Metrics
    acc_vote = accuracy_score(y_test, maj_vote_preds)
    p_vote, r_vote, f1_vote, _ = precision_recall_fscore_support(y_test, maj_vote_preds, average='macro')
    pw_vote, rw_vote, f1w_vote, _ = precision_recall_fscore_support(y_test, maj_vote_preds, average='weighted')
    
    vote_metrics = {
        "accuracy": acc_vote,
        "precision_macro": p_vote,
        "recall_macro": r_vote,
        "f1_macro": f1_vote,
        "precision_weighted": pw_vote,
        "recall_weighted": rw_vote,
        "f1_weighted": f1w_vote
    }
    
    with open(f"{RESULTS_DIR}/majority_vote_metrics.json", "w") as f:
        json.dump(vote_metrics, f, indent=4)
        
    # Majority Voting Classification Report
    vote_report = classification_report(y_test, maj_vote_preds, target_names=label_encoder.classes_)
    with open(f"{RESULTS_DIR}/majority_vote_classification_report.txt", "w") as f:
        f.write(vote_report)
        
    # Majority Voting Confusion Matrix
    cm_vote = confusion_matrix(y_test, maj_vote_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_vote, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('CNN Majority Voting Baseline Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/majority_vote_confusion_matrix.png")
    plt.close()
    
    print("Baseline models evaluation completed successfully!")
    print(f"Mean Probability Accuracy: {acc_mean:.4f} | Macro F1: {f1_mean:.4f}")
    print(f"Majority Voting Accuracy:  {acc_vote:.4f} | Macro F1: {f1_vote:.4f}")

if __name__ == "__main__":
    main()
