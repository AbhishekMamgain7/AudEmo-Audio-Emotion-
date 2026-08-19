import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paths
BASE_DIR = "./results"
COMPARISON_DIR = "./results/comparison"

def load_metrics_or_fallback(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    print(f"Warning: Metrics file {path} not found.")
    return None

def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    
    print("=== Comparing All Models ===")
    
    # Load all metrics
    vote_metrics = load_metrics_or_fallback(f"{BASE_DIR}/cnn_baseline/majority_vote_metrics.json")
    mean_metrics = load_metrics_or_fallback(f"{BASE_DIR}/cnn_baseline/mean_prob_metrics.json")
    lstm_metrics = load_metrics_or_fallback(f"{BASE_DIR}/cnn_lstm/metrics.json")
    gru_metrics = load_metrics_or_fallback(f"{BASE_DIR}/cnn_gru/metrics.json")
    
    data = []
    
    if vote_metrics:
        data.append({
            "Model": "CNN + Voting (Baseline)",
            "Accuracy": vote_metrics["accuracy"],
            "Precision (Macro)": vote_metrics["precision_macro"],
            "Recall (Macro)": vote_metrics["recall_macro"],
            "Macro F1": vote_metrics["f1_macro"],
            "Training Time (s)": 0.0,
            "Parameters": 0
        })
        
    if mean_metrics:
        data.append({
            "Model": "CNN + Mean Prob (Baseline)",
            "Accuracy": mean_metrics["accuracy"],
            "Precision (Macro)": mean_metrics["precision_macro"],
            "Recall (Macro)": mean_metrics["recall_macro"],
            "Macro F1": mean_metrics["f1_macro"],
            "Training Time (s)": 0.0,
            "Parameters": 0
        })
        
    if lstm_metrics:
        data.append({
            "Model": "CNN + LSTM",
            "Accuracy": lstm_metrics["accuracy"],
            "Precision (Macro)": lstm_metrics["precision_macro"],
            "Recall (Macro)": lstm_metrics["recall_macro"],
            "Macro F1": lstm_metrics["f1_macro"],
            "Training Time (s)": lstm_metrics["training_time_seconds"],
            "Parameters": lstm_metrics["trainable_parameters"]
        })
        
    if gru_metrics:
        data.append({
            "Model": "CNN + GRU",
            "Accuracy": gru_metrics["accuracy"],
            "Precision (Macro)": gru_metrics["precision_macro"],
            "Recall (Macro)": gru_metrics["recall_macro"],
            "Macro F1": gru_metrics["f1_macro"],
            "Training Time (s)": gru_metrics["training_time_seconds"],
            "Parameters": gru_metrics["trainable_parameters"]
        })
        
    if not data:
        print("Error: No metrics available for comparison. Run training/evaluation scripts first.")
        return
        
    df = pd.DataFrame(data)
    
    # Save comparison CSV
    df.to_csv(f"{COMPARISON_DIR}/model_comparison.csv", index=False)
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Model':<30} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'Macro F1':<8}")
    print("-"*80)
    for index, row in df.iterrows():
        print(f"{row['Model']:<30} | {row['Accuracy']:.4f}   | {row['Precision (Macro)']:.4f}    | {row['Recall (Macro)']:.4f}  | {row['Macro F1']:.4f}")
    print("="*80 + "\n")
    
    print("Model parameters and training times:")
    print("-"*80)
    for index, row in df.iterrows():
        if "Baseline" in row['Model']:
            print(f"{row['Model']:<30} | Params: N/A | Training Time: N/A")
        else:
            print(f"{row['Model']:<30} | Params: {row['Parameters']:,} | Training Time: {row['Training Time (s)']:.2f}s")
    print("="*80 + "\n")
    
    # Plot performance comparison
    df_melt = pd.melt(df, id_vars=['Model'], value_vars=['Accuracy', 'Macro F1'], var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=df_melt, palette='muted')
    plt.ylim(0.8, 1.02)  # Adjust to highlight small differences in the 80%-100% range
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Architecture')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{COMPARISON_DIR}/model_performance_comparison.png")
    plt.close()
    
    print(f"Comparison report saved to {COMPARISON_DIR}/")

if __name__ == "__main__":
    main()
