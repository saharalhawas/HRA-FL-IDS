import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def _safe_roc_auc(targets, probs):
    """Compute ROC-AUC with fallback for degenerate cases (NaN, single-class)."""
    try:
        return roc_auc_score(targets, probs) * 100
    except (ValueError, IndexError):
        return 50.0

def get_probabilities(model, dataloader, device):
    """Get predicted probabilities and true targets from a model."""
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Clamp to prevent NaN from exploded weights (adversarial scenarios)
            outputs = torch.clamp(outputs, min=-50.0, max=50.0)
            probs = torch.softmax(outputs, dim=1)[:, 1] # P(attack)
            
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Replace NaN/Inf probabilities with neutral 0.5
    nan_mask = ~np.isfinite(all_probs)
    if nan_mask.any():
        all_probs[nan_mask] = 0.5
    
    return all_targets, all_probs

def evaluate_with_threshold(all_targets, all_probs, threshold=0.5):
    """Compute metrics using a specific decision threshold."""
    # Ensure threshold handles strict probabilities
    all_preds = (all_probs >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds) * 100,
        'precision_binary': precision_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
        'recall_binary': recall_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
        'f1_binary': f1_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
        'roc_auc': _safe_roc_auc(all_targets, all_probs),
        'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist(),
    }
    return metrics

def sweep_thresholds(all_targets, all_probs, start=0.1, stop=0.9, step=0.05):
    """Sweep a range of thresholds and return the optimal one based on F1_binary."""
    thresholds = np.arange(start, stop + step/2, step)
    results = []
    
    best_f1 = -1
    best_threshold = 0.5
    
    for t in thresholds:
        metrics = evaluate_with_threshold(all_targets, all_probs, t)
        results.append({
            'threshold': t,
            'f1_binary': metrics['f1_binary'],
            'precision_binary': metrics['precision_binary'],
            'recall_binary': metrics['recall_binary'],
            'accuracy': metrics['accuracy']
        })
        
        if metrics['f1_binary'] > best_f1:
            best_f1 = metrics['f1_binary']
            best_threshold = t
            
    df = pd.DataFrame(results)
    return df, best_threshold, best_f1

def plot_threshold_curve(df, best_threshold, best_f1, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df['threshold'], df['f1_binary'], label='F1 Score', color='#C2185B', linewidth=2)
    plt.plot(df['threshold'], df['precision_binary'], label='Precision', color='#388E3C', linestyle='--')
    plt.plot(df['threshold'], df['recall_binary'], label='Recall', color='#F57C00', linestyle='-.')
    
    plt.axvline(x=best_threshold, color='black', linestyle=':', label=f'Best Threshold ({best_threshold:.2f})')
    plt.scatter([best_threshold], [best_f1], color='black', s=50, zorder=5)
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score (%)')
    plt.title('Metrics vs. Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
