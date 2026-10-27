import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json

from data_loader import FLDataset
from model import MLPModel
from evaluation import sweep_thresholds
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds) * 100,
        'loss': total_loss / max(num_batches, 1),
        'precision_binary': precision_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
        'recall_binary': recall_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
        'f1_binary': f1_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100,
        'roc_auc': roc_auc_score(all_targets, all_probs) * 100,
        'val_targets': all_targets,
        'val_probs': all_probs
    }
    return metrics

def run_centralized_baseline():
    print("=" * 80)
    print("  CENTRALIZED BASELINE: CIC-IDS-2017")
    print("  Finding the F1 mathematical ceiling without FedAvg interference.")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = FLDataset("data/cic", None, dataset_type='cicids2017', 
                        model_type='mlp', val_ratio=0.1, seed=42, subsample_size=500000)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(dataset.X_train_raw, dtype=torch.float32),
                      torch.tensor(dataset.y_train, dtype=torch.long)),
        batch_size=256, shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.tensor(dataset.X_val_raw, dtype=torch.float32),
                      torch.tensor(dataset.y_val, dtype=torch.long)),
        batch_size=512, shuffle=False
    )
    
    test_loader = DataLoader(
        TensorDataset(torch.tensor(dataset.X_test_original, dtype=torch.float32),
                      torch.tensor(dataset.y_test, dtype=torch.long)),
        batch_size=512, shuffle=False
    )
    
    model = MLPModel(input_dim=dataset.input_dim).to(device)
    
    # Global class weights for loss
    class_weights = getattr(dataset, 'global_class_weights', None)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) # AdamW standard
    
    EPOCHS = 15
    best_auc = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.2f}% | Val F1 (@0.5): {val_metrics['f1_binary']:.2f}%")
              
        if val_metrics['roc_auc'] > best_auc:
            best_auc = val_metrics['roc_auc']
            best_model_state = model.state_dict().copy()
            
    print("\n--- Training Complete ---")
    model.load_state_dict(best_model_state)
    val_metrics = evaluate(model, val_loader, device)
    
    # Threshold sweep on VAL
    sweep_df, best_thres, best_val_f1 = sweep_thresholds(val_metrics['val_targets'], val_metrics['val_probs'], step=0.02)
    print(f"\nBest Validation Threshold: {best_thres:.3f} -> Validation F1: {best_val_f1:.2f}%")
    
    # Apply to TEST
    test_metrics = evaluate(model, test_loader, device)
    test_preds = (test_metrics['val_probs'] >= best_thres).astype(int)
    
    final_f1 = f1_score(test_metrics['val_targets'], test_preds, pos_label=1) * 100
    final_precision = precision_score(test_metrics['val_targets'], test_preds, pos_label=1) * 100
    final_recall = recall_score(test_metrics['val_targets'], test_preds, pos_label=1) * 100
    final_accuracy = accuracy_score(test_metrics['val_targets'], test_preds) * 100
    
    print("\n" + "="*40)
    print("  CENTRALIZED TEST SET RESULTS")
    print("="*40)
    print(f"  F1_BINARY:  {final_f1:.2f}%")
    print(f"  PRECISION:  {final_precision:.2f}%")
    print(f"  RECALL:     {final_recall:.2f}%")
    print(f"  ACCURACY:   {final_accuracy:.2f}%")
    print(f"  ROC_AUC:    {test_metrics['roc_auc']:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_centralized_baseline()
