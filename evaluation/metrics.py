"""
FoldWise AI - Evaluation Metrics Engine
Computes standard classification and regression performance metrics,
fold-wise distributions, confusion matrices, and execution statistics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def calculate_classification_metrics(y_true, y_pred, y_proba=None, is_multiclass=False):
    """
    Calculate comprehensive evaluation metrics for classification tasks.
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted class labels
        y_proba (array-like, optional): Predicted class probabilities
        is_multiclass (bool): Whether task is multiclass or binary
        
    Returns:
        dict: Evaluated metrics including accuracy, precision, recall, f1, and confusion matrix
    """
    average_strategy = 'weighted' if is_multiclass else 'binary'
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average_strategy, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average_strategy, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average_strategy, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm
    }
    
    # Calculate ROC-AUC when probability outputs are available
    if y_proba is not None:
        try:
            if is_multiclass:
                auc = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
            else:
                # Binary classification: extract proba for positive class
                prob_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                auc = float(roc_auc_score(y_true, prob_pos))
            metrics['roc_auc'] = round(auc, 4)
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
        
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for continuous regression tasks.
    
    Args:
        y_true (array-like): Ground truth continuous values
        y_pred (array-like): Predicted continuous values
        
    Returns:
        dict: MAE, MSE, RMSE, and R2 score
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Safe MAPE calculation (avoiding zero division)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    non_zero_mask = y_true_arr != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100)
    else:
        mape = 0.0
        
    return {
        'mae': round(mae, 4),
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'r2_score': round(r2, 4),
        'mape': round(mape, 2)
    }

def aggregate_fold_metrics(fold_metrics_list, task_type='classification'):
    """
    Compute mean and standard deviation across fold metrics.
    
    Args:
        fold_metrics_list (list of dicts): Metric dictionary for each fold
        task_type (str): 'classification' or 'regression'
        
    Returns:
        dict: Aggregated summary statistics
    """
    if not fold_metrics_list:
        return {}
        
    primary_metric_key = 'accuracy' if task_type == 'classification' else 'rmse'
    scores = [m.get(primary_metric_key, 0.0) for m in fold_metrics_list]
    
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    
    summary = {
        'primary_metric': primary_metric_key,
        'mean_score': round(mean_score, 4),
        'std_score': round(std_score, 4),
        'scores_per_fold': [round(s, 4) for s in scores],
        'min_score': round(float(np.min(scores)), 4),
        'max_score': round(float(np.max(scores)), 4)
    }
    
    # Average all other metric keys that are numeric
    all_keys = fold_metrics_list[0].keys()
    for k in all_keys:
        if k in ['confusion_matrix', 'best_params', 'duration', 'fold']:
            continue
        first_val = fold_metrics_list[0].get(k)
        if isinstance(first_val, (int, float, np.number)):
            vals = [m[k] for m in fold_metrics_list if m.get(k) is not None and isinstance(m[k], (int, float, np.number))]
            if vals:
                summary[f'mean_{k}'] = round(float(np.mean(vals)), 4)
                summary[f'std_{k}'] = round(float(np.std(vals)), 4)
            
    return summary
