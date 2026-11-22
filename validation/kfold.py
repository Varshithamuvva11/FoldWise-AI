"""
FoldWise AI - Standard K-Fold Cross-Validation Engine
Evaluates machine learning models with configurable k, optional shuffling,
fold allocations for visual rendering, and zero-leakage pipeline fitting.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from preprocessing.processor import build_leak_free_pipeline
from evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    aggregate_fold_metrics
)

def run_kfold_cv(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    n_splits=5,
    shuffle=True,
    random_state=42,
    task_type='classification',
    is_multiclass=False,
    scaling='standard',
    imputation='mean'
):
    """
    Execute standard K-Fold Cross-Validation.
    
    Args:
        model: Base Scikit-learn estimator
        X (pd.DataFrame): Predictor feature matrix
        y (pd.Series/array): Target labels or values
        numerical_cols (list): Numeric feature column names
        categorical_cols (list): Categorical feature column names
        n_splits (int): Number of folds (default: 5)
        shuffle (bool): Whether to shuffle dataset before splitting
        random_state (int): Pseudorandom seed
        task_type (str): 'classification' or 'regression'
        is_multiclass (bool): Whether classification has >2 classes
        scaling (str): 'standard', 'minmax', or 'none'
        imputation (str): 'mean', 'median', or 'most_frequent'
        
    Returns:
        dict: Complete evaluation report including fold-wise scores,
              summary statistics, fold allocations, and execution runtime.
    """
    start_total_time = time.time()
    
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    fold_metrics = []
    fold_allocations = []
    class_distributions = []
    
    # Store predictions for accumulated confusion matrix
    all_y_true = []
    all_y_pred = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_start = time.time()
        
        # Split data without leakage
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Build fresh pipeline for each fold
        model_clone = clone(model)
        pipeline = build_leak_free_pipeline(
            model=model_clone,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            scaling=scaling,
            imputation_strategy=imputation
        )
        
        # Fit on training fold ONLY
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation fold
        y_pred = pipeline.predict(X_val_fold)
        
        fold_duration = time.time() - fold_start
        
        # Compute metrics based on task type
        if task_type == 'classification':
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_val_fold)
                except Exception:
                    y_proba = None
                    
            m = calculate_classification_metrics(y_val_fold, y_pred, y_proba=y_proba, is_multiclass=is_multiclass)
            m['fold'] = fold_idx + 1
            m['duration'] = round(fold_duration, 4)
            fold_metrics.append(m)
            
            all_y_true.extend(y_val_fold.tolist())
            all_y_pred.extend(y_pred.tolist())
            
            # Record class distribution in this validation fold
            val_dist = pd.Series(y_val_fold).value_counts().to_dict()
            class_distributions.append({
                'fold': fold_idx + 1,
                'distribution': {str(k): int(v) for k, v in val_dist.items()}
            })
        else:
            m = calculate_regression_metrics(y_val_fold, y_pred)
            m['fold'] = fold_idx + 1
            m['duration'] = round(fold_duration, 4)
            fold_metrics.append(m)
            
        # Record fold allocations for visualization
        fold_allocations.append({
            'fold': fold_idx + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'val_indices_preview': val_idx[:10].tolist()
        })
        
    total_execution_time = time.time() - start_total_time
    summary = aggregate_fold_metrics(fold_metrics, task_type=task_type)
    
    # Global confusion matrix for classification
    if task_type == 'classification' and all_y_true:
        from sklearn.metrics import confusion_matrix
        summary['accumulated_confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred).tolist()
        
    return {
        'method': 'K-Fold',
        'algorithm': model.__class__.__name__,
        'n_splits': n_splits,
        'shuffle': shuffle,
        'random_state': random_state,
        'task_type': task_type,
        'fold_metrics': fold_metrics,
        'summary': summary,
        'fold_allocations': fold_allocations,
        'class_distributions': class_distributions,
        'execution_time': round(total_execution_time, 4)
    }
