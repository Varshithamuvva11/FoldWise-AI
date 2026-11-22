"""
FoldWise AI - Stratified K-Fold Cross-Validation Engine
Preserves exact class distribution across all evaluation folds,
providing lower variance estimates for imbalanced classification tasks.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from preprocessing.processor import build_leak_free_pipeline
from evaluation.metrics import calculate_classification_metrics, aggregate_fold_metrics
from validation.kfold import run_kfold_cv

def run_stratified_kfold_cv(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    n_splits=5,
    shuffle=True,
    random_state=42,
    is_multiclass=False,
    scaling='standard',
    imputation='mean'
):
    """
    Execute Stratified K-Fold Cross-Validation.
    
    Guarantees that every fold contains approximately the exact same proportion
    of each target class as the complete dataset.
    
    Args:
        model: Base Scikit-learn classification estimator
        X (pd.DataFrame): Predictor feature matrix
        y (pd.Series/array): Discrete target class labels
        numerical_cols (list): Numeric feature column names
        categorical_cols (list): Categorical feature column names
        n_splits (int): Number of folds
        shuffle (bool): Whether to shuffle dataset before splitting
        random_state (int): Pseudorandom seed
        is_multiclass (bool): Whether target contains >2 classes
        scaling (str): 'standard', 'minmax', or 'none'
        imputation (str): 'mean', 'median', or 'most_frequent'
        
    Returns:
        dict: Complete stratified evaluation report
    """
    # Verify task is classification
    unique_classes = np.unique(y)
    if len(unique_classes) > 50:
        raise ValueError("Stratified K-Fold requires discrete class labels. For continuous targets, use standard K-Fold.")
        
    start_total_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    
    fold_metrics = []
    fold_allocations = []
    class_distributions = []
    
    all_y_true = []
    all_y_pred = []
    
    # Calculate baseline class proportions in full dataset
    full_counts = pd.Series(y).value_counts().to_dict()
    total_samples = len(y)
    baseline_proportions = {str(k): round((v / total_samples) * 100, 2) for k, v in full_counts.items()}
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        
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
        
        # Record preserved class proportions in this validation fold
        val_counts = pd.Series(y_val_fold).value_counts().to_dict()
        val_total = len(y_val_fold)
        val_proportions = {str(k): round((v / val_total) * 100, 2) for k, v in val_counts.items()}
        
        class_distributions.append({
            'fold': fold_idx + 1,
            'counts': {str(k): int(v) for k, v in val_counts.items()},
            'proportions': val_proportions
        })
        
        # Record fold allocations for visualization
        fold_allocations.append({
            'fold': fold_idx + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'val_indices_preview': val_idx[:10].tolist()
        })
        
    total_execution_time = time.time() - start_total_time
    summary = aggregate_fold_metrics(fold_metrics, task_type='classification')
    
    if all_y_true:
        summary['accumulated_confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred).tolist()
        
    return {
        'method': 'Stratified K-Fold',
        'algorithm': model.__class__.__name__,
        'n_splits': n_splits,
        'shuffle': shuffle,
        'random_state': random_state,
        'task_type': 'classification',
        'baseline_class_proportions': baseline_proportions,
        'fold_metrics': fold_metrics,
        'summary': summary,
        'fold_allocations': fold_allocations,
        'class_distributions': class_distributions,
        'execution_time': round(total_execution_time, 4)
    }

def compare_kfold_and_stratified(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    n_splits=5,
    shuffle=True,
    random_state=42,
    is_multiclass=False
):
    """
    Run K-Fold and Stratified K-Fold side-by-side on the exact same model and dataset
    to empirically demonstrate the variance and class-distribution difference.
    
    Returns:
        dict: Side-by-side comparison object with empirical variance contrast
    """
    kfold_res = run_kfold_cv(
        model=model,
        X=X,
        y=y,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        task_type='classification',
        is_multiclass=is_multiclass
    )
    
    strat_res = run_stratified_kfold_cv(
        model=model,
        X=X,
        y=y,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        is_multiclass=is_multiclass
    )
    
    kfold_std = kfold_res['summary']['std_score']
    strat_std = strat_res['summary']['std_score']
    
    return {
        'kfold': kfold_res,
        'stratified': strat_res,
        'variance_comparison': {
            'kfold_std': kfold_std,
            'stratified_std': strat_std,
            'std_difference': round(kfold_std - strat_std, 4),
            'stratified_is_more_stable': (strat_std <= kfold_std)
        }
    }
