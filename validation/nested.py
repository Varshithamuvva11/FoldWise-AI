"""
FoldWise AI - Genuine Nested Cross-Validation Engine
Implements strict double-loop cross-validation to decouple model selection
(hyperparameter tuning) from generalization performance estimation,
eliminating optimistic selection bias.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.base import clone
from preprocessing.processor import build_leak_free_pipeline
from evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    aggregate_fold_metrics
)
from models.model_config import get_param_grid

def run_nested_cv(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    param_grid=None,
    outer_splits=5,
    inner_splits=3,
    shuffle=True,
    random_state=42,
    task_type='classification',
    is_multiclass=False,
    scaling='standard',
    imputation='mean'
):
    """
    Execute genuine Nested Cross-Validation.
    
    Architecture:
    - Outer Loop (k_outer): Evaluates true generalization performance.
    - Inner Loop (k_inner): Runs GridSearchCV strictly inside the outer training fold.
    - Test Leakage Protection: Outer test fold is never touched during tuning.
    
    Args:
        model: Base Scikit-learn estimator
        X (pd.DataFrame): Features
        y (pd.Series/array): Target
        numerical_cols (list): Numeric column names
        categorical_cols (list): Categorical column names
        param_grid (dict, optional): Parameter search space. Defaults to model_config.
        outer_splits (int): Number of outer folds (generalization evaluation)
        inner_splits (int): Number of inner folds (hyperparameter optimization)
        shuffle (bool): Whether to shuffle partitions
        random_state (int): Seed
        task_type (str): 'classification' or 'regression'
        is_multiclass (bool): Whether classification target has >2 classes
        scaling (str): 'standard', 'minmax', or 'none'
        imputation (str): 'mean', 'median', or 'most_frequent'
        
    Returns:
        dict: Complete nested evaluation report with per-fold best parameters,
              inner scores, outer generalization scores, and selection bias analysis.
    """
    start_total_time = time.time()
    algorithm_name = model.__class__.__name__
    
    # Resolve parameter grid if not provided
    if not param_grid:
        param_grid = get_param_grid(algorithm_name)
        if not param_grid:
            # Fallback trivial grid if no grid defined
            param_grid = {}
            
    is_class = (task_type == 'classification')
    scoring_metric = 'accuracy' if is_class else 'neg_root_mean_squared_error'
    
    # Outer splitter
    if is_class:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    else:
        outer_cv = KFold(n_splits=outer_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
        
    outer_fold_metrics = []
    outer_allocations = []
    best_params_per_fold = []
    inner_best_scores = []
    
    all_y_true = []
    all_y_pred = []
    
    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        fold_start = time.time()
        
        # Outer split
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner splitter: operates exclusively on outer training data
        inner_seed = (random_state + outer_idx * 17) if random_state is not None else None
        if is_class:
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=shuffle, random_state=inner_seed if shuffle else None)
        else:
            inner_cv = KFold(n_splits=inner_splits, shuffle=shuffle, random_state=inner_seed if shuffle else None)
            
        # Build base leak-free pipeline for grid search
        base_pipeline = build_leak_free_pipeline(
            model=clone(model),
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            scaling=scaling,
            imputation_strategy=imputation
        )
        
        # Execute GridSearch strictly inside X_train_outer
        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring_metric,
            n_jobs=1,
            refit=True
        )
        grid_search.fit(X_train_outer, y_train_outer)
        
        # Extract best tuning results
        raw_best_params = grid_search.best_params_
        # Clean parameter names (strip 'model__' prefix for human readability)
        clean_best_params = {k.replace('model__', ''): v for k, v in raw_best_params.items()}
        best_params_per_fold.append(clean_best_params)
        
        inner_score = float(grid_search.best_score_)
        if not is_class:
            inner_score = float(-inner_score)  # convert back from neg_rmse to positive rmse
        inner_best_scores.append(round(inner_score, 4))
        
        # Evaluate the tuned pipeline on the completely held-out outer test fold
        best_pipeline = grid_search.best_estimator_
        y_pred_outer = best_pipeline.predict(X_test_outer)
        
        fold_duration = time.time() - fold_start
        
        if is_class:
            y_proba_outer = None
            if hasattr(best_pipeline, "predict_proba"):
                try:
                    y_proba_outer = best_pipeline.predict_proba(X_test_outer)
                except Exception:
                    y_proba_outer = None
                    
            m = calculate_classification_metrics(y_test_outer, y_pred_outer, y_proba=y_proba_outer, is_multiclass=is_multiclass)
            m['fold'] = outer_idx + 1
            m['duration'] = round(fold_duration, 4)
            m['best_params'] = clean_best_params
            m['inner_score'] = round(inner_score, 4)
            outer_fold_metrics.append(m)
            
            all_y_true.extend(y_test_outer.tolist())
            all_y_pred.extend(y_pred_outer.tolist())
        else:
            m = calculate_regression_metrics(y_test_outer, y_pred_outer)
            m['fold'] = outer_idx + 1
            m['duration'] = round(fold_duration, 4)
            m['best_params'] = clean_best_params
            m['inner_score'] = round(inner_score, 4)
            outer_fold_metrics.append(m)
            
        outer_allocations.append({
            'fold': outer_idx + 1,
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
            'test_indices_preview': test_idx[:10].tolist()
        })
        
    total_execution_time = time.time() - start_total_time
    summary = aggregate_fold_metrics(outer_fold_metrics, task_type=task_type)
    
    if is_class and all_y_true:
        from sklearn.metrics import confusion_matrix
        summary['accumulated_confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred).tolist()
        
    # Analyze inner tuning optimism vs outer generalization
    mean_inner = round(float(np.mean(inner_best_scores)), 4)
    std_inner = round(float(np.std(inner_best_scores)), 4)
    mean_outer = summary['mean_score']
    
    # Model selection bias calculation
    if is_class:
        optimistic_bias = round(mean_inner - mean_outer, 4)
    else:
        # For RMSE, lower is better, so inner < outer indicates optimistic underestimation of error
        optimistic_bias = round(mean_outer - mean_inner, 4)
        
    tuning_bias_analysis = {
        'mean_inner_score': mean_inner,
        'std_inner_score': std_inner,
        'mean_outer_generalization_score': mean_outer,
        'std_outer_score': summary['std_score'],
        'optimistic_bias': optimistic_bias,
        'explanation': (
            "Inner tuning scores represent parameters optimized to maximize validation folds. "
            "Outer generalization scores represent performance on strictly unseen data. "
            f"The observed difference of {optimistic_bias} reflects model selection bias."
        )
    }
    
    return {
        'method': 'Nested Cross-Validation',
        'algorithm': algorithm_name,
        'outer_splits': outer_splits,
        'inner_splits': inner_splits,
        'param_grid': {k.replace('model__', ''): v for k, v in param_grid.items()},
        'task_type': task_type,
        'best_params_per_fold': best_params_per_fold,
        'inner_best_scores': inner_best_scores,
        'fold_metrics': outer_fold_metrics,
        'summary': summary,
        'fold_allocations': outer_allocations,
        'tuning_bias_analysis': tuning_bias_analysis,
        'execution_time': round(total_execution_time, 4)
    }

def compare_nested_vs_non_nested(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    param_grid=None,
    outer_splits=5,
    inner_splits=3,
    shuffle=True,
    random_state=42,
    task_type='classification',
    is_multiclass=False,
    scaling='standard',
    imputation='mean'
):
    """
    Empirically compare genuine Nested Cross-Validation against standard Non-Nested CV
    using actual experimental execution.
    
    Demonstrates that non-nested GridSearchCV yields an overly optimistic estimate
    because the reported score was used for hyperparameter selection.
    
    Returns:
        dict: Comprehensive comparison report with bias measurements.
    """
    algorithm_name = model.__class__.__name__
    if not param_grid:
        param_grid = get_param_grid(algorithm_name)
        
    is_class = (task_type == 'classification')
    scoring_metric = 'accuracy' if is_class else 'neg_root_mean_squared_error'
    
    # 1. Execute genuine Nested CV
    nested_res = run_nested_cv(
        model=model,
        X=X,
        y=y,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        param_grid=param_grid,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        shuffle=shuffle,
        random_state=random_state,
        task_type=task_type,
        is_multiclass=is_multiclass,
        scaling=scaling,
        imputation=imputation
    )
    
    # 2. Execute standard Non-Nested CV (GridSearchCV across full dataset)
    start_non_nested = time.time()
    cv_splitter = StratifiedKFold(n_splits=outer_splits, shuffle=shuffle, random_state=random_state if shuffle else None) if is_class \
                  else KFold(n_splits=outer_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
                  
    base_pipeline = build_leak_free_pipeline(
        model=clone(model),
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        scaling=scaling,
        imputation_strategy=imputation
    )
    
    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring_metric,
        n_jobs=1,
        refit=True
    )
    grid.fit(X, y)
    non_nested_duration = time.time() - start_non_nested
    
    raw_non_nested_score = float(grid.best_score_)
    non_nested_score = raw_non_nested_score if is_class else float(-raw_non_nested_score)
    non_nested_best_params = {k.replace('model__', ''): v for k, v in grid.best_params_.items()}
    
    nested_score = nested_res['summary']['mean_score']
    
    if is_class:
        selection_bias = round(non_nested_score - nested_score, 4)
    else:
        selection_bias = round(nested_score - non_nested_score, 4)
        
    return {
        'nested': nested_res,
        'non_nested': {
            'best_score': round(non_nested_score, 4),
            'best_params': non_nested_best_params,
            'execution_time': round(non_nested_duration, 4)
        },
        'comparison': {
            'non_nested_score': round(non_nested_score, 4),
            'nested_generalization_score': round(nested_score, 4),
            'selection_bias': selection_bias,
            'is_optimistically_biased': (selection_bias > 0)
        }
    }
