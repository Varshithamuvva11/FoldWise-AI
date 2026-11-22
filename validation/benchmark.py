"""
FoldWise AI - Multi-Strategy Cross-Validation Benchmarking Engine
Directly benchmarks multiple cross-validation strategies on the same dataset and algorithm
to demonstrate empirically how strategy selection impacts reported performance, variance, and selection bias.
"""

import time
import numpy as np

from dataset.loader import load_dataset
from preprocessing.processor import prepare_dataset_for_modeling
from models.classifiers import get_classifier
from models.regressors import get_regressor
from validation.kfold import run_kfold_cv
from validation.stratified import run_stratified_kfold_cv
from validation.nested import run_nested_cv
from validation.timeseries import run_timeseries_cv

def benchmark_validation_strategies(
    dataset_name='breast_cancer',
    algorithm=None,
    target_column=None,
    n_splits=5,
    random_state=42
):
    """
    Execute compatible cross-validation strategies on the same model and data,
    contrasting reported performance and fold variance.
    
    Args:
        dataset_name (str): Dataset key
        algorithm (str, optional): Algorithm name
        target_column (str, optional): Target column override
        n_splits (int): Number of folds
        random_state (int): Reproducibility seed
        
    Returns:
        dict: Empirical strategy comparison scorecard and educational takeaways.
    """
    start_time = time.time()
    
    ds_dict = load_dataset(dataset_name, target_column=target_column)
    prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_column)
    
    X = prepared['X']
    y = prepared['y']
    task_type = prepared['task_type']
    num_cols = prepared['numerical_cols']
    cat_cols = prepared['categorical_cols']
    is_multiclass = ds_dict.get('is_multiclass', False)
    
    if not algorithm:
        algorithm = 'LogisticRegression' if task_type == 'classification' else 'RandomForestRegressor'
        
    if task_type == 'classification':
        model = get_classifier(algorithm, random_state=random_state)
    else:
        model = get_regressor(algorithm, random_state=random_state)
        
    strategy_results = {}
    
    if task_type == 'classification':
        # 1. Stratified K-Fold
        strat_res = run_stratified_kfold_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            n_splits=n_splits, shuffle=True, random_state=random_state,
            is_multiclass=is_multiclass
        )
        strategy_results['Stratified K-Fold'] = {
            'strategy': 'Stratified K-Fold',
            'summary': strat_res['summary'],
            'execution_time': strat_res['execution_time'],
            'methodological_status': 'Recommended Gold Standard',
            'badge': 'success'
        }
        
        # 2. Standard K-Fold
        kfold_res = run_kfold_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            n_splits=n_splits, shuffle=True, random_state=random_state,
            task_type='classification', is_multiclass=is_multiclass
        )
        strategy_results['Standard K-Fold'] = {
            'strategy': 'Standard K-Fold',
            'summary': kfold_res['summary'],
            'execution_time': kfold_res['execution_time'],
            'methodological_status': 'Prone to Fold Class Variance',
            'badge': 'warning'
        }
        
        # 3. Nested Cross-Validation (3 outer x 3 inner for responsive benchmark)
        nested_res = run_nested_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            outer_splits=min(n_splits, 4), inner_splits=3,
            shuffle=True, random_state=random_state,
            task_type='classification', is_multiclass=is_multiclass
        )
        strategy_results['Nested Cross-Validation'] = {
            'strategy': 'Nested Cross-Validation',
            'summary': nested_res['summary'],
            'execution_time': nested_res['execution_time'],
            'methodological_status': 'Unbiased Generalization (GridSearchCV)',
            'badge': 'info',
            'tuning_bias': nested_res.get('tuning_bias_analysis', {})
        }
        
    elif task_type == 'time_series':
        # 1. Time-Series CV
        ts_res = run_timeseries_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            n_splits=n_splits, gap=0,
            date_labels=prepared.get('date_labels', None)
        )
        strategy_results['Time-Series Cross-Validation'] = {
            'strategy': 'Time-Series Cross-Validation',
            'summary': ts_res['summary'],
            'execution_time': ts_res['execution_time'],
            'methodological_status': 'Strict Chronological Causality',
            'badge': 'success'
        }
        
        # 2. Naive Random K-Fold (to demonstrate lookahead leakage)
        naive_res = run_kfold_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            n_splits=n_splits, shuffle=True, random_state=random_state,
            task_type='regression'
        )
        strategy_results['Random K-Fold (Flawed)'] = {
            'strategy': 'Random K-Fold (Flawed)',
            'summary': naive_res['summary'],
            'execution_time': naive_res['execution_time'],
            'methodological_status': 'Artificially Optimistic (Future Leakage)',
            'badge': 'danger'
        }
        
    else: # Regression
        # 1. Standard K-Fold
        kfold_res = run_kfold_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            n_splits=n_splits, shuffle=True, random_state=random_state,
            task_type='regression'
        )
        strategy_results['Standard K-Fold'] = {
            'strategy': 'Standard K-Fold',
            'summary': kfold_res['summary'],
            'execution_time': kfold_res['execution_time'],
            'methodological_status': 'Recommended for Continuous Regression',
            'badge': 'success'
        }
        
        # 2. Nested Cross-Validation
        nested_res = run_nested_cv(
            model=model, X=X, y=y,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            outer_splits=min(n_splits, 4), inner_splits=3,
            shuffle=True, random_state=random_state,
            task_type='regression'
        )
        strategy_results['Nested Cross-Validation'] = {
            'strategy': 'Nested Cross-Validation',
            'summary': nested_res['summary'],
            'execution_time': nested_res['execution_time'],
            'methodological_status': 'Unbiased Generalization (GridSearchCV)',
            'badge': 'info',
            'tuning_bias': nested_res.get('tuning_bias_analysis', {})
        }
        
    total_time = round(time.time() - start_time, 4)
    
    # Synthesize empirical takeaway
    takeaways = []
    if 'Stratified K-Fold' in strategy_results and 'Standard K-Fold' in strategy_results:
        s_std = strategy_results['Stratified K-Fold']['summary']['std_score']
        k_std = strategy_results['Standard K-Fold']['summary']['std_score']
        if s_std <= k_std:
            takeaways.append(
                f"Stratified K-Fold reduced evaluation variance (\u03C3 = {s_std} vs \u03C3 = {k_std}) "
                "by preserving class distributions across folds."
            )
        else:
            takeaways.append("Stratified K-Fold maintained class proportions in all test partitions.")
            
    if 'Nested Cross-Validation' in strategy_results:
        tb = strategy_results['Nested Cross-Validation'].get('tuning_bias', {})
        bias = tb.get('optimistic_bias', 0.0)
        if bias > 0:
            takeaways.append(
                f"Non-nested hyperparameter selection suffered from an optimistic selection bias of +{bias}. "
                "Nested CV provides the true generalization estimate."
            )
            
    if 'Time-Series Cross-Validation' in strategy_results and 'Random K-Fold (Flawed)' in strategy_results:
        ts_rmse = strategy_results['Time-Series Cross-Validation']['summary']['mean_score']
        kf_rmse = strategy_results['Random K-Fold (Flawed)']['summary']['mean_score']
        leakage_gap = round(ts_rmse - kf_rmse, 4)
        takeaways.append(
            f"Random K-Fold reported an artificially low RMSE of {kf_rmse} compared to {ts_rmse} for Time-Series CV "
            f"(leakage depression: {leakage_gap}). Future observations leaked into the training folds."
        )

    return {
        'dataset_name': dataset_name,
        'algorithm': algorithm,
        'task_type': task_type,
        'n_splits': n_splits,
        'strategies': strategy_results,
        'takeaways': takeaways,
        'total_execution_time': total_time
    }
