"""
FoldWise AI - Model Comparison & Intelligent Recommendation Engine
Executes multi-model cross-validation evaluations across real datasets,
ranks algorithms by performance, stability, and computational efficiency,
and produces explainable, data-driven model recommendations.
"""

import time
import uuid
import numpy as np
import pandas as pd

from dataset.loader import load_dataset
from preprocessing.processor import prepare_dataset_for_modeling
from models.classifiers import get_classifier
from models.regressors import get_regressor
from models.model_config import CLASSIFIERS, REGRESSORS
from validation.kfold import run_kfold_cv
from validation.stratified import run_stratified_kfold_cv
from validation.nested import run_nested_cv
from validation.timeseries import run_timeseries_cv
from database.db import save_experiment, save_comparison
from evaluation.visualizer import generate_radar_chart_payload

# Metric optimization directions (True: higher is better, False: lower is better)
METRIC_DIRECTIONS = {
    'accuracy': True,
    'precision': True,
    'recall': True,
    'f1_score': True,
    'roc_auc': True,
    'r2_score': True,
    'mean_score': True, # Overridden per task
    'rmse': False,
    'mae': False,
    'mse': False,
    'mape': False
}

def validate_strategy_compatibility(task_type, validation_method, dataset_name=None):
    """
    Ensure the validation strategy is methodologically sound for the given task and dataset.
    
    Args:
        task_type (str): 'classification', 'regression', or 'time_series'
        validation_method (str): 'stratified', 'kfold', 'nested', 'timeseries'
        dataset_name (str, optional): Name of dataset
        
    Raises:
        ValueError: If strategy is methodologically incompatible
    """
    method = validation_method.lower().replace('-', '_')
    
    valid_methods = ['stratified', 'kfold', 'nested', 'timeseries', 'time_series']
    if method not in valid_methods:
        raise ValueError(f"Unknown validation method '{validation_method}'. Choose from: Stratified, K-Fold, Nested, Time-Series.")
        
    if task_type == 'classification':
        if method in ['timeseries', 'time_series']:
            raise ValueError(
                "TimeSeriesSplit is methodologically invalid for standard classification tasks. "
                "Use Stratified K-Fold or Nested Cross-Validation instead."
            )
            
    elif task_type == 'regression':
        if method == 'stratified':
            raise ValueError(
                "Stratified K-Fold is methodologically invalid for continuous regression targets. "
                "Use standard K-Fold or Nested Cross-Validation instead."
            )
            
    elif task_type == 'time_series':
        if method == 'stratified':
            raise ValueError(
                "Stratified K-Fold is incompatible with continuous time-series forecasting. "
                "Use Time-Series Cross-Validation to prevent lookahead leakage."
            )

def rank_and_recommend(model_results, task_type, primary_metric=None):
    """
    Rank models based on primary evaluation metric, cross-validation variance, and execution time.
    Generates explainable, context-aware recommendation text.
    
    Args:
        model_results (dict): Dictionary mapping algorithm_name -> execution results
        task_type (str): 'classification', 'regression', or 'time_series'
        primary_metric (str, optional): Metric name to prioritize
        
    Returns:
        dict: Rankings list and detailed recommendation narrative
    """
    if not model_results:
        return {'rankings': [], 'recommendation': {}}
        
    # Determine primary metric if not specified
    if not primary_metric:
        primary_metric = 'accuracy' if task_type == 'classification' else 'rmse'
        
    higher_is_better = METRIC_DIRECTIONS.get(primary_metric, (task_type == 'classification'))
    
    scored_models = []
    
    for algo_name, res in model_results.items():
        summary = res.get('summary', {})
        
        # Primary score extraction
        if primary_metric in summary:
            score = summary[primary_metric]
        elif f"mean_{primary_metric}" in summary:
            score = summary[f"mean_{primary_metric}"]
        else:
            score = summary.get('mean_score', 0.0)
            
        std = summary.get('std_score', summary.get(f"std_{primary_metric}", 0.0))
        scores_list = summary.get('scores_per_fold', [])
        min_score = summary.get('min_score', min(scores_list) if scores_list else score)
        max_score = summary.get('max_score', max(scores_list) if scores_list else score)
        exec_time = res.get('execution_time', 0.0)
        
        scored_models.append({
            'algorithm': algo_name,
            'primary_metric': primary_metric,
            'score': round(float(score), 4),
            'std': round(float(std), 4),
            'min_score': round(float(min_score), 4),
            'max_score': round(float(max_score), 4),
            'scores_per_fold': [round(float(s), 4) for s in scores_list],
            'execution_time': round(float(exec_time), 4),
            'metrics_summary': summary
        })
        
    # Sort models by primary metric, breaking ties with standard deviation (lower std = more stable)
    if higher_is_better:
        scored_models.sort(key=lambda m: (-m['score'], m['std'], m['execution_time']))
    else:
        scored_models.sort(key=lambda m: (m['score'], m['std'], m['execution_time']))
        
    # Assign ranks and badges
    for rank_idx, m in enumerate(scored_models, start=1):
        m['rank'] = rank_idx
        if rank_idx == 1:
            m['badge'] = 'Top Recommended'
            m['badge_class'] = 'success'
        elif rank_idx == 2:
            m['badge'] = 'Strong Alternative'
            m['badge_class'] = 'info'
        elif m['execution_time'] == min(x['execution_time'] for x in scored_models) and len(scored_models) > 2:
            m['badge'] = 'Fastest Inference'
            m['badge_class'] = 'warning'
        else:
            m['badge'] = f"Rank #{rank_idx}"
            m['badge_class'] = 'secondary'

    # Synthesize explainable recommendation
    winner = scored_models[0]
    runner_up = scored_models[1] if len(scored_models) > 1 else None
    fastest = min(scored_models, key=lambda x: x['execution_time'])
    most_stable = min(scored_models, key=lambda x: x['std'])
    
    metric_label = primary_metric.upper().replace('_', ' ')
    
    explanation_parts = []
    
    # 1. Performance margin
    if runner_up:
        if higher_is_better:
            margin = round(winner['score'] - runner_up['score'], 4)
            explanation_parts.append(
                f"<strong>{winner['algorithm']}</strong> ranks #1 with a mean {metric_label} of <strong>{winner['score']}</strong>, "
                f"outperforming runner-up {runner_up['algorithm']} ({runner_up['score']}) by a margin of +{margin}."
            )
        else:
            margin = round(runner_up['score'] - winner['score'], 4)
            explanation_parts.append(
                f"<strong>{winner['algorithm']}</strong> ranks #1 with the lowest {metric_label} of <strong>{winner['score']}</strong>, "
                f"achieving a lower error than {runner_up['algorithm']} ({runner_up['score']}) by -{margin}."
            )
    else:
        explanation_parts.append(
            f"<strong>{winner['algorithm']}</strong> achieved a mean {metric_label} of <strong>{winner['score']}</strong>."
        )
        
    # 2. Stability analysis
    if most_stable['algorithm'] == winner['algorithm']:
        explanation_parts.append(
            f"Additionally, {winner['algorithm']} is the <em>most stable model across folds</em> with minimal variance "
            f"(&sigma; = {winner['std']}), indicating superior generalization and low susceptibility to fold-wise data shifts."
        )
    else:
        explanation_parts.append(
            f"In terms of cross-validation variance, {most_stable['algorithm']} demonstrated the highest consistency "
            f"(&sigma; = {most_stable['std']} vs {winner['algorithm']} &sigma; = {winner['std']}). However, {winner['algorithm']} "
            f"remains superior overall due to higher mean predictive accuracy."
        )
        
    # 3. Efficiency trade-off
    if fastest['algorithm'] != winner['algorithm'] and fastest['execution_time'] < (winner['execution_time'] * 0.5):
        explanation_parts.append(
            f"<em>Efficiency note:</em> {fastest['algorithm']} was substantially faster to evaluate "
            f"({fastest['execution_time']}s vs {winner['execution_time']}s for {winner['algorithm']}). "
            f"If deploying in ultra-low-latency or constrained resource environments, {fastest['algorithm']} offers a viable trade-off."
        )
    else:
        explanation_parts.append(
            f"{winner['algorithm']} exhibited well-balanced computational execution at {winner['execution_time']}s."
        )

    recommendation = {
        'winner_algorithm': winner['algorithm'],
        'winner_score': winner['score'],
        'winner_std': winner['std'],
        'primary_metric': primary_metric,
        'metric_label': metric_label,
        'higher_is_better': higher_is_better,
        'most_stable_algorithm': most_stable['algorithm'],
        'fastest_algorithm': fastest['algorithm'],
        'summary_headline': f"{winner['algorithm']} is the recommended model for this configuration.",
        'detailed_narrative': " ".join(explanation_parts),
        'deployment_verdict': (
            f"Deploy {winner['algorithm']} for optimal generalization on {task_type} tasks. "
            f"Expected out-of-sample {metric_label}: {winner['score']} (\u00B1 {winner['std']})."
        )
    }
    
    return {
        'rankings': scored_models,
        'recommendation': recommendation
    }

def compare_models(
    dataset_name='iris',
    algorithms=None,
    validation_method=None,
    n_splits=5,
    primary_metric=None,
    random_state=42,
    target_column=None,
    scaling='standard',
    imputation='mean',
    gap=0,
    outer_splits=5,
    inner_splits=3
):
    """
    Execute genuine multi-model cross-validation comparison on real datasets.
    
    Guarantees:
    - Zero data leakage: Pipelines fitted inside training folds only.
    - Actual Scikit-learn execution for every selected model.
    - Methodological consistency across all compared models.
    - Saves comparison experiment record to SQLite database.
    
    Args:
        dataset_name (str): Dataset key ('iris', 'breast_cancer', 'wine_quality', 'air_passengers')
        algorithms (list, optional): List of algorithm names to compare
        validation_method (str, optional): 'stratified', 'kfold', 'nested', 'timeseries'
        n_splits (int): Number of folds (default: 5)
        primary_metric (str, optional): Metric to rank by (e.g. 'accuracy', 'rmse', 'f1_score')
        random_state (int): Pseudorandom reproducibility seed
        target_column (str, optional): Override target column
        scaling (str): 'standard', 'minmax', 'none'
        imputation (str): 'mean', 'median', 'most_frequent'
        gap (int): Temporal gap for time-series splits
        outer_splits (int): Outer folds for nested CV
        inner_splits (int): Inner folds for nested CV
        
    Returns:
        dict: Complete comparison evaluation report, rankings, charts, and recommendations.
    """
    start_total_time = time.time()
    comp_id = str(uuid.uuid4())[:12]
    
    # 1. Load actual dataset
    ds_dict = load_dataset(dataset_name, target_column=target_column)
    prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_column)
    
    X = prepared['X']
    y = prepared['y']
    task_type = prepared['task_type']
    num_cols = prepared['numerical_cols']
    cat_cols = prepared['categorical_cols']
    is_multiclass = ds_dict.get('is_multiclass', False)
    
    # 2. Determine default validation method if omitted
    if not validation_method:
        if task_type == 'classification':
            validation_method = 'stratified'
        elif task_type == 'time_series':
            validation_method = 'timeseries'
        else:
            validation_method = 'kfold'
            
    # Validate method compatibility
    validate_strategy_compatibility(task_type, validation_method, dataset_name=dataset_name)
    
    # 3. Determine algorithms to evaluate
    if not algorithms or not isinstance(algorithms, list):
        if task_type == 'classification':
            algorithms = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'SVC', 'KNeighborsClassifier']
        else:
            algorithms = ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor', 'SVR']
            
    # Filter algorithms to valid registry keys
    valid_registry = CLASSIFIERS if task_type == 'classification' else REGRESSORS
    active_algorithms = [a for a in algorithms if a in valid_registry]
    if not active_algorithms:
        raise ValueError(f"No valid algorithms specified for task '{task_type}'. Available: {list(valid_registry.keys())}")
        
    # 4. Determine default primary metric
    if not primary_metric:
        primary_metric = 'accuracy' if task_type == 'classification' else 'rmse'
        
    model_results = {}
    
    # 5. Execute cross-validation for each algorithm
    for algo_name in active_algorithms:
        algo_start = time.time()
        
        # Instantiate fresh model
        if task_type == 'classification':
            model = get_classifier(algo_name, random_state=random_state)
        else:
            model = get_regressor(algo_name, random_state=random_state)
            
        # Execute selected validation method
        if validation_method == 'stratified':
            res = run_stratified_kfold_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
                is_multiclass=is_multiclass,
                scaling=scaling,
                imputation=imputation
            )
        elif validation_method == 'kfold':
            res = run_kfold_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
                task_type=task_type,
                is_multiclass=is_multiclass,
                scaling=scaling,
                imputation=imputation
            )
        elif validation_method == 'nested':
            res = run_nested_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                shuffle=True,
                random_state=random_state,
                task_type=task_type,
                is_multiclass=is_multiclass,
                scaling=scaling,
                imputation=imputation
            )
        elif validation_method in ['timeseries', 'time_series']:
            date_labels = prepared.get('date_labels', None)
            res = run_timeseries_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                n_splits=n_splits,
                gap=gap,
                scaling=scaling,
                imputation=imputation,
                date_labels=date_labels
            )
        else:
            raise ValueError(f"Unsupported validation method: '{validation_method}'")
            
        model_results[algo_name] = res
        
        # Log individual experiment to database
        indiv_exp_id = str(uuid.uuid4())[:12]
        save_experiment({
            'experiment_id': indiv_exp_id,
            'dataset_name': dataset_name,
            'target_column': prepared['target_name'],
            'task_type': task_type,
            'algorithm': algo_name,
            'validation_method': res['method'],
            'hyperparameters': {},
            'metrics': res['summary'],
            'fold_scores': res['summary'].get('scores_per_fold', []),
            'best_params': res.get('best_params_per_fold', {}),
            'execution_time': res.get('execution_time', round(time.time() - algo_start, 4)),
            'notes': f"Part of comparison experiment {comp_id}"
        })
        
    total_execution_time = round(time.time() - start_total_time, 4)
    
    # 6. Rank and recommend models
    ranking_data = rank_and_recommend(model_results, task_type, primary_metric=primary_metric)
    rankings = ranking_data['rankings']
    recommendation = ranking_data['recommendation']
    
    # 7. Build chart datasets for Plotly
    chart_payload = {
        'algorithms': [m['algorithm'] for m in rankings],
        'scores': [m['score'] for m in rankings],
        'stds': [m['std'] for m in rankings],
        'execution_times': [m['execution_time'] for m in rankings],
        'metric_label': recommendation['metric_label'],
        'radar_data': generate_radar_chart_payload(rankings, task_type)
    }
    
    # 8. Save comparison summary record to SQLite
    save_comparison({
        'comparison_id': comp_id,
        'dataset_name': dataset_name,
        'task_type': task_type,
        'validation_method': validation_method,
        'primary_metric': primary_metric,
        'algorithms_compared': active_algorithms,
        'rankings': rankings,
        'winner_algorithm': recommendation.get('winner_algorithm', ''),
        'recommendation': recommendation,
        'execution_time': total_execution_time
    })
    
    return {
        'comparison_id': comp_id,
        'dataset_name': dataset_name,
        'target_column': prepared['target_name'],
        'task_type': task_type,
        'validation_method': validation_method,
        'primary_metric': primary_metric,
        'n_splits': n_splits,
        'models_compared_count': len(active_algorithms),
        'rankings': rankings,
        'recommendation': recommendation,
        'chart_data': chart_payload,
        'model_details': {k: {'summary': v['summary'], 'execution_time': v['execution_time']} for k, v in model_results.items()},
        'total_execution_time': total_execution_time
    }
