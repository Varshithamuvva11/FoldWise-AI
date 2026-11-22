"""
FoldWise AI - Advanced Multi-Metric Visualizer & Fold Allocation Engine
Generates Plotly-compatible payloads for 360-degree radar/spider charts,
cross-validation fold allocation diagrams, and multi-metric distribution curves.
"""

import numpy as np
from dataset.loader import load_dataset
from preprocessing.processor import prepare_dataset_for_modeling
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

def generate_radar_chart_payload(rankings, task_type='classification'):
    """
    Construct multi-axis radar (spider) chart data normalizing models across
    predictive quality, stability, and computational efficiency.
    
    Args:
        rankings (list of dicts): Ranked models from evaluation.comparison
        task_type (str): 'classification', 'regression', or 'time_series'
        
    Returns:
        dict: Radar categories and model polar traces for Plotly
    """
    if not rankings:
        return {'categories': [], 'traces': []}
        
    if task_type == 'classification':
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Stability', 'Speed']
    else:
        categories = ['Fit (R²)', 'Accuracy (1-MAPE)', 'Precision', 'Stability', 'Speed']
        
    max_time = max((m.get('execution_time', 0.1) for m in rankings), default=1.0)
    if max_time <= 0:
        max_time = 1.0
        
    traces = []
    
    for idx, m in enumerate(rankings):
        summary = m.get('metrics_summary', {})
        std = m.get('std', 0.0)
        t = m.get('execution_time', 0.1)
        
        # Stability: 100 is perfectly consistent (std=0), decreases as std grows
        stability_score = round(max(0.0, min(100.0, 100.0 * (1.0 - (std * 8.0)))), 1)
        
        # Speed: normalized 0-100 where fastest is 100
        speed_score = round(max(10.0, min(100.0, 100.0 * (1.0 - (t / (max_time * 1.5))))), 1)
        
        if task_type == 'classification':
            acc = round(summary.get('mean_accuracy', m.get('score', 0.0)) * 100.0, 1)
            prec = round(summary.get('mean_precision', acc / 100.0) * 100.0, 1)
            rec = round(summary.get('mean_recall', acc / 100.0) * 100.0, 1)
            f1 = round(summary.get('mean_f1_score', acc / 100.0) * 100.0, 1)
            values = [acc, prec, rec, f1, stability_score, speed_score]
        else:
            r2 = round(max(0.0, summary.get('mean_r2_score', 0.0)) * 100.0, 1)
            mape = summary.get('mean_mape', 15.0)
            acc_mape = round(max(0.0, min(100.0, 100.0 - mape)), 1)
            # Precision proxy for regression (normalized RMSE inversion)
            rmse = summary.get('mean_rmse', m.get('score', 1.0))
            prec_reg = round(max(10.0, min(100.0, 100.0 / (1.0 + rmse * 0.1))), 1)
            values = [r2, acc_mape, prec_reg, stability_score, speed_score]
            
        # Close the polygon loop for polar radar chart
        values_closed = values + [values[0]]
        
        colors = ['#10B981', '#6366F1', '#38BDF8', '#F59E0B', '#EC4899']
        color = colors[idx % len(colors)]
        
        traces.append({
            'name': m['algorithm'],
            'r': values_closed,
            'theta': categories + [categories[0]],
            'color': color,
            'fill': 'toself',
            'opacity': 0.2 if idx > 0 else 0.4
        })
        
    return {
        'categories': categories,
        'traces': traces
    }

def generate_fold_allocation_diagram(dataset_name='iris', validation_method='kfold', n_splits=5, gap=0):
    """
    Generate coordinate layout illustrating training vs validation fold allocations
    across all observations for visualization.
    
    Args:
        dataset_name (str): Dataset key
        validation_method (str): 'kfold', 'stratified', 'timeseries', 'nested'
        n_splits (int): Number of splits
        gap (int): Temporal gap
        
    Returns:
        dict: Matrix segments indicating fold assignments for plotting
    """
    ds_dict = load_dataset(dataset_name)
    prepared = prepare_dataset_for_modeling(ds_dict)
    X = prepared['X']
    y = prepared['y']
    n_samples = len(X)
    
    method = validation_method.lower().replace('-', '_')
    folds_data = []
    
    if method == 'stratified' and prepared['task_type'] == 'classification':
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitter = skf.split(X, y)
    elif method in ['timeseries', 'time_series']:
        tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        splitter = tss.split(X, y)
    else: # Default kfold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splitter = kf.split(X, y)
        
    for fold_idx, (train_idx, val_idx) in enumerate(splitter):
        folds_data.append({
            'fold': fold_idx + 1,
            'train_count': len(train_idx),
            'val_count': len(val_idx),
            'train_start': int(np.min(train_idx)),
            'train_end': int(np.max(train_idx)),
            'val_start': int(np.min(val_idx)),
            'val_end': int(np.max(val_idx)),
            'val_indices_preview': val_idx[:15].tolist()
        })
        
    return {
        'dataset_name': dataset_name,
        'validation_method': validation_method,
        'n_samples': n_samples,
        'n_splits': n_splits,
        'folds': folds_data
    }

def generate_dashboard_charts_payload(db_stats=None, registered_datasets=None):
    """
    Generate structured Plotly payload configurations for the overview dashboard.
    
    Args:
        db_stats (dict, optional): Database aggregate statistics
        registered_datasets (dict, optional): Registered datasets catalog
        
    Returns:
        dict: Payloads for strategy distribution, dataset landscape, and model matrix.
    """
    if db_stats is None:
        db_stats = {
            'total_experiments': 0,
            'total_comparisons': 0,
            'strategy_counts': {},
            'task_counts': {},
            'dataset_counts': {},
            'avg_execution_time': 0.0
        }
        
    strat_counts = db_stats.get('strategy_counts', {})
    
    # Map raw method names to formal labels
    strategy_mapping = [
        ('stratified', 'Stratified K-Fold', '#10B981'),
        ('kfold', 'Standard K-Fold', '#38BDF8'),
        ('nested', 'Nested CV', '#6366F1'),
        ('timeseries', 'Time-Series CV', '#F59E0B')
    ]
    
    strat_labels = [label for _, label, _ in strategy_mapping]
    strat_colors = [color for _, _, color in strategy_mapping]
    strat_values = []
    
    for key, label, _ in strategy_mapping:
        val = strat_counts.get(key, 0)
        if val == 0:
            for k, v in strat_counts.items():
                if key in k.lower():
                    val += v
        strat_values.append(val)
        
    has_strategy_data = sum(strat_values) > 0
    
    # Default landscape for built-in datasets
    default_landscape = [
        {'id': 'iris', 'name': 'Iris Flower', 'samples': 150, 'features': 4, 'task': 'Classification', 'color': '#10B981'},
        {'id': 'breast_cancer', 'name': 'Breast Cancer', 'samples': 569, 'features': 30, 'task': 'Classification', 'color': '#38BDF8'},
        {'id': 'wine_quality', 'name': 'Wine Quality', 'samples': 1599, 'features': 11, 'task': 'Regression', 'color': '#EC4899'},
        {'id': 'air_passengers', 'name': 'Air Passengers', 'samples': 144, 'features': 1, 'task': 'Time-Series', 'color': '#F59E0B'}
    ]
    
    model_matrix = {
        'total_algorithms': 9,
        'classifier_count': 5,
        'regressor_count': 4,
        'classifiers': [
            {'name': 'Logistic Regression', 'type': 'Linear', 'speed': 'Fast', 'interpretability': 'High'},
            {'name': 'Decision Tree Classifier', 'type': 'Tree', 'speed': 'Fast', 'interpretability': 'High'},
            {'name': 'Random Forest Classifier', 'type': 'Ensemble', 'speed': 'Moderate', 'interpretability': 'Medium'},
            {'name': 'Support Vector Machine (SVC)', 'type': 'Kernel', 'speed': 'Moderate', 'interpretability': 'Low'},
            {'name': 'K-Nearest Neighbors (KNN)', 'type': 'Instance-based', 'speed': 'Fast', 'interpretability': 'Medium'}
        ],
        'regressors': [
            {'name': 'Linear Regression', 'type': 'Linear', 'speed': 'Fast', 'interpretability': 'High'},
            {'name': 'Decision Tree Regressor', 'type': 'Tree', 'speed': 'Fast', 'interpretability': 'High'},
            {'name': 'Random Forest Regressor', 'type': 'Ensemble', 'speed': 'Moderate', 'interpretability': 'Medium'},
            {'name': 'Support Vector Regressor (SVR)', 'type': 'Kernel', 'speed': 'Moderate', 'interpretability': 'Low'}
        ]
    }
    
    return {
        'strategy_donut': {
            'labels': strat_labels,
            'values': strat_values,
            'colors': strat_colors,
            'has_data': has_strategy_data
        },
        'dataset_landscape': {
            'names': [d['name'] for d in default_landscape],
            'samples': [d['samples'] for d in default_landscape],
            'features': [d['features'] for d in default_landscape],
            'tasks': [d['task'] for d in default_landscape],
            'colors': [d['color'] for d in default_landscape]
        },
        'model_matrix': model_matrix
    }
