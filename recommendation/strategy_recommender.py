"""
FoldWise AI - Cross-Validation Strategy Recommender & Dataset Diagnostic Engine
Analyzes dataset characteristics (sample size, class balance, temporal dependencies, task type)
and provides automated, explainable recommendations on which cross-validation technique to deploy,
highlighting specific statistical hazards and pitfalls of inappropriate strategies.
"""

import numpy as np
import pandas as pd
from dataset.loader import load_dataset

def analyze_dataset_for_validation(dataset_dict, target_col=None):
    """
    Perform deep diagnostic analysis on a dataset to determine the optimal cross-validation strategy.
    
    Args:
        dataset_dict (dict): Dictionary returned by load_dataset()
        target_col (str, optional): Target column override
        
    Returns:
        dict: Diagnostic metrics, recommended strategy, fold recommendations,
              compatibility matrix, and detailed hazard warnings.
    """
    df = dataset_dict['df'].copy()
    dataset_name = dataset_dict.get('name') or dataset_dict.get('dataset_name') or dataset_dict.get('id', 'Dataset')
    target = target_col or dataset_dict.get('target')
    raw_task_type = dataset_dict.get('task_type')
    
    n_samples = len(df)
    n_features = len(df.columns) - (1 if target in df.columns else 0)
    
    # 1. Temporal dependency detection
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ['date', 'month', 'time', 'year', 'timestamp'])]
    is_sequential = (raw_task_type == 'time_series') or (len(date_candidates) > 0 and 'passengers' in dataset_name.lower())
    
    # 2. Target analysis
    target_series = df[target] if target in df.columns else df.iloc[:, -1]
    unique_vals = target_series.nunique()
    
    if is_sequential or raw_task_type == 'time_series':
        task_type = 'time_series'
    elif raw_task_type in ['classification', 'regression']:
        task_type = raw_task_type
    else:
        is_classification = (unique_vals <= 15 and not pd.api.types.is_float_dtype(target_series))
        task_type = 'classification' if is_classification else 'regression'
    
    # 3. Class distribution and imbalance analysis
    class_distribution = {}
    imbalance_ratio = 1.0
    minority_pct = 100.0
    is_imbalanced = False
    
    if task_type == 'classification':
        counts = target_series.value_counts()
        total_counts = len(target_series)
        class_distribution = {str(k): {'count': int(v), 'percentage': round((v / total_counts) * 100, 2)} for k, v in counts.items()}
        if len(counts) >= 2:
            maj_count = counts.iloc[0]
            min_count = counts.iloc[-1]
            imbalance_ratio = round(float(maj_count) / float(min_count), 2)
            minority_pct = round((float(min_count) / float(total_counts)) * 100, 2)
            is_imbalanced = (imbalance_ratio >= 1.35)
            
    # 4. Formulate recommendations based on statistical properties
    hazards = []
    compatibility_matrix = []
    
    if task_type == 'time_series':
        recommended_strategy = 'Time-Series Cross-Validation'
        recommended_k = 5 if n_samples >= 50 else 3
        confidence = 98
        primary_rationale = (
            f"The dataset '{dataset_name}' possesses chronological temporal dependencies. "
            "Observations must be evaluated using forward-expanding historical windows without random shuffling."
        )
        hazards.append({
            'strategy': 'Standard K-Fold',
            'severity': 'CRITICAL',
            'hazard': 'Lookahead Leakage',
            'description': 'Random shuffling causes future observations to be used to train models predicting past observations, causing falsely optimistic metrics.'
        })
        hazards.append({
            'strategy': 'Stratified K-Fold',
            'severity': 'CRITICAL',
            'hazard': 'Methodological Incompatibility',
            'description': 'Stratified K-Fold requires discrete class labels and shuffles data, destroying temporal causality.'
        })
        compatibility_matrix = [
            {'strategy': 'Time-Series Cross-Validation', 'status': 'RECOMMENDED', 'suitability': 'Essential', 'notes': 'Strictly maintains causality and zero-lookahead.'},
            {'strategy': 'Nested Cross-Validation', 'status': 'ACCEPTABLE (Sequential variant)', 'suitability': 'Feasible', 'notes': 'Requires inner sequential rolling splits.'},
            {'strategy': 'Standard K-Fold', 'status': 'INCOMPATIBLE / HAZARDOUS', 'suitability': 'Forbidden', 'notes': 'Causes catastrophic lookahead leakage.'},
            {'strategy': 'Stratified K-Fold', 'status': 'INCOMPATIBLE', 'suitability': 'Forbidden', 'notes': 'Incompatible with continuous temporal series.'}
        ]
        
    elif task_type == 'classification':
        if is_imbalanced:
            recommended_strategy = 'Stratified K-Fold'
            recommended_k = 5 if n_samples < 500 else 10
            confidence = 96
            primary_rationale = (
                f"Dataset contains class imbalance (ratio {imbalance_ratio}:1, minority class represents {minority_pct}%). "
                "Stratified K-Fold is required to preserve identical class proportions in every training and validation fold."
            )
            hazards.append({
                'strategy': 'Standard K-Fold',
                'severity': 'HIGH',
                'hazard': 'Fold Class Dropout & Evaluation Variance',
                'description': f"Random partitioning on an imbalanced dataset ({minority_pct}% minority) can leave some test folds with few or zero minority samples."
            })
            hazards.append({
                'strategy': 'Time-Series Split',
                'severity': 'CRITICAL',
                'hazard': 'Data Waste & Premature Truncation',
                'description': 'Non-sequential tabular observations must not be partitioned sequentially; early folds discard viable training data unnecessarily.'
            })
        else:
            recommended_strategy = 'Stratified K-Fold'
            recommended_k = 5 if n_samples < 250 else 10
            confidence = 92
            primary_rationale = (
                f"Even with balanced classes, Stratified K-Fold guarantees exact class distribution across all {recommended_k} folds, "
                "reducing cross-validation variance without any computational penalty."
            )
            
        compatibility_matrix = [
            {'strategy': 'Stratified K-Fold', 'status': 'RECOMMENDED', 'suitability': 'Optimal', 'notes': f'Preserves class balance across all folds (ratio {imbalance_ratio}:1).'},
            {'strategy': 'Nested Cross-Validation', 'status': 'RECOMMENDED (When Tuning)', 'suitability': 'Optimal for Tuning', 'notes': 'Eliminates hyperparameter selection bias.'},
            {'strategy': 'Standard K-Fold', 'status': 'ACCEPTABLE' if not is_imbalanced else 'HAZARDOUS', 'suitability': 'Suboptimal', 'notes': 'Random partitioning introduces unnecessary fold variance.'},
            {'strategy': 'Time-Series Split', 'status': 'INCOMPATIBLE', 'suitability': 'Forbidden', 'notes': 'Inappropriate for i.i.d. tabular data.'}
        ]
        
    else: # Regression
        recommended_strategy = 'Standard K-Fold'
        recommended_k = 5 if n_samples < 500 else 10
        confidence = 94
        primary_rationale = (
            f"Continuous target variable with {n_samples} samples. Standard K-Fold provides unbiased, low-variance estimation "
            "across continuous regression targets."
        )
        hazards.append({
            'strategy': 'Stratified K-Fold',
            'severity': 'CRITICAL',
            'hazard': 'Mathematical Incompatibility',
            'description': 'Stratified K-Fold cannot stratify continuous real-valued targets unless artificially binned into discrete bins.'
        })
        hazards.append({
            'strategy': 'Time-Series Split',
            'severity': 'HIGH',
            'hazard': 'Inefficient Historical Masking',
            'description': 'I.I.D. regression observations do not benefit from expanding chronological windows; early splits artificially restrict training set size.'
        })
        compatibility_matrix = [
            {'strategy': 'Standard K-Fold', 'status': 'RECOMMENDED', 'suitability': 'Optimal', 'notes': 'Standard unbiased evaluation for continuous targets.'},
            {'strategy': 'Nested Cross-Validation', 'status': 'RECOMMENDED (When Tuning)', 'suitability': 'Optimal for Tuning', 'notes': 'Required when optimizing regression hyperparameters.'},
            {'strategy': 'Stratified K-Fold', 'status': 'INCOMPATIBLE', 'suitability': 'Forbidden', 'notes': 'Cannot stratify continuous floating-point values.'},
            {'strategy': 'Time-Series Split', 'status': 'INCOMPATIBLE', 'suitability': 'Forbidden', 'notes': 'Inappropriate for non-sequential tabular observations.'}
        ]
        
    # Sample size reasoning
    if n_samples < 200:
        sample_size_advice = f"Small dataset (N={n_samples}). Use k=5 folds to ensure each validation fold contains at least {n_samples // 5} samples."
    elif n_samples <= 1000:
        sample_size_advice = f"Moderate dataset (N={n_samples}). k=5 or k=10 folds both provide balanced bias-variance tradeoffs."
    else:
        sample_size_advice = f"Large dataset (N={n_samples}). k=10 folds recommended to maximize training data volume (90% per fold) while keeping estimation variance low."

    return {
        'dataset_name': dataset_name,
        'task_type': task_type,
        'target_column': target,
        'n_samples': n_samples,
        'n_features': n_features,
        'diagnostics': {
            'is_sequential': is_sequential,
            'is_imbalanced': is_imbalanced,
            'imbalance_ratio': imbalance_ratio,
            'minority_class_percentage': minority_pct,
            'class_distribution': class_distribution,
            'sample_size_category': 'Small' if n_samples < 200 else ('Moderate' if n_samples <= 1000 else 'Large')
        },
        'recommendation': {
            'recommended_strategy': recommended_strategy,
            'recommended_k': recommended_k,
            'confidence_score': confidence,
            'primary_rationale': primary_rationale,
            'sample_size_advice': sample_size_advice,
            'tuning_note': "If hyperparameter tuning (GridSearchCV) is planned, upgrade to Nested Cross-Validation to eliminate optimistic model selection bias."
        },
        'hazards': hazards,
        'compatibility_matrix': compatibility_matrix
    }
