"""
FoldWise AI - Time-Series Cross-Validation & Sequential Forecasting Engine
Implements strict chronological TimeSeriesSplit with expanding historical windows,
zero shuffling, and no future-data lookahead leakage.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.base import clone
from preprocessing.processor import build_leak_free_pipeline, create_time_series_features
from evaluation.metrics import calculate_regression_metrics, aggregate_fold_metrics

def run_timeseries_cv(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    n_splits=5,
    max_train_size=None,
    test_size=None,
    gap=0,
    scaling='standard',
    imputation='mean',
    date_labels=None
):
    """
    Execute genuine chronological Time-Series Cross-Validation.
    
    Guarantees:
    1. Zero shuffling: Observational sequence is strictly forward-expanding.
    2. Zero lookahead leakage: max(train_index) < min(val_index) for every split.
    3. Scalers and transformers are fit solely on historical training windows.
    
    Args:
        model: Base Scikit-learn regression estimator
        X (pd.DataFrame): Time-series features (backward lags + temporal indices)
        y (pd.Series/array): Target continuous series to forecast
        numerical_cols (list): Numeric column names
        categorical_cols (list): Categorical column names
        n_splits (int): Number of rolling/expanding splits (default: 5)
        max_train_size (int, optional): Fixed sliding window size (None = expanding window)
        test_size (int, optional): Size of validation test window
        gap (int): Number of excluded observations between train and test windows
        scaling (str): 'standard', 'minmax', or 'none'
        imputation (str): 'mean', 'median', or 'most_frequent'
        date_labels (list, optional): Chronological timestamp strings for plotting
        
    Returns:
        dict: Complete time-series evaluation report with fold metrics,
              chronological window diagrams, and actual vs predicted forecasting curves.
    """
    start_total_time = time.time()
    
    # Configure TimeSeriesSplit
    tss = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=gap
    )
    
    fold_metrics = []
    fold_allocations = []
    chronological_windows = []
    
    # Track concatenated forecast points for full series plotting
    forecast_timeline = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tss.split(X, y)):
        fold_start = time.time()
        
        # Rigorous check for temporal causality: no future lookahead
        max_train_idx = int(np.max(train_idx))
        min_val_idx = int(np.min(val_idx))
        assert max_train_idx < min_val_idx, f"Temporal violation in fold {fold_idx + 1}: train index exceeds test index!"
        assert (min_val_idx - max_train_idx) >= (gap + 1), f"Gap constraint violation in fold {fold_idx + 1}"
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Build fresh pipeline for each expanding window
        model_clone = clone(model)
        pipeline = build_leak_free_pipeline(
            model=model_clone,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            scaling=scaling,
            imputation_strategy=imputation
        )
        
        # Fit on historical training window ONLY
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Forecast future validation window
        y_pred = pipeline.predict(X_val_fold)
        
        fold_duration = time.time() - fold_start
        
        # Compute regression & forecasting metrics (MAE, RMSE, MAPE)
        m = calculate_regression_metrics(y_val_fold, y_pred)
        m['fold'] = fold_idx + 1
        m['duration'] = round(fold_duration, 4)
        fold_metrics.append(m)
        
        # Build window metadata
        train_start_idx = int(train_idx[0])
        train_end_idx = int(train_idx[-1])
        val_start_idx = int(val_idx[0])
        val_end_idx = int(val_idx[-1])
        
        train_range_str = f"[{train_start_idx} : {train_end_idx}]"
        val_range_str = f"[{val_start_idx} : {val_end_idx}]"
        
        if date_labels and len(date_labels) >= len(X):
            train_date_str = f"{date_labels[train_start_idx]} to {date_labels[train_end_idx]}"
            val_date_str = f"{date_labels[val_start_idx]} to {date_labels[val_end_idx]}"
            val_dates = [date_labels[i] for i in val_idx]
        else:
            train_date_str = train_range_str
            val_date_str = val_range_str
            val_dates = [f"t_{i}" for i in val_idx]
            
        fold_window_info = {
            'fold': fold_idx + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_index_range': [train_start_idx, train_end_idx],
            'val_index_range': [val_start_idx, val_end_idx],
            'train_date_range': train_date_str,
            'val_date_range': val_date_str,
            'gap': gap,
            'actuals': [round(float(v), 2) for v in y_val_fold],
            'predictions': [round(float(v), 2) for v in y_pred],
            'dates': val_dates
        }
        chronological_windows.append(fold_window_info)
        
        fold_allocations.append({
            'fold': fold_idx + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'val_indices_preview': val_idx[:10].tolist()
        })
        
        # Record forecast points
        for i_pt in range(len(val_idx)):
            forecast_timeline.append({
                'date': val_dates[i_pt],
                'actual': round(float(y_val_fold.iloc[i_pt]), 2),
                'predicted': round(float(y_pred[i_pt]), 2),
                'fold': fold_idx + 1
            })
            
    total_execution_time = time.time() - start_total_time
    summary = aggregate_fold_metrics(fold_metrics, task_type='regression')
    
    return {
        'method': 'Time-Series Cross-Validation',
        'algorithm': model.__class__.__name__,
        'n_splits': n_splits,
        'gap': gap,
        'task_type': 'time_series',
        'fold_metrics': fold_metrics,
        'summary': summary,
        'fold_allocations': fold_allocations,
        'chronological_windows': chronological_windows,
        'forecast_timeline': forecast_timeline,
        'execution_time': round(total_execution_time, 4)
    }

def compare_timeseries_vs_random_kfold(
    model,
    X,
    y,
    numerical_cols,
    categorical_cols,
    n_splits=5,
    random_state=42
):
    """
    Demonstrate why random K-Fold is methodologically invalid for time-series forecasting.
    
    Random K-Fold shuffles future observations into training folds and predicts past observations,
    yielding falsely optimistic RMSE / R2 due to lookahead leakage.
    
    Returns:
        dict: Empirical contrast between Time-Series CV and Random K-Fold
    """
    # 1. Genuine Time-Series CV
    ts_res = run_timeseries_cv(
        model=model,
        X=X,
        y=y,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        n_splits=n_splits
    )
    
    # 2. Naive Random K-Fold (Methodologically Flawed for Time-Series)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipe = build_leak_free_pipeline(clone(model), numerical_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        
        m = calculate_regression_metrics(y_val, y_pred)
        kf_metrics.append(m)
        
    kf_summary = aggregate_fold_metrics(kf_metrics, task_type='regression')
    
    ts_rmse = ts_res['summary']['mean_score']
    kf_rmse = kf_summary['mean_score']
    
    return {
        'timeseries': ts_res,
        'random_kfold': {
            'method': 'Random K-Fold (Flawed for Time-Series)',
            'summary': kf_summary,
            'fold_metrics': kf_metrics
        },
        'comparison': {
            'timeseries_rmse': ts_rmse,
            'random_kfold_rmse': kf_rmse,
            'rmse_difference': round(ts_rmse - kf_rmse, 4),
            'explanation': (
                "Notice that Random K-Fold reports an artificially low RMSE compared to Time-Series CV. "
                "This occurs because Random K-Fold shuffles future observations into the training set, "
                "allowing the model to peek into the future (lookahead leakage). "
                "Time-Series CV strictly evaluates on future horizons, reflecting true deployment error."
            )
        }
    }

def create_isolated_time_series_split(
    df,
    target_col='Passengers',
    train_end_idx=100,
    val_end_idx=None,
    lag_orders=(1, 2, 3),
    gap=0
):
    """
    Generate training and validation feature matrices with strict temporal isolation.
    
    Guarantees:
    1. Training features (X_train) are generated strictly on df.iloc[:train_end_idx].
       Validation observations are never passed to the training feature generator.
    2. Validation features (X_val) for observation t only use past observations (t - 1, t - 2, ...),
       guaranteeing zero future lookahead.
    3. Respects any chronological gap between train_end_idx and val_start_idx.
    
    Returns:
        dict: {
            'X_train': pd.DataFrame,
            'y_train': pd.Series,
            'X_val': pd.DataFrame,
            'y_val': pd.Series,
            'feature_names': list,
            'train_dates': list,
            'val_dates': list,
            'train_samples': int,
            'val_samples': int
        }
    """
    max_lag = max(lag_orders)
    total_len = len(df)
    if val_end_idx is None:
        val_end_idx = total_len
        
    val_start_idx = train_end_idx + gap
    assert val_start_idx < val_end_idx, "val_start_idx must be before val_end_idx"
    assert val_start_idx - max_lag >= 0, "Not enough historical observations to construct lags for validation"
    
    # 1. Generate training features strictly on df.iloc[:train_end_idx]
    train_slice = df.iloc[:train_end_idx].copy()
    X_train, y_train, feature_cols, train_dates = create_time_series_features(
        train_slice, target_col=target_col, lag_orders=lag_orders, return_dates=True
    )
    
    # 2. Generate validation features using only historical context prior to validation points
    val_context = df.iloc[val_start_idx - max_lag : val_end_idx].copy()
    X_val, y_val, _, val_dates = create_time_series_features(
        val_context, target_col=target_col, lag_orders=lag_orders, return_dates=True
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'feature_names': feature_cols,
        'train_dates': train_dates,
        'val_dates': val_dates,
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }

