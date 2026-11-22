"""
FoldWise AI - Phase 6 Test Suite
Rigorous verification of Time-Series Cross-Validation, Chronological Integrity,
Zero-Lookahead Leakage, Forecasting Accuracy, and Empirical Leakage Contrast.
"""

import pytest
import numpy as np
import pandas as pd
from app import app
from dataset.loader import load_dataset
from preprocessing.processor import prepare_dataset_for_modeling, create_time_series_features
from validation.timeseries import (
    run_timeseries_cv,
    compare_timeseries_vs_random_kfold,
    create_isolated_time_series_split
)
from models.regressors import get_regressor

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_chronological_integrity_and_no_lookahead():
    """
    Test 1: Verify that every split in Time-Series CV satisfies:
    max(train_indices) < min(val_indices) with zero future lookahead.
    """
    ds = load_dataset('air_passengers')
    prepared = prepare_dataset_for_modeling(ds, target_col='Passengers', lag_orders=(1, 2, 3))
    
    X = prepared['X']
    y = prepared['y']
    num_cols = prepared['numerical_cols']
    cat_cols = prepared['categorical_cols']
    
    model = get_regressor('RandomForestRegressor', random_state=42)
    n_splits = 5
    res = run_timeseries_cv(
        model=model,
        X=X,
        y=y,
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        n_splits=n_splits,
        gap=0
    )
    
    assert res['method'] == 'Time-Series Cross-Validation'
    assert len(res['fold_metrics']) == n_splits
    assert len(res['chronological_windows']) == n_splits
    
    prev_train_size = 0
    for fold_info in res['chronological_windows']:
        train_start, train_end = fold_info['train_index_range']
        val_start, val_end = fold_info['val_index_range']
        
        # Chronological sequence constraint: train must precede validation
        assert train_end < val_start, f"Fold {fold_info['fold']} violated chronological ordering!"
        # Expanding window constraint: train size must expand
        assert fold_info['train_samples'] > prev_train_size
        prev_train_size = fold_info['train_samples']

def test_time_series_gap_constraint():
    """
    Test 2: Verify that when a horizon gap is configured,
    the gap constraint (min_val_idx - max_train_idx >= gap + 1) is strictly maintained.
    """
    ds = load_dataset('air_passengers')
    prepared = prepare_dataset_for_modeling(ds, target_col='Passengers')
    
    model = get_regressor('LinearRegression')
    gap = 3
    res = run_timeseries_cv(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=4,
        gap=gap
    )
    
    for fold_info in res['chronological_windows']:
        train_end = fold_info['train_index_range'][1]
        val_start = fold_info['val_index_range'][0]
        actual_gap = val_start - train_end - 1
        assert actual_gap == gap, f"Expected gap of {gap}, but got {actual_gap}"

def test_actual_air_passengers_forecasting_metrics():
    """
    Test 3: Verify that genuine forecasting metrics (MAE, RMSE, MAPE)
    are calculated safely using real Airline Passengers dataset without synthetic fabrication.
    """
    ds = load_dataset('air_passengers')
    prepared = prepare_dataset_for_modeling(ds, target_col='Passengers', lag_orders=(1, 2, 3))
    
    model = get_regressor('RandomForestRegressor', random_state=42)
    res = run_timeseries_cv(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=5
    )
    
    summary = res['summary']
    assert summary['mean_score'] > 0  # RMSE is strictly positive
    assert 'mean_mae' in summary
    assert 'mean_rmse' in summary
    assert 'mean_mape' in summary
    assert summary['mean_mae'] > 0
    assert summary['mean_mape'] > 0
    
    # Check that individual fold metrics contain MAE, RMSE, MAPE
    first_fold = res['fold_metrics'][0]
    assert 'mae' in first_fold
    assert 'rmse' in first_fold
    assert 'mape' in first_fold
    assert first_fold['mae'] > 0
    assert first_fold['rmse'] > 0
    
    # Check that predictions exist and timeline is populated
    assert len(res['forecast_timeline']) > 0
    sample_pt = res['forecast_timeline'][0]
    assert 'actual' in sample_pt
    assert 'predicted' in sample_pt
    assert 'date' in sample_pt

def test_empirical_timeseries_vs_random_kfold_contrast():
    """
    Test 4: Verify the empirical comparison engine demonstrating lookahead leakage in random K-Fold.
    Random K-Fold peeks into future data via random shuffling.
    """
    ds = load_dataset('air_passengers')
    prepared = prepare_dataset_for_modeling(ds, target_col='Passengers')
    
    model = get_regressor('RandomForestRegressor', random_state=42)
    contrast = compare_timeseries_vs_random_kfold(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=5,
        random_state=42
    )
    
    assert 'timeseries' in contrast
    assert 'random_kfold' in contrast
    assert 'comparison' in contrast
    
    comp = contrast['comparison']
    assert 'timeseries_rmse' in comp
    assert 'random_kfold_rmse' in comp
    assert 'explanation' in comp
    # Both RMSE values should be positive valid numbers
    assert comp['timeseries_rmse'] > 0
    assert comp['random_kfold_rmse'] > 0

def test_api_run_timeseries_experiment(client):
    """
    Test 5: Verify POST /api/experiments/run executes Time-Series CV end-to-end
    and records results to SQLite database.
    """
    payload = {
        'dataset_name': 'air_passengers',
        'target_column': 'Passengers',
        'validation_method': 'timeseries',
        'algorithm': 'RandomForestRegressor',
        'n_splits': 4,
        'gap': 1,
        'scaling': 'standard',
        'imputation': 'mean'
    }
    
    resp = client.post('/api/experiments/run', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    
    assert data['method'] == 'Time-Series Cross-Validation'
    assert data['n_splits'] == 4
    assert data['gap'] == 1
    assert data['task_type'] == 'time_series'
    assert len(data['fold_metrics']) == 4
    assert 'experiment_id' in data
    assert 'forecast_timeline' in data

def test_api_timeseries_compare_endpoint(client):
    """
    Test 6: Verify POST /api/validation/timeseries-compare returns valid empirical comparison.
    """
    payload = {
        'dataset_name': 'air_passengers',
        'target_column': 'Passengers',
        'algorithm': 'RandomForestRegressor',
        'n_splits': 4,
        'random_state': 42
    }
    
    resp = client.post('/api/validation/timeseries-compare', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    
    assert 'timeseries' in data
    assert 'random_kfold' in data
    assert 'comparison' in data
    assert data['comparison']['timeseries_rmse'] > 0

def test_isolated_time_series_split_features():
    """
    Test 7: Verify that lag features are generated separately on training data
    without accidentally using validation or future observations.
    Altering or corrupting future validation data must have ZERO impact on training features.
    """
    ds = load_dataset('air_passengers')
    df = ds['df']
    split_cutoff = 100
    
    # 1. Generate isolated split
    iso_split = create_isolated_time_series_split(
        df=df,
        target_col='Passengers',
        train_end_idx=split_cutoff,
        lag_orders=(1, 2, 3),
        gap=0
    )
    
    X_train_clean = iso_split['X_train']
    y_train_clean = iso_split['y_train']
    
    # 2. Corrupt future validation data with extreme values
    df_corrupted = df.copy()
    df_corrupted.loc[split_cutoff:, 'Passengers'] = 999999.0
    
    iso_corrupted = create_isolated_time_series_split(
        df=df_corrupted,
        target_col='Passengers',
        train_end_idx=split_cutoff,
        lag_orders=(1, 2, 3),
        gap=0
    )
    
    X_train_corrupted = iso_corrupted['X_train']
    
    # Verify training feature matrices are bit-for-bit identical (zero future leakage)
    max_diff = float((X_train_clean - X_train_corrupted).abs().max().max())
    assert max_diff == 0.0, f"Future validation contamination detected! diff={max_diff}"
    assert len(X_train_clean) == 97  # 100 minus max_lag 3
    assert len(iso_split['X_val']) == 44  # 144 - 100

def test_real_airline_passengers_forecasting_example():
    """
    Test 8: Verify an actual end-to-end forecasting pipeline on the real Airline Passengers dataset.
    Extract and verify true historical values, predicted values, and calculated metrics.
    """
    ds = load_dataset('air_passengers')
    df = ds['df']
    
    # Chronological partition: Train on observations up to 1957 (idx 108), forecast 1958-1960 (idx 108-144)
    split_cutoff = 108
    iso_split = create_isolated_time_series_split(
        df=df,
        target_col='Passengers',
        train_end_idx=split_cutoff,
        lag_orders=(1, 2, 3),
        gap=0
    )
    
    X_train = iso_split['X_train']
    y_train = iso_split['y_train']
    X_val = iso_split['X_val']
    y_val = iso_split['y_val']
    
    from sklearn.ensemble import RandomForestRegressor
    from evaluation.metrics import calculate_regression_metrics
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    metrics = calculate_regression_metrics(y_val, y_pred)
    
    # Ground truth checks against actual Airline Passengers data:
    # y_val starts at 1958-01 (historical value 340)
    assert float(y_val.iloc[0]) == 340.0
    assert len(y_pred) == 36  # 36 months of out-of-sample forecast
    
    # Verify regression metrics are strictly positive and mathematically sound
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    assert metrics['mape'] > 0
    assert metrics['mape'] < 30.0  # Reasonable forecast error under 30%

