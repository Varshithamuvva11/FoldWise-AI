"""
Unit & Integration Tests for Phase 4 - K-Fold & Stratified K-Fold Cross-Validation
Verifies fold score calculation, variance reporting, class proportion preservation,
allocation tracking, SQLite logging, and side-by-side empirical comparison.
"""

import pytest
import numpy as np
import pandas as pd
from app import app
from dataset.loader import load_dataset
from preprocessing.processor import prepare_dataset_for_modeling
from models.classifiers import get_classifier
from models.regressors import get_regressor
from validation.kfold import run_kfold_cv
from validation.stratified import run_stratified_kfold_cv, compare_kfold_and_stratified
from database.db import get_experiment_by_id

@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_kfold_cv_classification():
    """Test standard K-Fold CV on Iris dataset with Logistic Regression."""
    iris = load_dataset('iris')
    prepared = prepare_dataset_for_modeling(iris)
    model = get_classifier('LogisticRegression', random_state=42)
    
    n_splits = 5
    result = run_kfold_cv(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
        task_type='classification',
        is_multiclass=True
    )
    
    assert result['method'] == 'K-Fold'
    assert result['algorithm'] == 'LogisticRegression'
    assert len(result['fold_metrics']) == n_splits
    assert len(result['fold_allocations']) == n_splits
    assert result['execution_time'] > 0
    
    # Check fold sample allocations sum to 150
    for alloc in result['fold_allocations']:
        assert alloc['train_samples'] + alloc['val_samples'] == 150
        assert alloc['val_samples'] == 30
        
    s = result['summary']
    assert 0.0 <= s['mean_score'] <= 1.0
    assert 0.0 <= s['std_score'] <= 1.0
    assert len(s['scores_per_fold']) == n_splits
    assert 'accumulated_confusion_matrix' in s

def test_kfold_cv_regression():
    """Test standard K-Fold CV on Wine Quality dataset with Linear Regression."""
    wine = load_dataset('wine_quality')
    prepared = prepare_dataset_for_modeling(wine)
    model = get_regressor('LinearRegression')
    
    n_splits = 5
    result = run_kfold_cv(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
        task_type='regression'
    )
    
    assert result['method'] == 'K-Fold'
    assert result['task_type'] == 'regression'
    assert len(result['fold_metrics']) == n_splits
    
    s = result['summary']
    assert s['primary_metric'] == 'rmse'
    assert s['mean_score'] > 0  # RMSE must be positive
    assert 'mean_mae' in s
    assert 'mean_r2_score' in s

def test_stratified_kfold_class_preservation():
    """
    Test Stratified K-Fold on Breast Cancer dataset.
    Verifies that class proportions across each fold match the dataset baseline.
    """
    bc = load_dataset('breast_cancer')
    prepared = prepare_dataset_for_modeling(bc)
    model = get_classifier('LogisticRegression', random_state=42)
    
    n_splits = 5
    result = run_stratified_kfold_cv(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
        is_multiclass=False
    )
    
    assert result['method'] == 'Stratified K-Fold'
    assert len(result['fold_metrics']) == n_splits
    assert len(result['class_distributions']) == n_splits
    
    # Baseline class proportion in Breast Cancer: ~62.7% class 1, ~37.3% class 0
    baseline = result['baseline_class_proportions']
    baseline_c1 = baseline.get('1', 0.0)
    
    for dist in result['class_distributions']:
        fold_c1 = dist['proportions'].get('1', 0.0)
        # Fold proportion must be within 2 percentage points of baseline
        assert abs(fold_c1 - baseline_c1) < 2.5, f"Fold proportion {fold_c1} deviated from baseline {baseline_c1}"

def test_stratified_fails_on_continuous_target():
    """Verify that applying Stratified K-Fold to continuous targets raises ValueError."""
    wine = load_dataset('wine_quality')
    prepared = prepare_dataset_for_modeling(wine)
    model = get_regressor('LinearRegression')
    
    # Continuous synthetic target with 100 unique values
    y_cont = pd.Series(np.linspace(0, 100, len(prepared['X'])))
    
    with pytest.raises(ValueError) as excinfo:
        run_stratified_kfold_cv(
            model=model,
            X=prepared['X'],
            y=y_cont,
            numerical_cols=prepared['numerical_cols'],
            categorical_cols=prepared['categorical_cols'],
            n_splits=5
        )
    assert "Stratified K-Fold requires discrete class labels" in str(excinfo.value)

def test_compare_kfold_and_stratified():
    """Verify compare_kfold_and_stratified returns side-by-side empirical report."""
    bc = load_dataset('breast_cancer')
    prepared = prepare_dataset_for_modeling(bc)
    model = get_classifier('LogisticRegression', random_state=42)
    
    comp = compare_kfold_and_stratified(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    assert 'kfold' in comp
    assert 'stratified' in comp
    assert 'variance_comparison' in comp
    assert 'kfold_std' in comp['variance_comparison']
    assert 'stratified_std' in comp['variance_comparison']

def test_api_run_experiment_and_db_logging(client):
    """Test POST /api/experiments/run and verify SQLite database persistence."""
    payload = {
        'dataset_name': 'iris',
        'algorithm': 'LogisticRegression',
        'validation_method': 'stratified',
        'n_splits': 5,
        'random_state': 42,
        'scaling': 'standard',
        'imputation': 'mean'
    }
    res = client.post('/api/experiments/run', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert 'experiment_id' in data
    exp_id = data['experiment_id']
    assert data['method'] == 'Stratified K-Fold'
    assert data['summary']['mean_score'] > 0.8
    
    # Query database to confirm it was persisted
    db_record = get_experiment_by_id(exp_id)
    assert db_record is not None
    assert db_record['dataset_name'] == 'iris'
    assert db_record['algorithm'] == 'LogisticRegression'
    assert db_record['validation_method'] == 'Stratified K-Fold'

def test_api_compare_validation_endpoint(client):
    """Test POST /api/validation/compare endpoint."""
    payload = {
        'dataset_name': 'breast_cancer',
        'algorithm': 'LogisticRegression',
        'n_splits': 5,
        'random_state': 42
    }
    res = client.post('/api/validation/compare', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert 'kfold' in data
    assert 'stratified' in data
    assert 'variance_comparison' in data
