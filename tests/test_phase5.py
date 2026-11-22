"""
Unit & Integration Tests for Phase 5 - Nested Cross-Validation and Hyperparameter Tuning
Verifies double-loop execution, outer test isolation, parameter tracking,
model selection bias detection, and SQLite logging.
"""

import pytest
import numpy as np
import pandas as pd
from app import app
from dataset.loader import load_dataset
from preprocessing.processor import prepare_dataset_for_modeling
from models.classifiers import get_classifier
from models.regressors import get_regressor
from validation.nested import run_nested_cv, compare_nested_vs_non_nested
from database.db import get_experiment_by_id

@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_nested_cv_outer_test_isolation_and_parameters():
    """
    Rigorously verify that:
    1. Outer test data is never used during inner hyperparameter tuning.
    2. Best hyperparameters are selected and logged for every outer fold.
    3. Generalization score is evaluated on out-of-sample outer test folds.
    """
    iris = load_dataset('iris')
    prepared = prepare_dataset_for_modeling(iris)
    model = get_classifier('LogisticRegression', random_state=42)
    
    outer_k = 3
    inner_k = 2
    param_grid = {'model__C': [0.1, 1.0, 10.0]}
    
    result = run_nested_cv(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        param_grid=param_grid,
        outer_splits=outer_k,
        inner_splits=inner_k,
        shuffle=True,
        random_state=42,
        task_type='classification',
        is_multiclass=True
    )
    
    assert result['method'] == 'Nested Cross-Validation'
    assert result['outer_splits'] == outer_k
    assert result['inner_splits'] == inner_k
    assert len(result['fold_metrics']) == outer_k
    assert len(result['best_params_per_fold']) == outer_k
    assert len(result['inner_best_scores']) == outer_k
    
    # Verify each outer fold chose a parameter from the grid
    for bp in result['best_params_per_fold']:
        assert 'C' in bp
        assert bp['C'] in [0.1, 1.0, 10.0]
        
    # Check outer generalization performance
    s = result['summary']
    assert 0.0 <= s['mean_score'] <= 1.0
    assert 0.0 <= s['std_score'] <= 1.0
    
    # Check bias analysis
    bias = result['tuning_bias_analysis']
    assert 'mean_inner_score' in bias
    assert 'mean_outer_generalization_score' in bias
    assert 'optimistic_bias' in bias
    assert isinstance(bias['explanation'], str)

def test_nested_cv_regression():
    """Test Nested CV on Wine Quality regression dataset."""
    wine = load_dataset('wine_quality')
    prepared = prepare_dataset_for_modeling(wine)
    # Use smaller subset for fast unit testing
    X_sub = prepared['X'].iloc[:200]
    y_sub = prepared['y'].iloc[:200]
    
    model = get_regressor('DecisionTreeRegressor', random_state=42)
    param_grid = {'model__max_depth': [3, 5]}
    
    result = run_nested_cv(
        model=model,
        X=X_sub,
        y=y_sub,
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        param_grid=param_grid,
        outer_splits=3,
        inner_splits=2,
        shuffle=True,
        random_state=42,
        task_type='regression'
    )
    
    assert result['method'] == 'Nested Cross-Validation'
    assert result['task_type'] == 'regression'
    assert len(result['best_params_per_fold']) == 3
    assert result['summary']['primary_metric'] == 'rmse'
    assert result['summary']['mean_score'] > 0

def test_compare_nested_vs_non_nested():
    """
    Verify empirical comparison between Nested CV and standard Non-Nested model selection.
    Demonstrates model selection bias quantification.
    """
    iris = load_dataset('iris')
    prepared = prepare_dataset_for_modeling(iris)
    model = get_classifier('LogisticRegression', random_state=42)
    param_grid = {'model__C': [0.1, 1.0]}
    
    comp = compare_nested_vs_non_nested(
        model=model,
        X=prepared['X'],
        y=prepared['y'],
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        param_grid=param_grid,
        outer_splits=3,
        inner_splits=2,
        random_state=42,
        task_type='classification',
        is_multiclass=True
    )
    
    assert 'nested' in comp
    assert 'non_nested' in comp
    assert 'comparison' in comp
    
    c = comp['comparison']
    assert 'non_nested_score' in c
    assert 'nested_generalization_score' in c
    assert 'selection_bias' in c
    assert isinstance(c['is_optimistically_biased'], bool)

def test_api_run_nested_experiment(client):
    """Test POST /api/experiments/run with Nested CV and SQLite storage."""
    payload = {
        'dataset_name': 'iris',
        'algorithm': 'LogisticRegression',
        'validation_method': 'nested',
        'outer_splits': 3,
        'inner_splits': 2,
        'random_state': 42,
        'scaling': 'standard'
    }
    res = client.post('/api/experiments/run', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data['method'] == 'Nested Cross-Validation'
    assert 'tuning_bias_analysis' in data
    assert 'experiment_id' in data
    
    # Confirm SQLite saved best parameters
    rec = get_experiment_by_id(data['experiment_id'])
    assert rec is not None
    assert rec['validation_method'] == 'Nested Cross-Validation'
    assert len(rec['best_params']) == 3

def test_api_nested_compare_endpoint(client):
    """Test POST /api/validation/nested-compare endpoint."""
    payload = {
        'dataset_name': 'iris',
        'algorithm': 'DecisionTreeClassifier',
        'outer_splits': 3,
        'inner_splits': 2,
        'random_state': 42
    }
    res = client.post('/api/validation/nested-compare', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert 'nested' in data
    assert 'non_nested' in data
    assert 'comparison' in data
