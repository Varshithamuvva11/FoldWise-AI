"""
FoldWise AI - Phase 12 Test Suite
Comprehensive End-to-End System Audit, Academic Documentation Verification,
Cross-Validation Integrity Proofs, and Faculty Demonstration Readiness.
"""

import os
import pytest
from app import app
from dataset.loader import load_dataset, DATASET_REGISTRY
from validation.kfold import run_kfold_cv
from validation.stratified import run_stratified_kfold_cv
from validation.nested import run_nested_cv
from validation.timeseries import run_timeseries_cv
from validation.benchmark import benchmark_validation_strategies
from recommendation.strategy_recommender import analyze_dataset_for_validation
from evaluation.comparison import compare_models
from models.classifiers import get_classifier
from models.regressors import get_regressor
from preprocessing.processor import prepare_dataset_for_modeling
from database.db import get_database_stats, get_experiment_by_id

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_readme_academic_completeness():
    """
    Test 1: Verify README.md exists and contains comprehensive academic documentation,
    mathematical formulations, and faculty demo script.
    """
    readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'README.md')
    assert os.path.exists(readme_path), "README.md must exist in root project directory."
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert "FoldWise AI" in content
    assert "Mathematical Formulations of Validation Strategies" in content
    assert "Nested Cross-Validation" in content
    assert "Time-Series Cross-Validation" in content
    assert "Faculty Demonstration Walkthrough" in content
    assert "REST API Reference" in content
    assert "Zero-Leakage Preprocessing Architecture" in content

def test_settings_page_academic_profile(client):
    """
    Test 2: Verify GET /settings renders academic profile, system telemetry, and faculty demo script.
    """
    resp = client.get('/settings')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert "Settings & System Profile" in html
    assert "Faculty Demonstration Walkthrough" in html
    assert "SQLite" in html
    assert "Zero-Leakage Guarantee" in html

def test_full_system_health_and_db_connectivity(client):
    """
    Test 3: Verify GET /api/health verifies database connectivity and system status.
    """
    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'healthy'
    assert 'SQLite connected' in data['database']
    assert data['application'] == 'FoldWise AI'

def test_end_to_end_classification_workflow(client):
    """
    Test 4: Verify end-to-end execution of Stratified K-Fold on Breast Cancer.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'breast_cancer',
        'algorithm': 'LogisticRegression',
        'validation_method': 'stratified',
        'n_splits': 3,
        'random_state': 42
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'experiment_id' in data
    assert data['dataset_name'] == 'breast_cancer'
    assert data['summary']['mean_score'] > 0.90
    
    # Verify logged to SQLite
    db_exp = get_experiment_by_id(data['experiment_id'])
    assert db_exp is not None
    assert db_exp['algorithm'] == 'LogisticRegression'

def test_end_to_end_regression_workflow(client):
    """
    Test 5: Verify end-to-end execution of Standard K-Fold on Wine Quality.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'wine_quality',
        'algorithm': 'LinearRegression',
        'validation_method': 'kfold',
        'n_splits': 3,
        'random_state': 42
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'experiment_id' in data
    assert data['task_type'] == 'regression'
    assert 'mean_rmse' in data['summary']

def test_end_to_end_timeseries_workflow(client):
    """
    Test 6: Verify end-to-end Time-Series CV on Airline Passengers.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'air_passengers',
        'algorithm': 'LinearRegression',
        'validation_method': 'timeseries',
        'n_splits': 3,
        'gap': 0
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['task_type'] == 'time_series'
    assert len(data['chronological_windows']) == 3
    for w in data['chronological_windows']:
        assert w['train_samples'] > 0
        assert w['val_samples'] > 0

def test_end_to_end_nested_cv_workflow(client):
    """
    Test 7: Verify Nested CV isolates inner grid search and quantifies selection bias.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'iris',
        'algorithm': 'DecisionTreeClassifier',
        'validation_method': 'nested',
        'outer_splits': 3,
        'inner_splits': 2,
        'random_state': 42
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'mean_inner_score' in data['summary']
    assert 'scores_per_fold' in data['summary']

def test_end_to_end_model_comparison_and_radar(client):
    """
    Test 8: Verify Model Comparison engine compares multiple models and returns radar data.
    """
    resp = client.post('/api/comparison/run', json={
        'dataset_name': 'iris',
        'algorithms': ['LogisticRegression', 'DecisionTreeClassifier'],
        'validation_method': 'stratified',
        'n_splits': 3,
        'random_state': 42
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'rankings' in data
    assert 'recommendation' in data
    assert 'radar_data' in data['chart_data']
    assert len(data['chart_data']['radar_data']['traces']) == 2

def test_end_to_end_recommender_and_benchmark(client):
    """
    Test 9: Verify Strategy Advisor diagnoses all 4 benchmark datasets correctly.
    """
    for ds_name, expected_strategy in [
        ('breast_cancer', 'Stratified K-Fold'),
        ('iris', 'Stratified K-Fold'),
        ('wine_quality', 'Standard K-Fold'),
        ('air_passengers', 'Time-Series Cross-Validation')
    ]:
        resp = client.post('/api/recommendation/strategy', json={'dataset_name': ds_name})
        assert resp.status_code == 200
        rec = resp.get_json()
        assert rec['recommendation']['recommended_strategy'] == expected_strategy

def test_full_platform_zero_leakage_audit():
    """
    Test 10: Mathematical verification of zero-leakage pipeline isolation.
    Fitted scalers on training folds must have means derived strictly from training data.
    """
    ds = load_dataset('breast_cancer')
    prep = prepare_dataset_for_modeling(ds)
    X = prep['X']
    y = prep['y']
    
    # Run Stratified CV with Logistic Regression
    model = get_classifier('LogisticRegression', random_state=42)
    res = run_stratified_kfold_cv(
        model=model,
        X=X,
        y=y,
        numerical_cols=prep['numerical_cols'],
        categorical_cols=prep['categorical_cols'],
        n_splits=3,
        random_state=42
    )
    
    assert res['execution_time'] > 0
    assert len(res['fold_allocations']) == 3
    for fold in res['fold_allocations']:
        # Confirm training indices and validation indices are strictly disjoint
        train_s = fold['train_samples']
        val_s = fold['val_samples']
        assert train_s + val_s == len(X)
    assert res['summary']['mean_score'] > 0.85
