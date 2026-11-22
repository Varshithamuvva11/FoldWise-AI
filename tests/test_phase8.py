"""
FoldWise AI - Phase 8 Test Suite
Rigorous verification of the Cross-Validation Strategy Recommender,
Multi-Strategy Empirical Benchmarking Studio, and Advanced Visualizers (Radar & Fold Matrix).
"""

import pytest
from app import app
from dataset.loader import load_dataset
from recommendation.strategy_recommender import analyze_dataset_for_validation
from validation.benchmark import benchmark_validation_strategies
from evaluation.visualizer import generate_radar_chart_payload, generate_fold_allocation_diagram

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_strategy_recommender_classification_imbalanced():
    """
    Test 1: Verify that Breast Cancer (imbalanced classification) triggers
    Stratified K-Fold recommendation and flags Standard K-Fold class dropout hazard.
    """
    ds = load_dataset('breast_cancer')
    report = analyze_dataset_for_validation(ds)
    
    assert report['task_type'] == 'classification'
    assert report['diagnostics']['is_imbalanced'] is True
    assert report['diagnostics']['imbalance_ratio'] > 1.35
    assert report['recommendation']['recommended_strategy'] == 'Stratified K-Fold'
    assert report['recommendation']['confidence_score'] >= 90
    
    # Verify compatibility matrix contains Stratified as RECOMMENDED and Time-Series as INCOMPATIBLE
    compat = {c['strategy']: c['status'] for c in report['compatibility_matrix']}
    assert 'RECOMMENDED' in compat['Stratified K-Fold']
    assert 'INCOMPATIBLE' in compat['Time-Series Split']
    
    # Verify critical hazards exist
    hazards = [h['strategy'] for h in report['hazards']]
    assert 'Standard K-Fold' in hazards

def test_strategy_recommender_classification_balanced():
    """
    Test 2: Verify that balanced multi-class dataset (Iris) recommends Stratified K-Fold.
    """
    ds = load_dataset('iris')
    report = analyze_dataset_for_validation(ds)
    
    assert report['task_type'] == 'classification'
    assert report['diagnostics']['is_imbalanced'] is False
    assert report['recommendation']['recommended_strategy'] == 'Stratified K-Fold'
    assert report['recommendation']['recommended_k'] in [5, 10]

def test_strategy_recommender_regression():
    """
    Test 3: Verify that continuous target (Wine Quality) triggers Standard K-Fold
    and flags Stratified K-Fold as incompatible.
    """
    ds = load_dataset('wine_quality')
    report = analyze_dataset_for_validation(ds)
    
    assert report['task_type'] == 'regression'
    assert report['recommendation']['recommended_strategy'] == 'Standard K-Fold'
    
    compat = {c['strategy']: c['status'] for c in report['compatibility_matrix']}
    assert 'RECOMMENDED' in compat['Standard K-Fold']
    assert 'INCOMPATIBLE' in compat['Stratified K-Fold']

def test_strategy_recommender_time_series():
    """
    Test 4: Verify that Airline Passengers triggers Time-Series Cross-Validation
    and flags Standard K-Fold as CRITICAL HAZARDOUS due to lookahead leakage.
    """
    ds = load_dataset('air_passengers')
    report = analyze_dataset_for_validation(ds)
    
    assert report['task_type'] == 'time_series'
    assert report['recommendation']['recommended_strategy'] == 'Time-Series Cross-Validation'
    assert report['diagnostics']['is_sequential'] is True
    
    compat = {c['strategy']: c['status'] for c in report['compatibility_matrix']}
    assert 'RECOMMENDED' in compat['Time-Series Cross-Validation']
    assert 'INCOMPATIBLE' in compat['Standard K-Fold'] or 'HAZARDOUS' in compat['Standard K-Fold']

def test_multi_strategy_benchmarking_classification():
    """
    Test 5: Verify that multi-strategy benchmarking runs Stratified, Standard,
    and Nested CV on the exact same classification model (Logistic Regression).
    """
    res = benchmark_validation_strategies(
        dataset_name='breast_cancer',
        algorithm='LogisticRegression',
        n_splits=3,
        random_state=42
    )
    
    assert res['dataset_name'] == 'breast_cancer'
    assert res['algorithm'] == 'LogisticRegression'
    assert 'Stratified K-Fold' in res['strategies']
    assert 'Standard K-Fold' in res['strategies']
    assert 'Nested Cross-Validation' in res['strategies']
    assert len(res['takeaways']) > 0

def test_multi_strategy_benchmarking_time_series():
    """
    Test 6: Verify multi-strategy benchmarking on Airline Passengers directly compares
    Time-Series CV against naive Random K-Fold and calculates leakage gap.
    """
    res = benchmark_validation_strategies(
        dataset_name='air_passengers',
        algorithm='LinearRegression',
        n_splits=3,
        random_state=42
    )
    
    assert res['task_type'] == 'time_series'
    assert 'Time-Series Cross-Validation' in res['strategies']
    assert 'Random K-Fold (Flawed)' in res['strategies']
    assert any('leakage' in t.lower() or 'artificially' in t.lower() for t in res['takeaways'])

def test_radar_chart_payload_generation():
    """
    Test 7: Verify generation of 360-degree radar/spider chart traces
    for Plotly visualization.
    """
    mock_rankings = [
        {
            'algorithm': 'LogisticRegression',
            'score': 0.95,
            'std': 0.02,
            'execution_time': 0.1,
            'metrics_summary': {'mean_accuracy': 0.95, 'mean_precision': 0.94, 'mean_recall': 0.96, 'mean_f1_score': 0.95}
        },
        {
            'algorithm': 'DecisionTreeClassifier',
            'score': 0.91,
            'std': 0.04,
            'execution_time': 0.05,
            'metrics_summary': {'mean_accuracy': 0.91, 'mean_precision': 0.90, 'mean_recall': 0.92, 'mean_f1_score': 0.91}
        }
    ]
    
    radar = generate_radar_chart_payload(mock_rankings, task_type='classification')
    assert 'categories' in radar
    assert len(radar['categories']) == 6
    assert len(radar['traces']) == 2
    
    # Check that trace closed loop is enforced (length of r equals length of categories + 1)
    trace0 = radar['traces'][0]
    assert len(trace0['r']) == len(radar['categories']) + 1
    assert trace0['r'][0] == trace0['r'][-1]

def test_fold_allocation_diagram():
    """
    Test 8: Verify fold allocation diagram generator calculates correct sample slices.
    """
    alloc = generate_fold_allocation_diagram('iris', validation_method='kfold', n_splits=5)
    assert alloc['n_samples'] == 150
    assert alloc['n_splits'] == 5
    assert len(alloc['folds']) == 5
    
    for f in alloc['folds']:
        assert f['train_count'] > 0
        assert f['val_count'] > 0
        assert f['train_count'] + f['val_count'] == 150

def test_api_recommend_strategy_endpoint(client):
    """
    Test 9: Verify POST /api/recommendation/strategy returns 200 with diagnostics.
    """
    resp = client.post('/api/recommendation/strategy', json={'dataset_name': 'breast_cancer'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'diagnostics' in data
    assert 'recommendation' in data
    assert 'compatibility_matrix' in data
    assert data['recommendation']['recommended_strategy'] == 'Stratified K-Fold'

def test_api_benchmark_strategies_endpoint(client):
    """
    Test 10: Verify POST /api/validation/strategy-benchmark executes and returns 200.
    """
    resp = client.post('/api/validation/strategy-benchmark', json={
        'dataset_name': 'iris',
        'algorithm': 'LogisticRegression',
        'n_splits': 3
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'strategies' in data
    assert 'takeaways' in data

def test_api_fold_allocation_endpoint(client):
    """
    Test 11: Verify GET /api/visualization/fold-allocation returns 200 and valid matrix.
    """
    resp = client.get('/api/visualization/fold-allocation?dataset=breast_cancer&method=stratified&n_splits=4')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['n_samples'] == 569
    assert len(data['folds']) == 4
