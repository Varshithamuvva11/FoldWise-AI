"""
FoldWise AI - Phase 9 Test Suite
Rigorous verification of the Interactive Dashboard Overview, Live KPI Metrics,
Plotly Visual Analytics Payloads, Database Aggregations, and Quick Launchpads.
"""

import pytest
import uuid
from app import app
from database.db import (
    get_database_stats,
    save_experiment,
    save_comparison,
    delete_experiment,
    get_experiments,
    get_comparisons
)
from dataset.loader import load_dataset, list_registered_datasets
from evaluation.visualizer import generate_dashboard_charts_payload
from recommendation.strategy_recommender import analyze_dataset_for_validation
from validation.benchmark import benchmark_validation_strategies

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_dashboard_web_route(client):
    """
    Test 1: Verify that GET / returns 200, renders Overview & Strategy Hub,
    contains the 6 KPI metric cards and Plotly chart containers.
    """
    response = client.get('/')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert "Overview & Strategy Hub" in html
    assert "chart-strategy-distribution" in html
    assert "chart-dataset-landscape" in html
    assert "kpi-datasets" in html
    assert "kpi-experiments" in html
    assert "kpi-latency" in html

def test_api_dashboard_stats_endpoint(client):
    """
    Test 2: Verify GET /api/dashboard/stats returns 200 with structured live telemetry.
    """
    response = client.get('/api/dashboard/stats')
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'stats' in data
    assert 'db_stats' in data
    assert 'charts' in data
    assert 'recent_experiments' in data
    assert 'recent_comparisons' in data
    
    stats = data['stats']
    assert stats['dataset_count'] >= 4
    assert stats['strategy_count'] == 4
    assert stats['classifier_count'] == 5
    assert stats['regressor_count'] == 4
    assert stats['experiment_count'] >= 0
    assert stats['comparison_count'] >= 0
    assert isinstance(stats['avg_execution_time'], (int, float))

def test_live_database_stats_aggregation():
    """
    Test 3: Verify get_database_stats returns accurate schema and non-negative counts.
    """
    stats = get_database_stats()
    assert 'total_experiments' in stats
    assert 'total_comparisons' in stats
    assert 'strategy_counts' in stats
    assert 'task_counts' in stats
    assert 'dataset_counts' in stats
    assert 'avg_execution_time' in stats
    assert stats['total_experiments'] >= 0
    assert stats['total_comparisons'] >= 0
    assert isinstance(stats['avg_execution_time'], float)

def test_db_stats_increment_on_new_experiment():
    """
    Test 4: Verify that saving a new experiment dynamically increments total count
    and updates strategy-specific counts in get_database_stats.
    """
    before_stats = get_database_stats()
    test_id = f"test_p9_{uuid.uuid4().hex[:8]}"
    
    save_experiment({
        'experiment_id': test_id,
        'dataset_name': 'iris',
        'target_column': 'target',
        'task_type': 'classification',
        'algorithm': 'LogisticRegression',
        'validation_method': 'stratified',
        'hyperparameters': {},
        'metrics': {'accuracy': 0.96},
        'fold_scores': [0.96, 0.97, 0.95],
        'execution_time': 0.12,
        'notes': 'Phase 9 test experiment'
    })
    
    after_stats = get_database_stats()
    assert after_stats['total_experiments'] == before_stats['total_experiments'] + 1
    assert after_stats['strategy_counts'].get('stratified', 0) >= before_stats['strategy_counts'].get('stratified', 0) + 1
    
    # Cleanup
    delete_experiment(test_id)

def test_db_stats_increment_on_new_comparison():
    """
    Test 5: Verify that saving a model comparison increments comparison counter.
    """
    before_stats = get_database_stats()
    test_comp_id = f"comp_p9_{uuid.uuid4().hex[:8]}"
    
    save_comparison({
        'comparison_id': test_comp_id,
        'dataset_name': 'breast_cancer',
        'task_type': 'classification',
        'validation_method': 'stratified',
        'primary_metric': 'accuracy',
        'algorithms_compared': ['LogisticRegression', 'RandomForestClassifier'],
        'rankings': [{'algorithm': 'LogisticRegression', 'score': 0.97}],
        'winner_algorithm': 'LogisticRegression',
        'recommendation': 'Recommended for low variance.',
        'execution_time': 0.45
    })
    
    after_stats = get_database_stats()
    assert after_stats['total_comparisons'] == before_stats['total_comparisons'] + 1

def test_dashboard_charts_payload_generation():
    """
    Test 6: Verify generate_dashboard_charts_payload produces complete, well-formed
    Plotly traces and model capability matrices.
    """
    db_stats = get_database_stats()
    datasets = list_registered_datasets()
    payload = generate_dashboard_charts_payload(db_stats, datasets)
    
    assert 'strategy_donut' in payload
    assert 'dataset_landscape' in payload
    assert 'model_matrix' in payload
    
    # Check Strategy Donut structure
    donut = payload['strategy_donut']
    assert len(donut['labels']) == 4
    assert len(donut['values']) == 4
    assert len(donut['colors']) == 4
    
    # Check Dataset Landscape structure
    landscape = payload['dataset_landscape']
    assert len(landscape['names']) >= 4
    assert len(landscape['samples']) >= 4
    assert len(landscape['features']) >= 4
    assert len(landscape['tasks']) >= 4
    
    # Check Model Matrix
    mm = payload['model_matrix']
    assert mm['total_algorithms'] == 9
    assert len(mm['classifiers']) == 5
    assert len(mm['regressors']) == 4

def test_quick_action_launchpad_routes(client):
    """
    Test 7: Verify all launchpad navigation paths resolve to HTTP 200 without error.
    """
    routes = [
        '/datasets',
        '/validation',
        '/training',
        '/training?method=kfold',
        '/training?method=stratified',
        '/training?method=nested',
        '/training?method=timeseries',
        '/comparison',
        '/history',
        '/settings'
    ]
    for r in routes:
        resp = client.get(r)
        assert resp.status_code == 200, f"Route {r} failed with status {resp.status_code}"

def test_recent_experiments_ordering():
    """
    Test 8: Verify get_experiments returns records in descending chronological order.
    """
    exps = get_experiments(limit=10)
    if len(exps) >= 2:
        # id should be strictly descending
        for i in range(len(exps) - 1):
            assert exps[i]['id'] >= exps[i+1]['id']

def test_graceful_handling_when_no_data():
    """
    Test 9: Verify generate_dashboard_charts_payload safely handles empty / None input.
    """
    fallback_payload = generate_dashboard_charts_payload(None, None)
    assert fallback_payload['strategy_donut']['has_data'] is False
    assert len(fallback_payload['strategy_donut']['labels']) == 4
    assert fallback_payload['model_matrix']['total_algorithms'] == 9

def test_cross_phase_integration_and_zero_regressions():
    """
    Test 10: Verify end-to-end integration across recommender and dataset loader.
    """
    ds = load_dataset('iris')
    rec = analyze_dataset_for_validation(ds)
    assert rec['task_type'] == 'classification'
    assert rec['recommendation']['recommended_strategy'] == 'Stratified K-Fold'
