"""
Unit & Integration Tests for Phase 1 - FoldWise AI Shell
Tests Flask routes, database initialization, CRUD operations, and template rendering.
"""

import os
import pytest
import tempfile
from app import app
from database.db import (
    init_db,
    save_experiment,
    get_experiments,
    get_experiment_by_id,
    delete_experiment,
    clear_all_experiments
)

@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test /api/health endpoint returns 200 OK and valid JSON."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['application'] == 'FoldWise AI'

def test_all_web_routes(client):
    """Verify that all 8 core UI pages load with HTTP 200."""
    routes = [
        ('/', 'Overview & Strategy Hub'),
        ('/datasets', 'Dataset Explorer'),
        ('/preprocessing', 'Data Preprocessing'),
        ('/validation', 'Cross-Validation Lab'),
        ('/training', 'Model Training Studio'),
        ('/comparison', 'Cross-Validation Strategy Comparison'),
        ('/history', 'Experiment History'),
        ('/settings', 'Settings & System Profile'),
    ]
    
    for route, expected_text in routes:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} failed with status {response.status_code}"
        assert expected_text in response.get_data(as_text=True), f"Expected text '{expected_text}' not found in {route}"

def test_database_crud():
    """Verify SQLite database operations: save, get, and delete."""
    init_db()
    test_id = 'test-exp-001'
    
    # Save test record
    exp_data = {
        'experiment_id': test_id,
        'dataset_name': 'iris',
        'target_column': 'target',
        'task_type': 'classification',
        'algorithm': 'LogisticRegression',
        'validation_method': 'StratifiedKFold',
        'hyperparameters': {'C': 1.0, 'solver': 'lbfgs'},
        'metrics': {'accuracy': 0.9667, 'mean_score': 0.9667, 'std_score': 0.021},
        'fold_scores': [0.9667, 1.0, 0.9333, 0.9667, 0.9667],
        'best_params': {},
        'execution_time': 0.125,
        'status': 'COMPLETED',
        'notes': 'Unit test sample experiment'
    }
    
    saved_id = save_experiment(exp_data)
    assert saved_id == test_id
    
    # Fetch by ID
    record = get_experiment_by_id(test_id)
    assert record is not None
    assert record['dataset_name'] == 'iris'
    assert record['metrics']['accuracy'] == 0.9667
    assert len(record['fold_scores']) == 5
    
    # Delete test record
    deleted = delete_experiment(test_id)
    assert deleted is True
    assert get_experiment_by_id(test_id) is None

def test_export_csv_endpoint(client):
    """Test /api/experiments/export/csv returns valid CSV format."""
    response = client.get('/api/experiments/export/csv')
    assert response.status_code == 200
    assert response.mimetype == 'text/csv'
    assert 'Experiment ID' in response.get_data(as_text=True)
