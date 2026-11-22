"""
FoldWise AI - Phase 11 Test Suite
Rigorous verification of SQLite Ledger History, Advanced Multi-Parameter Filtering,
Full-Text Search, Sorting, Filtered CSV Export, and Inspection Modals.
"""

import pytest
import uuid
from app import app
from database.db import (
    save_experiment,
    delete_experiment,
    filter_experiments,
    get_experiment_by_id,
    get_experiments
)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_filter_experiments_by_dataset():
    """
    Test 1: Verify filter_experiments accurately isolates records for a given dataset.
    """
    records = filter_experiments(dataset='iris', limit=50)
    assert isinstance(records, list)
    for r in records:
        assert r['dataset_name'] == 'iris'

def test_filter_experiments_by_method():
    """
    Test 2: Verify filter_experiments filters by cross-validation strategy.
    """
    records = filter_experiments(validation_method='stratified', limit=50)
    assert isinstance(records, list)
    for r in records:
        assert 'stratified' in r['validation_method'].lower()

def test_filter_experiments_by_algorithm():
    """
    Test 3: Verify filter_experiments filters by ML algorithm.
    """
    records = filter_experiments(algorithm='LogisticRegression', limit=50)
    assert isinstance(records, list)
    for r in records:
        assert r['algorithm'].lower() == 'logisticregression'

def test_search_experiments_query():
    """
    Test 4: Verify full-text search across ID, dataset, algorithm, and notes.
    """
    unique_tag = f"tag_{uuid.uuid4().hex[:6]}"
    test_id = f"test_{uuid.uuid4().hex[:8]}"
    
    save_experiment({
        'experiment_id': test_id,
        'dataset_name': 'breast_cancer',
        'target_column': 'target',
        'task_type': 'classification',
        'algorithm': 'RandomForestClassifier',
        'validation_method': 'Stratified K-Fold',
        'hyperparameters': {},
        'metrics': {'accuracy': 0.95},
        'fold_scores': [0.95, 0.96],
        'execution_time': 0.22,
        'notes': f"Phase 11 search test with {unique_tag}"
    })
    
    # Search by unique tag in notes
    matched = filter_experiments(search=unique_tag)
    assert len(matched) >= 1
    assert matched[0]['experiment_id'] == test_id
    
    # Cleanup
    delete_experiment(test_id)

def test_api_experiments_filtering_endpoint(client):
    """
    Test 5: Verify GET /api/experiments with query parameters returns filtered JSON.
    """
    resp = client.get('/api/experiments?dataset=iris&limit=10')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    for row in data:
        assert row['dataset_name'] == 'iris'

def test_api_get_single_experiment_detail(client):
    """
    Test 6: Verify GET /api/experiments/<id> returns complete record, and 404 for missing.
    """
    all_exps = get_experiments(limit=5)
    if all_exps:
        target_id = all_exps[0]['experiment_id']
        resp = client.get(f'/api/experiments/{target_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['experiment_id'] == target_id
        assert 'metrics' in data
        assert 'fold_scores' in data
        assert 'algorithm' in data

    # 404 for non-existent
    bad_resp = client.get('/api/experiments/non_existent_fake_id_123')
    assert bad_resp.status_code == 404

def test_api_delete_single_experiment(client):
    """
    Test 7: Verify DELETE /api/experiments/<id> removes record from SQLite database.
    """
    test_id = f"temp_del_{uuid.uuid4().hex[:8]}"
    save_experiment({
        'experiment_id': test_id,
        'dataset_name': 'iris',
        'target_column': 'target',
        'task_type': 'classification',
        'algorithm': 'LogisticRegression',
        'validation_method': 'K-Fold',
        'hyperparameters': {},
        'metrics': {'accuracy': 0.93},
        'fold_scores': [0.93],
        'execution_time': 0.05
    })
    
    # Verify it exists
    assert get_experiment_by_id(test_id) is not None
    
    # Delete via API
    del_resp = client.delete(f'/api/experiments/{test_id}')
    assert del_resp.status_code == 200
    
    # Verify it was deleted
    assert get_experiment_by_id(test_id) is None

def test_filtered_csv_export(client):
    """
    Test 8: Verify GET /api/experiments/export/csv returns valid CSV matching active filters.
    """
    resp = client.get('/api/experiments/export/csv?dataset=iris')
    assert resp.status_code == 200
    assert 'text/csv' in resp.content_type
    
    csv_text = resp.data.decode('utf-8')
    lines = [line.strip() for line in csv_text.strip().split('\n') if line.strip()]
    assert len(lines) >= 1  # Header at least
    header = lines[0].split(',')
    assert 'Experiment ID' in header[0]
    assert 'Dataset' in header[1]

def test_history_web_route_with_filters(client):
    """
    Test 9: Verify GET /history renders correctly with search and filter parameters.
    """
    resp = client.get('/history?dataset=iris&method=stratified&q=iris')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert 'Experiment History & Storage' in html
    assert 'Filtered View' in html
    assert 'historyFilterForm' in html
    assert 'experimentDetailModal' in html

def test_sorting_orders_in_filter():
    """
    Test 10: Verify sort_by parameter applies proper column sorting.
    """
    records_fast = filter_experiments(sort_by='execution_time', sort_dir='asc', limit=10)
    if len(records_fast) >= 2:
        for i in range(len(records_fast) - 1):
            assert records_fast[i]['execution_time'] <= records_fast[i+1]['execution_time']
