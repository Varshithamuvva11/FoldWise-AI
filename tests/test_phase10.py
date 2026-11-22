"""
FoldWise AI - Phase 10 Test Suite
Rigorous verification of Frontend & Backend Integration Polish, Form Validations,
Error Toasts & Uniform Error Payloads, Responsive CSS, and Template Stability.
"""

import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_invalid_fold_count_validation(client):
    """
    Test 1: Verify API rejects n_splits < 2 or > 50 with 400 Bad Request and descriptive message.
    """
    resp_low = client.post('/api/experiments/run', json={
        'dataset_name': 'iris',
        'validation_method': 'kfold',
        'n_splits': 1
    })
    assert resp_low.status_code == 400
    assert 'Number of splits' in resp_low.get_json()['error']

    resp_high = client.post('/api/experiments/run', json={
        'dataset_name': 'iris',
        'validation_method': 'kfold',
        'n_splits': 99
    })
    assert resp_high.status_code == 400
    assert 'Number of splits' in resp_high.get_json()['error']

def test_nested_splits_validation(client):
    """
    Test 2: Verify Nested CV rejects outer or inner splits < 2.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'iris',
        'validation_method': 'nested',
        'outer_splits': 1,
        'inner_splits': 3
    })
    assert resp.status_code == 400
    assert 'Nested CV' in resp.get_json()['error']

def test_timeseries_negative_gap_validation(client):
    """
    Test 3: Verify Time-Series validation rejects negative horizon gaps.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'air_passengers',
        'algorithm': 'LinearRegression',
        'validation_method': 'timeseries',
        'gap': -2
    })
    assert resp.status_code == 400
    assert 'gap' in resp.get_json()['error'].lower()

def test_comparison_empty_algorithms_validation(client):
    """
    Test 4: Verify Comparison engine rejects empty algorithm list.
    """
    resp = client.post('/api/comparison/run', json={
        'dataset_name': 'breast_cancer',
        'algorithms': []
    })
    assert resp.status_code == 400
    assert 'algorithm' in resp.get_json()['error'].lower()

def test_comparison_invalid_splits_validation(client):
    """
    Test 5: Verify Comparison engine rejects n_splits < 2.
    """
    resp = client.post('/api/comparison/run', json={
        'dataset_name': 'iris',
        'algorithms': ['LogisticRegression'],
        'n_splits': 0
    })
    assert resp.status_code == 400
    assert 'Number of splits' in resp.get_json()['error']

def test_benchmark_invalid_splits_validation(client):
    """
    Test 6: Verify Strategy Benchmark studio rejects invalid fold counts.
    """
    resp = client.post('/api/validation/strategy-benchmark', json={
        'dataset_name': 'breast_cancer',
        'n_splits': 1
    })
    assert resp.status_code == 400
    assert 'Number of splits' in resp.get_json()['error']

def test_incompatible_strategy_error_handling(client):
    """
    Test 7: Verify attempting Stratified K-Fold on continuous regression target
    returns 400 with a clear methodological explanation.
    """
    resp = client.post('/api/experiments/run', json={
        'dataset_name': 'wine_quality',
        'validation_method': 'stratified',
        'algorithm': 'LinearRegression'
    })
    assert resp.status_code == 400
    assert 'Stratified K-Fold is only applicable' in resp.get_json()['error']

def test_static_assets_and_toast_css(client):
    """
    Test 8: Verify CSS and JS assets are accessible and contain toast and responsive rules.
    """
    css_resp = client.get('/static/css/style.css')
    assert css_resp.status_code == 200
    css_text = css_resp.data.decode('utf-8')
    assert '.toast' in css_text
    assert '.toast-container' in css_text
    assert '@media (max-width: 768px)' in css_text

    js_resp = client.get('/static/js/app.js')
    assert js_resp.status_code == 200
    js_text = js_resp.data.decode('utf-8')
    assert 'function showToast' in js_text
    assert 'initSidebarToggle' in js_text

def test_all_web_pages_render_without_error(client):
    """
    Test 9: Verify all primary web views render clean HTML with 200 status code.
    """
    pages = [
        ('/', 'Overview & Strategy Hub'),
        ('/datasets', 'Dataset Explorer'),
        ('/preprocessing', 'Data Preprocessing'),
        ('/validation', 'Cross-Validation Lab'),
        ('/training', 'Model Training Studio'),
        ('/comparison', 'Model Comparison'),
        ('/history', 'Experiment History'),
        ('/settings', 'Settings')
    ]
    for url, expected_text in pages:
        res = client.get(url)
        assert res.status_code == 200, f"Page {url} failed with status {res.status_code}"
        html = res.data.decode('utf-8')
        assert expected_text in html, f"Expected '{expected_text}' in page {url}"

def test_standardized_json_error_format(client):
    """
    Test 10: Verify API error responses consistently include an 'error' key.
    """
    # 1. Invalid dataset name
    resp1 = client.get('/api/datasets/non_existent_dataset_xyz')
    assert resp1.status_code == 404 or resp1.status_code == 400
    assert 'error' in resp1.get_json()

    # 2. Invalid validation method
    resp2 = client.post('/api/experiments/run', json={
        'dataset_name': 'iris',
        'validation_method': 'unknown_unsupported_strategy'
    })
    assert resp2.status_code == 400
    assert 'error' in resp2.get_json()
