"""
Unit & Integration Tests for Phase 2 - Dataset Loading, Caching, and Analysis
Tests Iris, Breast Cancer, Wine Quality, Airline Passengers, caching, and custom CSV upload.
"""

import os
import io
import pytest
import pandas as pd
from app import app
from dataset.loader import load_dataset, list_registered_datasets, save_uploaded_csv, CACHE_DIR
from dataset.analyzer import analyze_dataset

@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_load_iris_builtin():
    """Test loading Iris flower dataset."""
    data = load_dataset('iris')
    assert data['id'] == 'iris'
    assert data['source'] == 'builtin'
    assert data['n_samples'] == 150
    assert data['n_features'] == 4
    assert data['is_multiclass'] is True
    assert data['task_type'] == 'classification'
    assert data['target'] == 'target'
    assert isinstance(data['df'], pd.DataFrame)
    assert data['df'].shape == (150, 5)

def test_load_breast_cancer_builtin():
    """Test loading Breast Cancer Wisconsin dataset."""
    data = load_dataset('breast_cancer')
    assert data['id'] == 'breast_cancer'
    assert data['source'] == 'builtin'
    assert data['n_samples'] == 569
    assert data['n_features'] == 30
    assert data['is_multiclass'] is False
    assert data['task_type'] == 'classification'
    assert data['target'] == 'target'
    assert data['df'].shape == (569, 31)

def test_load_wine_quality_cached():
    """Test loading Wine Quality dataset from local cache."""
    data = load_dataset('wine_quality')
    assert data['id'] == 'wine_quality'
    assert data['source'] in ['cache', 'downloaded']
    assert data['n_samples'] == 1599
    assert data['n_features'] == 11
    assert data['target'] == 'quality'
    assert data['task_type'] == 'regression'
    assert os.path.exists(os.path.join(CACHE_DIR, 'winequality-red.csv'))

def test_load_air_passengers_cached():
    """Test loading Air Passengers dataset from local cache."""
    data = load_dataset('air_passengers')
    assert data['id'] == 'air_passengers'
    assert data['source'] in ['cache', 'downloaded']
    assert data['n_samples'] == 144
    assert data['target'] == 'Passengers'
    assert data['task_type'] == 'time_series'
    assert os.path.exists(os.path.join(CACHE_DIR, 'airline-passengers.csv'))

def test_dataset_analyzer():
    """Verify statistical analysis and distribution engine."""
    iris_data = load_dataset('iris')
    analysis = analyze_dataset(iris_data['df'], target_column='target', task_type='classification')
    
    assert analysis['shape']['rows'] == 150
    assert analysis['shape']['columns'] == 5
    assert len(analysis['preview']) == 10
    assert len(analysis['columns_info']) == 5
    assert analysis['duplicates']['count'] >= 0
    assert 'sepal length (cm)' in analysis['numerical_summary']
    assert analysis['numerical_summary']['sepal length (cm)']['count'] == 150
    assert len(analysis['correlation_matrix']['columns']) >= 4
    # Check 3 classes in target distribution
    assert len(analysis['target_distribution']['values']) == 3

def test_api_list_datasets(client):
    """Test GET /api/datasets returns registered benchmarks."""
    res = client.get('/api/datasets')
    assert res.status_code == 200
    data = res.get_json()
    assert 'iris' in data
    assert 'breast_cancer' in data
    assert 'wine_quality' in data
    assert 'air_passengers' in data

def test_api_get_dataset_details(client):
    """Test GET /api/datasets/<name> for each benchmark."""
    for ds_name in ['iris', 'breast_cancer', 'wine_quality', 'air_passengers']:
        res = client.get(f'/api/datasets/{ds_name}')
        assert res.status_code == 200, f"Failed for dataset {ds_name}"
        data = res.get_json()
        assert data['id'] == ds_name
        assert 'analysis' in data
        assert data['n_samples'] > 0
        assert data['n_features'] > 0

def test_custom_csv_upload(client):
    """Test uploading a custom CSV file."""
    # Create an in-memory sample CSV
    csv_content = b"feature_a,feature_b,feature_c,label\n1.2,3.4,5.6,0\n2.1,4.3,6.5,1\n3.0,5.0,7.0,0\n4.0,6.0,8.0,1\n"
    data = {
        'file': (io.BytesIO(csv_content), 'test_sample_dataset.csv'),
        'target_column': 'label'
    }
    res = client.post('/api/datasets/upload', data=data, content_type='multipart/form-data')
    assert res.status_code == 201
    res_data = res.get_json()
    assert res_data['samples'] == 4
    assert res_data['features'] == 3
    assert res_data['target'] == 'label'
    
    # Test loading the newly uploaded dataset
    load_res = client.get(f"/api/datasets/{res_data['dataset_id']}")
    assert load_res.status_code == 200
    load_json = load_res.get_json()
    assert load_json['n_samples'] == 4
    assert load_json['source'] == 'uploaded'

def test_invalid_dataset_name(client):
    """Test error handling when requesting non-existent dataset."""
    res = client.get('/api/datasets/non_existent_dataset_12345')
    assert res.status_code == 400
    data = res.get_json()
    assert 'error' in data
