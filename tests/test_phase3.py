"""
Unit & Integration Tests for Phase 3 - Data Preprocessing & Zero-Leakage Pipeline Architecture
Demonstrates strict in-fold fitting, no data leakage, categorical encoding,
and chronological time-series lag generation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from app import app
from dataset.loader import load_dataset
from preprocessing.processor import (
    identify_feature_types,
    create_preprocessor,
    build_leak_free_pipeline,
    create_time_series_features,
    prepare_dataset_for_modeling
)

@pytest.fixture
def client():
    """Create Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_identify_feature_types():
    """Verify accurate separation of numeric and categorical features."""
    df = pd.DataFrame({
        'num_1': [1.0, 2.5, 3.8],
        'num_2': [10, 20, 30],
        'cat_1': ['A', 'B', 'A'],
        'cat_2': ['cat', 'dog', 'bird']
    })
    num_cols, cat_cols = identify_feature_types(df)
    assert set(num_cols) == {'num_1', 'num_2'}
    assert set(cat_cols) == {'cat_1', 'cat_2'}

def test_zero_leakage_scaler_isolation():
    """
    Rigorously verify that test fold distribution never leaks into preprocessor parameters.
    Scaler fitted on X_train must reflect X_train mean, completely unaffected by X_test.
    """
    # Create training set with mean = 10.0
    X_train = pd.DataFrame({'feature': [8.0, 10.0, 12.0]})
    # Create test set with an extreme distribution, mean = 1000.0
    X_test = pd.DataFrame({'feature': [900.0, 1000.0, 1100.0]})
    
    preprocessor = create_preprocessor(numerical_cols=['feature'], categorical_cols=[], scaling='standard')
    
    # Fit preprocessor strictly on X_train
    preprocessor.fit(X_train)
    
    # Extract learned scaler mean
    scaler_mean = preprocessor.named_transformers_['num'].named_steps['scaler'].mean_[0]
    
    # Must equal 10.0 (X_train mean)
    assert np.isclose(scaler_mean, 10.0), f"Expected scaler mean 10.0, got {scaler_mean}"
    # Must NOT equal combined dataset mean (~505.0) or test mean (1000.0)
    assert not np.isclose(scaler_mean, 1000.0)
    assert not np.isclose(scaler_mean, 505.0)
    
    # Transform test set: (900 - 10) / std, strictly using train parameters
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape == (3, 1)

def test_numerical_imputation_and_scaling():
    """Verify SimpleImputer handles NaN and StandardScaler standardizes output."""
    X = pd.DataFrame({
        'feature_1': [1.0, np.nan, 3.0, 4.0, 5.0],
        'feature_2': [10.0, 20.0, 30.0, np.nan, 50.0]
    })
    preprocessor = create_preprocessor(numerical_cols=['feature_1', 'feature_2'], categorical_cols=[], scaling='standard', imputation_strategy='mean')
    
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape == (5, 2)
    assert not np.isnan(transformed).any(), "NaN values should be imputed"
    
    # On training data, standardized features have mean ~ 0.0
    assert np.allclose(np.mean(transformed, axis=0), 0.0, atol=1e-6)

def test_categorical_one_hot_encoding_unknown_handling():
    """Verify OneHotEncoder handles unknown categories in test data without error."""
    X_train = pd.DataFrame({
        'num': [1.0, 2.0, 3.0],
        'category': ['red', 'green', 'blue']
    })
    X_test = pd.DataFrame({
        'num': [4.0, 5.0],
        'category': ['yellow', 'green']  # 'yellow' was never seen in training
    })
    
    preprocessor = create_preprocessor(numerical_cols=['num'], categorical_cols=['category'])
    preprocessor.fit(X_train)
    
    # Must transform X_test with unknown category 'yellow' smoothly (encoded as all zeros)
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()

def test_build_leak_free_pipeline_execution():
    """Verify full Pipeline fits and predicts end-to-end on real benchmark dataset."""
    iris = load_dataset('iris')
    prepared = prepare_dataset_for_modeling(iris)
    
    X = prepared['X']
    y = prepared['y']
    
    pipeline = build_leak_free_pipeline(
        model=LogisticRegression(max_iter=200),
        numerical_cols=prepared['numerical_cols'],
        categorical_cols=prepared['categorical_cols'],
        scaling='standard'
    )
    
    # Train-test split
    X_train, X_test = X.iloc[:120], X.iloc[120:]
    y_train, y_test = y.iloc[:120], y.iloc[120:]
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)
    score = pipeline.score(X_test, y_test)
    assert 0.0 <= score <= 1.0

def test_time_series_lag_chronological_integrity():
    """
    Verify time series lag feature generation:
    1. Row t lag_1 equals row t-1 target.
    2. Strictly backwards-looking (no future data in features).
    3. Chronological row ordering preserved.
    """
    ts_data = pd.DataFrame({
        'Month': ['1949-01', '1949-02', '1949-03', '1949-04', '1949-05'],
        'Passengers': [112, 118, 132, 129, 121]
    })
    
    X, y, feature_names = create_time_series_features(ts_data, target_col='Passengers', lag_orders=[1, 2], drop_na=True)
    
    # Since max lag is 2, the first 2 rows (112, 118) are dropped, leaving 3 rows
    assert len(X) == 3
    assert len(y) == 3
    assert 'Passengers_lag_1' in feature_names
    assert 'Passengers_lag_2' in feature_names
    
    # First remaining row corresponds to 1949-03 (Passengers = 132)
    # lag_1 should be 118 (1949-02), lag_2 should be 112 (1949-01)
    assert y.iloc[0] == 132
    assert X.iloc[0]['Passengers_lag_1'] == 118
    assert X.iloc[0]['Passengers_lag_2'] == 112
    
    # Next row (1949-04, 129): lag_1 = 132, lag_2 = 118
    assert y.iloc[1] == 129
    assert X.iloc[1]['Passengers_lag_1'] == 132
    assert X.iloc[1]['Passengers_lag_2'] == 118

def test_api_inspect_preprocessing(client):
    """Test /api/preprocessing/inspect endpoint for all standard datasets."""
    for ds_name in ['iris', 'breast_cancer', 'wine_quality', 'air_passengers']:
        res = client.get(f'/api/preprocessing/inspect?dataset_name={ds_name}&scaling=standard&imputation=mean')
        assert res.status_code == 200
        data = res.get_json()
        assert data['dataset_name'] == ds_name
        assert 'numerical_features' in data
        assert 'categorical_features' in data
        assert 'pipeline_architecture' in data
        assert data['transformed_features_count'] > 0
