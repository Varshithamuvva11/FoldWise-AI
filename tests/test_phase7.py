"""
FoldWise AI - Phase 7 Test Suite
Comprehensive testing of Model Comparison Engine, Multi-Algorithm Cross-Validation,
Automated Model Ranking, Explainable Recommendation Engine, and SQLite Persistence.
"""

import pytest
from app import app
from evaluation.comparison import (
    compare_models,
    rank_and_recommend,
    validate_strategy_compatibility
)
from database.db import get_comparison_by_id, get_comparisons

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_classification_model_comparison():
    """
    Test 1: Verify multi-model comparison on a classification dataset (Iris)
    using Stratified K-Fold.
    """
    algos = ['LogisticRegression', 'DecisionTreeClassifier', 'SVC']
    res = compare_models(
        dataset_name='iris',
        algorithms=algos,
        validation_method='stratified',
        n_splits=3,
        primary_metric='accuracy',
        random_state=42
    )
    
    assert res['dataset_name'] == 'iris'
    assert res['task_type'] == 'classification'
    assert res['validation_method'] == 'stratified'
    assert res['models_compared_count'] == len(algos)
    assert len(res['rankings']) == len(algos)
    assert 'recommendation' in res
    assert res['recommendation']['winner_algorithm'] in algos
    
    # Check that each model has fold metrics and metrics summary
    for rank_entry in res['rankings']:
        assert rank_entry['primary_metric'] == 'accuracy'
        assert 0.0 <= rank_entry['score'] <= 1.0
        assert rank_entry['std'] >= 0.0
        assert len(rank_entry['scores_per_fold']) == 3

def test_regression_model_comparison():
    """
    Test 2: Verify multi-model comparison on a regression dataset (Wine Quality)
    using standard K-Fold and RMSE ranking.
    """
    algos = ['LinearRegression', 'RandomForestRegressor']
    res = compare_models(
        dataset_name='wine_quality',
        algorithms=algos,
        validation_method='kfold',
        n_splits=3,
        primary_metric='rmse',
        random_state=42
    )
    
    assert res['task_type'] == 'regression'
    assert res['primary_metric'] == 'rmse'
    assert len(res['rankings']) == len(algos)
    
    # In RMSE, lower is better. First ranked model should have <= RMSE than second
    rankings = res['rankings']
    assert rankings[0]['score'] <= rankings[1]['score']
    assert rankings[0]['rank'] == 1
    assert rankings[1]['rank'] == 2

def test_all_five_classifiers_execution():
    """
    Test 3: Verify that all 5 supported classification algorithms run successfully
    on Breast Cancer Wisconsin.
    """
    all_five = [
        'LogisticRegression',
        'DecisionTreeClassifier',
        'RandomForestClassifier',
        'SVC',
        'KNeighborsClassifier'
    ]
    res = compare_models(
        dataset_name='breast_cancer',
        algorithms=all_five,
        validation_method='stratified',
        n_splits=3,
        random_state=42
    )
    
    assert res['models_compared_count'] == 5
    evaluated_algos = [m['algorithm'] for m in res['rankings']]
    for a in all_five:
        assert a in evaluated_algos

def test_metric_calculations_mathematical_consistency():
    """
    Test 4: Verify mathematical consistency of aggregated metrics:
    min <= mean <= max and standard deviation >= 0.
    """
    res = compare_models(
        dataset_name='iris',
        algorithms=['LogisticRegression'],
        validation_method='stratified',
        n_splits=4,
        random_state=42
    )
    
    m = res['rankings'][0]
    assert m['min_score'] <= m['score'] <= m['max_score']
    assert m['std'] >= 0.0
    assert len(m['scores_per_fold']) == 4

def test_model_ranking_order_correctness():
    """
    Test 5: Verify that rank_and_recommend strictly orders models:
    - Descending for accuracy (higher is better)
    - Ascending for RMSE (lower is better)
    """
    # Dummy mock results for higher-is-better
    mock_class_results = {
        'ModelA': {'summary': {'mean_score': 0.85, 'std_score': 0.02, 'scores_per_fold': [0.83, 0.87]}, 'execution_time': 0.1},
        'ModelB': {'summary': {'mean_score': 0.95, 'std_score': 0.01, 'scores_per_fold': [0.94, 0.96]}, 'execution_time': 0.2},
        'ModelC': {'summary': {'mean_score': 0.90, 'std_score': 0.03, 'scores_per_fold': [0.87, 0.93]}, 'execution_time': 0.05}
    }
    
    ranked_class = rank_and_recommend(mock_class_results, task_type='classification', primary_metric='accuracy')
    r_list = ranked_class['rankings']
    assert r_list[0]['algorithm'] == 'ModelB'
    assert r_list[1]['algorithm'] == 'ModelC'
    assert r_list[2]['algorithm'] == 'ModelA'
    assert ranked_class['recommendation']['winner_algorithm'] == 'ModelB'
    
    # Dummy mock results for lower-is-better
    mock_reg_results = {
        'ModelX': {'summary': {'mean_score': 50.0, 'std_score': 5.0, 'scores_per_fold': [45.0, 55.0]}, 'execution_time': 0.1},
        'ModelY': {'summary': {'mean_score': 20.0, 'std_score': 2.0, 'scores_per_fold': [18.0, 22.0]}, 'execution_time': 0.2},
        'ModelZ': {'summary': {'mean_score': 35.0, 'std_score': 3.0, 'scores_per_fold': [32.0, 38.0]}, 'execution_time': 0.05}
    }
    
    ranked_reg = rank_and_recommend(mock_reg_results, task_type='regression', primary_metric='rmse')
    r_reg_list = ranked_reg['rankings']
    assert r_reg_list[0]['algorithm'] == 'ModelY'  # 20.0 lowest error
    assert r_reg_list[1]['algorithm'] == 'ModelZ'  # 35.0
    assert r_reg_list[2]['algorithm'] == 'ModelX'  # 50.0
    assert ranked_reg['recommendation']['winner_algorithm'] == 'ModelY'

def test_recommendation_engine_narrative_generation():
    """
    Test 6: Verify that the recommendation engine produces detailed, explainable narratives
    including performance margin, stability analysis, and efficiency notes.
    """
    res = compare_models(
        dataset_name='breast_cancer',
        algorithms=['LogisticRegression', 'DecisionTreeClassifier'],
        validation_method='stratified',
        n_splits=3,
        random_state=42
    )
    
    rec = res['recommendation']
    assert 'summary_headline' in rec
    assert 'detailed_narrative' in rec
    assert 'deployment_verdict' in rec
    assert rec['winner_algorithm'] in ['LogisticRegression', 'DecisionTreeClassifier']
    # Narrative must contain substantive text explaining margin and standard deviation
    assert len(rec['detailed_narrative']) > 50
    assert 'sigma' in rec['detailed_narrative'] or '&sigma;' in rec['detailed_narrative']

def test_validation_strategy_compatibility_checks():
    """
    Test 7: Verify that methodologically invalid strategies raise descriptive ValueErrors.
    """
    # Stratified K-Fold is invalid on continuous regression
    with pytest.raises(ValueError, match="Stratified K-Fold is methodologically invalid for continuous regression"):
        validate_strategy_compatibility('regression', 'stratified')
        
    # Time-Series Split is invalid on standard classification
    with pytest.raises(ValueError, match="TimeSeriesSplit is methodologically invalid for standard classification"):
        validate_strategy_compatibility('classification', 'timeseries')

def test_timeseries_model_comparison():
    """
    Test 8: Verify model comparison on chronological Airline Passengers dataset
    using Time-Series Cross-Validation.
    """
    algos = ['LinearRegression', 'RandomForestRegressor']
    res = compare_models(
        dataset_name='air_passengers',
        algorithms=algos,
        validation_method='timeseries',
        n_splits=3,
        primary_metric='rmse',
        random_state=42
    )
    
    assert res['task_type'] == 'time_series'
    assert res['validation_method'] == 'timeseries'
    assert len(res['rankings']) == len(algos)
    assert res['rankings'][0]['score'] > 0

def test_api_comparison_run_endpoint(client):
    """
    Test 9: Verify POST /api/comparison/run executes end-to-end and returns 200.
    """
    payload = {
        'dataset_name': 'iris',
        'algorithms': ['LogisticRegression', 'DecisionTreeClassifier'],
        'validation_method': 'stratified',
        'n_splits': 3,
        'primary_metric': 'accuracy'
    }
    
    resp = client.post('/api/comparison/run', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    
    assert 'comparison_id' in data
    assert 'rankings' in data
    assert 'recommendation' in data
    assert 'chart_data' in data
    assert data['models_compared_count'] == 2

def test_sqlite_comparison_persistence(client):
    """
    Test 10: Verify comparison runs are saved to SQLite and retrievable via API.
    """
    payload = {
        'dataset_name': 'breast_cancer',
        'algorithms': ['LogisticRegression', 'SVC'],
        'validation_method': 'stratified',
        'n_splits': 3,
        'primary_metric': 'accuracy'
    }
    
    resp = client.post('/api/comparison/run', json=payload)
    assert resp.status_code == 200
    comp_id = resp.get_json()['comparison_id']
    
    # Retrieve via API
    hist_resp = client.get('/api/comparison/history')
    assert hist_resp.status_code == 200
    history = hist_resp.get_json()
    assert any(c['comparison_id'] == comp_id for c in history)
    
    # Retrieve single record by ID
    single_resp = client.get(f'/api/comparison/{comp_id}')
    assert single_resp.status_code == 200
    single_record = single_resp.get_json()
    assert single_record['comparison_id'] == comp_id
    assert single_record['dataset_name'] == 'breast_cancer'

def test_error_handling_invalid_configurations(client):
    """
    Test 11: Verify API error handling for incompatible configurations.
    """
    # 1. Incompatible strategy: Stratified on continuous wine quality
    bad_strat_payload = {
        'dataset_name': 'wine_quality',
        'algorithms': ['LinearRegression'],
        'validation_method': 'stratified'
    }
    resp = client.post('/api/comparison/run', json=bad_strat_payload)
    assert resp.status_code == 400
    assert 'Stratified K-Fold is methodologically invalid' in resp.get_json()['error']
    
    # 2. Incompatible strategy: Time-Series on classification Iris
    bad_ts_payload = {
        'dataset_name': 'iris',
        'algorithms': ['LogisticRegression'],
        'validation_method': 'timeseries'
    }
    resp_ts = client.post('/api/comparison/run', json=bad_ts_payload)
    assert resp_ts.status_code == 400
    assert 'TimeSeriesSplit is methodologically invalid' in resp_ts.get_json()['error']

def test_existing_experiment_history_continues_to_work(client):
    """
    Test 12: Verify that previously saved experiment runs in SQLite continue
    to be retrievable without regressions.
    """
    resp = client.get('/api/experiments')
    assert resp.status_code == 200
    exps = resp.get_json()
    assert isinstance(exps, list)
