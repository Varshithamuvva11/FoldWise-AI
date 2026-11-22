"""
FoldWise AI - Regressor Factory
Instantiates and configures supported Scikit-learn regression models.
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from models.model_config import REGRESSORS

def get_regressor(algorithm_name, hyperparameters=None, random_state=42):
    """
    Factory function returning an instantiated Scikit-learn regressor.
    
    Args:
        algorithm_name (str): Key matching one of the 4 supported regressors
        hyperparameters (dict, optional): Custom hyperparameter overrides
        random_state (int): Reproducibility seed
        
    Returns:
        Regressor instance
    """
    if algorithm_name not in REGRESSORS:
        raise ValueError(f"Unsupported regressor '{algorithm_name}'. Available: {list(REGRESSORS.keys())}")
        
    params = REGRESSORS[algorithm_name]['default_params'].copy()
    if 'random_state' in params and random_state is not None:
        params['random_state'] = random_state
        
    if hyperparameters and isinstance(hyperparameters, dict):
        params.update(hyperparameters)
        
    if algorithm_name == 'LinearRegression':
        return LinearRegression(**params)
    elif algorithm_name == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**params)
    elif algorithm_name == 'RandomForestRegressor':
        return RandomForestRegressor(**params)
    elif algorithm_name == 'SVR':
        return SVR(**params)
        
    raise ValueError(f"Regressor implementation missing for '{algorithm_name}'")
