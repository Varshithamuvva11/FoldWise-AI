"""
FoldWise AI - Classifier Factory
Instantiates and configures supported Scikit-learn classification models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from models.model_config import CLASSIFIERS

def get_classifier(algorithm_name, hyperparameters=None, random_state=42):
    """
    Factory function returning an instantiated Scikit-learn classifier.
    
    Args:
        algorithm_name (str): Key matching one of the 5 supported classifiers
        hyperparameters (dict, optional): Custom hyperparameter overrides
        random_state (int): Reproducibility seed
        
    Returns:
        Classifier instance
    """
    if algorithm_name not in CLASSIFIERS:
        raise ValueError(f"Unsupported classifier '{algorithm_name}'. Available: {list(CLASSIFIERS.keys())}")
        
    params = CLASSIFIERS[algorithm_name]['default_params'].copy()
    if 'random_state' in params and random_state is not None:
        params['random_state'] = random_state
        
    if hyperparameters and isinstance(hyperparameters, dict):
        params.update(hyperparameters)
        
    if algorithm_name == 'LogisticRegression':
        return LogisticRegression(**params)
    elif algorithm_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**params)
    elif algorithm_name == 'RandomForestClassifier':
        return RandomForestClassifier(**params)
    elif algorithm_name == 'SVC':
        return SVC(**params)
    elif algorithm_name == 'KNeighborsClassifier':
        return KNeighborsClassifier(**params)
        
    raise ValueError(f"Classifier implementation missing for '{algorithm_name}'")
