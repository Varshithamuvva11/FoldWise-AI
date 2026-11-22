"""
FoldWise AI - Machine Learning Model Registry & Hyperparameter Config
Defines supported classification and regression algorithms, default parameters,
and parameter search spaces for Nested Cross-Validation.
"""

CLASSIFIERS = {
    'LogisticRegression': {
        'name': 'Logistic Regression',
        'default_params': {'max_iter': 500, 'random_state': 42},
        'param_grid': {
            'model__C': [0.1, 1.0, 10.0]
        }
    },
    'DecisionTreeClassifier': {
        'name': 'Decision Tree Classifier',
        'default_params': {'max_depth': 5, 'random_state': 42},
        'param_grid': {
            'model__max_depth': [3, 5, 8, None],
            'model__min_samples_split': [2, 5]
        }
    },
    'RandomForestClassifier': {
        'name': 'Random Forest Classifier',
        'default_params': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42},
        'param_grid': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [4, 8, None]
        }
    },
    'SVC': {
        'name': 'Support Vector Machine (SVC)',
        'default_params': {'probability': True, 'random_state': 42},
        'param_grid': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['linear', 'rbf']
        }
    },
    'KNeighborsClassifier': {
        'name': 'K-Nearest Neighbors (KNN)',
        'default_params': {'n_neighbors': 5},
        'param_grid': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance']
        }
    }
}

REGRESSORS = {
    'LinearRegression': {
        'name': 'Linear Regression',
        'default_params': {},
        'param_grid': {
            'model__fit_intercept': [True, False]
        }
    },
    'DecisionTreeRegressor': {
        'name': 'Decision Tree Regressor',
        'default_params': {'max_depth': 5, 'random_state': 42},
        'param_grid': {
            'model__max_depth': [3, 5, 8, None],
            'model__min_samples_split': [2, 5]
        }
    },
    'RandomForestRegressor': {
        'name': 'Random Forest Regressor',
        'default_params': {'n_estimators': 100, 'max_depth': 6, 'random_state': 42},
        'param_grid': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [4, 8, None]
        }
    },
    'SVR': {
        'name': 'Support Vector Regressor (SVR)',
        'default_params': {},
        'param_grid': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['linear', 'rbf']
        }
    }
}

def get_available_models(task_type='classification'):
    """Return model dictionary according to task type."""
    if task_type == 'classification':
        return CLASSIFIERS
    return REGRESSORS

def get_param_grid(algorithm_name):
    """Retrieve hyperparameter search grid for a specified algorithm."""
    if algorithm_name in CLASSIFIERS:
        return CLASSIFIERS[algorithm_name].get('param_grid', {})
    elif algorithm_name in REGRESSORS:
        return REGRESSORS[algorithm_name].get('param_grid', {})
    return {}
