"""
FoldWise AI - Data Preprocessing & Zero-Leakage Pipeline Architecture
Ensures all imputation, scaling, and categorical encoding are encapsulated
within Scikit-Learn Pipelines and ColumnTransformers so transformations are
strictly fitted within training folds with zero data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def identify_feature_types(X):
    """
    Separate features into numerical and categorical column lists.
    
    Args:
        X (pd.DataFrame): Input feature matrix
        
    Returns:
        tuple: (list of numerical column names, list of categorical column names)
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame to identify column types.")
        
    numerical_cols = []
    categorical_cols = []
    
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # If a numeric column has only 2 unique values and is treated as category, keep as numeric unless object
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return numerical_cols, categorical_cols

def create_preprocessor(numerical_cols, categorical_cols, scaling='standard', imputation_strategy='mean'):
    """
    Construct a leak-proof Scikit-Learn ColumnTransformer.
    
    Args:
        numerical_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
        scaling (str): 'standard' (StandardScaler), 'minmax' (MinMaxScaler), or 'none' (passthrough)
        imputation_strategy (str): 'mean', 'median', or 'most_frequent'
        
    Returns:
        ColumnTransformer: Preprocessing pipeline component
    """
    transformers = []
    
    # 1. Numerical Pipeline
    if numerical_cols:
        num_steps = [
            ('imputer', SimpleImputer(strategy=imputation_strategy))
        ]
        if scaling == 'standard':
            num_steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            num_steps.append(('scaler', MinMaxScaler()))
        # If 'none', no scaler is added
        
        num_pipe = Pipeline(steps=num_steps)
        transformers.append(('num', num_pipe, numerical_cols))
        
    # 2. Categorical Pipeline
    if categorical_cols:
        cat_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
        cat_pipe = Pipeline(steps=cat_steps)
        transformers.append(('cat', cat_pipe, categorical_cols))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor

def build_leak_free_pipeline(model, numerical_cols, categorical_cols, scaling='standard', imputation_strategy='mean'):
    """
    Assemble the complete zero-leakage Scikit-learn Pipeline.
    
    When pipeline.fit(X_train, y_train) is called inside cross-validation,
    the preprocessor computes mean/variance/imputation values SOLELY on X_train.
    During pipeline.predict(X_test), X_test is transformed using the parameters
    already learned from X_train, guaranteeing ZERO data leakage.
    
    Args:
        model: An instantiated Scikit-Learn estimator (classifier or regressor)
        numerical_cols (list): Column names for numeric transformers
        categorical_cols (list): Column names for categorical transformers
        scaling (str): 'standard', 'minmax', or 'none'
        imputation_strategy (str): 'mean', 'median', etc.
        
    Returns:
        Pipeline: Scikit-learn Pipeline instance
    """
    preprocessor = create_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        scaling=scaling,
        imputation_strategy=imputation_strategy
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

def create_time_series_features(df, target_col='Passengers', lag_orders=(1, 2, 3), drop_na=True, return_dates=False):
    """
    Generate backward-looking autoregressive lag features for time-series forecasting.
    
    Methodological Rules:
    1. Preserves strict chronological sequence (no shuffling).
    2. At timestep t, only observations from t-1, t-2, ... are available.
    3. No future observations or moving averages centered on future data are introduced.
    
    Args:
        df (pd.DataFrame): Input time series DataFrame
        target_col (str): Target column to forecast
        lag_orders (tuple/list): Number of past observation lags to generate (e.g. [1, 2, 3])
        drop_na (bool): Drop initial rows that lack past lag history
        return_dates (bool): If True, also return date strings as 4th element
        
    Returns:
        tuple: (X (DataFrame), y (Series), feature_names (list)) or with date_labels if return_dates=True
    """
    ts_df = df.copy()
    
    # Sort chronologically if a Date/Month column exists
    date_candidates = [c for c in ts_df.columns if 'month' in c.lower() or 'date' in c.lower() or 'time' in c.lower()]
    date_col = date_candidates[0] if date_candidates else None
    
    if date_col:
        try:
            ts_df['parsed_date'] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(by='parsed_date').reset_index(drop=True)
            # Add calendar month feature (cyclical predictor)
            ts_df['month_num'] = ts_df['parsed_date'].dt.month
            ts_df = ts_df.drop(columns=['parsed_date'])
        except Exception:
            pass
            
    # Ensure target is float
    ts_df[target_col] = pd.to_numeric(ts_df[target_col], errors='coerce')
    
    # Generate strictly past lags
    lag_feature_names = []
    for lag in sorted(lag_orders):
        col_name = f"{target_col}_lag_{lag}"
        ts_df[col_name] = ts_df[target_col].shift(lag)
        lag_feature_names.append(col_name)
        
    if drop_na:
        # Drop rows where lags are NaN (the first max(lag_orders) rows)
        ts_df = ts_df.dropna().reset_index(drop=True)
        
    # Extract date strings before dropping date column from features
    date_labels = ts_df[date_col].astype(str).tolist() if date_col else [str(i) for i in range(len(ts_df))]
    
    # Feature columns: all generated lags + month_num if present, excluding target and original raw date string
    exclude_cols = [target_col] + date_candidates
    feature_cols = [c for c in ts_df.columns if c not in exclude_cols]
    
    X = ts_df[feature_cols].copy()
    y = ts_df[target_col].copy()
    
    if return_dates:
        return X, y, feature_cols, date_labels
    return X, y, feature_cols

def prepare_dataset_for_modeling(dataset_dict, target_col=None, lag_orders=(1, 2, 3)):
    """
    Prepares X and y matrices and inspects feature types from a dataset dictionary.
    
    Args:
        dataset_dict (dict): Dictionary from dataset.loader.load_dataset
        target_col (str): Optional override for target column
        lag_orders (tuple): Lag periods for time series
        
    Returns:
        dict:
            - 'X': pd.DataFrame
            - 'y': pd.Series
            - 'numerical_cols': list
            - 'categorical_cols': list
            - 'task_type': str
            - 'target_name': str
    """
    df = dataset_dict['df'].copy()
    task_type = dataset_dict['task_type']
    target = target_col or dataset_dict['target']
    
    if task_type == 'time_series':
        X, y, feature_names, date_labels = create_time_series_features(df, target_col=target, lag_orders=lag_orders, return_dates=True)
        numerical_cols, categorical_cols = identify_feature_types(X)
        return {
            'X': X,
            'y': y,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'task_type': task_type,
            'target_name': target,
            'date_labels': date_labels
        }
        
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")
        
    X = df.drop(columns=[target])
    y = df[target]
    
    numerical_cols, categorical_cols = identify_feature_types(X)
    
    return {
        'X': X,
        'y': y,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'task_type': task_type,
        'target_name': target
    }
