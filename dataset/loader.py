"""
FoldWise AI - Dedicated Dataset Loader & Cache Manager
Handles automatic loading of Scikit-Learn built-ins, UCI downloads, GitHub datasets,
local disk caching, and secure custom CSV uploads.
"""

import os
import io
import time
import requests
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_CACHE = os.path.join(BASE_DIR, 'cache')
BUNDLED_UPLOAD = os.path.join(BASE_DIR, 'uploaded')

DATA_DIR = os.environ.get('DATA_DIR')
if DATA_DIR:
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')
    UPLOAD_DIR = os.path.join(DATA_DIR, 'uploaded')
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Seed cache from bundled cache if empty
    if os.path.exists(BUNDLED_CACHE):
        import shutil
        for f in os.listdir(BUNDLED_CACHE):
            src = os.path.join(BUNDLED_CACHE, f)
            dst = os.path.join(CACHE_DIR, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
else:
    CACHE_DIR = BUNDLED_CACHE
    UPLOAD_DIR = BUNDLED_UPLOAD

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dataset registry configuration
DATASET_REGISTRY = {
    'iris': {
        'name': 'Iris Flower Dataset',
        'type': 'builtin',
        'task_type': 'classification',
        'is_multiclass': True,
        'description': '150 samples, 4 features, 3 balanced flower classes (setosa, versicolor, virginica).',
        'default_target': 'target',
        'suitable_cv': ['kfold', 'stratified', 'nested']
    },
    'breast_cancer': {
        'name': 'Breast Cancer Wisconsin (Diagnostic)',
        'type': 'builtin',
        'task_type': 'classification',
        'is_multiclass': False,
        'description': '569 samples, 30 features, binary classification (malignant vs benign).',
        'default_target': 'target',
        'suitable_cv': ['kfold', 'stratified', 'nested']
    },
    'wine_quality': {
        'name': 'Wine Quality (Red)',
        'type': 'external',
        'task_type': 'regression',
        'is_multiclass': False,
        'description': '1,599 samples, 11 physiochemical properties predicting wine quality score (3-8).',
        'default_target': 'quality',
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'backup_urls': [
            'https://raw.githubusercontent.com/datasciencedojo/datasets/master/winequality-red.csv'
        ],
        'cache_filename': 'winequality-red.csv',
        'sep': ';',
        'suitable_cv': ['kfold', 'stratified', 'nested']
    },
    'air_passengers': {
        'name': 'International Airline Passengers',
        'type': 'external',
        'task_type': 'time_series',
        'is_multiclass': False,
        'description': '144 monthly observations (1949 to 1960) for chronological time-series forecasting.',
        'default_target': 'Passengers',
        'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
        'backup_urls': [
            'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv'
        ],
        'cache_filename': 'airline-passengers.csv',
        'sep': ',',
        'suitable_cv': ['timeseries']
    }
}

def list_registered_datasets():
    """Return catalog of all available standard datasets and uploaded files."""
    catalog = {}
    for key, info in DATASET_REGISTRY.items():
        cached = False
        if info['type'] == 'external':
            cached_path = os.path.join(CACHE_DIR, info['cache_filename'])
            cached = os.path.exists(cached_path)
        else:
            cached = True  # built-in is always available
            
        catalog[key] = {
            'id': key,
            'name': info['name'],
            'type': info['type'],
            'task_type': info['task_type'],
            'is_multiclass': info.get('is_multiclass', False),
            'description': info['description'],
            'default_target': info['default_target'],
            'cached': cached,
            'suitable_cv': info['suitable_cv']
        }
        
    # Discover custom uploads
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith('.csv'):
                upload_key = f"upload_{f}"
                catalog[upload_key] = {
                    'id': upload_key,
                    'name': f"Custom: {f}",
                    'type': 'uploaded',
                    'task_type': 'unknown',
                    'is_multiclass': False,
                    'description': f"User-uploaded CSV dataset: {f}",
                    'default_target': None,
                    'cached': True,
                    'suitable_cv': ['kfold', 'stratified', 'nested', 'timeseries']
                }
    return catalog

def download_dataset_with_retry(url, destination_path, backup_urls=None, max_retries=3, timeout=12):
    """
    Download external dataset with timeout, user-agent, retry loop, and backup URLs.
    Saves to destination_path only if download is successful and verified non-empty.
    """
    urls_to_try = [url] + (backup_urls or [])
    headers = {
        'User-Agent': 'FoldWiseAI-ML-Platform/1.0 (Academic Research; Windows)'
    }
    
    last_exception = None
    for target_url in urls_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(target_url, headers=headers, timeout=timeout)
                if response.status_code == 200 and len(response.content) > 50:
                    # Write to temporary file first, then atomically rename
                    temp_path = destination_path + f".tmp_{os.getpid()}"
                    with open(temp_path, 'wb') as f:
                        f.write(response.content)
                    if os.path.exists(destination_path):
                        os.remove(destination_path)
                    os.rename(temp_path, destination_path)
                    return True
                else:
                    last_exception = Exception(f"HTTP {response.status_code}: Empty or incomplete response from {target_url}")
            except Exception as e:
                last_exception = e
                time.sleep(0.5 * attempt)
                
    raise RuntimeError(f"Failed to download dataset after retrying {max_retries} times. Error: {str(last_exception)}")

def load_dataset(dataset_name, target_column=None, force_download=False):
    """
    Primary interface for loading datasets into Pandas DataFrames.
    Automatically handles built-ins, downloads, caching, and custom uploads.
    
    Returns:
        dict:
            - 'id': str
            - 'name': str
            - 'df': pd.DataFrame
            - 'features': list of column names (excluding target)
            - 'target': str (target column name)
            - 'task_type': 'classification' | 'regression' | 'time_series'
            - 'is_multiclass': bool
            - 'n_samples': int
            - 'n_features': int
            - 'source': 'builtin' | 'cache' | 'downloaded' | 'uploaded'
    """
    dataset_name = dataset_name.strip()
    
    # 1. Custom Uploaded Dataset
    if dataset_name.startswith('upload_') or dataset_name.endswith('.csv'):
        filename = dataset_name.replace('upload_', '')
        filepath = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Uploaded dataset file not found: {filename}")
        
        # Sniff delimiter
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(filepath)
            
        target = target_column if target_column and target_column in df.columns else df.columns[-1]
        features = [col for col in df.columns if col != target]
        
        # Auto-infer task type based on target values
        unique_targets = df[target].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[target])
        if unique_targets <= 15 or not is_numeric:
            task_type = 'classification'
            is_multiclass = unique_targets > 2
        else:
            task_type = 'regression'
            is_multiclass = False
            
        return {
            'id': dataset_name,
            'name': f"Custom: {filename}",
            'df': df,
            'features': features,
            'target': target,
            'task_type': task_type,
            'is_multiclass': is_multiclass,
            'n_samples': int(df.shape[0]),
            'n_features': int(len(features)),
            'source': 'uploaded'
        }
        
    # 2. Registered Benchmark Datasets
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")
        
    config = DATASET_REGISTRY[dataset_name]
    
    if dataset_name == 'iris':
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        target = target_column or 'target'
        features = [col for col in df.columns if col != target]
        return {
            'id': 'iris',
            'name': config['name'],
            'df': df,
            'features': features,
            'target': target,
            'task_type': 'classification',
            'is_multiclass': True,
            'n_samples': int(df.shape[0]),
            'n_features': int(len(features)),
            'source': 'builtin'
        }
        
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        target = target_column or 'target'
        features = [col for col in df.columns if col != target]
        return {
            'id': 'breast_cancer',
            'name': config['name'],
            'df': df,
            'features': features,
            'target': target,
            'task_type': 'classification',
            'is_multiclass': False,
            'n_samples': int(df.shape[0]),
            'n_features': int(len(features)),
            'source': 'builtin'
        }
        
    elif dataset_name == 'wine_quality':
        cache_path = os.path.join(CACHE_DIR, config['cache_filename'])
        source = 'cache'
        
        if force_download or not os.path.exists(cache_path):
            download_dataset_with_retry(config['url'], cache_path, config.get('backup_urls'))
            source = 'downloaded'
            
        # Semicolon delimited
        df = pd.read_csv(cache_path, sep=config.get('sep', ';'))
        # Clean column names (strip quotes/spaces)
        df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
        
        target = target_column or config['default_target']
        features = [c for c in df.columns if c != target]
        
        # Check if user wants classification categories for Wine (e.g. quality category)
        unique_targets = df[target].nunique()
        task_type = 'regression'
        is_multiclass = False
        
        return {
            'id': 'wine_quality',
            'name': config['name'],
            'df': df,
            'features': features,
            'target': target,
            'task_type': task_type,
            'is_multiclass': is_multiclass,
            'n_samples': int(df.shape[0]),
            'n_features': int(len(features)),
            'source': source
        }
        
    elif dataset_name == 'air_passengers':
        cache_path = os.path.join(CACHE_DIR, config['cache_filename'])
        source = 'cache'
        
        if force_download or not os.path.exists(cache_path):
            download_dataset_with_retry(config['url'], cache_path, config.get('backup_urls'))
            source = 'downloaded'
            
        df = pd.read_csv(cache_path, sep=config.get('sep', ','))
        df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
        
        # In Brownlee CSV, columns are Month and Passengers. Drop any footer/NaN rows
        df = df.dropna().reset_index(drop=True)
        # Ensure Passengers is numeric
        target = target_column or 'Passengers'
        df[target] = pd.to_numeric(df[target], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        
        features = [c for c in df.columns if c != target]
        
        return {
            'id': 'air_passengers',
            'name': config['name'],
            'df': df,
            'features': features,
            'target': target,
            'task_type': 'time_series',
            'is_multiclass': False,
            'n_samples': int(df.shape[0]),
            'n_features': int(len(features)),
            'source': source
        }
        
    raise ValueError(f"Dataset handler for '{dataset_name}' not implemented.")

def save_uploaded_csv(file_storage, target_col=None):
    """
    Save and validate uploaded CSV file.
    
    Args:
        file_storage: Werkzeug FileStorage object
        target_col: Optional chosen target column
        
    Returns:
        tuple: (dataset_id, metadata_dict)
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided for upload.")
        
    filename = file_storage.filename
    if not filename.lower().endswith('.csv'):
        raise ValueError("Invalid file extension. Only .csv files are supported.")
        
    # Sanitize filename
    safe_name = os.path.basename(filename).replace(' ', '_')
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    
    file_storage.save(save_path)
    
    # Validate CSV readability
    try:
        df = pd.read_csv(save_path, nrows=20)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise ValueError(f"Failed to parse CSV file: {str(e)}")
        
    dataset_id = f"upload_{safe_name}"
    meta = load_dataset(dataset_id, target_column=target_col)
    return dataset_id, meta
