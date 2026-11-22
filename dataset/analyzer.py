"""
FoldWise AI - Dataset Exploration & Statistical Analyzer
Extracts comprehensive statistical summaries, correlation structures,
class distributions, and data quality metrics from real DataFrames.
"""

import numpy as np
import pandas as pd

def analyze_dataset(df, target_column=None, task_type=None, max_preview_rows=10):
    """
    Perform deep statistical analysis on a given pandas DataFrame.
    
    Returns:
        dict:
            - preview: list of dicts (first N rows)
            - shape: {'rows': int, 'columns': int}
            - columns_info: list of dicts with name, dtype, null_count, null_percent, unique_count
            - duplicates: {'count': int, 'percent': float}
            - numerical_summary: dict of summary statistics for numeric columns
            - correlation_matrix: dict with columns and 2D matrix of correlation values
            - target_distribution: class counts or histogram distribution
    """
    if df is None or df.empty:
        raise ValueError("Cannot analyze an empty DataFrame.")
        
    rows, cols = df.shape
    target = target_column if (target_column and target_column in df.columns) else df.columns[-1]
    
    # 1. Column info & Missing Values
    columns_info = []
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        columns_info.append({
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': null_cnt,
            'null_percent': round((null_cnt / rows) * 100, 2),
            'unique_count': int(df[col].nunique()),
            'is_target': (col == target)
        })
        
    # 2. Duplicate Detection
    dup_count = int(df.duplicated().sum())
    dup_percent = round((dup_count / rows) * 100, 2)
    
    # 3. Statistical Summaries (Numeric features)
    numeric_df = df.select_dtypes(include=[np.number])
    numerical_summary = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()
        for col, stats in desc.items():
            numerical_summary[col] = {
                'count': int(stats.get('count', 0)),
                'mean': round(float(stats.get('mean', 0.0)), 4),
                'std': round(float(stats.get('std', 0.0)), 4) if not np.isnan(stats.get('std', 0.0)) else 0.0,
                'min': round(float(stats.get('min', 0.0)), 4),
                'q25': round(float(stats.get('25%', 0.0)), 4),
                'median': round(float(stats.get('50%', 0.0)), 4),
                'q75': round(float(stats.get('75%', 0.0)), 4),
                'max': round(float(stats.get('max', 0.0)), 4)
            }
            
    # 4. Correlation Matrix (Numeric features only)
    correlation_data = {'columns': [], 'matrix': []}
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().fillna(0).round(4)
        correlation_data = {
            'columns': list(corr_matrix.columns),
            'matrix': corr_matrix.values.tolist()
        }
        
    # 5. Target Analysis & Distribution
    target_series = df[target]
    is_numeric_target = pd.api.types.is_numeric_dtype(target_series)
    unique_target_count = target_series.nunique()
    
    target_distribution = {
        'column': target,
        'unique_count': unique_target_count,
        'is_numeric': is_numeric_target,
        'values': []
    }
    
    if (task_type == 'classification') or (unique_target_count <= 15 and not task_type == 'time_series'):
        # Discrete class distribution
        val_counts = target_series.value_counts(dropna=False).to_dict()
        for class_val, count in val_counts.items():
            target_distribution['values'].append({
                'label': str(class_val),
                'count': int(count),
                'percentage': round((count / rows) * 100, 2)
            })
    else:
        # Continuous distribution (histogram bins)
        clean_target = target_series.dropna()
        if not clean_target.empty and is_numeric_target:
            hist_counts, bin_edges = np.histogram(clean_target, bins=10)
            for i in range(len(hist_counts)):
                label = f"{round(bin_edges[i], 1)} - {round(bin_edges[i+1], 1)}"
                target_distribution['values'].append({
                    'label': label,
                    'count': int(hist_counts[i]),
                    'percentage': round((hist_counts[i] / len(clean_target)) * 100, 2)
                })
                
    # 6. Preview rows (converted to JSON-safe dictionary)
    preview_df = df.head(max_preview_rows).copy()
    # Replace NaN/inf with None for valid JSON serialization
    preview_df = preview_df.replace({np.nan: None, np.inf: None, -np.inf: None})
    preview = preview_df.to_dict(orient='records')
    
    return {
        'target': target,
        'shape': {'rows': rows, 'columns': cols},
        'columns_info': columns_info,
        'duplicates': {'count': dup_count, 'percent': dup_percent},
        'numerical_summary': numerical_summary,
        'correlation_matrix': correlation_data,
        'target_distribution': target_distribution,
        'preview': preview
    }
