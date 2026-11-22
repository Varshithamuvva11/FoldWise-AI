"""
FoldWise AI - SQLite Database Interface
Stores experiment logs, cross-validation metrics, fold scores, and configurations.
"""

import sqlite3
import json
import os
import shutil
from datetime import datetime

DATA_DIR = os.environ.get('DATA_DIR')
BUNDLED_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foldwise.db')

if os.environ.get('DATABASE_PATH'):
    DB_PATH = os.environ.get('DATABASE_PATH')
elif DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)
    DB_PATH = os.path.join(DATA_DIR, 'foldwise.db')
    # If persistent DB does not exist yet on attached disk, seed from bundled DB
    if not os.path.exists(DB_PATH) and os.path.exists(BUNDLED_DB):
        shutil.copy2(BUNDLED_DB, DB_PATH)
else:
    DB_PATH = BUNDLED_DB

def get_connection():
    """Get a connection to the SQLite database with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT UNIQUE NOT NULL,
            dataset_name TEXT NOT NULL,
            target_column TEXT NOT NULL,
            task_type TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            validation_method TEXT NOT NULL,
            hyperparameters TEXT,
            metrics TEXT NOT NULL,
            fold_scores TEXT,
            best_params TEXT,
            execution_time REAL NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'COMPLETED',
            notes TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison_id TEXT UNIQUE NOT NULL,
            dataset_name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            validation_method TEXT NOT NULL,
            primary_metric TEXT NOT NULL,
            algorithms_compared TEXT NOT NULL,
            rankings TEXT NOT NULL,
            winner_algorithm TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            execution_time REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_experiment(exp_data):
    """
    Save a completed experiment record to the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Serialize dict fields if necessary
    hyperparams_str = json.dumps(exp_data.get('hyperparameters', {})) if isinstance(exp_data.get('hyperparameters'), dict) else exp_data.get('hyperparameters', '{}')
    metrics_str = json.dumps(exp_data.get('metrics', {})) if isinstance(exp_data.get('metrics'), dict) else exp_data.get('metrics', '{}')
    fold_scores_str = json.dumps(exp_data.get('fold_scores', [])) if isinstance(exp_data.get('fold_scores'), (list, dict)) else exp_data.get('fold_scores', '[]')
    best_params_str = json.dumps(exp_data.get('best_params', {})) if isinstance(exp_data.get('best_params'), (dict, list)) else str(exp_data.get('best_params', '{}'))
    
    created_at = exp_data.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    cursor.execute("""
        INSERT INTO experiments (
            experiment_id, dataset_name, target_column, task_type,
            algorithm, validation_method, hyperparameters, metrics,
            fold_scores, best_params, execution_time, created_at, status, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exp_data['experiment_id'],
        exp_data['dataset_name'],
        exp_data['target_column'],
        exp_data.get('task_type', 'classification'),
        exp_data['algorithm'],
        exp_data['validation_method'],
        hyperparams_str,
        metrics_str,
        fold_scores_str,
        best_params_str,
        exp_data.get('execution_time', 0.0),
        created_at,
        exp_data.get('status', 'COMPLETED'),
        exp_data.get('notes', '')
    ))
    conn.commit()
    conn.close()
    return exp_data['experiment_id']

def filter_experiments(
    dataset=None,
    validation_method=None,
    task_type=None,
    algorithm=None,
    search=None,
    sort_by='created_at',
    sort_dir='desc',
    limit=200
):
    """
    Query experiments with flexible multi-parameter filtering, full-text search, and sorting.
    """
    conditions = []
    params = []
    
    if dataset:
        conditions.append("dataset_name = ?")
        params.append(dataset)
        
    if validation_method:
        conditions.append("LOWER(validation_method) LIKE ?")
        params.append(f"%{validation_method.lower()}%")
        
    if task_type:
        conditions.append("LOWER(task_type) = ?")
        params.append(task_type.lower())
        
    if algorithm:
        conditions.append("LOWER(algorithm) = ?")
        params.append(algorithm.lower())
        
    if search and search.strip():
        s_term = f"%{search.strip().lower()}%"
        conditions.append("""(
            LOWER(experiment_id) LIKE ? OR 
            LOWER(dataset_name) LIKE ? OR 
            LOWER(algorithm) LIKE ? OR 
            LOWER(validation_method) LIKE ? OR 
            LOWER(notes) LIKE ?
        )""")
        params.extend([s_term, s_term, s_term, s_term, s_term])
        
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    
    sort_columns = {
        'created_at': 'id',
        'id': 'id',
        'execution_time': 'execution_time',
        'algorithm': 'algorithm',
        'dataset_name': 'dataset_name'
    }
    col = sort_columns.get(sort_by, 'id')
    direction = 'ASC' if sort_dir and sort_dir.lower() == 'asc' else 'DESC'
    
    query = f"SELECT * FROM experiments{where_clause} ORDER BY {col} {direction} LIMIT ?"
    params.append(int(limit))
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        d['hyperparameters'] = json.loads(d['hyperparameters']) if d['hyperparameters'] else {}
        d['metrics'] = json.loads(d['metrics']) if d['metrics'] else {}
        d['fold_scores'] = json.loads(d['fold_scores']) if d['fold_scores'] else []
        d['best_params'] = json.loads(d['best_params']) if d['best_params'] else {}
        results.append(d)
    return results

def get_experiments(limit=100):
    """Retrieve all experiments ordered by date descending."""
    return filter_experiments(limit=limit)

def get_experiment_by_id(exp_id):
    """Retrieve a single experiment by its experiment_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (exp_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d['hyperparameters'] = json.loads(d['hyperparameters']) if d['hyperparameters'] else {}
    d['metrics'] = json.loads(d['metrics']) if d['metrics'] else {}
    d['fold_scores'] = json.loads(d['fold_scores']) if d['fold_scores'] else []
    d['best_params'] = json.loads(d['best_params']) if d['best_params'] else {}
    return d

def delete_experiment(exp_id):
    """Delete an experiment by its experiment_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (exp_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def save_comparison(comp_data):
    """
    Save a model comparison result record to the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    algos_str = json.dumps(comp_data.get('algorithms_compared', [])) if isinstance(comp_data.get('algorithms_compared'), list) else str(comp_data.get('algorithms_compared', '[]'))
    rankings_str = json.dumps(comp_data.get('rankings', [])) if isinstance(comp_data.get('rankings'), (list, dict)) else str(comp_data.get('rankings', '[]'))
    rec_str = json.dumps(comp_data.get('recommendation', {})) if isinstance(comp_data.get('recommendation'), dict) else str(comp_data.get('recommendation', ''))
    created_at = comp_data.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    cursor.execute("""
        INSERT INTO comparisons (
            comparison_id, dataset_name, task_type, validation_method,
            primary_metric, algorithms_compared, rankings, winner_algorithm,
            recommendation, execution_time, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        comp_data['comparison_id'],
        comp_data['dataset_name'],
        comp_data.get('task_type', 'classification'),
        comp_data['validation_method'],
        comp_data.get('primary_metric', 'accuracy'),
        algos_str,
        rankings_str,
        comp_data.get('winner_algorithm', ''),
        rec_str,
        comp_data.get('execution_time', 0.0),
        created_at
    ))
    conn.commit()
    conn.close()
    return comp_data['comparison_id']

def get_comparisons(limit=50):
    """Retrieve all comparisons ordered by date descending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM comparisons ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        d = dict(r)
        d['algorithms_compared'] = json.loads(d['algorithms_compared']) if d['algorithms_compared'] else []
        d['rankings'] = json.loads(d['rankings']) if d['rankings'] else []
        try:
            d['recommendation'] = json.loads(d['recommendation'])
        except Exception:
            pass
        results.append(d)
    return results

def get_comparison_by_id(comp_id):
    """Retrieve a single comparison record by comparison_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM comparisons WHERE comparison_id = ?", (comp_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d['algorithms_compared'] = json.loads(d['algorithms_compared']) if d['algorithms_compared'] else []
    d['rankings'] = json.loads(d['rankings']) if d['rankings'] else []
    try:
        d['recommendation'] = json.loads(d['recommendation'])
    except Exception:
        pass
    return d

def clear_all_experiments():
    """Clear all records from experiments table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments")
    conn.commit()
    conn.close()
    return True

def get_database_stats():
    """
    Retrieve live aggregated statistics from the SQLite database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Total counts
    cursor.execute("SELECT COUNT(*) FROM experiments")
    total_experiments = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM comparisons")
    total_comparisons = cursor.fetchone()[0]
    
    # 2. Strategy breakdown
    cursor.execute("""
        SELECT validation_method, COUNT(*) as count 
        FROM experiments 
        GROUP BY validation_method 
        ORDER BY count DESC
    """)
    strategy_counts = {row['validation_method']: row['count'] for row in cursor.fetchall()}
    
    # 3. Task type breakdown
    cursor.execute("""
        SELECT task_type, COUNT(*) as count 
        FROM experiments 
        GROUP BY task_type 
        ORDER BY count DESC
    """)
    task_counts = {row['task_type']: row['count'] for row in cursor.fetchall()}
    
    # 4. Dataset usage breakdown
    cursor.execute("""
        SELECT dataset_name, COUNT(*) as count 
        FROM experiments 
        GROUP BY dataset_name 
        ORDER BY count DESC
    """)
    dataset_counts = {row['dataset_name']: row['count'] for row in cursor.fetchall()}
    
    # 5. Average execution time
    cursor.execute("SELECT AVG(execution_time) FROM experiments")
    avg_row = cursor.fetchone()
    avg_execution_time = round(avg_row[0], 4) if avg_row and avg_row[0] is not None else 0.0
    
    conn.close()
    
    return {
        'total_experiments': total_experiments,
        'total_comparisons': total_comparisons,
        'strategy_counts': strategy_counts,
        'task_counts': task_counts,
        'dataset_counts': dataset_counts,
        'avg_execution_time': avg_execution_time
    }

# Initialize database on module load
init_db()
