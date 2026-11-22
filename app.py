"""
FoldWise AI - Core Flask Web Application
Intelligent Cross-Validation and Machine Learning Model Evaluation Platform
"""

import os
import io
import csv
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from database.db import (
    init_db,
    save_experiment,
    get_experiments,
    get_experiment_by_id,
    delete_experiment,
    clear_all_experiments,
    save_comparison,
    get_comparisons,
    get_comparison_by_id,
    get_database_stats,
    filter_experiments
)
from dataset.loader import (
    load_dataset,
    list_registered_datasets,
    save_uploaded_csv,
    DATASET_REGISTRY
)
from dataset.analyzer import analyze_dataset
from preprocessing.processor import (
    prepare_dataset_for_modeling,
    create_preprocessor,
    build_leak_free_pipeline,
    identify_feature_types
)
from models.classifiers import get_classifier
from models.regressors import get_regressor
from validation.kfold import run_kfold_cv
from validation.stratified import run_stratified_kfold_cv, compare_kfold_and_stratified
from validation.nested import run_nested_cv, compare_nested_vs_non_nested
from validation.timeseries import run_timeseries_cv, compare_timeseries_vs_random_kfold
from validation.benchmark import benchmark_validation_strategies
from evaluation.comparison import compare_models
from evaluation.visualizer import generate_fold_allocation_diagram, generate_radar_chart_payload, generate_dashboard_charts_payload
from recommendation.strategy_recommender import analyze_dataset_for_validation

# Initialize application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'foldwise-ai-secret-key-2026-ml-platform')

# Configure persistent data directory if specified (e.g. Render Persistent Disk mount)
DATA_DIR = os.environ.get('DATA_DIR')
if DATA_DIR:
    app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploaded')
    app.config['CACHE_FOLDER'] = os.path.join(DATA_DIR, 'cache')
else:
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'uploaded')
    app.config['CACHE_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'cache')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload and cache directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)

# Ensure database is initialized
init_db()

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------------
# Web Page Routes
# -------------------------------------------------------------

@app.route('/')
def index():
    """Overview & Strategy Hub dashboard."""
    db_stats = get_database_stats()
    datasets = list_registered_datasets()
    recent_experiments = get_experiments(limit=10)
    recent_comparisons = get_comparisons(limit=5)
    charts_payload = generate_dashboard_charts_payload(db_stats, datasets)
    
    stats = {
        'dataset_count': len(datasets),
        'experiment_count': db_stats['total_experiments'],
        'comparison_count': db_stats['total_comparisons'],
        'avg_execution_time': db_stats['avg_execution_time'],
        'classifier_count': 5,
        'regressor_count': 4,
        'strategy_count': 4
    }
    return render_template(
        'dashboard.html',
        active_page='dashboard',
        stats=stats,
        db_stats=db_stats,
        recent_experiments=recent_experiments,
        recent_comparisons=recent_comparisons,
        charts_payload=charts_payload
    )

@app.route('/datasets')
def datasets_view():
    """Dataset Explorer page."""
    return render_template('datasets.html', active_page='datasets')

@app.route('/preprocessing')
def preprocessing_view():
    """Data Preprocessing & Pipeline Isolation overview."""
    return render_template('preprocessing.html', active_page='preprocessing')

@app.route('/validation')
def validation_view():
    """Cross-Validation Lab comparative interface."""
    return render_template('validation.html', active_page='validation')

@app.route('/training')
def training_view():
    """Model Training & Experimentation Studio."""
    return render_template('training.html', active_page='training')

@app.route('/comparison')
def comparison_view():
    """Cross-validation strategy comparative evaluation engine."""
    experiments = get_experiments(limit=100)
    return render_template('comparison.html', active_page='comparison', experiments=experiments)

@app.route('/history')
def history_view():
    """Experiment history & SQLite logs with dynamic filtering support."""
    dataset = request.args.get('dataset', None)
    method = request.args.get('method', None)
    task_type = request.args.get('task_type', None)
    algorithm = request.args.get('algorithm', None)
    search = request.args.get('q', None)
    sort_by = request.args.get('sort_by', 'created_at')
    sort_dir = request.args.get('sort_dir', 'desc')
    
    experiments = filter_experiments(
        dataset=dataset,
        validation_method=method,
        task_type=task_type,
        algorithm=algorithm,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=200
    )
    datasets = list_registered_datasets()
    db_stats = get_database_stats()
    
    return render_template(
        'history.html',
        active_page='history',
        experiments=experiments,
        datasets=datasets,
        db_stats=db_stats,
        current_filters={
            'dataset': dataset or '',
            'method': method or '',
            'task_type': task_type or '',
            'algorithm': algorithm or '',
            'q': search or '',
            'sort_by': sort_by,
            'sort_dir': sort_dir
        }
    )

@app.route('/settings')
def settings_view():
    """System preferences and project metadata."""
    db_stats = get_database_stats()
    return render_template('settings.html', active_page='settings', db_stats=db_stats)

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route('/api/health')
def health_check():
    """Health check endpoint for platform verification."""
    return jsonify({
        'status': 'healthy',
        'application': 'FoldWise AI',
        'version': '1.0.0',
        'database': 'SQLite connected'
    }), 200

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats_endpoint():
    """Retrieve live aggregated KPI metrics, strategy distribution, and activity stream."""
    db_stats = get_database_stats()
    datasets = list_registered_datasets()
    recent_experiments = get_experiments(limit=10)
    recent_comparisons = get_comparisons(limit=5)
    charts = generate_dashboard_charts_payload(db_stats, datasets)
    
    return jsonify({
        'stats': {
            'dataset_count': len(datasets),
            'experiment_count': db_stats['total_experiments'],
            'comparison_count': db_stats['total_comparisons'],
            'avg_execution_time': db_stats['avg_execution_time'],
            'classifier_count': 5,
            'regressor_count': 4,
            'strategy_count': 4
        },
        'db_stats': db_stats,
        'recent_experiments': recent_experiments,
        'recent_comparisons': recent_comparisons,
        'charts': charts
    }), 200

# -------------------------------------------------------------
# Dataset Management Endpoints
# -------------------------------------------------------------

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """List all registered benchmark datasets and uploaded CSVs."""
    datasets = list_registered_datasets()
    return jsonify(datasets), 200

@app.route('/api/datasets/<dataset_name>', methods=['GET'])
def get_dataset_details(dataset_name):
    """Load and return comprehensive statistical analysis of a dataset."""
    target_col = request.args.get('target', None)
    try:
        data_dict = load_dataset(dataset_name, target_column=target_col)
        analysis = analyze_dataset(
            df=data_dict['df'],
            target_column=data_dict['target'],
            task_type=data_dict['task_type']
        )
        return jsonify({
            'id': data_dict['id'],
            'name': data_dict['name'],
            'target': data_dict['target'],
            'features': data_dict['features'],
            'task_type': data_dict['task_type'],
            'is_multiclass': data_dict['is_multiclass'],
            'n_samples': data_dict['n_samples'],
            'n_features': data_dict['n_features'],
            'source': data_dict['source'],
            'analysis': analysis
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/datasets/<dataset_name>/retry-download', methods=['POST'])
def retry_download(dataset_name):
    """Force re-download of an external dataset."""
    try:
        data_dict = load_dataset(dataset_name, force_download=True)
        return jsonify({
            'message': f"Dataset '{dataset_name}' downloaded and cached successfully.",
            'id': data_dict['id'],
            'samples': data_dict['n_samples'],
            'features': data_dict['n_features']
        }), 200
    except Exception as e:
        return jsonify({'error': f"Failed to download '{dataset_name}': {str(e)}"}), 502

@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Handle custom CSV file upload securely."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Only CSV files (.csv) are supported'}), 400
        
    target_col = request.form.get('target_column', None)
    try:
        dataset_id, meta = save_uploaded_csv(file, target_col=target_col)
        return jsonify({
            'message': 'File uploaded and parsed successfully',
            'dataset_id': dataset_id,
            'name': meta['name'],
            'samples': meta['n_samples'],
            'features': meta['n_features'],
            'target': meta['target'],
            'task_type': meta['task_type']
        }), 201
    except Exception as e:
        return jsonify({'error': f"Upload processing error: {str(e)}"}), 400

# -------------------------------------------------------------
# Preprocessing Inspection Endpoint
# -------------------------------------------------------------

@app.route('/api/preprocessing/inspect', methods=['GET', 'POST'])
def inspect_preprocessing():
    """Inspect and test the leak-free preprocessing pipeline on a chosen dataset."""
    if request.method == 'POST':
        data = request.get_json() or {}
        dataset_name = data.get('dataset_name', 'iris')
        target_col = data.get('target_column', None)
        scaling = data.get('scaling', 'standard')
        imputation = data.get('imputation', 'mean')
        lags = data.get('lags', [1, 2, 3])
    else:
        dataset_name = request.args.get('dataset_name', 'iris')
        target_col = request.args.get('target_column', None)
        scaling = request.args.get('scaling', 'standard')
        imputation = request.args.get('imputation', 'mean')
        lags = [1, 2, 3]

    try:
        ds_dict = load_dataset(dataset_name, target_column=target_col)
        prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_col, lag_orders=lags)
        
        X = prepared['X']
        y = prepared['y']
        num_cols = prepared['numerical_cols']
        cat_cols = prepared['categorical_cols']
        
        preprocessor = create_preprocessor(
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            scaling=scaling,
            imputation_strategy=imputation
        )
        
        # Test fitting preprocessor on a sample to verify pipeline integrity
        X_trans = preprocessor.fit_transform(X)
        output_features_count = int(X_trans.shape[1]) if hasattr(X_trans, 'shape') else len(num_cols) + len(cat_cols)
        
        # Feature matrix sample preview
        preview_rows = X.head(5).to_dict(orient='records')
        
        return jsonify({
            'dataset_name': dataset_name,
            'task_type': prepared['task_type'],
            'target_name': prepared['target_name'],
            'total_samples': int(len(X)),
            'numerical_features': num_cols,
            'categorical_features': cat_cols,
            'num_count': len(num_cols),
            'cat_count': len(cat_cols),
            'scaling': scaling,
            'imputation': imputation,
            'transformed_features_count': output_features_count,
            'preview': preview_rows,
            'pipeline_architecture': {
                'numerical': [
                    {'step': 'SimpleImputer', 'strategy': imputation},
                    {'step': 'Scaler', 'type': 'StandardScaler' if scaling == 'standard' else ('MinMaxScaler' if scaling == 'minmax' else 'None')}
                ],
                'categorical': [
                    {'step': 'SimpleImputer', 'strategy': 'most_frequent'},
                    {'step': 'OneHotEncoder', 'handle_unknown': 'ignore'}
                ],
                'isolation': 'Wrapped in ColumnTransformer & Pipeline; fitted strictly inside training fold indices'
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# -------------------------------------------------------------
# Experiment Execution & Validation Endpoints
# -------------------------------------------------------------

@app.route('/api/experiments/run', methods=['POST'])
def run_experiment_endpoint():
    """Execute a real Machine Learning cross-validation experiment and log to SQLite."""
    data = request.get_json() or {}
    
    dataset_name = data.get('dataset_name', 'iris')
    target_col = data.get('target_column', None)
    algorithm = data.get('algorithm', 'LogisticRegression')
    validation_method = data.get('validation_method', 'stratified').lower()
    n_splits = int(data.get('n_splits', 5))
    random_state = int(data.get('random_state', 42))
    scaling = data.get('scaling', 'standard')
    imputation = data.get('imputation', 'mean')
    hyperparams = data.get('hyperparameters', {})
    
    if n_splits < 2 or n_splits > 50:
        return jsonify({'error': 'Number of splits (n_splits) must be between 2 and 50.'}), 400
        
    if validation_method in ['timeseries', 'time_series']:
        gap = int(data.get('gap', 0))
        if gap < 0:
            return jsonify({'error': 'Horizon gap cannot be negative.'}), 400
    
    try:
        # 1. Load actual dataset
        ds_dict = load_dataset(dataset_name, target_column=target_col)
        prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_col)
        
        X = prepared['X']
        y = prepared['y']
        task_type = prepared['task_type']
        num_cols = prepared['numerical_cols']
        cat_cols = prepared['categorical_cols']
        is_multiclass = ds_dict.get('is_multiclass', False)
        
        # 2. Instantiate actual ML model
        if task_type == 'classification':
            model = get_classifier(algorithm, hyperparameters=hyperparams, random_state=random_state)
        else:
            model = get_regressor(algorithm, hyperparameters=hyperparams, random_state=random_state)
            
        # 3. Execute selected validation strategy
        if validation_method == 'kfold':
            result = run_kfold_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
                task_type=task_type,
                is_multiclass=is_multiclass,
                scaling=scaling,
                imputation=imputation
            )
        elif validation_method == 'stratified':
            if task_type != 'classification':
                return jsonify({'error': 'Stratified K-Fold is only applicable to classification tasks with discrete labels.'}), 400
                
            result = run_stratified_kfold_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
                is_multiclass=is_multiclass,
                scaling=scaling,
                imputation=imputation
            )
        elif validation_method == 'nested':
            outer_splits = int(data.get('outer_splits', n_splits if n_splits else 5))
            inner_splits = int(data.get('inner_splits', 3))
            if outer_splits < 2 or inner_splits < 2:
                return jsonify({'error': 'Nested CV outer and inner splits must each be at least 2.'}), 400
            
            result = run_nested_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                shuffle=True,
                random_state=random_state,
                task_type=task_type,
                is_multiclass=is_multiclass,
                scaling=scaling,
                imputation=imputation
            )
        elif validation_method in ['timeseries', 'time_series']:
            gap = int(data.get('gap', 0))
            if gap < 0:
                return jsonify({'error': 'Horizon gap cannot be negative.'}), 400
            date_labels = prepared.get('date_labels', None)
            
            result = run_timeseries_cv(
                model=model,
                X=X,
                y=y,
                numerical_cols=num_cols,
                categorical_cols=cat_cols,
                n_splits=n_splits,
                gap=gap,
                scaling=scaling,
                imputation=imputation,
                date_labels=date_labels
            )
        else:
            return jsonify({'error': f"Validation method '{validation_method}' not recognized."}), 400
            
        # 4. Save experiment to SQLite database
        exp_id = str(uuid.uuid4())[:12]
        best_params_record = result.get('best_params_per_fold', {})
        db_record = {
            'experiment_id': exp_id,
            'dataset_name': dataset_name,
            'target_column': prepared['target_name'],
            'task_type': task_type,
            'algorithm': algorithm,
            'validation_method': result['method'],
            'hyperparameters': hyperparams,
            'metrics': result['summary'],
            'fold_scores': result['summary']['scores_per_fold'],
            'best_params': best_params_record,
            'execution_time': result['execution_time'],
            'notes': f"Executed via {result['method']}"
        }
        save_experiment(db_record)
        
        result['experiment_id'] = exp_id
        result['dataset_name'] = dataset_name
        result['target_column'] = prepared['target_name']
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/validation/compare', methods=['POST'])
def compare_validation_methods():
    """Compare K-Fold vs Stratified K-Fold side-by-side on the same model and data."""
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'breast_cancer')
    target_col = data.get('target_column', None)
    algorithm = data.get('algorithm', 'LogisticRegression')
    n_splits = int(data.get('n_splits', 5))
    random_state = int(data.get('random_state', 42))
    
    if n_splits < 2 or n_splits > 50:
        return jsonify({'error': 'Number of splits must be between 2 and 50.'}), 400
    
    try:
        ds_dict = load_dataset(dataset_name, target_column=target_col)
        if ds_dict['task_type'] != 'classification':
            return jsonify({'error': 'Comparison between K-Fold and Stratified K-Fold requires a classification dataset.'}), 400
            
        prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_col)
        model = get_classifier(algorithm, random_state=random_state)
        
        comparison = compare_kfold_and_stratified(
            model=model,
            X=prepared['X'],
            y=prepared['y'],
            numerical_cols=prepared['numerical_cols'],
            categorical_cols=prepared['categorical_cols'],
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
            is_multiclass=ds_dict.get('is_multiclass', False)
        )
        return jsonify(comparison), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/validation/nested-compare', methods=['POST'])
def compare_nested_endpoint():
    """Empirically compare Nested CV against Non-Nested CV to demonstrate selection bias."""
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'iris')
    target_col = data.get('target_column', None)
    algorithm = data.get('algorithm', 'LogisticRegression')
    outer_splits = int(data.get('outer_splits', 5))
    inner_splits = int(data.get('inner_splits', 3))
    random_state = int(data.get('random_state', 42))
    
    try:
        ds_dict = load_dataset(dataset_name, target_column=target_col)
        prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_col)
        task_type = prepared['task_type']
        
        if task_type == 'classification':
            model = get_classifier(algorithm, random_state=random_state)
        else:
            model = get_regressor(algorithm, random_state=random_state)
            
        comparison = compare_nested_vs_non_nested(
            model=model,
            X=prepared['X'],
            y=prepared['y'],
            numerical_cols=prepared['numerical_cols'],
            categorical_cols=prepared['categorical_cols'],
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            shuffle=True,
            random_state=random_state,
            task_type=task_type,
            is_multiclass=ds_dict.get('is_multiclass', False)
        )
        return jsonify(comparison), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/validation/timeseries-compare', methods=['POST'])
def compare_timeseries_endpoint():
    """Empirically compare Time-Series CV vs naive random K-Fold on Airline Passengers."""
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'air_passengers')
    target_col = data.get('target_column', None)
    algorithm = data.get('algorithm', 'RandomForestRegressor')
    n_splits = int(data.get('n_splits', 5))
    random_state = int(data.get('random_state', 42))
    
    if n_splits < 2 or n_splits > 50:
        return jsonify({'error': 'Number of splits must be between 2 and 50.'}), 400
    
    try:
        ds_dict = load_dataset(dataset_name, target_column=target_col)
        prepared = prepare_dataset_for_modeling(ds_dict, target_col=target_col)
        model = get_regressor(algorithm, random_state=random_state)
        
        comparison = compare_timeseries_vs_random_kfold(
            model=model,
            X=prepared['X'],
            y=prepared['y'],
            numerical_cols=prepared['numerical_cols'],
            categorical_cols=prepared['categorical_cols'],
            n_splits=n_splits,
            random_state=random_state
        )
        return jsonify(comparison), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/comparison/run', methods=['POST'])
def run_model_comparison_endpoint():
    """Execute multi-model cross-validation comparison and generate recommendations."""
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'iris')
    algorithms = data.get('algorithms', None)
    validation_method = data.get('validation_method', None)
    n_splits = int(data.get('n_splits', 5))
    primary_metric = data.get('primary_metric', None)
    random_state = int(data.get('random_state', 42))
    target_column = data.get('target_column', None)
    scaling = data.get('scaling', 'standard')
    imputation = data.get('imputation', 'mean')
    gap = int(data.get('gap', 0))
    outer_splits = int(data.get('outer_splits', 5))
    inner_splits = int(data.get('inner_splits', 3))
    
    if algorithms is not None and (not isinstance(algorithms, list) or len(algorithms) == 0):
        return jsonify({'error': 'Algorithms must be a non-empty list of algorithm names.'}), 400
    if n_splits < 2 or n_splits > 50:
        return jsonify({'error': 'Number of splits (n_splits) must be between 2 and 50.'}), 400
        
    if validation_method in ['timeseries', 'time_series']:
        gap = int(data.get('gap', 0))
        if gap < 0:
            return jsonify({'error': 'Horizon gap cannot be negative.'}), 400
    
    try:
        report = compare_models(
            dataset_name=dataset_name,
            algorithms=algorithms,
            validation_method=validation_method,
            n_splits=n_splits,
            primary_metric=primary_metric,
            random_state=random_state,
            target_column=target_column,
            scaling=scaling,
            imputation=imputation,
            gap=gap,
            outer_splits=outer_splits,
            inner_splits=inner_splits
        )
        return jsonify(report), 200
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"Comparison error: {str(e)}"}), 500

@app.route('/api/comparison/history', methods=['GET'])
def list_comparison_history():
    """Retrieve saved model comparisons."""
    limit = request.args.get('limit', 50, type=int)
    history = get_comparisons(limit=limit)
    return jsonify(history), 200

@app.route('/api/comparison/<comparison_id>', methods=['GET'])
def get_comparison_detail(comparison_id):
    """Retrieve a single saved comparison record."""
    comp = get_comparison_by_id(comparison_id)
    if not comp:
        return jsonify({'error': 'Comparison record not found'}), 404
    return jsonify(comp), 200

# -------------------------------------------------------------
# Strategy Recommender & Visualization Endpoints
# -------------------------------------------------------------

@app.route('/api/recommendation/strategy', methods=['POST'])
def recommend_strategy_endpoint():
    """Diagnose dataset characteristics and recommend optimal validation strategy."""
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'iris')
    target_column = data.get('target_column', None)
    
    try:
        ds_dict = load_dataset(dataset_name, target_column=target_column)
        report = analyze_dataset_for_validation(ds_dict, target_col=target_column)
        return jsonify(report), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/validation/strategy-benchmark', methods=['POST'])
def benchmark_strategies_endpoint():
    """Directly benchmark compatible validation strategies on the same model and data."""
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'breast_cancer')
    algorithm = data.get('algorithm', None)
    target_column = data.get('target_column', None)
    n_splits = int(data.get('n_splits', 5))
    random_state = int(data.get('random_state', 42))
    
    if n_splits < 2 or n_splits > 50:
        return jsonify({'error': 'Number of splits must be between 2 and 50.'}), 400
    
    try:
        bench_res = benchmark_validation_strategies(
            dataset_name=dataset_name,
            algorithm=algorithm,
            target_column=target_column,
            n_splits=n_splits,
            random_state=random_state
        )
        return jsonify(bench_res), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualization/fold-allocation', methods=['GET'])
def get_fold_allocation():
    """Return coordinate matrix of train and validation segments for plotting."""
    dataset_name = request.args.get('dataset', 'iris')
    validation_method = request.args.get('method', 'kfold')
    n_splits = request.args.get('n_splits', 5, type=int)
    gap = request.args.get('gap', 0, type=int)
    
    try:
        alloc = generate_fold_allocation_diagram(
            dataset_name=dataset_name,
            validation_method=validation_method,
            n_splits=n_splits,
            gap=gap
        )
        return jsonify(alloc), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    """Retrieve list of completed experiments with filtering, search, and sorting."""
    dataset = request.args.get('dataset', None)
    method = request.args.get('method', None)
    task_type = request.args.get('task_type', None)
    algorithm = request.args.get('algorithm', None)
    search = request.args.get('q', None)
    sort_by = request.args.get('sort_by', 'created_at')
    sort_dir = request.args.get('sort_dir', 'desc')
    limit = request.args.get('limit', 200, type=int)
    
    experiments = filter_experiments(
        dataset=dataset,
        validation_method=method,
        task_type=task_type,
        algorithm=algorithm,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=limit
    )
    return jsonify(experiments), 200

@app.route('/api/experiments/<experiment_id>', methods=['GET'])
def get_experiment_details(experiment_id):
    """Retrieve detailed single experiment record."""
    exp = get_experiment_by_id(experiment_id)
    if not exp:
        return jsonify({'error': 'Experiment not found'}), 404
    return jsonify(exp), 200

@app.route('/api/experiments/<experiment_id>', methods=['DELETE'])
def remove_experiment(experiment_id):
    """Delete an experiment record."""
    success = delete_experiment(experiment_id)
    if not success:
        return jsonify({'error': 'Experiment not found or could not be deleted'}), 404
    return jsonify({'message': f'Experiment {experiment_id} deleted successfully'}), 200

@app.route('/api/experiments/clear', methods=['POST'])
def clear_history():
    """Clear all experiment history records."""
    clear_all_experiments()
    return jsonify({'message': 'All experiment history cleared'}), 200

@app.route('/api/experiments/export/csv')
def export_experiments_csv():
    """Export experiment results as a CSV file with filtering support."""
    dataset = request.args.get('dataset', None)
    method = request.args.get('method', None)
    task_type = request.args.get('task_type', None)
    algorithm = request.args.get('algorithm', None)
    search = request.args.get('q', None)
    sort_by = request.args.get('sort_by', 'created_at')
    sort_dir = request.args.get('sort_dir', 'desc')
    limit = request.args.get('limit', 1000, type=int)
    
    experiments = filter_experiments(
        dataset=dataset,
        validation_method=method,
        task_type=task_type,
        algorithm=algorithm,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=limit
    )
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Experiment ID', 'Dataset', 'Target', 'Task Type', 'Algorithm',
        'Validation Method', 'Execution Time (s)', 'Primary Metric', 'Fold Mean Score',
        'Fold Std Dev', 'Min Score', 'Max Score', 'Created At'
    ])
    
    for exp in experiments:
        m = exp.get('metrics', {})
        writer.writerow([
            exp['experiment_id'],
            exp['dataset_name'],
            exp['target_column'],
            exp['task_type'],
            exp['algorithm'],
            exp['validation_method'],
            exp['execution_time'],
            m.get('accuracy', m.get('rmse', 'N/A')),
            m.get('mean_score', 'N/A'),
            m.get('std_score', 'N/A'),
            m.get('min_score', 'N/A'),
            m.get('max_score', 'N/A'),
            exp['created_at']
        ])
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=foldwise_experiments.csv"}
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
