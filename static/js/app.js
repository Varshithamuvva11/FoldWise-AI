/**
 * FoldWise AI - Core Frontend Utilities & Interactions
 * Manages theme, dataset exploration, live Plotly visualizations, and CSV uploads.
 */

let currentLoadedDataset = null;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initSidebarToggle();
    initDatasetExplorer();
});

// Theme Management (Dark / Light)
function initTheme() {
    const savedTheme = localStorage.getItem('foldwise_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeToggleUI(savedTheme);

    const toggleBtn = document.getElementById('themeToggleBtn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('foldwise_theme', newTheme);
            updateThemeToggleUI(newTheme);
            
            // Re-render plotly charts with current theme template if open
            if (currentLoadedDataset && currentLoadedDataset.analysis) {
                renderTargetDistribution(currentLoadedDataset.analysis.target_distribution);
                renderCorrelationHeatmap(currentLoadedDataset.analysis.correlation_matrix);
            }
        });
    }
}

function updateThemeToggleUI(theme) {
    const toggleBtn = document.getElementById('themeToggleBtn');
    if (!toggleBtn) return;
    
    if (theme === 'light') {
        toggleBtn.innerHTML = '<i class="bi bi-moon-stars"></i> <span>Dark Mode</span>';
    } else {
        toggleBtn.innerHTML = '<i class="bi bi-sun"></i> <span>Light Mode</span>';
    }
}

// Sidebar toggle for mobile / responsive
function initSidebarToggle() {
    const mobileBtn = document.getElementById('mobileSidebarToggle');
    const sidebar = document.querySelector('.app-sidebar');
    
    if (mobileBtn && sidebar) {
        mobileBtn.addEventListener('click', () => {
            sidebar.classList.toggle('show');
        });
    }
}

// Toast Notification Helper
function showToast(title, message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    const iconClass = type === 'success' ? 'bi-check-circle-fill text-success' :
                      type === 'danger' ? 'bi-exclamation-triangle-fill text-danger' :
                      type === 'warning' ? 'bi-exclamation-circle-fill text-warning' :
                      'bi-info-circle-fill text-primary';
                      
    const toastId = 'toast_' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center fw-card border-0 mb-2" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body d-flex align-items-start gap-2">
                    <i class="bi ${iconClass} fs-5"></i>
                    <div>
                        <strong class="d-block text-primary">${title}</strong>
                        <span class="text-secondary small">${message}</span>
                    </div>
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', toastHtml);
    const toastElem = document.getElementById(toastId);
    if (window.bootstrap && window.bootstrap.Toast) {
        const bsToast = new window.bootstrap.Toast(toastElem, { delay: 4000 });
        bsToast.show();
        toastElem.addEventListener('hidden.bs.toast', () => toastElem.remove());
    } else {
        setTimeout(() => toastElem.remove(), 4000);
    }
}

// -------------------------------------------------------------
// Dataset Explorer Management
// -------------------------------------------------------------
function initDatasetExplorer() {
    // 1. Load Dataset Buttons
    document.querySelectorAll('.load-dataset-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const datasetName = this.getAttribute('data-dataset');
            loadAndDisplayDataset(datasetName);
        });
    });

    // 2. Retry Download Buttons
    document.querySelectorAll('.retry-download-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const datasetName = this.getAttribute('data-dataset');
            retryDatasetDownload(datasetName);
        });
    });

    // 3. Reanalyze on Target Change
    const reanalyzeBtn = document.getElementById('reanalyzeBtn');
    if (reanalyzeBtn) {
        reanalyzeBtn.addEventListener('click', () => {
            const targetSelect = document.getElementById('activeTargetSelect');
            if (currentLoadedDataset && targetSelect) {
                const newTarget = targetSelect.value;
                loadAndDisplayDataset(currentLoadedDataset.id, newTarget);
            }
        });
    }

    // 4. Custom CSV Upload Form
    const uploadForm = document.getElementById('uploadCsvForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('csvFileInput');
            const targetInput = document.getElementById('targetColInput');
            const feedback = document.getElementById('uploadFeedback');
            const submitBtn = document.getElementById('submitUploadBtn');
            
            if (!fileInput.files || fileInput.files.length === 0) {
                feedback.textContent = 'Please choose a CSV file to upload.';
                feedback.classList.remove('d-none');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            if (targetInput && targetInput.value.trim()) {
                formData.append('target_column', targetInput.value.trim());
            }
            
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
            feedback.classList.add('d-none');
            
            fetch('/api/datasets/upload', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json().then(data => ({ status: res.status, body: data })))
            .then(({ status, body }) => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="bi bi-upload"></i> <span>Upload & Ingest</span>';
                
                if (status === 201) {
                    // Close modal
                    const modalElem = document.getElementById('uploadModal');
                    const modalInstance = bootstrap.Modal.getInstance(modalElem);
                    if (modalInstance) modalInstance.hide();
                    uploadForm.reset();
                    
                    showToast('CSV Uploaded', `Ingested ${body.name} (${body.samples} rows, ${body.features} features)`, 'success');
                    // Automatically load the uploaded dataset
                    loadAndDisplayDataset(body.dataset_id);
                } else {
                    feedback.textContent = body.error || 'Upload failed.';
                    feedback.classList.remove('d-none');
                }
            })
            .catch(err => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="bi bi-upload"></i> <span>Upload & Ingest</span>';
                feedback.textContent = 'Network or server error during upload.';
                feedback.classList.remove('d-none');
            });
        });
    }
}

function loadAndDisplayDataset(datasetName, targetCol = null) {
    const placeholder = document.getElementById('datasetDetailsPlaceholder');
    const activeDetails = document.getElementById('datasetDetailsActive');
    const titleElem = document.getElementById('activeDatasetTitle');
    const subtitleElem = document.getElementById('activeDatasetSubtitle');
    const targetWrapper = document.getElementById('targetSelectWrapper');
    
    if (placeholder) {
        placeholder.innerHTML = `
            <div class="spinner-border text-primary my-4" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-secondary small">Fetching and calculating live statistical summaries for <strong>${datasetName}</strong>...</p>
        `;
        placeholder.style.display = 'block';
    }
    if (activeDetails) activeDetails.style.display = 'none';

    let url = `/api/datasets/${datasetName}`;
    if (targetCol) {
        url += `?target=${encodeURIComponent(targetCol)}`;
    }

    fetch(url)
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || 'Failed to load dataset'); });
            }
            return res.json();
        })
        .then(data => {
            currentLoadedDataset = data;
            
            if (placeholder) placeholder.style.display = 'none';
            if (activeDetails) activeDetails.style.display = 'block';
            
            // Header titles
            if (titleElem) titleElem.innerHTML = `<i class="bi bi-check-circle-fill text-success"></i> <span>${data.name}</span>`;
            if (subtitleElem) subtitleElem.textContent = `Loaded from ${data.source.toUpperCase()} &bull; Target: ${data.target} &bull; ${data.n_samples} samples, ${data.n_features} features`;

            // Summary Pills
            const a = data.analysis;
            document.getElementById('statRows').textContent = a.shape.rows.toLocaleString();
            document.getElementById('statFeatures').textContent = data.n_features;
            document.getElementById('statTaskType').textContent = data.task_type;
            
            const totalNulls = a.columns_info.reduce((acc, col) => acc + col.null_count, 0);
            document.getElementById('statMissing').textContent = totalNulls;
            document.getElementById('statDuplicates').textContent = a.duplicates.count;
            document.getElementById('statSource').textContent = data.source;

            // Target selector dropdown
            if (targetWrapper) {
                targetWrapper.style.display = 'flex';
                const targetSelect = document.getElementById('activeTargetSelect');
                targetSelect.innerHTML = '';
                const allCols = a.columns_info.map(c => c.name);
                allCols.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    if (col === data.target) opt.selected = true;
                    targetSelect.appendChild(opt);
                });
            }

            // Training redirect button parameter
            const trainBtn = document.getElementById('proceedToTrainingBtn');
            if (trainBtn) {
                trainBtn.href = `/training?dataset=${data.id}&target=${encodeURIComponent(data.target)}`;
            }

            // Render all tabs
            renderDataPreview(a.preview, data.target);
            renderNumericalStats(a.numerical_summary);
            renderTargetDistribution(a.target_distribution);
            renderCorrelationHeatmap(a.correlation_matrix);
            renderSchemaTable(a.columns_info, data.target);

            showToast('Dataset Loaded', `Successfully inspected ${data.name}`, 'success');
        })
        .catch(err => {
            if (placeholder) {
                placeholder.innerHTML = `
                    <div class="alert alert-danger mx-auto mt-3" style="max-width: 500px;">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        <strong>Error loading dataset:</strong> ${err.message}
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-danger retry-download-btn" data-dataset="${datasetName}">
                                <i class="bi bi-arrow-repeat me-1"></i> Retry Download / Reload
                            </button>
                        </div>
                    </div>
                `;
                // reattach retry listener
                placeholder.querySelectorAll('.retry-download-btn').forEach(b => {
                    b.addEventListener('click', () => retryDatasetDownload(datasetName));
                });
            }
            showToast('Loading Error', err.message, 'danger');
        });
}

function retryDatasetDownload(datasetName) {
    showToast('Download Initiated', `Attempting fresh fetch for ${datasetName}...`, 'info');
    fetch(`/api/datasets/${datasetName}/retry-download`, { method: 'POST' })
        .then(res => res.json().then(data => ({ status: res.status, body: data })))
        .then(({ status, body }) => {
            if (status === 200) {
                showToast('Download Success', body.message, 'success');
                loadAndDisplayDataset(datasetName);
            } else {
                showToast('Download Failed', body.error || 'Server error', 'danger');
            }
        })
        .catch(err => {
            showToast('Download Error', err.message, 'danger');
        });
}

// -------------------------------------------------------------
// Data Tab Renderers
// -------------------------------------------------------------

function renderDataPreview(rows, targetCol) {
    const thead = document.getElementById('previewTableHead');
    const tbody = document.getElementById('previewTableBody');
    if (!rows || rows.length === 0) return;

    const columns = Object.keys(rows[0]);
    thead.innerHTML = '<tr>' + columns.map(c => 
        `<th class="${c === targetCol ? 'text-primary fw-bold' : ''}">${c} ${c === targetCol ? '<span class="badge bg-primary ms-1">Target</span>' : ''}</th>`
    ).join('') + '</tr>';

    tbody.innerHTML = rows.map(r => 
        '<tr>' + columns.map(c => 
            `<td class="${c === targetCol ? 'fw-semibold text-info' : ''}">${r[c] !== null ? r[c] : '<span class="text-muted">NaN</span>'}</td>`
        ).join('') + '</tr>'
    ).join('');
}

function renderNumericalStats(stats) {
    const tbody = document.getElementById('statsTableBody');
    if (!stats || Object.keys(stats).length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted py-3">No continuous numeric features found in this dataset.</td></tr>';
        return;
    }

    tbody.innerHTML = Object.entries(stats).map(([feat, s]) => `
        <tr>
            <td><strong>${feat}</strong></td>
            <td>${s.count}</td>
            <td class="text-light">${s.mean}</td>
            <td class="text-secondary">&plusmn;${s.std}</td>
            <td>${s.min}</td>
            <td>${s.q25}</td>
            <td class="text-info fw-semibold">${s.median}</td>
            <td>${s.q75}</td>
            <td>${s.max}</td>
        </tr>
    `).join('');
}

function renderTargetDistribution(targetData) {
    const plotElem = document.getElementById('targetDistributionPlot');
    if (!plotElem || !targetData || !targetData.values) return;

    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const isDark = currentTheme === 'dark';

    const labels = targetData.values.map(v => v.label);
    const counts = targetData.values.map(v => v.count);
    const percentages = targetData.values.map(v => v.percentage);

    const trace = {
        x: labels,
        y: counts,
        text: percentages.map(p => `${p}%`),
        textposition: 'auto',
        type: 'bar',
        marker: {
            color: 'rgba(99, 102, 241, 0.85)',
            line: { color: 'rgba(99, 102, 241, 1)', width: 1.5 }
        }
    };

    const layout = {
        title: {
            text: `Target Class / Frequency Distribution (${targetData.column})`,
            font: { color: isDark ? '#F9FAFB' : '#0F172A', size: 14 }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: isDark ? '#9CA3AF' : '#475569', family: 'Inter, sans-serif' },
        xaxis: {
            title: targetData.column,
            gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
        },
        yaxis: {
            title: 'Sample Count',
            gridcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
        },
        margin: { l: 50, r: 20, t: 40, b: 50 }
    };

    Plotly.newPlot(plotElem, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderCorrelationHeatmap(corrData) {
    const plotElem = document.getElementById('correlationHeatmapPlot');
    if (!plotElem || !corrData || !corrData.columns || corrData.columns.length === 0) {
        if (plotElem) plotElem.innerHTML = '<div class="text-center py-5 text-muted">Insufficient numeric features for correlation matrix.</div>';
        return;
    }

    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const isDark = currentTheme === 'dark';

    const trace = {
        z: corrData.matrix,
        x: corrData.columns,
        y: corrData.columns,
        type: 'heatmap',
        colorscale: [
            [0.0, '#3B82F6'],
            [0.5, '#1E293B'],
            [1.0, '#EF4444']
        ],
        zmin: -1,
        zmax: 1
    };

    const layout = {
        title: {
            text: 'Pearson Feature Correlation Matrix',
            font: { color: isDark ? '#F9FAFB' : '#0F172A', size: 14 }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: isDark ? '#9CA3AF' : '#475569', family: 'Inter, sans-serif' },
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' },
        margin: { l: 120, r: 20, t: 40, b: 100 }
    };

    Plotly.newPlot(plotElem, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderSchemaTable(columns, targetCol) {
    const tbody = document.getElementById('schemaTableBody');
    if (!tbody || !columns) return;

    tbody.innerHTML = columns.map(c => `
        <tr class="${c.name === targetCol ? 'table-active' : ''}">
            <td><strong>${c.name}</strong></td>
            <td><code>${c.dtype}</code></td>
            <td>
                ${c.name === targetCol 
                    ? '<span class="badge-pill primary">Target Variable</span>' 
                    : '<span class="badge bg-secondary-subtle text-light">Predictor Feature</span>'}
            </td>
            <td>${c.null_count}</td>
            <td>${c.null_percent}%</td>
            <td>${c.unique_count}</td>
        </tr>
    `).join('');
}
