"""
modelling_tuning.py
===================
Kriteria 2 — Advance: Eksperimen & Tracking dengan MLflow + DagsHub

Proyek Akhir: Membangun Sistem Machine Learning — Dicoding
Dataset     : Telco Customer Churn (hasil preprocessing Kriteria 1)
Model       : Random Forest Classifier
Tuning      : RandomizedSearchCV (efisien untuk search space besar)
Tracking    : MLflow manual logging (mlflow.log_param / mlflow.log_metric)
Remote      : DagsHub MLflow Tracking Server

Artefak yang di-log ke MLflow Run:
  - confusion_matrix.png
  - roc_curve.png
  - feature_importance.png
  - classification_report.txt
  - best_model (sklearn flavor)

Struktur Eksperimen:
  Experiment : "Telco-Churn-RandomForest"
  Parent Run : "RandomizedSearchCV-Tuning"
    └─ Child Runs : satu run per kandidat hyperparameter
"""

# ──────────────────────────────────────────────────────────────
# 0. IMPORT & KONFIGURASI
# ──────────────────────────────────────────────────────────────
import os
import json
import logging
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend agar aman di server/CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sns.set_theme(style='whitegrid', palette='Set2')

# ──────────────────────────────────────────────────────────────
# 1. KONFIGURASI — SESUAIKAN BAGIAN INI
# ──────────────────────────────────────────────────────────────

# ── DagsHub ──
DAGSHUB_OWNER  = 'layanan.rumahku'   # Ganti dengan username DagsHub Anda
DAGSHUB_REPO   = 'Membangun-Sistem-Machine-Learning-VanesHasan'    # Ganti dengan nama repositori DagsHub Anda

# ── Path dataset hasil preprocessing (Kriteria 1) ──
X_PATH = os.getenv('X_PATH', 'telco_preprocessing/X_processed.csv')
Y_PATH = os.getenv('Y_PATH', 'telco_preprocessing/y_processed.csv')

# ── Experiment & Run ──
EXPERIMENT_NAME = 'Telco-Churn-RandomForest'
PARENT_RUN_NAME = 'RandomizedSearchCV-Tuning'

# ── Split & Reprodusibilitas ──
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Tuning ──
N_ITER       = 20    # Jumlah kombinasi yang dicoba RandomizedSearchCV
CV_FOLDS     = 5     # Stratified K-Fold
SCORING      = 'roc_auc'

# ── Folder artefak lokal sementara ──
ARTIFACT_DIR = 'artifacts'


# ──────────────────────────────────────────────────────────────
# 2. INISIALISASI DAGSHUB + MLFLOW (BYPASS MANUAL)
# ──────────────────────────────────────────────────────────────

def init_dagshub_mlflow() -> None:
    """
    Inisialisasi DagsHub MLflow Tracking Server secara Manual.
    Metode ini menghindari OAuth flow browser yang sering mengalami Gateway Timeout.
    """
    logger.info('Menginisialisasi DagsHub MLflow Tracking secara Manual...')

    # Amankan kredensial langsung ke environment internal MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "layanan.rumahku"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "1021c4589c32c4ed4d115c0a89bc816581444b49"

    # Set remote tracking URI secara langsung
    remote_url = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(remote_url)
    
    logger.info(f'MLflow Tracking URI berhasil diarahkan ke: {mlflow.get_tracking_uri()}')


# ──────────────────────────────────────────────────────────────
# 3. LOAD DATA
# ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Memuat X dan y dari hasil preprocessing Kriteria 1."""
    logger.info(f'Memuat data dari:\n  X → {X_PATH}\n  y → {Y_PATH}')

    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise FileNotFoundError(
            'File preprocessing tidak ditemukan. '
            'Pastikan sudah menjalankan automate_Nama-siswa.py terlebih dahulu.'
        )

    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).squeeze()   # DataFrame satu kolom → Series

    logger.info(f'Data dimuat — X: {X.shape}, y: {y.shape}')
    logger.info(f'Distribusi target:\n{y.value_counts().to_string()}')
    return X, y


# ──────────────────────────────────────────────────────────────
# 4. FUNGSI PEMBUATAN ARTEFAK GAMBAR
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> str:
    """Membuat dan menyimpan confusion matrix sebagai PNG."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['No Churn (0)', 'Churn (1)']

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap='Blues',
        colorbar=True,
        values_format='d',
    )

    tn, fp, fn, tp = cm.ravel()
    ax.set_title(
        f'Confusion Matrix — Random Forest\n'
        f'TP={tp} | TN={tn} | FP={fp} | FN={fn}',
        fontsize=13, fontweight='bold', pad=15,
    )
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Confusion matrix disimpan → {save_path}')
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_score: float,
    save_path: str,
) -> str:
    """Membuat dan menyimpan ROC Curve sebagai PNG."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr, tpr,
        color='#e74c3c', lw=2.5,
        label=f'Random Forest (AUC = {auc_score:.4f})',
    )
    ax.plot(
        [0, 1], [0, 1],
        color='grey', lw=1.5, linestyle='--',
        label='Random Classifier (AUC = 0.50)',
    )

    ax.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')

    j_scores    = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        color='#c0392b', s=100, zorder=5,
        label=f'Optimal Threshold = {thresholds[optimal_idx]:.3f}',
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(
        'ROC Curve — Random Forest Classifier\nTelco Customer Churn',
        fontsize=13, fontweight='bold', pad=15,
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'ROC Curve disimpan → {save_path}')
    return save_path


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    save_path: str,
    top_n: int = 20,
) -> str:
    """Membuat bar plot feature importance dari model Random Forest."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    top_features    = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = sns.color_palette('RdYlGn', n_colors=top_n)[::-1]

    bars = ax.barh(
        range(top_n), top_importances[::-1],
        color=colors, edgecolor='white', linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
    ax.set_title(
        f'Top {top_n} Feature Importances — Random Forest',
        fontsize=13, fontweight='bold', pad=15,
    )

    for bar, val in zip(bars, top_importances[::-1]):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha='left', fontsize=8,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Feature importance disimpan → {save_path}')
    return save_path


# ──────────────────────────────────────────────────────────────
# 5. HITUNG SEMUA METRIK EVALUASI
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Menghitung metrik evaluasi komprehensif."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc    = (recall_score(y_true, y_pred) + specificity) / 2

    return {
        'accuracy'         : accuracy_score(y_true, y_pred),
        'precision'        : precision_score(y_true, y_pred, zero_division=0),
        'recall'           : recall_score(y_true, y_pred, zero_division=0),
        'f1_score'         : f1_score(y_true, y_pred, zero_division=0),
        'roc_auc'          : roc_auc_score(y_true, y_prob),
        'specificity'      : specificity,
        'balanced_accuracy': balanced_acc,
        'tp': float(tp), 'tn': float(tn),
        'fp': float(fp), 'fn': float(fn),
    }


# ──────────────────────────────────────────────────────────────
# 6. LOGGING MANUAL KE MLFLOW
# ──────────────────────────────────────────────────────────────

def log_params_manual(params: dict) -> None:
    """Log hyperparameter satu per satu menggunakan mlflow.log_param()."""
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics_manual(metrics: dict, prefix: str = '') -> None:
    """Log metrik satu per satu menggunakan mlflow.log_metric()."""
    for key, value in metrics.items():
        metric_name = f'{prefix}{key}' if prefix else key
        mlflow.log_metric(metric_name, value)


# ──────────────────────────────────────────────────────────────
# 7. PIPELINE UTAMA — TUNING + LOGGING
# ──────────────────────────────────────────────────────────────

def run_experiment() -> None:
    """Pipeline utama tuning hyperparameter dan MLflow logging."""
    # ── 7.1 Inisialisasi DagsHub secara Manual ──
    init_dagshub_mlflow()

    # ── 7.2 Set / Buat Experiment ──
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f'Experiment: "{EXPERIMENT_NAME}"')

    # ── 7.3 Load & Split Data ──
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f'Split selesai — Train: {X_train.shape}, Test: {X_test.shape}'
    )

    # ── 7.4 Definisi Search Space Hyperparameter ──
    param_dist = {
        'n_estimators'     : [100, 200, 300, 500, 750],
        'max_depth'        : [None, 5, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf' : [1, 2, 4, 8],
        'max_features'     : ['sqrt', 'log2', 0.3, 0.5],
        'class_weight'     : ['balanced', 'balanced_subsample', None],
        'bootstrap'        : [True, False],
        'criterion'        : ['gini', 'entropy'],
    }

    # ── 7.5 Cross-Validation Strategy ──
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ════════════════════════════════════════════
    # PARENT RUN — Orkestrasi seluruh tuning
    # ════════════════════════════════════════════
    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f'Parent Run ID: {parent_run_id}')

        mlflow.log_param('experiment_strategy', 'RandomizedSearchCV')
        mlflow.log_param('n_iter', N_ITER)
        mlflow.log_param('cv_folds', CV_FOLDS)
        mlflow.log_param('cv_strategy', 'StratifiedKFold')
        mlflow.log_param('scoring_metric', SCORING)
        mlflow.log_param('test_size', TEST_SIZE)
        mlflow.log_param('random_state', RANDOM_STATE)
        mlflow.log_param('train_samples', X_train.shape[0])
        mlflow.log_param('test_samples', X_test.shape[0])
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('target_class_distribution', y_train.value_counts().to_dict())

        base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

        logger.info(
            f'Memulai RandomizedSearchCV — {N_ITER} iterasi, '
            f'{CV_FOLDS}-fold StratifiedKFold ...'
        )

        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=N_ITER,
            scoring=SCORING,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            refit=True,
        )
        search.fit(X_train, y_train)

        logger.info('RandomizedSearchCV selesai.')
        logger.info(f'Best CV {SCORING}: {search.best_score_:.4f}')
        logger.info(f'Best Params: {search.best_params_}')

        # ── 7.7 Log setiap kandidat sebagai Child Run ──
        cv_results = pd.DataFrame(search.cv_results_)

        for idx, row in cv_results.iterrows():
            candidate_params = {
                k.replace('param_', ''): v
                for k, v in row.items()
                if k.startswith('param_')
            }
            with mlflow.start_run(
                run_name=f'candidate_{idx + 1:02d}',
                nested=True,
            ) as child_run:
                log_params_manual(candidate_params)
                mlflow.log_param('candidate_index', idx + 1)

                mlflow.log_metric('cv_mean_roc_auc', row['mean_test_score'])
                mlflow.log_metric('cv_std_roc_auc', row['std_test_score'])
                mlflow.log_metric('cv_mean_train_roc_auc', row.get('mean_train_score', -1))
                mlflow.log_metric('cv_rank', int(row['rank_test_score']))

                is_best = (int(row['rank_test_score']) == 1)
                mlflow.set_tag('is_best_candidate', str(is_best))

        logger.info(f'Selesai logging {len(cv_results)} child runs.')

        # ════════════════════════════════════════════
        # 7.8 EVALUASI MODEL TERBAIK
        # ════════════════════════════════════════════
        best_model  = search.best_estimator_
        best_params = search.best_params_

        y_pred_train = best_model.predict(X_train)
        y_prob_train = best_model.predict_proba(X_train)[:, 1]

        y_pred_test  = best_model.predict(X_test)
        y_prob_test  = best_model.predict_proba(X_test)[:, 1]

        train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train)
        test_metrics  = compute_metrics(y_test,  y_pred_test,  y_prob_test)

        logger.info('=== METRIK EVALUASI ===')
        logger.info(f"  Train — Acc: {train_metrics['accuracy']:.4f} | "
                    f"AUC: {train_metrics['roc_auc']:.4f} | "
                    f"F1: {train_metrics['f1_score']:.4f}")
        logger.info(f"  Test  — Acc: {test_metrics['accuracy']:.4f} | "
                    f"AUC: {test_metrics['roc_auc']:.4f} | "
                    f"F1: {test_metrics['f1_score']:.4f}")

        log_params_manual({f'best_{k}': v for k, v in best_params.items()})
        mlflow.log_param('best_cv_score', round(search.best_score_, 6))

        log_metrics_manual(train_metrics, prefix='train_')
        log_metrics_manual(test_metrics,  prefix='test_')

        overfit_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
        mlflow.log_metric('overfit_gap_roc_auc', round(overfit_gap, 6))

        # ════════════════════════════════════════════
        # 7.9 BUAT & LOG ARTEFAK GAMBAR — MANUAL
        # ════════════════════════════════════════════
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(cm_path, artifact_path='evaluation_plots')

        roc_path = os.path.join(ARTIFACT_DIR, 'roc_curve.png')
        plot_roc_curve(y_test, y_prob_test, test_metrics['roc_auc'], roc_path)
        mlflow.log_artifact(roc_path, artifact_path='evaluation_plots')

        fi_path = os.path.join(ARTIFACT_DIR, 'feature_importance.png')
        plot_feature_importance(best_model, X.columns.tolist(), fi_path)
        mlflow.log_artifact(fi_path, artifact_path='evaluation_plots')

        report_str  = classification_report(
            y_test, y_pred_test,
            target_names=['No Churn (0)', 'Churn (1)'],
        )
        report_path = os.path.join(ARTIFACT_DIR, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write('=== Classification Report — Best Random Forest ===\n\n')
            f.write(report_str)
            f.write('\n\n=== Best Hyperparameters ===\n')
            f.write(json.dumps(best_params, indent=2, default=str))
            f.write('\n\n=== Test Metrics ===\n')
            f.write(json.dumps({k: round(v, 6) for k, v in test_metrics.items()}, indent=2))
        mlflow.log_artifact(report_path, artifact_path='reports')
        logger.info(f'Classification report disimpan → {report_path}')

        cv_csv_path = os.path.join(ARTIFACT_DIR, 'cv_results.csv')
        cv_results.to_csv(cv_csv_path, index=False)
        mlflow.log_artifact(cv_csv_path, artifact_path='reports')

        # ════════════════════════════════════════════
        # 7.10 LOG & REGISTER MODEL TERBAIK (Sesuai Kriteria Skilled)
        # ════════════════════════════════════════════
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='model', 
            registered_model_name='TelcoChurn-RandomForest',
            input_example=X_test.head(5),
            metadata={
                'dataset'  : 'Telco Customer Churn',
                'framework': 'scikit-learn',
                'algorithm': 'RandomForestClassifier',
                'tuning'   : 'RandomizedSearchCV',
            },
        )
        logger.info(f'Model diregistrasi — URI: {model_info.model_uri}')

        # ════════════════════════════════════════════
        # 7.11 RINGKASAN AKHIR
        # ════════════════════════════════════════════
        logger.info('=' * 60)
        logger.info('  EKSPERIMEN SELESAI')
        logger.info('=' * 60)
        logger.info(f'  Parent Run ID  : {parent_run_id}')
        logger.info(f'  Experiment     : {EXPERIMENT_NAME}')
        logger.info(f'  Best CV AUC    : {search.best_score_:.4f}')
        logger.info(f'  Test Accuracy  : {test_metrics["accuracy"]:.4f}')
        logger.info(f'  Test Precision : {test_metrics["precision"]:.4f}')
        logger.info(f'  Test Recall    : {test_metrics["recall"]:.4f}')
        logger.info(f'  Test F1-Score  : {test_metrics["f1_score"]:.4f}')
        logger.info(f'  Test ROC-AUC   : {test_metrics["roc_auc"]:.4f}')
        logger.info(f'  Overfit Gap    : {overfit_gap:.4f}')
        logger.info('=' * 60)
        logger.info(
            f'Lihat hasil di DagsHub:\n'
            f'  https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}/experiments'
        )


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_experiment()