"""
train.py
========
Entry point untuk MLflow Project (Kriteria 3 — Advance).

Script ini dipanggil oleh `mlflow run` via konfigurasi MLProject.
Menerima semua hyperparameter sebagai argumen CLI, melatih model
Random Forest, melakukan logging manual ke MLflow/DagsHub, dan
menyimpan model serta artefak evaluasi.

Penggunaan manual (tanpa mlflow run):
  python train.py --n_estimators 300 --max_depth None ...

Penggunaan via MLflow Project:
  mlflow run . -P n_estimators=300 -P max_depth=None ...
"""

import os
import sys
import json
import logging
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report, ConfusionMatrixDisplay,
)

import mlflow
import mlflow.sklearn
import dagshub

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = 'artifacts'


# ──────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Telco Churn — MLflow Project Retraining Script'
    )

    # Data
    parser.add_argument('--x_path', type=str,
                        default='telco_preprocessing/X_processed.csv')
    parser.add_argument('--y_path', type=str,
                        default='telco_preprocessing/y_processed.csv')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)

    # Hyperparameter RF
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--max_depth', type=str, default='None')
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--max_features', type=str, default='sqrt')
    parser.add_argument('--class_weight', type=str, default='balanced')
    parser.add_argument('--criterion', type=str, default='gini')

    # MLflow / DagsHub
    parser.add_argument('--experiment_name', type=str,
                        default='Telco-Churn-Retraining')
    parser.add_argument('--run_name', type=str,
                        default='mlproject-retrain')
    parser.add_argument('--dagshub_owner', type=str,
                        default=os.environ.get('DAGSHUB_OWNER', 'layanan.rumahku'))
    parser.add_argument('--dagshub_repo', type=str,
                        default=os.environ.get('DAGSHUB_REPO', 'Membangun-Sistem-Machine-Learning-VanesHasan'))

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# HELPER: KONVERSI ARGUMEN STRING
# ──────────────────────────────────────────────────────────────

def parse_max_depth(value: str):
    """Konversi string 'None' → Python None, angka → int."""
    if value is None or str(value).strip().lower() == 'none':
        return None
    return int(value)


def parse_class_weight(value: str):
    """Konversi string 'None' → Python None."""
    if value is None or str(value).strip().lower() == 'none':
        return None
    return value


# ──────────────────────────────────────────────────────────────
# ARTEFAK: CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: str) -> str:
    cm     = confusion_matrix(y_true, y_pred)
    labels = ['No Churn (0)', 'Churn (1)']
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    ax.set_title(
        f'Confusion Matrix — Random Forest\n'
        f'TP={tp} | TN={tn} | FP={fp} | FN={fn}',
        fontsize=13, fontweight='bold', pad=15,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Confusion matrix → {save_path}')
    return save_path


# ──────────────────────────────────────────────────────────────
# ARTEFAK: ROC CURVE
# ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, auc_score: float, save_path: str) -> str:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores    = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'Random Forest (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--',
            label='Random Classifier (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#e74c3c')
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               color='#c0392b', s=100, zorder=5,
               label=f'Optimal Threshold = {thresholds[optimal_idx]:.3f}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Telco Customer Churn',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'ROC Curve → {save_path}')
    return save_path


# ──────────────────────────────────────────────────────────────
# HITUNG METRIK
# ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy'         : accuracy_score(y_true, y_pred),
        'precision'        : precision_score(y_true, y_pred, zero_division=0),
        'recall'           : recall_score(y_true, y_pred, zero_division=0),
        'f1_score'         : f1_score(y_true, y_pred, zero_division=0),
        'roc_auc'          : roc_auc_score(y_true, y_prob),
        'specificity'      : specificity,
        'balanced_accuracy': (recall_score(y_true, y_pred, zero_division=0)
                              + specificity) / 2,
        'tp': float(tp), 'tn': float(tn),
        'fp': float(fp), 'fn': float(fn),
    }


# ──────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Inisialisasi DagsHub + MLflow ──
    # ── Inisialisasi DagsHub + MLflow ──
    logger.info('Inisialisasi DagsHub MLflow Tracking ...')
    dagshub.init(
        repo_owner=args.dagshub_owner,
        repo_name=args.dagshub_repo,
        mlflow=True,
        root_dir=os.path.abspath(os.path.join(os.getcwd(), "../..")) # Paksa tunjuk ke root repo utama
    )
    logger.info(f'Tracking URI: {mlflow.get_tracking_uri()}')
    mlflow.set_experiment(args.experiment_name)

    # ── Load Data ──
    logger.info(f'Memuat data — X: {args.x_path} | y: {args.y_path}')
    if not os.path.exists(args.x_path) or not os.path.exists(args.y_path):
        raise FileNotFoundError(
            'Dataset preprocessing tidak ditemukan. '
            'Pastikan automate_Nama-siswa.py sudah dijalankan.'
        )
    X = pd.read_csv(args.x_path)
    y = pd.read_csv(args.y_path).squeeze()
    logger.info(f'Data dimuat — X: {X.shape} | y: {y.shape}')

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # ── Konversi hyperparameter ──
    max_depth    = parse_max_depth(args.max_depth)
    class_weight = parse_class_weight(args.class_weight)
    max_features = (
        args.max_features
        if args.max_features in ('sqrt', 'log2')
        else float(args.max_features)
    )

    # ── Bangun Model ──
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        criterion=args.criterion,
        random_state=args.random_state,
        n_jobs=-1,
    )

    # ════════════════════════════════════════════
    # MLFLOW RUN — logging manual penuh
    # ════════════════════════════════════════════
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        logger.info(f'MLflow Run ID: {run_id}')

        # ── Log Params — MANUAL ──
        mlflow.log_param('n_estimators',      args.n_estimators)
        mlflow.log_param('max_depth',         str(max_depth))
        mlflow.log_param('min_samples_split', args.min_samples_split)
        mlflow.log_param('min_samples_leaf',  args.min_samples_leaf)
        mlflow.log_param('max_features',      args.max_features)
        mlflow.log_param('class_weight',      str(class_weight))
        mlflow.log_param('criterion',         args.criterion)
        mlflow.log_param('test_size',         args.test_size)
        mlflow.log_param('random_state',      args.random_state)
        mlflow.log_param('train_samples',     X_train.shape[0])
        mlflow.log_param('test_samples',      X_test.shape[0])
        mlflow.log_param('n_features',        X_train.shape[1])

        # ── Cross-Validation (5-Fold) ──
        logger.info('Menjalankan 5-Fold Stratified Cross-Validation ...')
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=args.random_state)
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
            return_train_score=True,
            n_jobs=-1,
        )

        # Log CV metrics — MANUAL
        for metric in ('accuracy', 'f1', 'roc_auc', 'precision', 'recall'):
            mlflow.log_metric(f'cv_mean_{metric}',
                              float(cv_results[f'test_{metric}'].mean()))
            mlflow.log_metric(f'cv_std_{metric}',
                              float(cv_results[f'test_{metric}'].std()))

        logger.info(
            f'CV — AUC: {cv_results["test_roc_auc"].mean():.4f} '
            f'(±{cv_results["test_roc_auc"].std():.4f})'
        )

        # ── Training Final ──
        logger.info('Training model pada seluruh training set ...')
        model.fit(X_train, y_train)

        # ── Prediksi ──
        y_pred_test  = model.predict(X_test)
        y_prob_test  = model.predict_proba(X_test)[:, 1]
        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)[:, 1]

        # ── Hitung & Log Metrik — MANUAL ──
        test_metrics  = compute_metrics(y_test,  y_pred_test,  y_prob_test)
        train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train)

        for k, v in test_metrics.items():
            mlflow.log_metric(f'test_{k}', round(float(v), 6))
        for k, v in train_metrics.items():
            mlflow.log_metric(f'train_{k}', round(float(v), 6))

        overfit_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
        mlflow.log_metric('overfit_gap_roc_auc', round(overfit_gap, 6))

        logger.info(
            f'Test — Acc: {test_metrics["accuracy"]:.4f} | '
            f'AUC: {test_metrics["roc_auc"]:.4f} | '
            f'F1: {test_metrics["f1_score"]:.4f}'
        )

        # ── Artefak: Confusion Matrix ──
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(cm_path, artifact_path='evaluation_plots')

        # ── Artefak: ROC Curve ──
        roc_path = os.path.join(ARTIFACT_DIR, 'roc_curve.png')
        plot_roc_curve(y_test, y_prob_test, test_metrics['roc_auc'], roc_path)
        mlflow.log_artifact(roc_path, artifact_path='evaluation_plots')

        # ── Artefak: Classification Report ──
        report_str  = classification_report(
            y_test, y_pred_test,
            target_names=['No Churn (0)', 'Churn (1)'],
        )
        report_path = os.path.join(ARTIFACT_DIR, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write('=== Classification Report ===\n\n')
            f.write(report_str)
            f.write('\n\n=== Hyperparameters ===\n')
            f.write(json.dumps({
                'n_estimators'     : args.n_estimators,
                'max_depth'        : str(max_depth),
                'min_samples_split': args.min_samples_split,
                'min_samples_leaf' : args.min_samples_leaf,
                'max_features'     : args.max_features,
                'class_weight'     : str(class_weight),
                'criterion'        : args.criterion,
            }, indent=2))
        mlflow.log_artifact(report_path, artifact_path='reports')

        # ── Simpan model ke file lokal (untuk upload ke GitHub) ──
        model_local_dir = 'saved_model'
        mlflow.sklearn.save_model(
            sk_model=model,
            path=model_local_dir,
        )
        logger.info(f'Model disimpan lokal → {model_local_dir}/')

        # ── Log & Register model ke MLflow/DagsHub ──
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            registered_model_name='TelcoChurn-RF-MLProject',
            input_example=X_test.head(3),
        )
        logger.info(f'Model URI: {model_info.model_uri}')

        # ── Tags ──
        mlflow.set_tag('source',      'MLProject')
        mlflow.set_tag('algorithm',   'RandomForestClassifier')
        mlflow.set_tag('dataset',     'Telco Customer Churn')
        mlflow.set_tag('author',      args.dagshub_owner)
        mlflow.set_tag('status',      'completed')

        # ── Ringkasan ──
        logger.info('=' * 55)
        logger.info('  TRAINING SELESAI')
        logger.info('=' * 55)
        logger.info(f'  Run ID     : {run_id}')
        logger.info(f'  Accuracy   : {test_metrics["accuracy"]:.4f}')
        logger.info(f'  Precision  : {test_metrics["precision"]:.4f}')
        logger.info(f'  Recall     : {test_metrics["recall"]:.4f}')
        logger.info(f'  F1-Score   : {test_metrics["f1_score"]:.4f}')
        logger.info(f'  ROC-AUC    : {test_metrics["roc_auc"]:.4f}')
        logger.info(f'  Overfit Gap: {overfit_gap:.4f}')
        logger.info('=' * 55)


if __name__ == '__main__':
    main()
