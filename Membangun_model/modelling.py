"""
modelling.py
============
Kriteria 2 — Basic: Melatih model dengan MLflow Tracking UI (LOKAL)

Proyek Akhir: Membangun Sistem Machine Learning — Dicoding
Dataset     : Telco Customer Churn (hasil preprocessing Kriteria 1)
Model       : Random Forest Classifier
Tracking    : MLflow autolog — disimpan LOKAL (tanpa DagsHub, tanpa tuning)

Cara menjalankan:
  1. python modelling.py
  2. mlflow ui          (buka http://localhost:5000 untuk lihat dashboard)
"""

import os
import logging
import warnings
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# ──────────────────────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────────────────────
X_PATH       = os.getenv('X_PATH', 'telco_preprocessing/X_processed.csv')
Y_PATH       = os.getenv('Y_PATH', 'telco_preprocessing/y_processed.csv')
TEST_SIZE    = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = 'Telco-Churn-RandomForest'

# ──────────────────────────────────────────────────────────────
# TRACKING LOKAL — tidak menggunakan DagsHub atau remote server
# Hasil disimpan di folder ./mlruns secara otomatis
# ──────────────────────────────────────────────────────────────
# TIDAK perlu mlflow.set_tracking_uri() — default sudah lokal

def load_data():
    logger.info(f'Memuat data dari: {X_PATH} dan {Y_PATH}')
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).squeeze()
    logger.info(f'Data dimuat — X: {X.shape}, y: {y.shape}')
    return X, y


def run_experiment():
    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f'Experiment: "{EXPERIMENT_NAME}"')

    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f'Split — Train: {X_train.shape}, Test: {X_test.shape}')

    # ── AUTOLOG — wajib untuk kriteria Basic ──
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name='RandomForest-Basic'):
        logger.info('Memulai training model...')

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        logger.info('\n' + classification_report(
            y_test, y_pred,
            target_names=['No Churn (0)', 'Churn (1)'],
        ))

        logger.info('Training selesai. Semua metrik & model otomatis di-log oleh autolog.')

    logger.info('='*55)
    logger.info('  SELESAI — Jalankan: mlflow ui')
    logger.info('  Buka browser: http://localhost:5000')
    logger.info('='*55)


if __name__ == '__main__':
    run_experiment()
