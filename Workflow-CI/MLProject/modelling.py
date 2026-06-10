"""
modelling.py — MLflow Project (Workflow CI)
============================================
Dijalankan oleh: mlflow run . --env-manager=local
Path data menggunakan path RELATIF agar kompatibel dengan GitHub Actions runner.
"""

import os
import logging
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ── Path RELATIF — wajib agar tidak error di GitHub Actions ──
# Script ini dijalankan dari dalam folder MLProject oleh mlflow run
X_PATH       = 'telco_preprocessing/X_processed.csv'
Y_PATH       = 'telco_preprocessing/y_processed.csv'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

def main():
    # Tracking lokal — mlruns tersimpan di dalam folder MLProject
    mlflow.set_experiment('Telco-Churn-CI')
    mlflow.sklearn.autolog()

    logger.info(f'Membaca data dari: {X_PATH}')
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).squeeze()
    logger.info(f'Data dimuat — X: {X.shape}, y: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    with mlflow.start_run(run_name='RF-CI-Run'):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_f1', f1)
        mlflow.log_metric('test_roc_auc', auc)

        logger.info(f'Accuracy : {acc:.4f}')
        logger.info(f'F1 Score : {f1:.4f}')
        logger.info(f'ROC-AUC  : {auc:.4f}')
        logger.info('Training selesai — semua artifact tersimpan lokal di mlruns/')

if __name__ == '__main__':
    main()
