"""
modelling.py — Workflow CI
===========================
File ini digunakan oleh MLflow Project (MLProject) pada Kriteria 3.
Kompatibel dengan GitHub Actions CI pipeline.
"""

import os
import logging
import warnings
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

X_PATH       = os.getenv('X_PATH', 'telco_preprocessing/X_processed.csv')
Y_PATH       = os.getenv('Y_PATH', 'telco_preprocessing/y_processed.csv')
TEST_SIZE    = float(os.getenv('TEST_SIZE', '0.2'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))

mlflow.sklearn.autolog()

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    logger.info(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')
    mlflow.log_metric('test_accuracy', acc)
    mlflow.log_metric('test_f1', f1)
    mlflow.log_metric('test_roc_auc', auc)

    mlflow.sklearn.log_model(model, artifact_path='model')
    logger.info('Model berhasil disimpan.')
