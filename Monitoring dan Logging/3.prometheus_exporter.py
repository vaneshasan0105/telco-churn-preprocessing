"""
3.prometheus_exporter.py  (alias: 7.Inference.py)
==================================================
Kriteria 4 — Advance: Serving & Monitoring Model Telco Customer Churn

Stack:
  - FastAPI          : REST API serving endpoint
  - prometheus_client: Ekspos 10 metrik kustom ke /metrics
  - MLflow sklearn   : Load model dari saved_model/ folder

Endpoint:
  POST /predict   → inferensi tunggal atau batch
  GET  /health    → health check
  GET  /metrics   → Prometheus scrape target
  GET  /model-info→ info model yang dimuat

10 Metrik Prometheus Kustom:
  1.  http_requests_total            (Counter)   total request masuk
  2.  http_request_duration_seconds  (Histogram) latensi end-to-end request
  3.  prediction_churn_yes_total     (Counter)   prediksi Churn = 1
  4.  prediction_churn_no_total      (Counter)   prediksi Churn = 0
  5.  prediction_errors_total        (Counter)   request error (per tipe)
  6.  prediction_confidence_avg      (Gauge)     rata-rata probabilitas batch terakhir
  7.  model_load_time_seconds        (Gauge)     durasi load model saat startup
  8.  active_requests                (Gauge)     request sedang diproses saat ini
  9.  batch_size_histogram           (Histogram) distribusi ukuran batch
  10. prediction_latency_seconds     (Summary)   latensi murni inferensi model

Setup:
  pip install fastapi uvicorn prometheus_client mlflow scikit-learn pandas numpy
  python 3.prometheus_exporter.py
  # API      : http://localhost:8000/docs
  # Metrics  : http://localhost:8000/metrics
"""

import os
import sys
import time
import logging
import warnings
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

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
MODEL_PATH  = os.getenv('MODEL_PATH', 'saved_model')
HOST        = os.getenv('HOST', '0.0.0.0')
PORT        = int(os.getenv('PORT', '8000'))
APP_VERSION = '1.0.0'


# ══════════════════════════════════════════════════════════════
# 10 METRIK PROMETHEUS KUSTOM
# ══════════════════════════════════════════════════════════════

# 1. Counter — total HTTP request (per method, endpoint, status_code)
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests yang diterima server',
    ['method', 'endpoint', 'status_code'],
)

# 2. Histogram — latensi end-to-end tiap HTTP request
HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'Durasi HTTP request dari masuk hingga response dikirim (detik)',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# 3. Counter — total prediksi Churn = YES (kelas positif)
PREDICTION_CHURN_YES = Counter(
    'prediction_churn_yes_total',
    'Akumulasi prediksi Churn = Yes (label 1)',
)

# 4. Counter — total prediksi Churn = NO (kelas negatif)
PREDICTION_CHURN_NO = Counter(
    'prediction_churn_no_total',
    'Akumulasi prediksi Churn = No (label 0)',
)

# 5. Counter — total error prediksi (per tipe error)
PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Total request yang menghasilkan error saat prediksi',
    ['error_type'],
)

# 6. Gauge — rata-rata confidence (probabilitas) batch prediksi terakhir
PREDICTION_CONFIDENCE_AVG = Gauge(
    'prediction_confidence_avg',
    'Rata-rata probabilitas prediksi Churn pada batch terakhir (0.0–1.0)',
)

# 7. Gauge — waktu load model saat startup
MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Waktu memuat model dari disk saat startup (detik)',
)

# 8. Gauge — jumlah request yang sedang aktif diproses
ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Jumlah HTTP request yang sedang aktif diproses server saat ini',
)

# 9. Histogram — distribusi ukuran batch per request prediksi
BATCH_SIZE_HISTOGRAM = Histogram(
    'batch_size_histogram',
    'Distribusi jumlah sampel dalam satu request prediksi',
    buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500],
)

# 10. Summary — latensi murni inferensi model (tanpa overhead HTTP/network)
PREDICTION_LATENCY = Summary(
    'prediction_latency_seconds',
    'Latensi murni model.predict() tanpa overhead HTTP (detik)',
)


# ──────────────────────────────────────────────────────────────
# STATE GLOBAL APLIKASI
# ──────────────────────────────────────────────────────────────
class AppState:
    model        = None
    model_uri    : str = ''
    loaded_at    : str = ''
    model_type   : str = 'unknown'


state = AppState()


# ──────────────────────────────────────────────────────────────
# LIFESPAN: LOAD MODEL SAAT STARTUP
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat server startup, catat waktu load ke Gauge."""
    logger.info('=' * 58)
    logger.info('  Telco Churn Inference Server  |  Starting Up')
    logger.info('=' * 58)

    t_start = time.time()
    try:
        import mlflow.sklearn
        logger.info(f'Memuat model dari path: {MODEL_PATH}')
        state.model     = mlflow.sklearn.load_model(MODEL_PATH)
        state.model_uri = MODEL_PATH
        state.loaded_at = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        state.model_type= type(state.model).__name__

        load_time = time.time() - t_start
        MODEL_LOAD_TIME.set(load_time)

        logger.info(f'✅ Model dimuat: {state.model_type}  ({load_time:.3f}s)')
    except Exception as exc:
        logger.error(f'❌ Gagal memuat model: {exc}')
        logger.error(f'   Pastikan folder "{MODEL_PATH}" valid.')

    logger.info(f'API Docs : http://localhost:{PORT}/docs')
    logger.info(f'Metrics  : http://localhost:{PORT}/metrics')
    logger.info('=' * 58)

    yield  # ← server berjalan di sini

    logger.info('Server shutting down ...')


# ──────────────────────────────────────────────────────────────
# INISIALISASI FASTAPI
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title='Telco Customer Churn — Inference & Monitoring API',
    description=(
        'Serving endpoint untuk prediksi Churn pelanggan telekomunikasi. '
        'Terintegrasi dengan Prometheus untuk monitoring real-time. '
        'Kriteria 4 — Proyek Akhir Dicoding: Membangun Sistem ML.'
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────
# MIDDLEWARE: Auto-tracking latensi & jumlah request
# ──────────────────────────────────────────────────────────────
@app.middleware('http')
async def prometheus_middleware(request: Request, call_next):
    """
    Middleware otomatis yang mencatat:
    - HTTP_REQUESTS_TOTAL      per (method, path, status_code)
    - HTTP_REQUEST_DURATION    per (method, path)
    - ACTIVE_REQUESTS          naik saat request masuk, turun saat selesai
    """
    method   = request.method
    endpoint = request.url.path

    ACTIVE_REQUESTS.inc()
    t_start = time.perf_counter()

    try:
        response    = await call_next(request)
        status_code = str(response.status_code)
        duration    = time.perf_counter() - t_start

        HTTP_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=method, endpoint=endpoint
        ).observe(duration)

        return response

    except Exception as exc:
        duration = time.perf_counter() - t_start
        HTTP_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code='500'
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=method, endpoint=endpoint
        ).observe(duration)
        raise exc

    finally:
        ACTIVE_REQUESTS.dec()


# ──────────────────────────────────────────────────────────────
# PYDANTIC SCHEMA
# ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Body POST /predict — list of dict, satu dict per pelanggan."""
    data: List[dict] = Field(
        ...,
        example=[{
            'tenure': 24,
            'MonthlyCharges': 79.85,
            'TotalCharges': 1914.0,
            'SeniorCitizen': 0,
            'Partner': 1,
            'Dependents': 0,
            'PhoneService': 1,
            'PaperlessBilling': 1,
            'Contract_One year': 0,
            'Contract_Two year': 0,
            'PaymentMethod_Credit card (automatic)': 0,
            'PaymentMethod_Electronic check': 1,
            'PaymentMethod_Mailed check': 0,
        }],
    )


class PredictResponse(BaseModel):
    predictions      : List[int]
    probabilities    : List[float]
    labels           : List[str]
    batch_size       : int
    inference_time_ms: float
    churn_count      : int
    no_churn_count   : int


# ──────────────────────────────────────────────────────────────
# ENDPOINT: HEALTH CHECK
# ──────────────────────────────────────────────────────────────
@app.get('/health', tags=['System'], summary='Health check server & model')
async def health_check():
    model_ok = state.model is not None
    return JSONResponse(
        status_code=200 if model_ok else 503,
        content={
            'status'      : 'ok' if model_ok else 'degraded',
            'model_status': 'loaded' if model_ok else 'not_loaded',
            'model_type'  : state.model_type,
            'model_uri'   : state.model_uri,
            'loaded_at'   : state.loaded_at,
            'version'     : APP_VERSION,
        },
    )


# ──────────────────────────────────────────────────────────────
# ENDPOINT: MODEL INFO
# ──────────────────────────────────────────────────────────────
@app.get('/model-info', tags=['System'], summary='Detail informasi model')
async def model_info():
    if state.model is None:
        raise HTTPException(status_code=503, detail='Model belum dimuat.')

    info: dict = {
        'model_type': state.model_type,
        'model_uri' : state.model_uri,
        'loaded_at' : state.loaded_at,
    }
    if hasattr(state.model, 'get_params'):
        info['hyperparameters'] = {
            k: str(v) for k, v in state.model.get_params().items()
        }
    if hasattr(state.model, 'feature_importances_'):
        fi = state.model.feature_importances_
        info['n_features']         = int(len(fi))
        info['top_feature_importance'] = float(fi.max())

    return JSONResponse(info)


# ──────────────────────────────────────────────────────────────
# ENDPOINT: PREDIKSI (CORE)
# ──────────────────────────────────────────────────────────────
@app.post(
    '/predict',
    response_model=PredictResponse,
    tags=['Inference'],
    summary='Prediksi Customer Churn',
)
async def predict(request: PredictRequest):
    """
    Prediksi apakah pelanggan akan churn.

    **Input**: list of dict — setiap dict mewakili satu baris fitur pelanggan.
    **Output**: prediksi (0/1), probabilitas churn, dan label teks.
    """
    if state.model is None:
        PREDICTION_ERRORS.labels(error_type='model_not_loaded').inc()
        raise HTTPException(
            status_code=503,
            detail='Model belum dimuat. Cek log server.',
        )

    try:
        df_input = pd.DataFrame(request.data)
        batch_sz = len(df_input)

        # Catat ukuran batch
        BATCH_SIZE_HISTOGRAM.observe(batch_sz)

        # ── Inferensi — timing murni model ──
        t_infer = time.perf_counter()
        with PREDICTION_LATENCY.time():
            preds  = state.model.predict(df_input)
            probs  = state.model.predict_proba(df_input)[:, 1]
        infer_ms = (time.perf_counter() - t_infer) * 1000

        # ── Update metrik Prometheus ──
        n_yes = int(np.sum(preds == 1))
        n_no  = int(np.sum(preds == 0))

        PREDICTION_CHURN_YES.inc(n_yes)
        PREDICTION_CHURN_NO.inc(n_no)
        PREDICTION_CONFIDENCE_AVG.set(float(np.mean(probs)))

        labels = ['Churn' if p == 1 else 'No Churn' for p in preds]

        logger.info(
            f'/predict  batch={batch_sz} | '
            f'Churn={n_yes} | NoChurn={n_no} | '
            f'{infer_ms:.2f}ms'
        )

        return PredictResponse(
            predictions      = preds.tolist(),
            probabilities    = [round(float(p), 4) for p in probs],
            labels           = labels,
            batch_size       = batch_sz,
            inference_time_ms= round(infer_ms, 3),
            churn_count      = n_yes,
            no_churn_count   = n_no,
        )

    except KeyError as exc:
        PREDICTION_ERRORS.labels(error_type='missing_feature').inc()
        raise HTTPException(
            status_code=422,
            detail=f'Kolom fitur tidak ditemukan: {exc}. '
                   f'Pastikan nama kolom sesuai dengan data training.',
        )
    except ValueError as exc:
        PREDICTION_ERRORS.labels(error_type='invalid_value').inc()
        raise HTTPException(
            status_code=422,
            detail=f'Nilai input tidak valid: {exc}',
        )
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type='internal_error').inc()
        logger.error(f'Prediction error: {exc}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f'Internal server error: {str(exc)}',
        )


# ──────────────────────────────────────────────────────────────
# ENDPOINT: PROMETHEUS METRICS SCRAPE TARGET
# ──────────────────────────────────────────────────────────────
@app.get(
    '/metrics',
    response_class=PlainTextResponse,
    tags=['Monitoring'],
    summary='Prometheus metrics scrape endpoint',
    include_in_schema=True,
)
async def metrics():
    """
    Endpoint yang di-scrape oleh Prometheus setiap `scrape_interval`.
    Dikonfigurasi sebagai target di prometheus.yml.
    """
    return PlainTextResponse(
        content=generate_latest(REGISTRY).decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST,
    )


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
        log_level='info',
    )
