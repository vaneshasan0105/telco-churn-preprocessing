"""
Skrip otomatisasi preprocessing dataset Telco Customer Churn.

Proyek Akhir Kelas: Membangun Sistem Machine Learning - Dicoding
Dataset  : WA_Fn-UseC_-Telco-Customer-Churn.csv
Target   : Churn (Yes/No)

Alur:
  1. load_data()            → Baca CSV mentah
  2. fix_total_charges()    → Tangani spasi kosong & konversi ke numerik
  3. drop_identifier_cols() → Hapus kolom ID (tidak informatif)
  4. impute_missing()       → Imputasi median untuk numerik
  5. encode_categoricals()  → LabelEncoder (biner) + OHE (multi-kelas)
  6. scale_numerics()       → StandardScaler pada fitur numerik asli
  7. save_outputs()         → Simpan X dan y ke folder output
  8. run_pipeline()         → Orkestrasi semua langkah di atas
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ──────────────────────────────────────────────
# KONFIGURASI LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log', mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────
TARGET_COL      = 'Churn'
ID_COLS         = ['customerID']
NUM_COLS_SCALE  = ['tenure', 'MonthlyCharges', 'TotalCharges']
OUTPUT_DIR      = 'telco_preprocessing'   # Nama folder output


# ══════════════════════════════════════════════
# FUNGSI-FUNGSI MODULAR
# ══════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Membaca dataset CSV dari path yang diberikan.

    Parameters
    ----------
    filepath : str
        Lokasi file CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame mentah.
    """
    logger.info(f'[1/7] Memuat data dari: {filepath}')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'File tidak ditemukan: {filepath}')

    df = pd.read_csv(filepath)
    logger.info(f'      Data dimuat — Shape: {df.shape}')
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memperbaiki kolom 'TotalCharges' yang terbaca sebagai object
    akibat adanya nilai berupa spasi kosong (' ').

    Langkah:
      - Ganti string kosong/spasi → NaN
      - Konversi kolom ke tipe numerik (float64)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame dengan TotalCharges sudah bertipe float.
    """
    logger.info('[2/7] Memperbaiki kolom TotalCharges ...')
    df = df.copy()

    # Hitung baris bermasalah sebelum perbaikan
    if df['TotalCharges'].dtype == object:
        n_bad = (df['TotalCharges'].str.strip() == '').sum()
        logger.info(f'      Ditemukan {n_bad} baris dengan spasi kosong.')

        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        logger.info(f'      TotalCharges → float64. NaN sekarang: {df["TotalCharges"].isna().sum()}')
    else:
        logger.info('      TotalCharges sudah numerik, dilewati.')

    return df


def drop_identifier_cols(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Menghapus kolom identifier yang tidak memiliki nilai prediktif.

    Parameters
    ----------
    df   : pd.DataFrame
    cols : list, opsional
           Daftar kolom yang akan di-drop. Default: ['customerID'].

    Returns
    -------
    pd.DataFrame
    """
    cols = cols or ID_COLS
    logger.info(f'[3/7] Menghapus kolom identifier: {cols}')
    df = df.copy()

    existing = [c for c in cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    logger.info(f'      Kolom dihapus: {existing}. Shape baru: {df.shape}')
    return df


def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Mengimputasi missing value pada kolom numerik.

    Median dipilih karena lebih robust terhadap outlier
    dibandingkan mean.

    Parameters
    ----------
    df       : pd.DataFrame
    strategy : str, default 'median'
               Strategi imputasi ('mean', 'median', 'most_frequent').

    Returns
    -------
    pd.DataFrame
    """
    logger.info(f'[4/7] Imputasi missing value (strategi={strategy}) ...')
    df = df.copy()

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    total_missing_before = df[num_cols].isnull().sum().sum()
    logger.info(f'      Total NaN pada kolom numerik (sebelum): {total_missing_before}')

    if total_missing_before > 0:
        imputer = SimpleImputer(strategy=strategy)
        df[num_cols] = imputer.fit_transform(df[num_cols])
        logger.info(f'      Imputasi selesai. NaN tersisa: {df.isnull().sum().sum()}')
    else:
        logger.info('      Tidak ada missing value, dilewati.')

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Mengenkode variabel kategorikal:
    - Kolom BINER  (2 nilai unik) → LabelEncoder
    - Kolom MULTI  (>2 nilai unik) → pd.get_dummies (OneHotEncoding)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_encoded  : pd.DataFrame    — DataFrame setelah encoding
    encoding_info : dict          — Info mapping untuk referensi/audit
    """
    logger.info('[5/7] Encoding variabel kategorikal ...')
    df = df.copy()
    encoding_info = {'label_encoding': {}, 'ohe_columns': []}

    cat_cols  = df.select_dtypes(include=['object']).columns.tolist()
    bin_cols  = [c for c in cat_cols if df[c].nunique() == 2]
    multi_cols = [c for c in cat_cols if df[c].nunique() > 2]

    logger.info(f'      Kolom kategorikal    : {cat_cols}')
    logger.info(f'      Biner (LabelEncoder) : {bin_cols}')
    logger.info(f'      Multi-kelas (OHE)    : {multi_cols}')

    # ── LabelEncoder untuk kolom biner ──
    le = LabelEncoder()
    for col in bin_cols:
        df[col] = le.fit_transform(df[col])
        encoding_info['label_encoding'][col] = dict(
            zip(le.classes_, map(int, le.transform(le.classes_)))
        )

    # ── OneHotEncoding untuk kolom multi-kelas ──
    # drop_first=True menghindari dummy variable trap
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        encoding_info['ohe_columns'] = [
            c for c in df.columns
            if any(c.startswith(mc + '_') for mc in multi_cols)
        ]

    logger.info(f'      Shape setelah encoding: {df.shape}')
    return df, encoding_info


def scale_numerics(
    X: pd.DataFrame,
    cols_to_scale: list = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Menerapkan StandardScaler pada fitur numerik asli.

    Hanya kolom numerik asli yang di-scale (bukan hasil OHE yang
    sudah berskala 0/1 atau kolom target).

    Parameters
    ----------
    X             : pd.DataFrame  — Fitur (tanpa target)
    cols_to_scale : list, opsional

    Returns
    -------
    X_scaled : pd.DataFrame
    scaler   : StandardScaler (fitted) — untuk dipakai saat inferensi
    """
    cols_to_scale = cols_to_scale or NUM_COLS_SCALE
    cols_exist    = [c for c in cols_to_scale if c in X.columns]

    logger.info(f'[6/7] Scaling kolom numerik: {cols_exist}')
    X = X.copy()

    scaler = StandardScaler()
    X[cols_exist] = scaler.fit_transform(X[cols_exist])

    logger.info('      Scaling selesai.')
    return X, scaler


def save_outputs(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Menyimpan hasil preprocessing (X dan y) ke folder output.

    Files yang dihasilkan:
    - X_processed.csv  : Fitur siap latih
    - y_processed.csv  : Label target

    Parameters
    ----------
    X          : pd.DataFrame
    y          : pd.Series
    output_dir : str
    """
    logger.info(f'[7/7] Menyimpan hasil ke folder: {output_dir}/')
    os.makedirs(output_dir, exist_ok=True)

    x_path = os.path.join(output_dir, 'X_processed.csv')
    y_path = os.path.join(output_dir, 'y_processed.csv')

    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    logger.info(f'      ✅ X_processed.csv disimpan → {x_path}  (Shape: {X.shape})')
    logger.info(f'      ✅ y_processed.csv disimpan → {y_path}  (Shape: {y.shape})')


# ══════════════════════════════════════════════
# PIPELINE UTAMA
# ══════════════════════════════════════════════

def run_pipeline(filepath: str, output_dir: str = OUTPUT_DIR) -> None:
    """
    Mengorkestrasi seluruh langkah preprocessing secara berurutan.

    Parameters
    ----------
    filepath   : str  — Path ke file CSV mentah
    output_dir : str  — Folder tujuan output
    """
    logger.info('=' * 60)
    logger.info('  PIPELINE PREPROCESSING: Telco Customer Churn')
    logger.info('=' * 60)

    # Step 1 – Load
    df = load_data(filepath)

    # Step 2 – Perbaiki TotalCharges
    df = fix_total_charges(df)

    # Step 3 – Drop kolom identifier
    df = drop_identifier_cols(df)

    # Step 4 – Imputasi missing value
    df = impute_missing(df)

    # Step 5 – Encoding kategorikal
    df, enc_info = encode_categoricals(df)

    # Step 6 – Pisah fitur & target, lalu scaling
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X, scaler = scale_numerics(X)

    # Step 7 – Simpan output
    save_outputs(X, y, output_dir=output_dir)

    logger.info('=' * 60)
    logger.info('  PIPELINE SELESAI')
    logger.info(f'  X shape: {X.shape} | y shape: {y.shape}')
    logger.info(f'  Output tersimpan di: ./{output_dir}/')
    logger.info('=' * 60)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Otomatisasi Preprocessing - Telco Customer Churn'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='WA_Fn-UseC_-Telco-Customer-Churn.csv',
        help='Path ke file CSV dataset mentah.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_DIR,
        help='Folder tujuan hasil preprocessing.'
    )
    args = parser.parse_args()

    run_pipeline(filepath=args.input, output_dir=args.output)
