"""
automate_Student.py
Skrip otomatisasi preprocessing dataset Heart Disease UCI.

Tahapan IDENTIK dengan notebook Eksperimen_Student.ipynb:
  Step 1 - Handle Missing Values
  Step 2 - Remove Duplicates
  Step 3 - Handle Outliers (IQR Method)
  Step 4 - Feature Engineering (Binning & Fitur Turunan)
  Step 5 - Encoding Data Kategorikal
  Step 6 - Normalisasi / Standarisasi Fitur (StandardScaler)
  Step 7 - Train-Test Split
  Step 8 - Simpan Preprocessed Data

Usage:
    python automate_Student.py --input heart_raw/heart_raw.csv --output_dir heart_preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import logging
import os

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# STEP 0 - Load Dataset  (sama dengan Bagian 3 notebook)
# ─────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari path yang diberikan (CSV atau Excel)."""
    logger.info(f"Memuat dataset dari: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Format file tidak didukung: {ext}")
    logger.info(f"  Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    logger.info(f"  Kolom: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 1 - Handle Missing Values
# ─────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame,
                           target_col: str = "target") -> pd.DataFrame:
    """
    Menangani missing values dengan imputasi median pada kolom numerik.
    Identik dengan Step 1 di notebook.
    """
    logger.info("STEP 1: Handle Missing Values")
    df = df.copy()

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    missing_total = df[num_cols].isnull().sum().sum()
    if missing_total > 0:
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])
        logger.info(f"  {missing_total} missing values diimputasi (strategi: median)")
    else:
        logger.info("  Tidak ada missing values — dataset sudah bersih")

    logger.info(f"  Shape setelah handling missing: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 - Remove Duplicates
# ─────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat.
    Identik dengan Step 2 di notebook.
    """
    logger.info("STEP 2: Remove Duplicates")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    logger.info(f"  Sebelum : {before} baris")
    logger.info(f"  Sesudah : {len(df)} baris")
    logger.info(f"  Dihapus : {removed} baris duplikat")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 - Handle Outliers (IQR Method)
# ─────────────────────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame,
                    columns: list = None) -> pd.DataFrame:
    """
    Mendeteksi dan menghapus outlier menggunakan metode IQR.
    Identik dengan Step 3 di notebook.
    """
    logger.info("STEP 3: Handle Outliers (IQR Method)")
    if columns is None:
        columns = ["trestbps", "chol", "thalach", "oldpeak"]

    df_out = df.copy()
    total_removed = 0

    for col in columns:
        if col not in df_out.columns:
            continue
        Q1  = df_out[col].quantile(0.25)
        Q3  = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before_n = len(df_out)
        df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
        removed_n = before_n - len(df_out)
        total_removed += removed_n
        if removed_n > 0:
            logger.info(f"  {col:12s} → {removed_n} outlier dihapus "
                        f"[{lower:.1f}, {upper:.1f}]")

    logger.info(f"  Total dihapus : {total_removed} baris")
    logger.info(f"  Shape sesudah : {df_out.reset_index(drop=True).shape}")
    return df_out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 4 - Feature Engineering (Binning & Fitur Turunan)
# ─────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur baru:
      - age_group         : binning usia ke 4 kelompok
      - chol_thalach_ratio: rasio kolesterol per detak jantung
    Identik dengan Step 4 di notebook.
    """
    logger.info("STEP 4: Feature Engineering (Binning & Fitur Turunan)")
    df = df.copy()

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 40, 55, 65, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)
        logger.info("  Fitur baru: age_group  (binning usia → 4 kelompok)")
        logger.info("       0=≤40 | 1=41-55 | 2=56-65 | 3=>65")

    if "chol" in df.columns and "thalach" in df.columns:
        df["chol_thalach_ratio"] = df["chol"] / (df["thalach"] + 1)
        logger.info("  Fitur baru: chol_thalach_ratio  (chol / thalach)")

    new_feats = ["age_group", "chol_thalach_ratio"]
    logger.info(f"  Shape setelah feature engineering: {df.shape}")
    logger.info(f"  Kolom baru: {[c for c in new_feats if c in df.columns]}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 5 - Encoding Data Kategorikal
# ─────────────────────────────────────────────────────────────
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan Label Encoding pada kolom bertipe object/category.
    Identik dengan Step 5 di notebook.
    """
    logger.info("STEP 5: Encoding Data Kategorikal")
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"  {col} → Label Encoded")
    else:
        logger.info("  Semua kolom sudah dalam format numerik")
        logger.info(f"  Tipe data: {df.dtypes.value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 - Normalisasi / Standarisasi Fitur (StandardScaler)
# ─────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame,
                   target_col: str = "target") -> tuple:
    """
    Menerapkan StandardScaler pada seluruh fitur (kecuali target).
    Identik dengan Step 6 di notebook.

    Returns:
        df_scaled (DataFrame), scaler (StandardScaler)
    """
    logger.info("STEP 6: Normalisasi / Standarisasi Fitur (StandardScaler)")
    df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    logger.info(f"  Metode       : StandardScaler (mean=0, std=1)")
    logger.info(f"  Jumlah fitur : {len(feature_cols)}")
    logger.info(f"  Fitur        : {feature_cols}")
    return df, scaler


# ─────────────────────────────────────────────────────────────
# STEP 7 - Train-Test Split
# ─────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame,
               target_col: str = "target",
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Membagi dataset menjadi train dan test set dengan stratifikasi.
    Identik dengan Step 7 di notebook.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("STEP 7: Train-Test Split")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"  Test size    : {int(test_size*100)}%")
    logger.info(f"  Random state : {random_state}")
    logger.info(f"  Stratify     : Ya (menjaga proporsi kelas)")
    logger.info(f"  X_train      : {X_train.shape}")
    logger.info(f"  X_test       : {X_test.shape}")
    logger.info(f"  y_train dist : {y_train.value_counts().to_dict()}")
    logger.info(f"  y_test dist  : {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# STEP 8 - Simpan Preprocessed Data
# ─────────────────────────────────────────────────────────────
def save_preprocessed(X_train, X_test, y_train, y_test,
                       output_dir: str = "heart_preprocessing") -> tuple:
    """
    Menyimpan dataset train dan test yang sudah siap dilatih ke CSV.
    Identik dengan Step 8 di notebook.

    Returns:
        (train_path, test_path)
    """
    logger.info("STEP 8: Simpan Preprocessed Data")
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train.values
    test_df  = X_test.copy()
    test_df["target"]  = y_test.values

    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    logger.info(f"  📁 {train_path}  → {len(train_df)} baris, {train_df.shape[1]} kolom")
    logger.info(f"  📁 {test_path}   → {len(test_df)} baris,  {test_df.shape[1]} kolom")
    return train_path, test_path


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_preprocessing(input_path: str,
                      output_dir: str = "heart_preprocessing",
                      target_col: str = "target",
                      outlier_cols: list = None) -> tuple:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.
    Tahapan identik dengan notebook Eksperimen_Student.ipynb (Step 1–8).

    Returns:
        (train_path, test_path)
    """
    logger.info("=" * 60)
    logger.info("MULAI PREPROCESSING PIPELINE")
    logger.info("Dataset    : Heart Disease UCI")
    logger.info(f"Input      : {input_path}")
    logger.info(f"Output dir : {output_dir}")
    logger.info("=" * 60)

    df = load_data(input_path)                              # Load
    df = handle_missing_values(df, target_col)             # Step 1
    df = remove_duplicates(df)                             # Step 2
    df = handle_outliers(df, outlier_cols)                 # Step 3
    df = feature_engineering(df)                           # Step 4
    df = encode_categorical(df)                            # Step 5
    df, _ = scale_features(df, target_col)                 # Step 6
    X_train, X_test, y_train, y_test = split_data(         # Step 7
        df, target_col
    )
    train_path, test_path = save_preprocessed(             # Step 8
        X_train, X_test, y_train, y_test, output_dir
    )

    logger.info("=" * 60)
    logger.info("✅ PREPROCESSING SELESAI")
    logger.info(f"   Dataset siap dilatih tersedia di: {output_dir}/")
    logger.info("=" * 60)
    return train_path, test_path


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing Heart Disease UCI — identik dengan notebook"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="heart_raw/heart_raw.csv",
        help="Path ke dataset raw (default: heart_raw/heart_raw.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="heart_preprocessing",
        help="Direktori output hasil preprocessing"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="target",
        help="Nama kolom target (default: target)"
    )
    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_dir=args.output_dir,
        target_col=args.target_col,
    )
