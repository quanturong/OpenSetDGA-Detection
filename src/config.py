"""
config.py – Centralized configuration for the OpenSetDGA pipeline.

All paths, hyperparameters, and constants are defined here.
Import this module instead of hardcoding values in individual scripts.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
KNOWN_DIR       = PROCESSED_DIR / "known"
UNKNOWN_FAM_DIR = PROCESSED_DIR / "unknown_family"
UNKNOWN_OOD_DIR = PROCESSED_DIR / "unknown_ood"
FEATURE_CACHE   = PROCESSED_DIR / "_feature_cache"
BASELINE_OUT    = ROOT_DIR / "baseline_out"

# ── Dataset constants ─────────────────────────────────────────────────────────
N_KNOWN_CLASSES  = 19          # after removing families with < MIN_SAMPLES samples
MIN_SAMPLES      = 50          # minimum samples per family to be included in known split
RANDOM_SEED      = 42
TRAIN_SIZE       = 320_000
VAL_SIZE         = 40_000
TEST_KNOWN_SIZE  = 40_000
UNKNOWN_FAM_SIZE = 50_000
UNKNOWN_OOD_SIZE = 100_000

# ── Tokenization ──────────────────────────────────────────────────────────────
CHARS    = "abcdefghijklmnopqrstuvwxyz0123456789-."
VOCAB_SIZE = len(CHARS) + 2    # 0=PAD, 1=UNK
MAX_LEN  = 75

# ── Neural hyperparameters ────────────────────────────────────────────────────
EMBED_DIM  = 32
FEAT_DIM   = 128
BATCH_SIZE = 512
LR         = 1e-3
MAX_EPOCHS = 30
PATIENCE   = 5
WEIGHT_DECAY = 1e-4

# BiLSTM specific
BILSTM_HIDDEN_DIM  = 64
BILSTM_NUM_LAYERS  = 1
BILSTM_DROPOUT     = 0.2

# ── OOD scoring parameters ────────────────────────────────────────────────────
ENERGY_TEMPERATURE = 1.0
ODIN_TEMPERATURE   = 1000
ODIN_EPS_VALUES    = [0.0, 0.001, 0.005]
KNN_K_VALUES       = [5, 20, 50]
KNN_SUBSAMPLE      = 50_000
REACT_PERCENTILES  = [65, 75, 85, 90, 95]

# Outlier Exposure
LAMBDA_OE       = 0.5
OE_TRAIN_SIZE   = 50_000

# ── LightGBM parameters ───────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators":  1000,
    "learning_rate": 0.05,
    "num_leaves":    63,
    "max_depth":     -1,
    "random_state":  RANDOM_SEED,
    "n_jobs":        -1,
    "verbose":       -1,
}

LGBM_MULTICLASS_PARAMS = {
    **LGBM_PARAMS,
    "n_estimators": 800,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT   = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
LOG_DATEFMT  = "%Y-%m-%d %H:%M:%S"
