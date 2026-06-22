import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_RESULTS_DIR, MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Science Configuration
SIGMA_CLIP_LIMIT = 3.0  # Outlier removal threshold (standard deviations)

# Detrending (Savitzky-Golay Filter)
DETREND_WINDOW_DAYS = 0.5  # Window size in days (gets converted to number of points)
DETREND_POLYORDER = 2      # Polynomial order

# Transit Search (Box Least Squares)
BLS_MIN_PERIOD = 0.5       # Minimum search period in days
BLS_MAX_PERIOD = 15.0      # Maximum search period in days
BLS_OVERSAMPLE = 2.0       # Period grid oversampling factor
BLS_DURATION_MIN = 0.05    # Minimum transit duration fraction of period
BLS_DURATION_MAX = 0.15    # Maximum transit duration fraction of period

# ML Classification Configuration
MODEL_PATH = MODEL_DIR / "classifier.pkl"
TRAINING_SET_SIZE = 200    # Number of synthetic light curves to generate for training
RANDOM_STATE = 42

# Target Classes
CLASSES = {
    0: "instrumental_artifact",  # noise/artifact
    1: "stellar_variability",   # spots/pulsations
    2: "eclipsing_binary",     # stellar companion (V-shaped or secondary eclipse)
    3: "exoplanet_transit"     # planet (U-shaped, flat bottom)
}
