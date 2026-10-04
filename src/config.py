"""Central place for paths and constants so nothing is hard-coded elsewhere."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "Churn"
ID_COL = "customerID"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
