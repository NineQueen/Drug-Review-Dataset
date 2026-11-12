from pathlib import Path

# Resolved from this file, independent of the notebook working directory.
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PARQUET = "train.parquet"
VALID_PARQUET = "valid.parquet"
TEST_PARQUET = "test.parquet"
TRAIN_RAW_PARQUET = "train_raw.parquet"

TRAIN_PREDICTION_PARQUET = "train_predictions.parquet"
VALID_PREDICTION_PARQUET = "valid_predictions.parquet"
TEST_PREDICTION_PARQUET = "test_predictions.parquet"

RAW_CSV = ROOT / "data" / "raw" / "drugsComTrain_raw.csv"
WEBMD_CSV = ROOT / "data" / "raw" / "webmd.csv.gz"
REPLACEMENT_JSON = ROOT / "replacement.json"

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
# Fitted Spark models are kept in git. Generated frames under data/processed are not.
EMBEDDING_MODELS_DIR = ROOT / "src" / "embeddings" / "models"
ANALYTIC_MODELS_DIR = ROOT / "src" / "analytics" / "models"
SPLIT_DIR = ROOT / "data" / "processed" / "split"
EMBEDDING_DIR = ROOT / "data" / "processed" / "embeddings"
PREDICTION_DIR = ROOT / "data" / "processed" / "predictions"
TEMP_DIR = ROOT / "data" / "processed" / "temp"
# Out-of-fold base scores and fold embedding models for the second level.
STACKING_DIR = ROOT / "data" / "processed" / "stacking"
STACKING_EMBEDDING_DIR = STACKING_DIR / "models"
STACKING_MANIFEST = "manifest.json"
STACKING_ORDINAL_FILE = ROOT / "src" / "stacking" / "ordinal.json"

PARAMETER_DIR = ROOT / "data" / "parameter"