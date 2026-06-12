from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data/raw/customer_support_tickets.csv"

INTERIM_DIR = ROOT / "data/interim"
PROCESSED_DIR = ROOT / "data/processed"

MODEL_DIR = ROOT / "outputs/models"
DOSSIER_DIR = ROOT / "outputs/dossiers"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEVERITY_MAP = {
    "Low":0,
    "Medium":1,
    "High":2,
    "Critical":3
}

INV_SEVERITY_MAP = {
    0:"Low",
    1:"Medium",
    2:"High",
    3:"Critical"
}

RANDOM_STATE = 42