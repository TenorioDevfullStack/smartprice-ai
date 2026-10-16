from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREA = "Área Construída (m2)"
CEP = "CEP_8"
STANDARD = "Padrão (IPTU)"
YEAR = "ACC (IPTU)"
TARGET = "Valor de Transação (declarado pelo contribuinte)"
DATE = "Data de Transação"
FEATURES = [AREA, CEP, STANDARD, YEAR]
DATA_PATH = ROOT / "data/processed/smartprice_demo.csv.gz"
MODEL_PATH = ROOT / "models/smartprice_model.joblib"
METRICS_PATH = ROOT / "models/metrics.json"
REPORTS = ROOT / "reports"
HOLDOUT_START = "2025-10-22"
WINDOWS = [("2025-05-31", "2025-07-31"), ("2025-07-31", "2025-08-31"), ("2025-08-31", "2025-10-21")]
SEED = 42
PRICE_BINS = [0, 500000, 1000000, 2000000, 5000000, float("inf")]
PRICE_LABELS = ["Até R$ 500 mil", "R$ 500 mil–1 mi", "R$ 1–2 mi", "R$ 2–5 mi", "Acima de R$ 5 mi"]
