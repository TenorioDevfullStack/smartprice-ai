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
INFERENCE_MANIFEST_PATH = ROOT / "models/inference_manifest_v1.json"
# Raiz de confiança da release: mudar apenas após revisão do novo conjunto.
INFERENCE_MANIFEST_SHA256 = "b5d57117605bcea3575e2846436dd64a6645fca9d08ae408c46ab602eba98ffd"
REPORTS = ROOT / "reports"
HOLDOUT_START = "2025-10-22"
WINDOWS = [("2025-05-31", "2025-07-31"), ("2025-07-31", "2025-08-31"), ("2025-08-31", "2025-10-21")]
SEED = 42
PRICE_BINS = [0, 500000, 1000000, 2000000, 5000000, float("inf")]
PRICE_LABELS = ["Até R$ 500 mil", "R$ 500 mil–1 mi", "R$ 1–2 mi", "R$ 2–5 mi", "Acima de R$ 5 mi"]
