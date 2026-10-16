"""Uma única transformação entre entradas brutas e previsão."""
import json

import joblib
import pandas as pd

from src.config import AREA, CEP, STANDARD, YEAR, MODEL_PATH, METRICS_PATH
from src.evaluation import observation_band
from src.feature_engineering import validate_inputs


def load_bundle(model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    if not model_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Artefatos ausentes. Execute python -m src.train na raiz do projeto.")
    return joblib.load(model_path), json.loads(metrics_path.read_text(encoding="utf-8"))


def estimate_price(area, cep, standard, year, bundle=None):
    model, metadata = bundle if bundle is not None else load_bundle()
    frame = validate_inputs(pd.DataFrame([{AREA:area,CEP:cep,STANDARD:standard,YEAR:year}]))
    predicted = float(model.predict(frame)[0])
    lower,upper = observation_band(predicted,metadata["calibration"])
    warnings = []
    if frame[CEP].iloc[0] not in metadata["training_categories"][CEP]:
        warnings.append("CEP inédito no treino: a estimativa usa a representação disponível e merece cautela adicional.")
    if frame[STANDARD].iloc[0] not in metadata["training_categories"][STANDARD]:
        warnings.append("Padrão IPTU inédito no treino.")
    bounds = metadata["training_area_range"]
    if not bounds[0] <= float(frame[AREA].iloc[0]) <= bounds[1]:
        warnings.append("Área fora da faixa observada no treino; extrapolação.")
    if predicted > 5000000:
        warnings.append("Imóveis de alto valor têm menos exemplos e erro histórico elevado.")
    return {"prediction":predicted,"lower":lower,"upper":upper,"warnings":warnings}
