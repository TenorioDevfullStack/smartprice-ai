"""Congela a V3 auditada, sem reexecutar notebooks ou remover observações."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, DATA_PATH, FEATURES, AREA, CEP, YEAR, TARGET, DATE, WINDOWS
from src.feature_engineering import validate_inputs


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recovered_data():
    manifest = json.loads((ROOT / "models/baseline_v1_manifesto.json").read_text(encoding="utf-8"))
    path = ROOT / "data/processed/baseline_v1_particoes.joblib"
    if sha256(path) != manifest["artifact_sha256"][path.relative_to(ROOT).as_posix()]:
        raise ValueError("Partições V1 divergentes do SHA-256 auditado.")
    parts = joblib.load(path)
    frames = []
    for split in ["train", "test"]:
        x, y, dates = [parts[f"{key}_{split}"] for key in ["X", "y", "datas"]]
        if not x.index.equals(y.index) or not x.index.equals(dates.index):
            raise ValueError("Partições desalinhadas.")
        frames.append(x.assign(**{TARGET:y, DATE:dates}))
    return pd.concat(frames).sort_index().rename_axis("indice_original")


def validate_dataset(frame):
    if set(frame.columns) != set(FEATURES + [TARGET, DATE]):
        raise ValueError("Schema de modelagem inválido; somente features, target e data.")
    if not frame.index.is_unique:
        raise ValueError("Identificadores técnicos duplicados.")
    for column in [AREA,YEAR,TARGET]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Schema: {column} deve ter tipo numérico.")
    validate_inputs(frame)
    dates = pd.to_datetime(frame[DATE], errors="coerce")
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    if dates.isna().any() or not dates.dt.year.eq(2025).all():
        raise ValueError("Data inválida ou fora da coorte de 2025.")
    if not np.isfinite(y).all() or not y.ge(10000).all():
        raise ValueError("Preço inválido; V3 exige valor >= R$ 10 mil.")
    if not frame[YEAR].le(dates.dt.year).all():
        raise ValueError("ACC IPTU posterior à transação.")
    return {
        "rows": len(frame), "start": str(dates.min().date()), "end":str(dates.max().date()),
        "dtypes":{c:str(t) for c,t in frame.dtypes.items()},
        "missing":{c:int(n) for c,n in frame.isna().sum().items()},
        "duplicate_technical_ids":int(frame.index.duplicated().sum()),
        "duplicate_reduced_rows":int(frame.duplicated().sum()),
        "duplicate_policy":"Não remover: projeção reduzida não identifica duplicatas de transações. V3 removeu cópias exatas usando todas as colunas originais.",
        "area_kind":"Área Construída (m2) cadastral; não equivale necessariamente à área útil.",
        "area":frame[AREA].describe().to_dict(), "price":y.describe().to_dict(),
        "price_m2":(y/frame[AREA]).describe().to_dict(), "unique_ceps":int(frame[CEP].nunique()),
        "location":"CEP cadastral; sem bairro inferido artificialmente, endereço ou SQL.",
    }


def load_modeling_data(path=DATA_PATH):
    if path.exists():
        frame = pd.read_csv(path, index_col="indice_original", dtype={CEP:"string", FEATURES[2]:"string"}, parse_dates=[DATE], float_precision="round_trip")
    else:
        frame = load_recovered_data()
    validate_dataset(frame)
    return frame


def freeze_dataset():
    source = load_recovered_data()
    report = validate_dataset(source)
    if DATA_PATH.exists():
        pd.testing.assert_frame_equal(source, load_modeling_data(), check_dtype=False)
    else:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(DATA_PATH, index_label="indice_original", float_format="%.17g", compression={"method":"gzip", "mtime":0})
        pd.testing.assert_frame_equal(source, load_modeling_data(), check_dtype=False)
    report["sha256"] = sha256(DATA_PATH)
    return source, report


def temporal_windows(frame):
    dates = pd.to_datetime(frame[DATE])
    return [(frame.loc[dates.le(cut)].copy(), frame.loc[dates.gt(cut) & dates.le(end)].copy()) for cut,end in WINDOWS]
