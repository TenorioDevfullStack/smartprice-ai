"""Somente atributos cadastrais disponíveis antes da negociação."""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import AREA, CEP, STANDARD, YEAR, FEATURES


def normalize_cep(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        text = str(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not value.is_integer():
            return None
        text = str(int(value))
    else:
        text = str(value).strip()
        if re.fullmatch(r"[0-9]{5}-[0-9]{3}", text):
            text = text.replace("-", "")
    return text.zfill(8) if re.fullmatch(r"[0-9]{7,8}", text) else None


def validate_inputs(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Informe pelo menos um imóvel.")
    result = frame[FEATURES].copy()
    for column in [AREA, YEAR]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"{column}: informe um número finito.")
    if not result[AREA].gt(0).all():
        raise ValueError("Área construída deve ser maior que zero.")
    if not result[YEAR].between(1900, 2025).all() or not result[YEAR].mod(1).eq(0).all():
        raise ValueError("ACC IPTU deve ser um ano inteiro entre 1900 e 2025 (coorte do modelo).")
    result[CEP] = result[CEP].map(normalize_cep)
    if result[CEP].isna().any():
        raise ValueError("CEP inválido: use oito dígitos ou 00000-000.")
    if result[STANDARD].isna().any() or result[STANDARD].astype(str).str.strip().eq("").any():
        raise ValueError("Informe o padrão IPTU.")
    result[CEP] = result[CEP].astype("string")
    result[STANDARD] = result[STANDARD].astype("string").str.strip()
    return result


class RawInputs(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_inputs(X)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        return validate_inputs(X)


class PropertyFeatures(RawInputs):
    def transform(self, X):
        result = validate_inputs(X)
        result["log_area"] = np.log1p(result[AREA])
        result["cep_prefix3"] = result[CEP].str[:3]
        return result
