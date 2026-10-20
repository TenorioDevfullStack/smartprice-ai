"""Inferência verificada e versionada, sem ajuste ou troca do hist_log."""
import io
import math
from decimal import Decimal
from numbers import Real

import joblib
import numpy as np
import pandas as pd

from src.config import AREA, CEP, STANDARD, YEAR
from src.artifact_contract import InferenceBundle, verify_artifacts
from src.evaluation import observation_band
from src.feature_engineering import validate_inputs
from src.inference_errors import (
    ArtifactIntegrityError, InputValidationError, PredictionError, fail,
)


def load_bundle(model_path=None, metrics_path=None, manifest_path=None, *, expected_fingerprint=None):
    checked = verify_artifacts(model_path, metrics_path, manifest_path)
    if expected_fingerprint is not None and checked.fingerprint != expected_fingerprint:
        fail(ArtifactIntegrityError, "cache_fingerprint")
    try:
        # Usa os mesmos bytes conferidos; não reabre um path mutável após o hash.
        model = joblib.load(io.BytesIO(checked.model_bytes))
    except Exception as exc:
        fail(ArtifactIntegrityError, "deserialize", exc)
    return InferenceBundle(model, checked.metadata_bytes, checked.manifest_bytes, checked.fingerprint)


def _input_frame(area, cep, standard, year):
    try:
        for value in [area, cep, standard, year]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, Real, Decimal)):
                raise ValueError("Tipo de entrada inválido")
            if not isinstance(value, str) and not math.isfinite(float(value)):
                raise ValueError("Entrada não finita")
        frame = pd.DataFrame([{AREA: area, CEP: cep, STANDARD: standard, YEAR: year}])
        return validate_inputs(frame)
    except (ValueError, TypeError, OverflowError) as exc:
        fail(InputValidationError, "input_validation", exc)


def estimate_price(area, cep, standard, year, bundle=None):
    frame = _input_frame(area, cep, standard, year)
    bundle = load_bundle() if bundle is None else bundle
    if not isinstance(bundle, InferenceBundle):
        fail(ArtifactIntegrityError, "unverified_bundle")
    metadata, manifest = bundle.metadata, bundle.manifest
    warnings, codes = [], []

    def warn(code, message):
        codes.append(code)
        warnings.append(message)

    known_cep = frame[CEP].iloc[0] in metadata["training_categories"][CEP]
    if not known_cep:
        warn("UNSEEN_CEP", "CEP inédito no treino: a estimativa usa a representação disponível e merece cautela adicional.")
        warn("GEOGRAPHY_UNVERIFIED", "Não foi possível confirmar a cobertura geográfica deste CEP. O modelo foi desenvolvido somente para apartamentos de São Paulo.")
    if frame[STANDARD].iloc[0] not in metadata["training_categories"][STANDARD]:
        warn("UNSEEN_STANDARD", "Padrão IPTU inédito no treino; a estimativa tem cobertura limitada.")
    bounds = metadata["training_area_range"]
    if not bounds[0] <= float(frame[AREA].iloc[0]) <= bounds[1]:
        warn("AREA_EXTRAPOLATION", "Área fora da faixa observada no treino; extrapolação.")
    year_bounds = manifest["domain"]["year_range"]
    if not year_bounds[0] <= float(frame[YEAR].iloc[0]) <= year_bounds[1]:
        warn("YEAR_EXTRAPOLATION", "ACC IPTU fora da faixa observada no treino; extrapolação.")
    try:
        predictions = np.asarray(bundle.model.predict(frame), dtype=float)
        if predictions.shape != (1,) or not np.isfinite(predictions).all() or predictions[0] < 0:
            raise ValueError("Saída inválida")
        predicted = float(predictions[0])
        lower, upper = observation_band(predicted, metadata["calibration"])
        if not all(math.isfinite(v) for v in [lower, upper]) or not 0 <= lower <= predicted <= upper:
            raise ValueError("Faixa inválida")
    except Exception as exc:
        fail(PredictionError, "predict_or_band", exc)
    if predicted > 5000000:
        warn("HIGH_VALUE", "Imóveis de alto valor têm menos exemplos e erro histórico elevado.")
    return {
        "prediction": predicted, "lower": lower, "upper": upper, "warnings": warnings,
        "warning_codes": codes, "status": "WARNING" if warnings else "ACCEPT",
        "model_version": manifest["model_version"], "schema_version": manifest["schema_version"],
        "algorithm": manifest["algorithm"], "reference_period": manifest["reference_period"],
        "training_period": manifest["training_period"], "calibration_id": manifest["calibration"]["id"],
        "band_kind": manifest["calibration"]["band_kind"], "limitations": manifest["limitations"],
        "geographic_status": "SEEN_IN_TRAINING" if known_cep else "UNVERIFIED",
    }
