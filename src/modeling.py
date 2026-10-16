"""Candidatos enxutos e política física explícita para a demonstração."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits

from src.config import ROOT, AREA, CEP, STANDARD, YEAR, SEED
from src.feature_engineering import RawInputs, PropertyFeatures, validate_inputs

CANDIDATES = ["median", "cep_m2", "ridge_v1", "ridge_log", "forest", "forest_log", "hist", "hist_log"]


class CepM2Regressor(RegressorMixin, BaseEstimator):
    def __init__(self, min_count=5):
        self.min_count = min_count

    def fit(self, X, y):
        frame = validate_inputs(X)
        values = np.asarray(y, dtype=float) / frame[AREA].to_numpy()
        self.global_m2_ = float(np.median(values))
        grouped = pd.DataFrame({"cep":frame[CEP].to_numpy(), "m2":values}).groupby("cep")["m2"].agg(["median", "count"])
        self.cep_m2_ = grouped.loc[grouped["count"].ge(self.min_count), "median"].to_dict()
        return self

    def predict(self, X):
        frame = validate_inputs(X)
        return frame[CEP].map(self.cep_m2_).fillna(self.global_m2_).to_numpy(dtype=float) * frame[AREA].to_numpy()


class NonNegativeRegressor(RegressorMixin, BaseEstimator):
    """Clipping público >=0; o método unbounded permite auditar o efeito."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict_unbounded(self, X):
        validate_inputs(X)
        with threadpool_limits(limits=2):
            predictions = np.asarray(self.estimator_.predict(X), dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError("Modelo retornou previsão não finita.")
        return predictions

    def predict(self, X):
        return np.maximum(0, self.predict_unbounded(X))


def make_candidate(name):
    if name not in CANDIDATES:
        raise ValueError(f"Candidato desconhecido: {name}")
    if name == "median":
        return Pipeline([("inputs",RawInputs()), ("regressor",DummyRegressor(strategy="median"))])
    if name == "cep_m2":
        return CepM2Regressor()
    if name.startswith("ridge"):
        original = joblib.load(ROOT / "models/baseline_v1_recuperado.joblib")
        estimator = Pipeline([("inputs",RawInputs()), ("ridge",clone(original))])
    else:
        categorical = ["cep_prefix3",STANDARD]
        numeric = [AREA,YEAR,"log_area"]
        if name.startswith("hist"):
            # Códigos categóricos não ordenados: -1 é missing para HGB.
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1, max_categories=254)
            regressor = HistGradientBoostingRegressor(max_iter=180, learning_rate=.08, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=1, categorical_features=[3,4],
                early_stopping=False, random_state=SEED)
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
            regressor = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=5,
                max_features=.8, n_jobs=2, random_state=SEED)
        pre = ColumnTransformer([("numeric",SimpleImputer(strategy="median"),numeric), ("categorical",encoder,categorical)], sparse_threshold=0)
        estimator = Pipeline([("features",PropertyFeatures()), ("preprocessing",pre), ("regressor",regressor)])
    if name.endswith("_log"):
        return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1, check_inverse=True)
    return estimator
