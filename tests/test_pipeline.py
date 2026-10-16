import importlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest


def module(name):
    assert importlib.util.find_spec(name) is not None, f"Implementar {name}"
    return importlib.import_module(name)


@pytest.fixture
def sample():
    return pd.DataFrame({
        "Área Construída (m2)": [40, 70, 100, 130, 180, 60, 80, 120],
        "CEP_8": ["01234000", "01234000", "04567000", "04567000"] * 2,
        "Padrão (IPTU)": ["21", "22", "23", "24"] * 2,
        "ACC (IPTU)": [1980, 2000, 2010, 2020] * 2,
    })


def test_freeze_load_schema_and_temporal_windows():
    dp = module("src.data_processing")
    config = module("src.config")
    frame = dp.load_modeling_data()
    assert len(frame) == 63164
    assert frame.index.is_unique
    assert set(frame.columns) == set(config.FEATURES + [config.TARGET, config.DATE])
    report = dp.validate_dataset(frame)
    assert report["rows"] == 63164
    windows = dp.temporal_windows(frame)
    assert len(windows) == 3
    for train, validation in windows:
        assert train[config.DATE].max() < validation[config.DATE].min()
        assert validation[config.DATE].max() <= pd.Timestamp("2025-10-21")
        assert train.index.intersection(validation.index).empty


@pytest.mark.parametrize("value,expected", [(1234567,"01234567"), ("01234-567","01234567"), ("12A34567",None), ("123",None), (np.nan,None)])
def test_cep_normalization(value, expected):
    assert module("src.feature_engineering").normalize_cep(value) == expected


def test_features_do_not_leak_or_use_transaction_date(sample):
    fe = module("src.feature_engineering")
    original = sample.copy(deep=True)
    transformed = fe.PropertyFeatures().fit_transform(sample)
    contaminated = sample.assign(Preco_m2=999999, **{"Valor de Transação (declarado pelo contribuinte)":999999, "Data de Transação":"2099-01-01"})
    pd.testing.assert_frame_equal(transformed, fe.PropertyFeatures().fit_transform(contaminated))
    pd.testing.assert_frame_equal(sample, original)
    assert "log_area" in transformed
    np.testing.assert_allclose(transformed["log_area"], np.log1p(sample["Área Construída (m2)"]))


@pytest.mark.parametrize("change", [{"Área Construída (m2)":0}, {"Área Construída (m2)":np.inf}, {"ACC (IPTU)":1800}, {"CEP_8":"bad"}])
def test_inference_rejects_invalid_inputs(sample, change):
    fe = module("src.feature_engineering")
    frame = sample.iloc[:1].copy().assign(**change)
    with pytest.raises(ValueError):
        fe.validate_inputs(frame)


@pytest.mark.parametrize("name", ["median", "cep_m2", "ridge_v1", "ridge_log", "forest", "forest_log", "hist", "hist_log"])
def test_candidates_unknown_categories_fit_predict_and_persistence(name, sample, tmp_path):
    modeling = module("src.modeling")
    y = np.array([200000, 400000, 800000, 1000000, 1500000, 300000, 500000, 900000])
    model = modeling.make_candidate(name)
    model.fit(sample, y)
    unknown = sample.iloc[:2].copy().assign(CEP_8="09999999", **{"Padrão (IPTU)":"INEDITO"})
    pred = model.predict(unknown)
    assert np.isfinite(pred).all()
    destination = tmp_path / "model.joblib"
    joblib.dump(model, destination)
    # RF soma árvores em paralelo: a ordem de soma pode variar em ~1e-9 real.
    # Tolerância subcentavo, sem exigir identidade binária de ponto flutuante.
    np.testing.assert_allclose(pred, joblib.load(destination).predict(unknown), rtol=1e-12, atol=1e-6)


def test_m2_baseline_only_learns_train_prices(sample):
    model = module("src.modeling").make_candidate("cep_m2")
    model.fit(sample, np.asarray(sample["Área Construída (m2)"]) * 10000)
    unknown = sample.iloc[:1].copy().assign(CEP_8="09999999")
    assert model.predict(unknown)[0] == 400000


def test_evaluation_segments_and_empty():
    evaluation = module("src.evaluation")
    y = np.array([100000,500000,1000000,2000000,5000000,6000000])
    pred = y * .8
    stats = evaluation.regression_metrics(y, pred)
    assert stats["negative_predictions"] == 0
    assert stats["MedAE"] == 300000
    bands = evaluation.price_segments(y, pred)
    assert sum(b["n"] for b in bands) == len(y)
    assert bands[-1]["n"] == 1 and bands[-1]["underestimate_pct"] == 100
    assert evaluation.regression_metrics([], [])["n"] == 0


def test_uncertainty_groups_by_prediction_not_true_target():
    ev = module("src.evaluation")
    calibration = ev.calibrate_error(np.array([9000000.,200000.]), np.array([100000.,100000.]), min_group=1)
    assert calibration["groups"][0]["n"] == 2
    assert calibration["groups"][-1]["n"] == 0


def test_nonnegative_policy_is_explicit(sample):
    from sklearn.dummy import DummyRegressor
    wrapper = module("src.modeling").NonNegativeRegressor(DummyRegressor(strategy="constant",constant=-50)).fit(sample, np.ones(len(sample)))
    assert np.all(wrapper.predict(sample) == 0)
    assert np.all(wrapper.predict_unbounded(sample) == -50)


def test_dashboard_helpers_empty_filters_and_money(sample):
    helpers = module("app.helpers")
    frame = sample.assign(**{"Data de Transação":pd.Timestamp("2025-06-01"), "Valor de Transação (declarado pelo contribuinte)":500000})
    assert helpers.filter_market(frame, (pd.Timestamp("2025-01-01"),pd.Timestamp("2025-12-31")), (0,1000000),(0,300),["99999999"]).empty
    assert helpers.money(1234567.89) == "R$ 1.234.568"
    assert helpers.market_summary(frame)["transactions"] == 8


def test_invalid_dataset_fails_instead_of_dropping_rows():
    dp = module("src.data_processing")
    config = module("src.config")
    frame = dp.load_modeling_data().iloc[:10].copy()
    frame.loc[frame.index[0],config.TARGET] = -10
    with pytest.raises(ValueError):
        dp.validate_dataset(frame)


def test_final_artifact_and_inference():
    inference = module("src.inference")
    config = module("src.config")
    assert config.MODEL_PATH.exists(), "Modelo final deve acompanhar o checkout"
    bundle = inference.load_bundle()
    result = inference.estimate_price(70,"01234-000","22",2009,bundle)
    assert np.isfinite(result["prediction"]) and result["prediction"] >= 0
    assert 0 <= result["lower"] <= result["prediction"] <= result["upper"]
    with pytest.raises(ValueError):
        inference.estimate_price(0,"01234-000","22",2009,bundle)


def test_inference_accepts_numeric_text_after_validation():
    inference = module("src.inference")
    config = module("src.config")
    assert config.MODEL_PATH.exists(), "Modelo final deve acompanhar o checkout"
    numeric = inference.estimate_price(70,"01234-000","22",2009)
    text = inference.estimate_price("70","01234-000","22","2009")
    assert numeric["prediction"] == text["prediction"]


def test_metrics_reject_nonfinite_or_misaligned_predictions():
    ev = module("src.evaluation")
    with pytest.raises(ValueError):
        ev.regression_metrics([10,20],[10])
    with pytest.raises(ValueError):
        ev.regression_metrics([10],[np.inf])


def test_empty_schema_and_duplicate_ids_are_rejected():
    dp = module("src.data_processing")
    frame = dp.load_modeling_data().iloc[:2].copy()
    with pytest.raises(ValueError):
        dp.validate_dataset(frame.iloc[:0])
    with pytest.raises(ValueError):
        dp.validate_dataset(pd.concat([frame,frame]))
    with pytest.raises(ValueError):
        dp.validate_dataset(frame.drop(columns=frame.columns[0]))


def test_dataset_schema_rejects_numeric_text_with_valueerror():
    dp = module("src.data_processing")
    frame = dp.load_modeling_data().iloc[:2].copy()
    frame["Área Construída (m2)"] = frame["Área Construída (m2)"].astype(str)
    with pytest.raises(ValueError, match="numérico"):
        dp.validate_dataset(frame)
