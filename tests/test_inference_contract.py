"""Contratos de inferência; nenhum ajuste de modelos ou alteração histórica."""
import hashlib
import json
import logging
from unittest.mock import Mock

import numpy as np
import pytest

from src import config, inference


@pytest.fixture
def bundle():
    return inference.load_bundle()


@pytest.fixture
def release(tmp_path, monkeypatch):
    from src import artifact_contract as contract

    model = tmp_path / "model.joblib"
    metadata = tmp_path / "metrics.json"
    manifest = tmp_path / "manifest.json"
    model.write_bytes(config.MODEL_PATH.read_bytes())
    metadata.write_bytes(config.METRICS_PATH.read_bytes())
    manifest.write_bytes(config.INFERENCE_MANIFEST_PATH.read_bytes())

    def approve(change_manifest=None, change_metadata=None):
        """Simula aprovação do código apenas no teste; não é uma API de produção."""
        doc = json.loads(manifest.read_bytes())
        meta = json.loads(metadata.read_bytes())
        if change_metadata:
            change_metadata(meta)
            metadata.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            doc["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        if change_manifest:
            change_manifest(doc)
        manifest.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(config, "INFERENCE_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())

    return contract, model, metadata, manifest, approve


def test_versioned_response_and_legacy_keys(bundle):
    result = inference.estimate_price(70, "03362-000", "22", 2009, bundle)
    assert {"prediction", "lower", "upper", "warnings"} <= result.keys()
    assert result["status"] == "ACCEPT"
    assert result["model_version"] == "hist_log-2025-v1"
    assert result["schema_version"] == "1.0.0"
    assert result["reference_period"] == {"start": "2025-01-02", "end": "2025-12-31"}
    assert result["training_period"]["end"] == "2025-10-21"
    assert result["calibration_id"]
    assert result["limitations"]
    assert result["band_kind"] == "historical_absolute_error"
    assert result["geographic_status"] == "SEEN_IN_TRAINING"
    assert 0 <= result["lower"] <= result["prediction"] <= result["upper"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("args,code", [
    ((0, "03362000", "22", 2009), "INVALID_INPUT"),
    ((np.inf, "03362000", "22", 2009), "INVALID_INPUT"),
    ((True, "03362000", "22", 2009), "INVALID_INPUT"),
    ((70, ["03362000"], "22", 2009), "INVALID_INPUT"),
    ((70, "bad", "22", 2009), "INVALID_INPUT"),
    ((70, "03362000", {}, 2009), "INVALID_INPUT"),
    ((70, "03362000", " ", 2009), "INVALID_INPUT"),
    ((70, "03362000", "22", 2026), "INVALID_INPUT"),
    ((70, "03362000", "22", 2009.5), "INVALID_INPUT"),
])
def test_rejected_input_is_controlled_valueerror(bundle, args, code):
    from src.inference_errors import InputValidationError

    with pytest.raises(InputValidationError) as caught:
        inference.estimate_price(*args, bundle=bundle)
    assert isinstance(caught.value, ValueError)
    assert caught.value.code == code
    assert caught.value.status == "REJECT"


@pytest.mark.parametrize("field,value", [
    (0, np.datetime64("2025-01-01")),
    (0, np.timedelta64(70, "D")),
    (0, complex(70, 1)),
    (3, complex(2009, 0)),
    (1, complex(3362000, 0)),
    (2, np.inf),
])
def test_non_real_or_temporal_attributes_are_rejected_before_prediction(bundle, monkeypatch, field, value):
    from src.inference_errors import InputValidationError

    model, _ = bundle
    predict = Mock()
    monkeypatch.setattr(model, "predict", predict)
    args = [70, "03362000", "22", 2009]
    args[field] = value
    with pytest.raises(InputValidationError):
        inference.estimate_price(*args, bundle=bundle)
    predict.assert_not_called()


@pytest.mark.parametrize("args,warning", [
    ((70, "99999-999", "22", 2009), "GEOGRAPHY_UNVERIFIED"),
    ((2000, "03362000", "22", 2009), "AREA_EXTRAPOLATION"),
    ((1, "03362000", "22", 2009), "AREA_EXTRAPOLATION"),
    ((70, "03362000", "NOVO", 2009), "UNSEEN_STANDARD"),
    ((70, "03362000", "22", 1900), "YEAR_EXTRAPOLATION"),
])
def test_warning_policy_uses_training_domain(bundle, args, warning):
    result = inference.estimate_price(*args, bundle=bundle)
    assert result["status"] == "WARNING"
    assert warning in result["warning_codes"]
    if warning == "GEOGRAPHY_UNVERIFIED":
        assert result["geographic_status"] == "UNVERIFIED"
        assert "UNSEEN_CEP" in result["warning_codes"]


def test_numeric_text_prediction_and_band_stay_identical(bundle):
    numeric = inference.estimate_price(70, "01234-000", "22", 2009, bundle)
    text = inference.estimate_price("70", "01234000", "22", "2009", bundle)
    assert [numeric[k] for k in ["prediction", "lower", "upper"]] == [text[k] for k in ["prediction", "lower", "upper"]]
    assert numeric["prediction"] == 322553.95493870456
    assert numeric["lower"] == 210398.14670868727
    assert numeric["upper"] == 434709.7631687218


@pytest.mark.parametrize("which", ["model", "metadata", "manifest"])
def test_missing_artifacts_never_deserialize(release, monkeypatch, which):
    from src.inference_errors import ArtifactMissingError

    _, model, metadata, manifest, _ = release
    {"model": model, "metadata": metadata, "manifest": manifest}[which].unlink()
    loader = Mock()
    monkeypatch.setattr(inference.joblib, "load", loader)
    with pytest.raises(ArtifactMissingError):
        inference.load_bundle(model, metadata, manifest)
    loader.assert_not_called()


@pytest.mark.parametrize("which", ["model", "metadata", "manifest"])
def test_unapproved_bytes_never_deserialize(release, monkeypatch, which):
    from src.inference_errors import ArtifactIntegrityError

    _, model, metadata, manifest, _ = release
    {"model": model, "metadata": metadata, "manifest": manifest}[which].write_bytes(b"untrusted bytes")
    loader = Mock()
    monkeypatch.setattr(inference.joblib, "load", loader)
    with pytest.raises(ArtifactIntegrityError):
        inference.load_bundle(model, metadata, manifest)
    loader.assert_not_called()


@pytest.mark.parametrize("change,expected", [
    (lambda d: d.update(schema_version="9.0.0"), "SCHEMA_INCOMPATIBLE"),
    (lambda d: d.update(required_inputs=["unexpected"]), "SCHEMA_INCOMPATIBLE"),
    (lambda d: d["calibration"].update(sha256="0" * 64), "CALIBRATION_INCOMPATIBLE"),
    (lambda d: d["calibration"].update(price_bins=[0, 1]), "CALIBRATION_INCOMPATIBLE"),
    (lambda d: d["dependencies"].update(joblib="0.0.0"), "DEPENDENCY_INCOMPATIBLE"),
    (lambda d: d.update(limitations="invalid"), "METADATA_INVALID"),
])
def test_incompatible_approved_manifest_blocks_load(release, monkeypatch, change, expected):
    from src.inference_errors import InferenceError

    _, model, metadata, manifest, approve = release
    approve(change_manifest=change)
    loader = Mock()
    monkeypatch.setattr(inference.joblib, "load", loader)
    with pytest.raises(InferenceError) as caught:
        inference.load_bundle(model, metadata, manifest)
    assert caught.value.code == expected
    loader.assert_not_called()


@pytest.mark.parametrize("change,expected", [
    (lambda d: d.pop("training_categories"), "METADATA_INVALID"),
    (lambda d: d.update(training_area_range=[1544, 6]), "METADATA_INVALID"),
    (lambda d: d.update(selected_model="ridge_v1"), "SCHEMA_INCOMPATIBLE"),
    (lambda d: d["calibration"]["groups"][0].update(radius=-1), "CALIBRATION_INCOMPATIBLE"),
    (lambda d: d["calibration"].update(statistical_confidence_interval=True), "CALIBRATION_INCOMPATIBLE"),
    (lambda d: d.update(holdout_comparison={}), "METADATA_INVALID"),
    (lambda d: d["prediction_policy"].pop("ridge_original_negative_holdout"), "METADATA_INVALID"),
    (lambda d: d["final_holdout_metrics"].update(MAE=10**500), "METADATA_INVALID"),
])
def test_invalid_approved_metadata_blocks_load(release, monkeypatch, change, expected):
    from src.inference_errors import InferenceError

    _, model, metadata, manifest, approve = release
    approve(change_metadata=change)
    loader = Mock()
    monkeypatch.setattr(inference.joblib, "load", loader)
    with pytest.raises(InferenceError) as caught:
        inference.load_bundle(model, metadata, manifest)
    assert caught.value.code == expected
    loader.assert_not_called()


def test_fingerprint_checks_files_even_after_bundle_was_loaded(release):
    from src.inference_errors import ArtifactIntegrityError

    contract, model, metadata, manifest, _ = release
    key = contract.artifact_fingerprint(model, metadata, manifest)
    inference.load_bundle(model, metadata, manifest, expected_fingerprint=key)
    model.write_bytes(b"changed after cache")
    with pytest.raises(ArtifactIntegrityError):
        contract.artifact_fingerprint(model, metadata, manifest)


def test_verified_bundle_defends_metadata_from_consumer_mutation(bundle):
    _, metadata = bundle
    metadata["calibration"]["groups"][0]["radius"] = -999
    assert inference.estimate_price(70, "03362000", "22", 2009, bundle)["lower"] >= 0


def test_unverified_tuple_cannot_bypass_loading(bundle):
    from src.inference_errors import ArtifactIntegrityError

    with pytest.raises(ArtifactIntegrityError):
        inference.estimate_price(70, "03362000", "22", 2009, tuple(bundle))


def test_prediction_failure_is_sanitized(bundle, monkeypatch, caplog):
    from src.inference_errors import PredictionError

    model, _ = bundle
    monkeypatch.setattr(model, "predict", Mock(side_effect=RuntimeError("private CEP 03362000 internal /secret/path")))
    with caplog.at_level(logging.ERROR), pytest.raises(PredictionError) as caught:
        inference.estimate_price(70, "03362000", "22", 2009, bundle)
    assert caught.value.code == "PREDICTION_FAILED"
    assert "03362000" not in str(caught.value)
    assert "/secret" not in caplog.text
    assert "03362000" not in caplog.text
    assert "PREDICTION_FAILED" in caplog.text


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -1, [1, 2]])
def test_bad_estimator_output_is_controlled(bundle, monkeypatch, prediction):
    from src.inference_errors import PredictionError

    model, _ = bundle
    value = prediction if isinstance(prediction, list) else [prediction]
    monkeypatch.setattr(model, "predict", Mock(return_value=value))
    with pytest.raises(PredictionError):
        inference.estimate_price(70, "03362000", "22", 2009, bundle)
