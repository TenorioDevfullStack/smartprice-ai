"""Valida a release antes do joblib. O digest aprovado no código ancora a confiança."""
from dataclasses import dataclass
from datetime import date
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import platform
import re

from src import config
from src.inference_errors import (
    ArtifactMissingError, ArtifactIntegrityError, MetadataError,
    SchemaCompatibilityError, CalibrationCompatibilityError,
    DependencyCompatibilityError, fail,
)

SCHEMA_VERSION = "1.0.0"
DEPENDENCIES = {"numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"}


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read(path):
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as exc:
        fail(ArtifactMissingError, "read_artifact", exc)
    except (OSError, TypeError, ValueError) as exc:
        fail(ArtifactIntegrityError, "read_artifact", exc)


def _json(raw):
    def reject_constant(value):
        raise ValueError("JSON não finito")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Chave JSON duplicada")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _period(value, count=False):
    if not isinstance(value, dict):
        return False
    try:
        start, end = date.fromisoformat(value["start"]), date.fromisoformat(value["end"])
        return start <= end and (not count or type(value["n"]) is int and value["n"] > 0)
    except (KeyError, ValueError, TypeError):
        return False


def _range(value, positive=False, integer=False):
    return (isinstance(value, list) and len(value) == 2
            and all(_finite(v) for v in value) and value[0] <= value[1]
            and (not positive or value[0] > 0)
            and (not integer or all(float(v).is_integer() for v in value)))


def _validate_contract(manifest, metadata):
    """Valida os campos utilizados pela inferência e pelo dashboard."""
    try:
        if (manifest["manifest_version"] != "1.0.0" or manifest["schema_version"] != SCHEMA_VERSION
                or manifest["required_inputs"] != config.FEATURES or metadata["features"] != config.FEATURES
                or manifest["selected_model"] != "hist_log" or metadata["selected_model"] != "hist_log"
                or manifest["algorithm"] != "HistGradientBoostingRegressor"
                or manifest["target_transform"] != "log1p/expm1"
                or metadata["target"] != config.TARGET or manifest["target"] != config.TARGET):
            fail(SchemaCompatibilityError, "input_schema")
        if (not _text(manifest["model_version"]) or not isinstance(manifest["limitations"], list)
                or not manifest["limitations"] or not all(map(_text, manifest["limitations"]))):
            raise ValueError("Identificação inválida")
        if not _period(manifest["reference_period"]) or not _period(manifest["training_period"], count=True):
            raise ValueError("Períodos inválidos")
        dataset, training = metadata["dataset"], metadata["split"]["development"]
        if (manifest["reference_period"] != {k: dataset[k] for k in ["start", "end"]}
                or manifest["training_period"] != training
                or manifest["dataset_sha256"] != dataset["sha256"]
                or metadata["model_sha256"] != manifest["model_sha256"]):
            raise ValueError("Proveniência incompatível")
        categories = metadata["training_categories"]
        for name in [config.CEP, config.STANDARD]:
            values = categories[name]
            if not isinstance(values, list) or not values or not all(map(_text, values)) or len(set(values)) != len(values):
                raise ValueError("Categorias inválidas")
        if not all(re.fullmatch(r"[0-9]{8}", cep) for cep in categories[config.CEP]):
            raise ValueError("Vocabulário postal inválido")
        domain = manifest["domain"]
        if (not _range(metadata["training_area_range"], positive=True)
                or metadata["training_area_range"] != domain["area_range"]
                or not _range(domain["year_range"], integer=True)
                or domain["valid_year_range"] != [1900, 2025]
                or not 1900 <= domain["year_range"][0] <= domain["year_range"][1] <= 2025
                or domain["geography_verification"] != "training_membership_only"
                or domain["city"] != "São Paulo/SP"
                or domain["property_type"] != "apartamento"
                or domain["transaction"] != "compra e venda integral"):
            raise ValueError("Domínio inválido")
        if metadata["prediction_policy"]["nonnegative_clip"] is not True or metadata["prediction_policy"]["minimum"] != 0:
            raise ValueError("Política física incompatível")
        for key in ["ridge_original_negative_holdout", "selected_raw_negative_holdout", "selected_clipped_holdout"]:
            count = metadata["prediction_policy"][key]
            if type(count) is not int or count < 0:
                raise ValueError("Contagem de política inválida")
        for name in ["MAE", "RMSE", "R2", "MedAE", "MdAPE_pct", "negative_predictions", "n"]:
            if not _finite(metadata["final_holdout_metrics"][name]):
                raise ValueError("Métricas inválidas")
        comparison = metadata["holdout_comparison"]
        if not isinstance(comparison, dict) or "hist_log" not in comparison:
            raise ValueError("Comparação ausente")
        for name, values in comparison.items():
            if not _text(name) or not all(_finite(values[k]) for k in ["MAE", "RMSE", "R2", "MedAE", "MdAPE_pct", "negative_predictions"]):
                raise ValueError("Comparação inválida")
        if len(metadata["final_holdout_segments"]) != len(config.PRICE_LABELS):
            raise ValueError("Segmentos inválidos")
        for row in metadata["final_holdout_segments"]:
            if not all(_finite(row[k]) for k in ["n", "MAE", "underestimate_pct"]):
                raise ValueError("Segmento inválido")
        for value in metadata["unseen_categories"].values():
            if not all(type(value[k]) is int and value[k] >= 0 for k in ["unique", "rows"]):
                raise ValueError("Cobertura inválida")
        if not _text(metadata["selection"]["caveat"]) or not metadata["selection"]["ranking"]:
            raise ValueError("Seleção inválida")
        for row in metadata["selection"]["ranking"]:
            if (not _text(row["model"]) or not _finite(row["fold_mae_std"])
                    or not all(_finite(row["metrics"][k]) for k in ["MAE", "RMSE"])):
                raise ValueError("Ranking inválido")
        if {row["model"] for row in metadata["selection"]["ranking"]} != set(comparison):
            raise ValueError("Ranking e comparação desalinhados")
        coverage = metadata["observational_band_holdout_coverage_pct"]
        if not _finite(coverage) or not 0 <= coverage <= 100:
            raise ValueError("Cobertura inválida")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        fail(MetadataError, "metadata_schema", exc)


def _validate_calibration(manifest, metadata):
    try:
        contract, calibration = manifest["calibration"], metadata["calibration"]
        bins = [None if math.isinf(v) else v for v in config.PRICE_BINS]
        if (not _text(contract["id"]) or contract["price_bins"] != bins
                or contract["price_labels"] != config.PRICE_LABELS
                or contract["quantile"] != 0.8 or calibration["quantile"] != contract["quantile"]
                or contract["band_kind"] != "historical_absolute_error"
                or contract["statistical_confidence_interval"] is not False
                or calibration["statistical_confidence_interval"] is not False
                or not _text(calibration["label"]) or not _text(calibration["source"])
                or not _finite(calibration["global_radius"]) or calibration["global_radius"] < 0
                or len(calibration["groups"]) != len(config.PRICE_LABELS)):
            raise ValueError("Calibração incompatível")
        for label, group in zip(config.PRICE_LABELS, calibration["groups"]):
            if (group["segment"] != label or not _finite(group["radius"]) or group["radius"] < 0
                    or type(group["n"]) is not int or group["n"] < 0
                    or type(group["fallback_global"]) is not bool
                    or group["fallback_global"] != (group["n"] < 50)
                    or group["fallback_global"] and group["radius"] != calibration["global_radius"]):
                raise ValueError("Grupo incompatível")
        if canonical_sha256(calibration) != contract["sha256"]:
            raise ValueError("Digest de calibração divergente")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        fail(CalibrationCompatibilityError, "calibration", exc)


def _validate_environment(manifest, metadata):
    try:
        expected = manifest["dependencies"]
        if (manifest["python"] != ".".join(platform.python_version_tuple()[:2])
                or set(expected) != DEPENDENCIES
                or not metadata["environment"]["python"].startswith(manifest["python"] + ".")):
            raise ValueError("Runtime incompatível")
        for name, requirement in expected.items():
            if not _text(requirement) or version(name) != requirement:
                raise ValueError("Dependência incompatível")
            if name != "threadpoolctl" and metadata["environment"]["libraries"][name] != requirement:
                raise ValueError("Ambiente de treino incompatível")
    except (KeyError, TypeError, ValueError, PackageNotFoundError, AttributeError) as exc:
        fail(DependencyCompatibilityError, "dependencies", exc)


@dataclass(frozen=True)
class VerifiedArtifacts:
    model_bytes: bytes | None
    metadata_bytes: bytes
    manifest_bytes: bytes
    fingerprint: str


def verify_artifacts(model_path=None, metrics_path=None, manifest_path=None, *, include_model=True):
    """Retorna bytes aprovados, sem interpretar pickle/joblib."""
    model_path = config.MODEL_PATH if model_path is None else model_path
    metrics_path = config.METRICS_PATH if metrics_path is None else metrics_path
    manifest_path = config.INFERENCE_MANIFEST_PATH if manifest_path is None else manifest_path
    manifest_bytes = _read(manifest_path)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != config.INFERENCE_MANIFEST_SHA256:
        fail(ArtifactIntegrityError, "manifest_trust")
    try:
        manifest = _json(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("Manifesto inválido")
        for key in ["model_sha256", "metadata_sha256", "dataset_sha256"]:
            if not isinstance(manifest[key], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
                raise ValueError("Digest inválido")
    except (KeyError, TypeError, ValueError) as exc:
        fail(MetadataError, "manifest_json", exc)
    metadata_bytes = _read(metrics_path)
    if hashlib.sha256(metadata_bytes).hexdigest() != manifest["metadata_sha256"]:
        fail(ArtifactIntegrityError, "metadata_digest")
    try:
        metadata = _json(metadata_bytes)
        if not isinstance(metadata, dict):
            raise ValueError("Metadados inválidos")
    except (TypeError, ValueError) as exc:
        fail(MetadataError, "metadata_json", exc)
    _validate_contract(manifest, metadata)
    _validate_calibration(manifest, metadata)
    _validate_environment(manifest, metadata)
    model_bytes = _read(model_path) if include_model else None
    if include_model and hashlib.sha256(model_bytes).hexdigest() != manifest["model_sha256"]:
        fail(ArtifactIntegrityError, "model_digest")
    fingerprint = canonical_sha256({"manifest": manifest_hash, "model": manifest["model_sha256"],
                                    "metadata": manifest["metadata_sha256"],
                                    "dependencies": manifest["dependencies"], "python": manifest["python"]})
    return VerifiedArtifacts(model_bytes, metadata_bytes, manifest_bytes, fingerprint)


def artifact_fingerprint(model_path=None, metrics_path=None, manifest_path=None):
    return verify_artifacts(model_path, metrics_path, manifest_path).fingerprint


def load_metadata(metrics_path=None, manifest_path=None):
    """Análises usam metadados confiáveis mesmo se o modelo estiver ausente."""
    checked = verify_artifacts(metrics_path=metrics_path, manifest_path=manifest_path, include_model=False)
    return _json(checked.metadata_bytes)


@dataclass(frozen=True)
class InferenceBundle:
    model: object
    _metadata_bytes: bytes
    _manifest_bytes: bytes
    fingerprint: str

    @property
    def metadata(self):
        return _json(self._metadata_bytes)

    @property
    def manifest(self):
        return _json(self._manifest_bytes)

    def __iter__(self):
        return iter((self.model, self.metadata))

    def __getitem__(self, index):
        return (self.model, self.metadata)[index]

    def __len__(self):
        return 2
