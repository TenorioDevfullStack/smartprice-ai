"""Erros públicos estáveis; diagnóstico técnico sem conteúdo das consultas."""
import logging

LOGGER = logging.getLogger("smartprice.inference")


class InferenceError(Exception):
    code = "INFERENCE_UNAVAILABLE"
    status = "ERROR"
    message = "Estimativa temporariamente indisponível. Tente novamente mais tarde."

    def __init__(self):
        super().__init__(self.message)


class ArtifactMissingError(InferenceError, FileNotFoundError):
    code = "ARTIFACT_MISSING"


class ArtifactIntegrityError(InferenceError):
    code = "ARTIFACT_INTEGRITY_FAILED"


class MetadataError(InferenceError):
    code = "METADATA_INVALID"


class SchemaCompatibilityError(InferenceError):
    code = "SCHEMA_INCOMPATIBLE"


class CalibrationCompatibilityError(InferenceError):
    code = "CALIBRATION_INCOMPATIBLE"


class DependencyCompatibilityError(InferenceError):
    code = "DEPENDENCY_INCOMPATIBLE"


class InputValidationError(InferenceError, ValueError):
    code = "INVALID_INPUT"
    status = "REJECT"
    message = "Entrada inválida. Confira área positiva, CEP, padrão IPTU e ACC inteiro entre 1900 e 2025."


class PredictionError(InferenceError):
    code = "PREDICTION_FAILED"


def fail(error_type, stage, cause=None):
    """Não registra valores, paths, mensagem arbitrária ou traceback da causa."""
    error = error_type()
    level = logging.WARNING if error.status == "REJECT" else logging.ERROR
    LOGGER.log(level, "%s stage=%s exception_type=%s", error.code, stage,
               type(cause).__name__ if cause is not None else "none")
    raise error from None
