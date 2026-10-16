"""Recupera o V1 executando as células originais, sem diagnóstico ou tuning.

Uso, da raiz do projeto: python src/recuperar_baseline_v1.py
Recusa sobrescrever artefatos. A validação independente não treina modelos:
python -m unittest discover -s tests -v
"""
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/01_exploracao.ipynb"
# Somente dependências da preparação e treinamento/avaliação originais.
# Células 97 e 98 (diagnóstico de erros e gráficos) não são executadas.
CELLS = [0, 1, 2, 10, 11, 15, 31, 32, 33, 34, 36, 51, 52, 72, 73,
         77, 80, 86, 88, 90, 92, 95, 96]
PATHS = {
    "pipeline": ROOT / "models/baseline_v1_recuperado.joblib",
    "partitions": ROOT / "data/processed/baseline_v1_particoes.joblib",
    "predictions": ROOT / "data/processed/baseline_v1_previsoes_teste.csv",
    "manifest": ROOT / "models/baseline_v1_manifesto.json",
}
PREVIOUS = {"MAE (R$)": 288838.43, "RMSE (R$)": 954108.08, "R²": .6064,
            "Mediana do erro absoluto (R$)": 124050.79,
            "Erro percentual absoluto mediano (%)": 25.15}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(frame):
    return {
        "rows": len(frame),
        "columns": frame.columns.tolist() if isinstance(frame, pd.DataFrame) else [frame.name],
        "sha256_values_and_index": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(),
        "sha256_ordered_index": hashlib.sha256(frame.index.to_numpy(dtype="int64").tobytes()).hexdigest(),
    }


def recover():
    existing = [str(p.relative_to(ROOT)) for p in PATHS.values() if p.exists()]
    if existing:
        raise FileExistsError(f"Recuperação interrompida sem treinar/sobrescrever: {existing}. Valide os artefatos existentes com unittest.")
    notebook_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    requirements_bytes = (ROOT / "requirements.txt").read_bytes()
    requirements_encoding = "utf-16" if requirements_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    requirements_lines = requirements_bytes.decode(requirements_encoding).splitlines()
    sources = {i: "".join(notebook["cells"][i]["source"]) for i in CELLS}
    assert "Ridge(alpha=1.0, solver=\"lsqr\")" in sources[95]
    assert "baseline_v1.fit(X_train, y_train)" in sources[95]
    assert "data_corte_treino = (acumulado_treino - alvo_treino).abs().idxmin()" in sources[92]
    scope = {"__name__": "__baseline_recovery__", "display": lambda *args, **kwargs: None}
    os.environ.setdefault("MPLBACKEND", "Agg")
    previous_cwd = Path.cwd()
    signatures_before_fit = None
    try:
        os.chdir(NOTEBOOK.parent)
        for i in CELLS:
            print(f"Executando célula original {i}...", flush=True)
            if i == 95:
                signatures_before_fit = {name: signature(scope[name]) for name in ["df_v1", "df_v2", "df_v3", "df_modelagem", "X_train", "X_test", "y_train", "y_test"]}
            with redirect_stdout(io.StringIO()):
                exec(compile(sources[i], f"{NOTEBOOK.name}:cell_{i}", "exec"), scope)
    finally:
        os.chdir(previous_cwd)

    assert signatures_before_fit is not None
    for name, before in signatures_before_fit.items():
        assert signature(scope[name]) == before, f"Conteúdo/índices de {name} alterados pelo fit"
    x_train, x_test, y_train, y_test = [scope[k] for k in ["X_train", "X_test", "y_train", "y_test"]]
    dates = scope["datas_modelagem"]
    # Conferência diretamente nos índices da V3 e nas datas, sem reset_index.
    assert x_train.index.equals(scope["df_v3"].index[dates.le(pd.Timestamp("2025-10-21"))])
    assert x_test.index.equals(scope["df_v3"].index[dates.gt(pd.Timestamp("2025-10-21"))])
    assert (len(x_train), len(x_test)) == (50598, 12566)
    assert str(dates.loc[x_train.index].min().date()) == "2025-01-02"
    assert str(dates.loc[x_test.index].max().date()) == "2025-12-31"
    raw_path = Path(scope["arquivo_itbi"])
    if not raw_path.is_absolute():
        raw_path = (NOTEBOOK.parent / raw_path).resolve()
    partitions = {"X_train": x_train, "X_test": x_test, "y_train": y_train, "y_test": y_test,
                  "datas_train": dates.loc[x_train.index].copy(), "datas_test": dates.loc[x_test.index].copy()}
    predictions = scope["avaliacao_baseline_v1"][["Data", "Valor real (R$)", "Valor previsto (R$)"]].copy()
    assert predictions.index.equals(x_test.index) and x_test.index.equals(y_test.index)
    np.testing.assert_array_equal(predictions["Valor real (R$)"].to_numpy(), y_test.to_numpy())
    metrics = {k: float(v) for k, v in scope["metricas_baseline_v1"].items()}
    comparison = {k: {"previous_reported": v, "recovered": metrics[k], "difference": metrics[k]-v,
                       "same_at_reported_precision": round(metrics[k], 4 if k == "R²" else 2) == v}
                  for k, v in PREVIOUS.items()}

    # Exclusão por schema: não exporta bases brutas, SQL ou campos de endereço.
    for p in PATHS.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    with PATHS["pipeline"].open("xb") as stream:
        joblib.dump(scope["baseline_v1"], stream, compress=3)
    with PATHS["partitions"].open("xb") as stream:
        joblib.dump(partitions, stream, compress=3)
    with PATHS["predictions"].open("x", encoding="utf-8", newline="") as stream:
        predictions.to_csv(stream, index_label="indice_original", float_format="%.17g")

    loaded = joblib.load(PATHS["pipeline"])
    loaded_data = joblib.load(PATHS["partitions"])
    loaded_predictions = pd.read_csv(PATHS["predictions"], index_col="indice_original", parse_dates=["Data"], float_precision="round_trip")
    for key, original in partitions.items():
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(original, loaded_data[key])
        else:
            pd.testing.assert_series_equal(original, loaded_data[key])
    np.testing.assert_array_equal(loaded.predict(loaded_data["X_test"]), scope["previsoes_baseline_v1"])
    np.testing.assert_array_equal(loaded_predictions["Valor previsto (R$)"], scope["previsoes_baseline_v1"])
    np.testing.assert_array_equal(loaded_predictions["Valor real (R$)"], y_test.to_numpy())
    assert loaded_predictions.index.equals(x_test.index)
    assert NOTEBOOK.read_bytes() == notebook_bytes
    assert all(signature(scope[name]) == sig for name, sig in signatures_before_fit.items())
    versions = {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "openpyxl", "matplotlib"]}
    parameters = scope["baseline_v1"].get_params(deep=True)
    parameters = {k: v if v is None or isinstance(v, (str, int, bool)) or isinstance(v, float) and np.isfinite(v) else repr(v) for k, v in parameters.items()}
    manifest = {
        "created_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "model": "Baseline V1 — Ridge original, uma reexecução autorizada",
        "executed_original_cell_indices": CELLS,
        "source_sha256": {"notebook_before_recovery_record": hashlib.sha256(notebook_bytes).hexdigest(), "raw_workbook": sha256(raw_path)},
        "executed_cell_source_sha256": {str(i): hashlib.sha256(s.encode("utf-8")).hexdigest() for i, s in sources.items()},
        "raw_workbook": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "sheet_order": scope["planilhas_2025"],
        "features": scope["FEATURES_APROVADAS"],
        "target": scope["col_target"],
        "split": {"train_rows": len(x_train), "test_rows": len(x_test), "train_start": str(dates.loc[x_train.index].min().date()), "train_end": "2025-10-21", "test_start": "2025-10-22", "test_end": "2025-12-31", "rule": "Acumulado por data inteira mais próximo de round(0.80*N); preservada a ordem/índice original"},
        "historical_membership_validation": "Reconstruída pelo código e fonte originais; contagens e datas conferidas com saídas históricas. Não havia listas/hashes históricos das partições para comparação independente registro a registro.",
        "signatures_before_and_after_fit": signatures_before_fit,
        "partition_signatures": {k: signature(v) for k, v in partitions.items()},
        "metrics_recovered": metrics,
        "comparison": comparison,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "libraries": versions, "notebook_reported_python": notebook["metadata"]["language_info"]["version"], "previous_requirements": requirements_lines, "requirements_encoding": requirements_encoding, "original_venv_unavailable": True},
        "pipeline_parameters": parameters,
        "verification": {"load_test": "OK", "predictions_bitwise_equal": True, "max_absolute_prediction_difference": 0.0, "partition_dtypes_and_indices_preserved": True, "base_signatures_preserved": True},
        "artifact_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for k, p in PATHS.items() if k != "manifest"},
        "exported_data_policy": "Somente quatro features aprovadas, target, data e índice técnico; sem SQL, endereço, complemento ou financiamento.",
        "deep_error_diagnosis_executed": False,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    with PATHS["manifest"].open("x", encoding="utf-8") as stream:
        stream.write(manifest_json + "\n")
    print(json.dumps({"metrics": metrics, "comparison": comparison, "versions": versions, "load_test": "OK"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    recover()
