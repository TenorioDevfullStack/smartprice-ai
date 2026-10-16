"""Verificação sem treinamento dos artefatos locais do Baseline V1."""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Área Construída (m2)", "CEP_8", "Padrão (IPTU)", "ACC (IPTU)"]


class BaselineArtifactsTest(unittest.TestCase):
    def test_existing_artifacts_prevent_training_and_overwrite(self):
        paths = [ROOT / "models/baseline_v1_recuperado.joblib", ROOT / "models/baseline_v1_manifesto.json",
                 ROOT / "data/processed/baseline_v1_particoes.joblib", ROOT / "data/processed/baseline_v1_previsoes_teste.csv"]
        self.assertTrue(all(p.is_file() for p in paths))
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        result = subprocess.run([sys.executable, str(ROOT / "src/recuperar_baseline_v1.py")], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)
        self.assertNotIn("Executando", result.stdout)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_round_trip_alignment_metrics_and_methodology(self):
        manifest_path = ROOT / "models/baseline_v1_manifesto.json"
        self.assertTrue(manifest_path.is_file(), "Artefatos do Baseline V1 ainda não foram recuperados")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        notebook = json.loads((ROOT / "notebooks/01_exploracao.ipynb").read_bytes())
        for i, expected_hash in manifest["executed_cell_source_sha256"].items():
            self.assertEqual(hashlib.sha256("".join(notebook["cells"][int(i)]["source"]).encode("utf-8")).hexdigest(), expected_hash)
        for rel, expected_hash in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), expected_hash)
        data = joblib.load(ROOT / "data/processed/baseline_v1_particoes.joblib")
        self.assertEqual(set(data), {"X_train", "X_test", "y_train", "y_test", "datas_train", "datas_test"})
        for part, count in [("train", 50598), ("test", 12566)]:
            x, y, dates = [data[f"{prefix}_{part}"] for prefix in ["X", "y", "datas"]]
            self.assertEqual(len(x), count)
            self.assertEqual(x.columns.tolist(), FEATURES)
            self.assertTrue(x.index.is_unique)
            self.assertTrue(x.index.equals(y.index) and x.index.equals(dates.index))
            self.assertTrue(dates.dt.year.eq(2025).all())
        self.assertTrue(data["X_train"].index.intersection(data["X_test"].index).empty)
        self.assertEqual(str(data["datas_train"].max().date()), "2025-10-21")
        self.assertEqual(str(data["datas_test"].min().date()), "2025-10-22")
        model = joblib.load(ROOT / "models/baseline_v1_recuperado.joblib")
        self.assertEqual(model.named_steps["regressao"].alpha, 1.0)
        self.assertEqual(model.named_steps["regressao"].solver, "lsqr")
        pre = model.named_steps["preprocessamento"]
        self.assertEqual(pre.named_transformers_["cep"].named_steps["one_hot"].min_frequency, 5)
        self.assertEqual(pre.named_transformers_["cep"].named_steps["one_hot"].handle_unknown, "infrequent_if_exist")
        self.assertEqual(pre.named_transformers_["padrao"].named_steps["one_hot"].handle_unknown, "ignore")
        pred = pd.read_csv(ROOT / "data/processed/baseline_v1_previsoes_teste.csv", index_col="indice_original", parse_dates=["Data"], float_precision="round_trip")
        self.assertEqual(pred.columns.tolist(), ["Data", "Valor real (R$)", "Valor previsto (R$)"])
        self.assertTrue(pred.index.equals(data["X_test"].index))
        np.testing.assert_array_equal(pred["Valor real (R$)"].to_numpy(), data["y_test"].to_numpy())
        np.testing.assert_array_equal(pred["Data"].to_numpy(), data["datas_test"].to_numpy())
        predictions = model.predict(data["X_test"])
        np.testing.assert_array_equal(predictions, pred["Valor previsto (R$)"].to_numpy())
        real = data["y_test"].to_numpy()
        metrics = {"MAE (R$)": mean_absolute_error(real, predictions), "RMSE (R$)": np.sqrt(mean_squared_error(real, predictions)), "R²": r2_score(real, predictions), "Mediana do erro absoluto (R$)": np.median(np.abs(predictions-real))}
        for key, value in metrics.items():
            self.assertEqual(value, manifest["metrics_recovered"][key])
        for key, frame in data.items():
            self.assertEqual(hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(), manifest["partition_signatures"][key]["sha256_values_and_index"])


if __name__ == "__main__":
    unittest.main()
