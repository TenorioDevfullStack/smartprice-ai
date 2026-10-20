"""Falhas de inferência não removem navegação, análises ou tratamento público."""
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src import config, inference
from src.inference_errors import PredictionError


def dashboard():
    return AppTest.from_file(str(config.ROOT / "app/dashboard.py"), default_timeout=30).run()


def test_missing_model_has_controlled_message_and_keeps_seven_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing-private-model.joblib")
    app = dashboard()
    assert not app.exception
    assert len(app.sidebar.radio[0].options) == 7
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        if page == "Modelo":
            assert app.error
            assert "temporariamente indisponível" in app.error[0].value
            assert "missing-private" not in app.error[0].value


def test_warm_cache_does_not_hide_corrupted_model(monkeypatch, tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(config.MODEL_PATH.read_bytes())
    monkeypatch.setattr(config, "MODEL_PATH", path)
    app = dashboard()
    app.sidebar.radio[0].set_value("Modelo").run()
    app.button[0].click().run()
    assert any(m.label == "Preço estimado" for m in app.metric)
    path.write_bytes(b"corrupted despite cached resource")
    app.run()
    assert not app.exception
    assert app.error
    assert not any(m.label == "Preço estimado" for m in app.metric)


def test_invalid_metadata_does_not_break_market_navigation(monkeypatch, tmp_path):
    path = tmp_path / "private-metadata.json"
    path.write_bytes(b"invalid json")
    monkeypatch.setattr(config, "METRICS_PATH", path)
    app = dashboard()
    assert not app.exception
    assert len(app.metric) == 4
    for page in ["Mercado", "Preço por m²", "Distribuições", "Modelo", "Performance", "Diagnóstico"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        if page in ["Modelo", "Performance", "Diagnóstico"]:
            assert app.error


def test_prediction_failure_is_controlled_in_dashboard(monkeypatch):
    monkeypatch.setattr(inference, "estimate_price", Mock(side_effect=PredictionError()))
    app = dashboard()
    app.sidebar.radio[0].set_value("Modelo").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error


def test_dashboard_exposes_reference_and_understandable_warning():
    app = dashboard()
    app.sidebar.radio[0].set_value("Modelo").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Não foi possível confirmar" in w.value for w in app.warning)
    assert any("Referência: 2025" in c.value for c in app.caption)
