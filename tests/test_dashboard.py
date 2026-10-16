from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_navigation_filters_and_estimator():
    assert (ROOT/"app/dashboard.py").exists(), "Implementar dashboard"
    assert (ROOT/"models/metrics.json").exists(), "Métricas devem acompanhar o checkout"
    app = AppTest.from_file(str(ROOT/"app/dashboard.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric)==4
    for page in ["Mercado", "Preço por m²", "Distribuições", "Modelo", "Performance", "Diagnóstico"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        if page=="Mercado":
            app.multiselect[0].set_value(["05144085"]).run()
            assert not app.exception
            assert int(app.metric[0].value.replace(".","")) < 63164
        if page=="Modelo":
            app.button[0].click().run()
            assert not app.exception
            assert any(m.label=="Preço estimado" for m in app.metric)
            app.text_input[0].set_value("123").run()
            app.button[0].click().run()
            assert app.error and not app.exception
