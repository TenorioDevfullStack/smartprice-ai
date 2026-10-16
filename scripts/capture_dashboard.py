"""Inicie Streamlit na porta 8501; execute com Playwright instalado."""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def capture():
    folder=ROOT/"docs/screenshots"
    folder.mkdir(parents=True,exist_ok=True)
    failures=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,channel="chrome")
        page=browser.new_page(viewport={"width":1440,"height":1100},device_scale_factor=1)
        page.on("pageerror",lambda error:failures.append(str(error)))
        started=time.perf_counter()
        page.goto("http://127.0.0.1:8501",wait_until="domcontentloaded")
        page.get_by_role("heading",name="O mercado, pelos dados.").wait_for(timeout=60000)
        page.locator(".js-plotly-plot").first.wait_for(timeout=30000)
        page.wait_for_timeout(1200)
        startup=round(time.perf_counter()-started,2)
        page.screenshot(path=str(folder/"dashboard-overview.png"),full_page=True)
        page.get_by_text("Mercado",exact=True).click()
        page.get_by_role("heading",name="Explore suas referências.").wait_for()
        page.locator(".js-plotly-plot").first.wait_for()
        page.wait_for_timeout(800)
        page.screenshot(path=str(folder/"dashboard-market.png"),full_page=True)
        page.get_by_role("combobox",name="CEPs").fill("05144085")
        page.get_by_role("option",name="05144085",exact=True).click()
        page.wait_for_timeout(1000)
        assert page.get_by_test_id("stMetricValue").first.inner_text()=="200"
        page.get_by_text("Modelo",exact=True).click()
        page.get_by_role("heading",name="Uma referência para o imóvel.").wait_for()
        page.get_by_role("button",name="Estimar preço",exact=True).click()
        page.get_by_text("Preço estimado",exact=True).wait_for(timeout=30000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(folder/"dashboard-model.png"),full_page=True)
        page.get_by_role("textbox",name="CEP",exact=True).fill("123")
        page.get_by_role("button",name="Estimar preço",exact=True).click()
        page.get_by_text("CEP inválido: use oito dígitos ou 00000-000.",exact=True).wait_for()
        page.get_by_text("Performance",exact=True).click()
        page.get_by_role("heading",name="Performance sem atalhos.").wait_for()
        page.locator(".js-plotly-plot").first.wait_for()
        page.wait_for_timeout(1200)
        page.screenshot(path=str(folder/"dashboard-performance.png"),full_page=True)
        page.get_by_text("Diagnóstico",exact=True).click()
        page.get_by_role("heading",name="Onde o modelo erra.").wait_for()
        page.get_by_text("Visão geral",exact=True).click()
        page.get_by_role("heading",name="O mercado, pelos dados.").wait_for()
        page.set_viewport_size({"width":390,"height":844})
        page.wait_for_timeout(1200)
        mobile=page.evaluate("({width:window.innerWidth,scrollWidth:document.documentElement.scrollWidth})")
        assert mobile["scrollWidth"]<=mobile["width"]+2, mobile
        page.screenshot(path=str(folder/"dashboard-mobile.png"),full_page=True)
        assert not failures, failures
        browser.close()
    result={"browser":"Chrome / Playwright","overview_render_seconds":startup,"screenshots":4,"additional_mobile_screenshot":True,
        "desktop_viewport":[1440,1100],"mobile_viewport":[390,844],
        "tested":["startup","main pages","CEP filter: 200 rows","valid inference","invalid CEP","charts","basic mobile horizontal overflow"],"page_errors":failures}
    (ROOT/"reports/browser_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    capture()
