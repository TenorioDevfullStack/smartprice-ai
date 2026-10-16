# Streamlit Community Cloud

Use o repositório `TenorioDevfullStack/smartprice-ai`, branch `main`, arquivo `app/dashboard.py`. Escolha **Python 3.13** em Advanced settings. `.python-version` documenta o runtime local; a seleção de versão na interface da Cloud continua necessária. Não existem secrets ou variáveis de ambiente obrigatórias.

Referências oficiais: [deploy](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependências](https://docs.streamlit.io/deploy/concepts/dependencies).

O app lê `data/processed/smartprice_demo.csv.gz`, `models/metrics.json`, `models/smartprice_model.joblib` e relatórios em `reports/`. Todos devem estar no Git. O dashboard não lê o XLSX original, não treina e não executa notebooks no startup. Joblib importa as classes de `src/`; mantenha a estrutura e as versões pinadas. Não carregue modelos joblib de origem não confiável.

Teste de ambiente limpo:

```bash
python -m venv .venv-cloud
# Ative o ambiente conforme seu sistema
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
python -m src.train
python -m streamlit run app/dashboard.py --server.port 8501
```

Os arquivos anteriores de partições e pipeline Ridge, agora explicitamente incluídos no Git, permitem retreinar sem XLSX. A auditoria opcional `python -m src.verify_raw` usa o XLSX preservado. A reprodução da exploração original e gráficos do diagnóstico requer `requirements-dev.txt`.

A validação local usa Windows/Python 3.13, ambiente criado somente com requirements.txt, AppTest e navegador Chrome desktop/mobile. Isso verifica imports, paths, carga e fluxos básicos; não representa uma publicação executada na infraestrutura Linux da Cloud. Após publicar, conferir logs, tempo/memória de startup e os mesmos fluxos no endereço público.

Portfólio: quatro screenshots reais estão em `docs/screenshots/`; existe uma captura adicional mobile. O script de captura requer Playwright e Chrome. O arquivo `reports/browser_verification.json` registra o teste local, sem alegar deploy público.
