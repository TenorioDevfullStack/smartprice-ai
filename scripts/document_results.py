"""Atualiza documentação e notebook 02 com os resultados persistidos."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def generate():
    m=json.loads((ROOT/"models/metrics.json").read_text(encoding="utf-8"))
    b=m["holdout_comparison"]["ridge_v1"]
    f=m["final_holdout_metrics"]
    tail=m["final_holdout_segments"][-1]
    def br(value,decimals=2):
        return f"{value:,.{decimals}f}".replace(",","_").replace(".",",").replace("_",".")
    table="| Modelo | MAE (R$) | RMSE (R$) | R² | MedAE (R$) | Negativas |\n|---|---:|---:|---:|---:|---:|\n"
    for name,metrics in m["holdout_comparison"].items():
        table+=f"| {name} | {br(metrics['MAE'])} | {br(metrics['RMSE'])} | {br(metrics['R2'],4)} | {br(metrics['MedAE'])} | {metrics['negative_predictions']} |\n"
    validation="| Modelo | MAE validação ponderado (R$) | Desvio entre janelas (R$) |\n|---|---:|---:|\n"
    for row in m["selection"]["ranking"]:
        validation+=f"| {row['model']} | {br(row['metrics']['MAE'])} | {br(row['fold_mae_std'])} |\n"
    segments="| Faixa real | n | MAE (R$) | MedAE (R$) | Erro médio (R$) | Subestimação | Superestimação |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in m["final_holdout_segments"]:
        segments+=f"| {row['segment']} | {row['n']} | {br(row['MAE'])} | {br(row['MedAE'])} | {br(row['mean_error'])} | {br(row['underestimate_pct'])}% | {br(row['overestimate_pct'])}% |\n"
    windows="| Janela | Treino expansivo | n treino | Validação | n validação |\n|---|---|---:|---|---:|\n"
    for i,row in enumerate(m["split"]["windows"],1):
        t,v=row["train"],row["validation"]
        windows+=f"| {i} | {t['start']} a {t['end']} | {t['n']} | {v['start']} a {v['end']} | {v['n']} |\n"
    readme=f'''# SmartPrice AI

Análise de transações de apartamentos e estimativa experimental de preços em São Paulo com dados públicos de ITBI, validação temporal e dashboard Streamlit.

[Código](https://github.com/TenorioDevfullStack/smartprice-ai) · [Notebook de modelagem](notebooks/02_modelagem.ipynb) · [Métricas completas](models/metrics.json) · [Guia de deploy](docs/deploy.md)

![Visão geral do dashboard](docs/screenshots/dashboard-overview.png)

## Visão geral

O projeto parte de declarações reais, preserva a exploração anterior e entrega um fluxo reproduzível: dados → processamento → features → baselines → candidatos → validação temporal → seleção → holdout → pipeline persistido → aplicação.

## Problema

Como criar uma referência de preço com atributos cadastrais limitados e avaliar seus erros sem misturar passado e futuro? O objetivo é demonstrar uma metodologia auditável, incluindo falhas em imóveis de alto valor e categorias inéditas.

## Dataset

Fonte: [Dados das Transações Imobiliárias com recolhimento de ITBI, Prefeitura de São Paulo](https://prefeitura.sp.gov.br/web/fazenda/w/acesso_a_informacao/31501). Arquivo local original: `GUIAS DE ITBI PAGAS (28012026) XLS.xlsx`, preservado. A publicação mensal acompanha o recolhimento do imposto e pode conter transações de outros períodos; a coorte usa a **data da transação**.

O conjunto V3 possui **{m['dataset']['rows']:,} registros** entre **02/01/2025 e 31/12/2025**, de compras e vendas integrais de apartamentos, com valor declarado ≥ R$ 10 mil e área positiva. Sem valores ausentes nas quatro features, preço e data. Há 7.533 CEPs distintos.

O arquivo `data/processed/smartprice_demo.csv.gz` contém a coorte completa com quatro features, preço, data e índice técnico. Mede {((ROOT/'data/processed/smartprice_demo.csv.gz').stat().st_size/1024):.0f} KiB; o dashboard não lê o XLSX de 46,7 MB. Não exportamos endereço, número, complemento, SQL ou informações de financiamento nos novos artefatos.

**Área construída cadastral não é necessariamente área útil.** O valor declarado para ITBI também pode divergir de um valor de mercado. O CSV público é uma projeção das declarações, sem enriquecimento com dados pessoais.

## Pipeline de dados

1. Consolidar as 12 planilhas de 2025 na ordem original.
2. Filtrar uso de apartamento, compra e venda, proporção transmitida de 100%, área e preço positivos.
3. Manter preço ≥ R$ 10 mil e data da transação em 2025.
4. Remover cópias exatamente iguais em todas as colunas V2, incluindo `Mes_Arquivo` (48 cópias).
5. Normalizar CEP para texto com oito dígitos; validar schema, finitude e ACC cadastral.
6. Congelar as mesmas observações e índices das partições recuperadas; verificar SHA-256 e round-trip.

Existem 750 coincidências na projeção reduzida. Elas **não foram removidas**, pois não comprovam transações duplicadas. O critério histórico não deduplica operações semelhantes entre planilhas mensais. `python -m src.verify_raw` reconstrói a V3 sem executar o notebook e compara todos os índices, atributos, preços e datas às partições recuperadas.

## EDA

O dashboard explora preço, área construída, preço/m², CEPs mais frequentes e medianas mensais. A mediana de preço é R$ 448 mil, a área mediana é 102 m² e a mediana do preço/m² é aproximadamente R$ 4.901. Não são medidas de valorização de um imóvel constante: composição das transações varia por mês.

## Feature engineering

Entradas brutas: `Área Construída (m2)`, `CEP_8`, `Padrão (IPTU)` e `ACC (IPTU)` (ano cadastral de construção).

- Ridge usa as quatro features V1: escala numérica e CEP completo em one-hot com categorias raras agrupadas (`min_frequency=5`, `infrequent_if_exist`); padrão aceita categorias novas com `ignore`.
- Árvores recebem área, `log1p(área)`, ACC, prefixo postal de três dígitos e padrão IPTU. RF usa one-hot; HGB usa categorias nativas via ordinal encoder com código -1 para desconhecidos. Não atribuímos nomes de bairros aos prefixos.
- Data define splits e gráficos; não é entrada da estimativa. Não acrescentamos dormitórios, vagas, bairro ou região inexistentes na projeção aprovada.
- Preço/m² e variáveis monetárias fiscais ou de financiamento são excluídos dos modelos ML. A comparação de log mantém a mesma representação dentro de cada família; diferenças entre famílias incluem suas representações.

## Baselines

**Ridge V1 preservado:** alpha=1, solver=lsqr. As previsões persistidas foram reproduzidas exatamente e seus artefatos anteriores mantidos por hash.

Baselines adicionais: mediana global do preço e mediana de preço/m² por CEP × área, calculadas somente no treino. CEPs com menos de cinco exemplos ou inéditos usam a mediana global de preço/m² do treino. A baseline por CEP é metodologicamente separada do uso de preço/m² como input ML.

## Machine Learning

Oito candidatos: mediana global, baseline por CEP, Ridge original/log, RandomForest original/log e HistGradientBoosting original/log. RF: 100 árvores, profundidade máxima 16, folha mínima 5. HGB: 180 iterações, learning rate 0,08, 31 folhas, folha mínima 30, early stopping desativado para evitar validação aleatória interna. Seeds e parâmetros ficam no código.

`log1p(y)` é invertido com `expm1` antes das métricas. Não há correção lognormal de viés, e não se promete que target em log sempre melhore. O Ridge log mostrou instabilidade entre janelas. Nenhuma biblioteca de boosting externa foi adicionada.

## Validação temporal

{windows}
Seleção por MAE ponderado nas 25.151 previsões fora da amostra dessas três janelas; estabilidade e diagnósticos são reportados. O vencedor foi **{m['selected_model']}**. A vantagem sobre o Ridge aparece nas três janelas; a diferença para as melhores árvores é pequena, sem afirmação de superioridade estatística.

{validation}
Após a seleção registrada, o escolhido foi reajustado em **50.598 registros, 02/01–21/10/2025**. O holdout contém **12.566 registros, 22/10–31/12/2025**. Não usamos split aleatório como avaliação principal. A publicação dos resultados de todos os candidatos no holdout é descritiva; a escolha não muda com eles.

**Limitação histórica:** o holdout do Ridge já havia sido examinado no diagnóstico anterior. A etapa atual não usa esses resultados na seleção, mas não representa um teste prospectivo completamente cego.

## Resultados

Métricas calculadas em reais, no holdout completo. Candidatos abaixo sem clipping:

{table}
O modelo final possui MAE **R$ {br(f['MAE'])}**, RMSE **R$ {br(f['RMSE'])}**, R² **{br(f['R2'],4)}**, MedAE **R$ {br(f['MedAE'])}** e MdAPE **{br(f['MdAPE_pct'])}%**. A redução descritiva de MAE frente ao Ridge V1 é {br((1-f['MAE']/b['MAE'])*100)}% neste holdout; não é garantia de desempenho futuro.

## Diagnóstico

{segments}
Na cauda acima de R$ 5 milhões: **{tail['n']} casos**, MAE **R$ {br(tail['MAE'])}**, erro médio **R$ {br(tail['mean_error'])}** e **{br(tail['underestimate_pct'])}% de subestimação**. Ridge V1: {br(m['holdout_segments']['ridge_v1'][-1]['underestimate_pct'])}% de subestimação nessa cauda. A fragilidade persiste.

Ridge original: **326 previsões negativas**. Escolhido antes/depois da política física: **0/0**. O pipeline final aplica limite explícito ≥ 0; os diagnósticos dos candidatos mantêm suas previsões sem esse limite. Há **413 CEPs inéditos em 532 registros** e nenhum padrão IPTU inédito no holdout.

Faixa demonstrativa: percentil 80 do erro absoluto em validação, agrupado pelo preço **previsto**; grupos com menos de 50 casos usam erro global. Rotulada “faixa baseada no erro histórico do modelo”. Não é intervalo de confiança nem garantia. Cobertura observada no holdout: **{br(m['observational_band_holdout_coverage_pct'])}%**. Seleção e calibração compartilham validação, o que pode produzir otimismo.

## Limitações

- Uma cidade, apartamentos e compras e vendas integrais; sem validação para casas, terrenos ou outras cidades.
- Coorte de 2025; não há inflação, atualização de mercado ou validação em 2026.
- Valor declarado, área cadastral, amostragem administrativa e possível repetição de operações entre meses.
- Sem dormitórios, vagas, conservação, atributos de condomínio, coordenadas ou comparação de imóveis idênticos.
- Amostra pequena e erros grandes na cauda superior; relações descritivas não são causais.
- O agrupamento postal das árvores perde parte da resolução do CEP completo.
- Validação temporal não impede que imóveis/empreendimentos semelhantes apareçam em janelas diferentes; não há split por imóvel identificado.

**Modelo experimental baseado em transações de ITBI de São Paulo. Não substitui avaliação imobiliária profissional.**

## Dashboard

Sete páginas: Visão geral, Mercado, Preço por m², Distribuições, Modelo, Performance e Diagnóstico. Filtros de período, preço, área e CEP; formulário das quatro features; mensagens para entradas inválidas, CEPs inéditos e extrapolação. Métricas completas; gráficos de resíduos e real × previsto usam amostra determinística de até 2.500 casos.

![Explorador de mercado](docs/screenshots/dashboard-market.png)
![Estimador](docs/screenshots/dashboard-model.png)
![Performance](docs/screenshots/dashboard-performance.png)

## Como executar

Python **3.13**, na raiz do repositório:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
python -m streamlit run app/dashboard.py
```

Os artefatos compactos já estão incluídos, então o dashboard pode iniciar sem retreinar ou executar notebook. O treino reproduz as comparações a partir da V3 congelada; não altera o Ridge V1. Num checkout existente com `.venv` antigo, crie outro ambiente em vez de sobrescrevê-lo.

```bash
# Verificar a fonte bruta e os artefatos anteriores (opcional)
python -m src.verify_raw
python src/diagnosticar_baseline_v1.py --inspect
# Exploração, notebook e screenshots (opcional)
python -m pip install -r requirements-dev.txt
python scripts/document_results.py
python scripts/execute_modeling_notebook.py
python scripts/capture_dashboard.py
```

A captura espera Streamlit em `127.0.0.1:8501` e Chrome instalado. Dependências de notebook/gráficos históricos ficam em requirements-dev, não no startup do app. Arquivos joblib são carregados apenas de artefatos confiáveis deste repositório.

## Deploy

Preparado para Streamlit Community Cloud: branch `main`, arquivo **`app/dashboard.py`**, Python **3.13** escolhido em Advanced settings e dependências pinadas em `requirements.txt`. [Instruções oficiais](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Sem `.env`, segredos, API externa ou caminhos Windows no runtime. Modelo final ~277 KiB; CSV ~797 KiB. [Checklist e limites de verificação](docs/deploy.md). Preparação e testes locais não significam que uma aplicação pública já foi publicada.

## Estrutura do projeto

```text
app/                 dashboard Streamlit e helpers
src/                 contratos, features, modelos, avaliação, treino, inferência
data/raw/            XLSX original preservado
data/processed/      V3 compacta e partições/previsões V1 preservadas
models/              Ridge V1, modelo final e métricas
reports/             qualidade, seleção e diagnóstico agregado
notebooks/           01 original + 02 modelagem
tests/               contratos, inferência, persistência e AppTest
docs/screenshots/    capturas reais desktop e mobile
scripts/             documentação e verificação visual
.streamlit/          tema e configuração
```

## Próximos passos

Validação prospectiva em nova coorte; deduplicação baseada em identidade de imóvel com cuidado de privacidade; atributos físicos adicionais legítimos; calibração separada da seleção; comparar diferentes níveis de agrupamento geográfico e estabilidade por segmento. Só adotar mudanças depois de validá-las temporalmente.

## Autor

[TenorioDevfullStack](https://github.com/TenorioDevfullStack). Projeto de portfólio em Data Science e Machine Learning. A autoria é identificada pelo perfil do repositório; não inferimos nome civil ou credenciais.
'''
    (ROOT/"README.md").write_text(readme,encoding="utf-8")
    def markdown(text):
        return {"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)}
    def code(text):
        return {"cell_type":"code","metadata":{},"source":text.splitlines(keepends=True),"execution_count":None,"outputs":[]}
    cells=[markdown("# SmartPrice AI — modelagem temporal\n\nNotebook documental. A lógica reutilizável vive em `src/`. O notebook 01 e artefatos V1 permanecem preservados. Execute `python -m src.train` para reproduzir a comparação completa; este notebook lê e confere os resultados salvos."),
        code("from pathlib import Path\nimport sys, json\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nimport numpy as np\nfrom src.config import *\nfrom src.data_processing import load_modeling_data, validate_dataset, temporal_windows\nfrom src.evaluation import regression_metrics\nfrom src.inference import load_bundle, estimate_price\nmetadata = json.loads(METRICS_PATH.read_text(encoding='utf-8'))\ndata = load_modeling_data()\nvalidate_dataset(data)"),
        markdown("## Dataset congelado e EDA\n63.164 registros, 02/01–31/12/2025. Área construída cadastral; preço declarado, não necessariamente preço de mercado. Apenas quatro atributos cadastrais; sem proxies monetárias no modelo."),
        code("data[[AREA, TARGET]].describe(percentiles=[.01,.5,.95,.99])"),
        code("data.assign(preco_m2=data[TARGET]/data[AREA]).groupby(data[DATE].dt.to_period('M')).agg(n=(TARGET,'size'),preco_mediano=(TARGET,'median'),m2_mediano=('preco_m2','median'))"),
        markdown("## Features e split\nRidge preserva features e encoder V1; árvores acrescentam log de área e agrupamento postal de três dígitos. Data somente para split/EDA. CEP inédito aceito. Mesma representação em original/log dentro de cada família.\n\n"+windows),
        code("assert set(FEATURES).isdisjoint({TARGET, 'Preco_m2', DATE})\nfor train, valid in temporal_windows(data):\n    assert train[DATE].max() < valid[DATE].min()\n    assert valid[DATE].max() < pd.Timestamp(HOLDOUT_START)\nmetadata['split']"),
        markdown("## Baselines, candidatos e validação\nMediana global, mediana de preço/m² por CEP treinada sem holdout, Ridge V1 e candidatos RF/HGB original/log. Escolha por MAE ponderado; estatísticas em reais após expm1.\n\n"+validation),
        code("pd.DataFrame([{'model':r['model'], **r['metrics'], 'fold_mae_std':r['fold_mae_std']} for r in metadata['selection']['ranking']])"),
        markdown("## Seleção e holdout\nVencedor: "+m['selected_model']+". Reajuste até 21/10/2025; holdout 22/10–31/12/2025. O holdout já havia sido examinado no diagnóstico histórico: não é teste prospectivo totalmente cego."),
        code("holdout = data.loc[data[DATE].ge(HOLDOUT_START)]\nmodel, saved = load_bundle()\npredicted = model.predict(holdout[FEATURES])\ncomputed = regression_metrics(holdout[TARGET], predicted)\nassert computed == saved['final_holdout_metrics']\ncomputed"),
        code("pd.DataFrame(metadata['holdout_comparison']).T"),
        markdown("## Cauda superior e categorias novas\nA melhoria média não elimina erros na cauda. Não esconder negativos do Ridge.\n\n"+segments),
        code("metadata['prediction_policy'], metadata['unseen_categories']"),
        markdown("## Inferência e incerteza\nFaixa baseada no erro histórico de validação, por preço previsto, com fallback para grupos pequenos. Não é intervalo de confiança ou garantia. Modelo experimental de ITBI de São Paulo; não substitui avaliação profissional."),
        code("estimate_price(70, '01234-000', '22', 2009)"),
        markdown("## Limitações e próximos passos\nUma coorte, área cadastral, valor declarado, poucas features, perda geográfica no prefixo postal e calibração compartilhando seleção. Validar prospectivamente antes de extrapolar para outras cidades/períodos. Leia README e reports/selection.json para metodologia e parâmetros completos.")]
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}},"nbformat":4,"nbformat_minor":5}
    for i,cell in enumerate(cells): cell["id"]=f"smartprice-{i:02}"
    (ROOT/"notebooks/02_modelagem.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    (ROOT/"docs/resultados.md").write_text("# Resultados verificados\n\nFonte programática: models/metrics.json e reports/selection.json.\n\n"+windows+"\n"+validation+"\n"+table+"\n"+segments+"\nFaixa observacional: cobertura no holdout "+br(m['observational_band_holdout_coverage_pct'])+"%. Limitações e reprodução no README.\n",encoding="utf-8")
    print("README, notebook 02 e docs/resultados.md atualizados dos artefatos.")


if __name__=="__main__":
    generate()
