# SmartPrice AI — plano de finalização

**Objetivo:** concluir o roteiro de 30 fases fornecido pelo usuário no workspace existente.
**Arquitetura:** congelar as partições V3 verificadas em CSV gzip com schema mínimo; manter o Ridge original e seus hashes. Implementar contratos em src, comparação temporal em três janelas anteriores a 22/10/2025, seleção por MAE ponderado, calibração observacional por previsões fora da amostra e avaliação final. O dashboard lê somente artefatos compactos, sem notebooks ou XLSX.
**Stack:** Python 3.13, pandas, numpy, scikit-learn, joblib, Streamlit, Plotly, pytest.

O roteiro do usuário autoriza implementação, revisão, commits e push ao final. Trabalhar no checkout local para preservar as alterações não commitadas, sem criar outro projeto. Execução sequencial com testes e revisão independente antes de publicar.

- [x] Auditoria: registrar status, inventário e SHA-256; validar Ridge e dados persistidos; manter notebook 01 e scripts originais intactos.
- [x] Contratos: testes primeiro para schema, carregamento, CEP, entradas inválidas, exclusão de proxies monetárias e splits sem sobreposição. Criar src/config.py, data_processing.py e feature_engineering.py.
- [x] Modelos: testes de fit/predict/persistência/categorias inéditas; criar modeling.py e evaluation.py. Comparar mediana, mediana preço/m² de treino por CEP, Ridge original/log, RandomForest original/log e HistGradientBoosting original/log. Árvores usam prefixo de CEP de três dígitos e log de área; documentar a diferença de representação.
- [x] Treino: criar src/train.py; executar três janelas com cortes em 31/05, 31/07 e 31/08; validar junho-julho, agosto e setembro–21/outubro. Seleção por MAE ponderado somente nessas janelas; registrar estabilidade, cauda e previsões negativas. Reajustar escolhido até 21/10; abrir holdout de 22/10–31/12 após seleção. O holdout já foi examinado no diagnóstico histórico: informar essa limitação.
- [x] Artefatos: CSV gzip mínimo completo, diagnóstico agregado e amostra determinística, modelo comprimido e métricas JSON. Faixa de erro absoluto p80 por faixa do preço previsto em validação, sem calibrar no holdout. Política explícita de não negatividade na inferência.
- [x] Aplicação: criar app/helpers.py e app/dashboard.py com navegação, filtros, EDA, formulário das quatro features, performance e diagnóstico. Testar com Streamlit AppTest e navegador desktop/mobile; gerar quatro screenshots reais.
- [x] Documentação: README com resultados calculados, notebook 02 usando src, guia de deploy Python 3.13, configuração Streamlit, dependências pinadas e relatório final.
- [x] Verificação: python -m pytest -q; python -m src.train; ambiente limpo e servidor Streamlit; verificar hashes preservados, git diff --check e tamanhos. Revisão independente antes de commits seletivos; git push origin main sem force e registrar resultado.

Comandos locais usam `git -c safe.directory=U:/smartprice-ai` devido à divergência de proprietário do volume. Nenhuma configuração global de Git será alterada.
