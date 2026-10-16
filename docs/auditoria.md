# Auditoria e preservação

Estado inicial: `.gitignore` e `notebooks/01_exploracao.ipynb` modificados; `docs/`, `models/`, `src/` e `tests/` não rastreados. README vazio. Não havia implementação do app em `app/`, nem pasta `reports/`. requirements.txt continha um freeze extenso de ambiente Jupyter. `.venv` apontava para um Python inexistente de outro usuário; foi preservado e os testes de ambiente limpo usaram `.venv-cloud` ignorado.

Arquivos anteriores e seus hashes estão em `auditoria_inicial_2026-10-16.json`. A fonte principal foi o workspace, com HEAD inicial `20c0511`. Notebook 01, XLSX, scripts de recuperação/diagnóstico, testes V1, modelos e previsões V1, diagnósticos e planos anteriores foram preservados byte a byte. README, requirements e gitignore receberam as mudanças pedidas; os novos módulos não sobrescrevem os scripts anteriores.

Os arquivos V1 originalmente ignorados eram `models/baseline_v1_recuperado.joblib`, `data/processed/baseline_v1_particoes.joblib` e `data/processed/baseline_v1_previsoes_teste.csv`. Eles agora têm exceções específicas no gitignore: são pequenos, contêm somente o schema aprovado e permitem reproduzir a comparação e o fit sem executar notebooks. A exceção do modelo final também é específica; outros modelos grandes continuam ignorados.

O carregador valida o hash das partições. A auditoria independente `python -m src.verify_raw --report` lê as 12 planilhas, replica filtros e deduplicação documentados e compara todos os índices e valores aos artefatos recuperados; o relatório agregado fica em `reports/raw_verification.json`. A V3 tem 63.164 linhas. Foram removidas 48 cópias exatamente iguais na base original; coincidências na projeção reduzida não são excluídas.

O Ridge V1 foi reproduzido exatamente com as versões locais originais: MAE 288838,43365456635; RMSE 954108,0794786493; R² 0,6064064201089863; MedAE 124050,78926702146. O notebook original não é executado pelo treino novo. Seu script de recuperação mantém a recusa a sobrescrever artefatos anteriores.

Revisão independente: sem leakage de seleção identificado; corrigida a inferência que comparava uma área textual com limites numéricos e reforçado o schema para rejeitar tipos não numéricos com ValueError. Os resultados ficam em README, `docs/resultados.md` e no relatório final ao usuário.

O teste de persistência dos candidatos aceita diferenças numéricas abaixo de um milionésimo de real, com tolerância relativa de 1e-12, pois a soma paralela das árvores de RandomForest variou em cerca de 1e-9 real entre chamadas. O modelo final HGB e as previsões Ridge preservadas também foram conferidos por igualdade exata. Essa diferença de soma não muda seleção ou métricas na precisão reportada.
