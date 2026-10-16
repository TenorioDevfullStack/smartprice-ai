# Recuperação do Baseline V1

**Objetivo:** executar somente a preparação e o Ridge originais, persistir os dados mínimos e verificar a reprodução sem iniciar o diagnóstico.

**Arquivos:** `src/recuperar_baseline_v1.py`, `tests/test_baseline_v1_artifacts.py`, artefatos em `models/` e `data/processed/`, registro adicional no notebook.

- [x] Criar verificação dos artefatos, executar antes da recuperação e confirmar falha por ausência dos arquivos.
- [x] Executar as células originais necessárias em processo separado; preservar filtros, ordenação, tipos, índices e parâmetros. Não executar células 97–98 de diagnóstico/gráficos.
- [x] Conferir contagens, corte, identidade por índices e assinaturas das bases. Documentar a ausência de hashes históricos das partições.
- [x] Salvar pipeline, partições mínimas, previsões alinhadas e manifesto com métricas, diferenças, parâmetros, versões e hashes, sem sobrescrever arquivos válidos.
- [x] Carregar em outro processo, conferir identidade de índices/dados e reprodução das previsões, executar testes e registrar resultado no notebook sem alterar células anteriores.

Validação: `python -m unittest discover -s tests -v`. Execução: `python src/recuperar_baseline_v1.py`. Arquivos gerados devem conter apenas features aprovadas, índice técnico, data, target e previsão; nenhum SQL, endereço, nome, complemento ou financiamento.
