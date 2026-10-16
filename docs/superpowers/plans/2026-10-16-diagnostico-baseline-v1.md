# Diagnóstico do Baseline V1

**Objetivo:** analisar exclusivamente os quatro artefatos recuperados, sem fit, alteração de dados ou implementação de V2.

**Execução:** análise nesta sessão, conforme autorização expressa do usuário. Nenhuma integração de branch ou publicação faz parte da entrega.

**Arquivos novos:** `src/diagnosticar_baseline_v1.py` (validação e cálculos reproduzíveis) e `docs/diagnostico_baseline_v1_2026-10-16/` (relatório, resumo JSON e até dois PNGs). Não editar notebook, script de recuperação ou artefatos.

- [x] Confirmar caminhos absolutos; conferir SHA-256 dos arquivos e das seis partições (conteúdo e índice ordenado), alinhamento de X/y/datas/CSV, divisão temporal, parâmetros e reprodução exata das previsões. Interromper se houver divergência.
- [x] Calcular métricas globais e por valor real, previsão negativa e cobertura do CEP. Comparar características com denominadores explícitos; tratar ausentes separadamente se existirem.
- [x] Inspecionar categorias aprendidas, agrupamento infrequente e transformações de CEPs inéditos; decompor previsões usando apenas transform/predict do pipeline ajustado.
- [x] Gerar relatório organizado e no máximo dois gráficos; fundamentar até três recomendações condicionais para V2, sem causalidade.
- [x] Executar `python -m unittest discover -s tests -v` e `python src/diagnosticar_baseline_v1.py`; conferir hashes antes/depois, cálculos e arquivos de entrega. Não reconstruir bases, treinar ou executar células do notebook.

**Resultado:** dois testes passaram. Conferência independente por NumPy/pandas confirmou contagens, MAE, RMSE, vieses e cobertura; quatro hashes e hash do notebook preservados. Inspeção de AST confirmou ausência de chamadas de fit/fit_transform/partial_fit/set_params no diagnóstico. Dois PNGs inspecionados visualmente. Entrega em `docs/diagnostico_baseline_v1_2026-10-16/relatorio.md` e `resumo_metricas.json`.

**Ocorrência de ambiente:** a primeira tentativa de criar o diretório do relatório recebeu WinError 5; todas as validações já haviam passado e não houve gravação parcial. A execução com a permissão de escrita necessária concluiu a entrega. O teste existente de proteção chama o script de recuperação, que recusa arquivos existentes antes de qualquer fit; nenhuma recuperação foi realizada.
