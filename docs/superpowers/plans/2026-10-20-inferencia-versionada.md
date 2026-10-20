# Inferência versionada — plano técnico

**Objetivo:** fortalecer a inferência do hist_log existente sem treinamento, alteração dos dados, substituição de modelos, commit ou push.

**Arquitetura:** um manifesto novo e imutável referencia os hashes dos bytes do modelo e dos metadados históricos. Seu próprio SHA-256 fica aprovado no código. A verificação antecede a desserialização, que utiliza os mesmos bytes conferidos em memória. Um bundle verificado mantém modelo, metadados e contrato associados; as chaves históricas da resposta permanecem disponíveis.

**Stack:** biblioteca padrão Python 3.13, dependências já instaladas, pytest e Streamlit AppTest. Execução incremental nesta sessão, conforme autorização do pedido. Os testes existentes ajustam apenas estimadores sintéticos temporários; nenhum treino do projeto será executado.

## Decisões

- Manifesto adicional em `models/inference_manifest_v1.json`; nenhum JSON histórico será regravado.
- `src/artifact_contract.py`: confiança, schema, calibração, ambiente, leitura verificada e bundle.
- `src/inference_errors.py`: códigos estáveis e mensagens públicas, sem caminhos ou dados da consulta.
- `src/inference.py`: ACCEPT/WARNING, rejeições por exceção compatível com ValueError, resposta versionada e isolamento de falhas de previsão.
- CEP visto no treino representa cobertura histórica, não confirmação postal atual. CEP inédito recebe WARNING e geografia não verificada. Não haverá lista geográfica inventada.
- Área fora de [6, 1544] recebe WARNING. Ano fora do contrato histórico 1900–2025 recebe REJECT; eventual extrapolação dentro desse contrato usa a distribuição do treino como referência.
- Cache Streamlit usa fingerprints verificados de manifesto, modelo, metadados e ambiente. Nenhum argumento ignorado pelo cache carregará uma versão diferente.
- Confiança depende da integridade do código distribuído. Um checksum aprovado no mesmo diretório não seria suficiente; alterações coordenadas do código também permanecem fora deste mecanismo.

## Etapas e validação

- [x] Inspecionar módulos, metadados, testes, dashboard e estado Git; registrar hashes e 24 previsões anteriores em memória.
- [x] Executar suíte existente para estabelecer baseline: 41 aprovados.
- [x] Criar testes que falhem para integridade pré-joblib, metadados/schema/calibração/ambiente, resposta, entradas e previsão.
- [x] Criar manifesto a partir dos artefatos existentes e apenas da distribuição de treino; aprovar seu digest no código.
- [x] Implementar verificação e bundle imutável; rodar `python -B -m pytest tests/test_inference_contract.py -q -p no:cacheprovider`: 50 aprovados após revisão.
- [x] Integrar resposta versionada e política de domínio, mantendo prediction/lower/upper/warnings e desempacotamento do bundle.
- [x] Acrescentar AppTests de indisponibilidade, entrada inválida, navegação e invalidação do cache; verificar falha antes de alterar o dashboard.
- [x] Integrar Streamlit preservando sete páginas, gráficos, filtros e estilo.
- [x] Documentar contrato, limites da confiança e processo manual de aprovação de futuras versões em `docs/inferencia.md`.
- [x] Executar suíte completa, `pip check`, revisão de código e comparação exata das 24 previsões anteriores: 96 testes aprovados, dependências consistentes e nenhuma diferença numérica.
- [x] Conferir hashes de todos os arquivos preservados (incluindo V2), `git diff --check` e `git status`; somente três arquivos anteriores modificados, todos no escopo.

## Resultado da revisão

Revisão independente encontrou conversão indevida de escalares temporais/complexos e lacunas de validação de metadados de futuras releases. Testes de regressão reproduziram as falhas antes das correções. A segunda revisão confirmou os ajustes e não encontrou pendências críticas ou importantes. A execução da suíte completa exigiu permissões para temporários e pipes do Windows; nenhuma dependência foi instalada e nenhum treinamento do projeto foi executado.

## Critérios de aceite

Hash, origem não aprovada, schema, calibração e dependências incompatíveis bloqueiam joblib.load. O consumidor recebe códigos controlados; logs não incluem valores das entradas nem mensagens arbitrárias da exceção do estimador. Consultas válidas mantêm previsão e faixa exatamente iguais às anteriores. A indisponibilidade é apresentada sem traceback na interface. Modelo, dados, notebooks, relatórios históricos e experimentos V2 mantêm seus hashes.
