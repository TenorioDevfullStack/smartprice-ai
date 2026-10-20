# Inferência verificada — contrato 1.0.0

Esta etapa mantém o hist_log existente. Não executa treinamento, tuning ou alteração de partições. O manifesto adicional `models/inference_manifest_v1.json` descreve a release `hist_log-2025-v1`; modelo, métricas e calibração históricos permanecem intactos.

## Carga e confiança

1. Ler o manifesto e exigir o SHA-256 aprovado em `src/config.py`.
2. Conferir o hash dos bytes de `models/metrics.json` antes de interpretá-los.
3. Validar schema, identificação do modelo, proveniência, categorias, períodos, domínio, estruturas consumidas pelo dashboard e política física.
4. Conferir os grupos, limites, identificação e SHA-256 canônico da calibração.
5. Exigir Python 3.13 e as versões exatas de numpy, pandas, scikit-learn, scipy, joblib e threadpoolctl registradas no manifesto. Patches Python 3.13 são aceitos; dependências Python são verificadas por metadados da distribuição instalada.
6. Conferir os bytes do modelo e só então desserializar o mesmo conteúdo em memória. Nenhum arquivo é reaberto entre a conferência do hash e `joblib.load`.

A origem aprovada é a release auditada que acompanha o código. Cópias byte a byte podem ser realocadas; fornecer um caminho não autoriza seu conteúdo. Não existe parâmetro público para aprovar um novo hash. O digest do manifesto fica no código, e não em um sidecar substituível junto dos arquivos.

Isso depende da integridade da distribuição do código e do ambiente instalado. Não é assinatura digital, sandbox de pickle, nem proteção contra alguém capaz de alterar o próprio código, suas dependências ou objetos Python no processo. Joblib continua restrito a artefatos confiáveis.

`load_bundle()` retorna um `InferenceBundle` que associa modelo, metadados, manifesto e fingerprint. `model, metadata = load_bundle()` e indexação `[0]`/`[1]` continuam funcionando. Os JSONs são mantidos como bytes imutáveis; acessos devolvem cópias. Tuplas arbitrárias não são aceitas como bundles de inferência. Um bundle mantido em memória representa aquela versão verificada, sem reler o disco em cada `estimate_price(..., bundle=...)`.

## Entradas e domínio

As entradas continuam sendo área construída cadastral, CEP, código de padrão IPTU e ACC IPTU. Números textuais e CEP com hífen continuam aceitos quando normalizáveis. Coleções, booleanos, valores numéricos não finitos, área não positiva, CEP malformado, padrão ausente e ACC fracionário ou fora de 1900–2025 são recusados.

| Estado | Política |
|---|---|
| ACCEPT | Entrada válida, categorias vistas e área/ACC dentro da distribuição observada de desenvolvimento |
| WARNING | CEP ou padrão inédito, extrapolação de área/ACC, ou previsão superior ao patamar de alto valor já usado no diagnóstico histórico |
| REJECT | Entrada inválida ou incompatível com o contrato físico/temporal existente; não executa previsão |

Área observada: **6–1.544 m²**, conforme `training_area_range` dos metadados. ACC observado: **1919–2025**, derivado exclusivamente dos 50.598 registros com data anterior a 22/10/2025 na projeção congelada. O hash do dataset e o critério constam no manifesto. O contrato histórico aceita ACC 1900–2025; anos anteriores a 1919 nessa faixa geram aviso, sem inventar um corte estatístico para recusa.

Não há cadastro postal oficial neste projeto. Um CEP visto nas transações de treino recebe `SEEN_IN_TRAINING`, evidência de cobertura histórica, não confirmação atual de município ou existência postal. CEP inédito recebe `UNVERIFIED`, `UNSEEN_CEP` e `GEOGRAPHY_UNVERIFIED`; oito dígitos não demonstram pertencimento a São Paulo. Não existe informação suficiente para criar uma lista de CEPs externos a recusar. A consulta pressupõe apartamento em compra e venda integral de São Paulo; as quatro entradas não permitem comprovar tipo ou natureza de transação.

## Resposta

```python
from src.inference import load_bundle, estimate_price
from src.inference_errors import InferenceError

try:
    bundle = load_bundle()
    result = estimate_price(70, "03362-000", "22", 2009, bundle)
except InferenceError as error:
    public_message = str(error)
    stable_code = error.code
    query_status = error.status  # REJECT para entrada; ERROR para indisponibilidade
```

Campos preservados: `prediction`, `lower`, `upper`, `warnings`.

Campos novos: `status`, `warning_codes`, `model_version`, `schema_version`, `algorithm`, `reference_period`, `training_period`, `calibration_id`, `band_kind`, `limitations`, `geographic_status`.

`reference_period` descreve a coorte de 02/01–31/12/2025; `training_period` explicita o ajuste até 21/10/2025. A faixa é baseada no erro absoluto histórico por preço previsto (`historical_absolute_error`). Não é intervalo de confiança nem garantia de preço. Valores, transformação e raios de calibração permanecem os anteriores.

Entradas recusadas levantam `InputValidationError`, também compatível com `ValueError`. Falhas de infraestrutura levantam `InferenceError` com mensagem pública genérica e código específico:

- `ARTIFACT_MISSING`
- `ARTIFACT_INTEGRITY_FAILED`
- `METADATA_INVALID`
- `SCHEMA_INCOMPATIBLE`
- `CALIBRATION_INCOMPATIBLE`
- `DEPENDENCY_INCOMPATIBLE`
- `INVALID_INPUT`
- `PREDICTION_FAILED`

O logger `smartprice.inference` registra código, etapa e classe da causa. Não registra entradas, valores previstos, paths, texto arbitrário da causa ou traceback com dados. O coletor e a retenção de logs dependem da configuração de operação; não foram criados serviços externos.

## Streamlit e cache

As sete páginas, gráficos, filtros e identidade visual foram preservados. O estimador mostra avisos e referência temporal; falhas do modelo recebem mensagem controlada. Os metadados das análises também são verificados. Uma falha nesses metadados mantém as quatro páginas de mercado disponíveis, enquanto estimador/performance/diagnóstico indicam indisponibilidade.

A função externa `model_bundle()` verifica manifesto, metadados, ambiente e bytes do modelo em cada acesso, antes de consultar o cache. A função cacheada recebe o fingerprint como argumento real e exige o mesmo fingerprint ao carregar. Alterações, remoções ou corrupção são bloqueadas mesmo com recurso previamente aquecido. As análises históricas podem continuar com seus metadados confiáveis se apenas o arquivo do modelo estiver ausente.

## Aprovação de uma futura release

Não atualizar hashes automaticamente ao detectar divergências. Uma divergência bloqueia o serviço e exige diagnóstico.

Uma futura versão deve preservar os artefatos antigos, registrar seus novos caminhos/hashes, justificar schema e política, verificar calibração, registrar o ambiente e comparar previsões. Criar outro manifesto, revisar seus bytes e atualizar a referência/digest aprovado no código como uma mudança revisável. Publicar código e artefatos juntos e reiniciar o processo. Alterações de transformação exigem testes de regressão, pois classes Python fazem parte do comportamento persistido.

O treino histórico permanece fora desse fluxo: executá-lo não aprova uma release. Se ele sobrescrever seus caminhos existentes no futuro, os hashes deixarão de coincidir e a inferência será bloqueada até uma aprovação explícita.

## Verificação

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
.venv-cloud/Scripts/python.exe -B -m pytest tests -q -p no:cacheprovider
.venv-cloud/Scripts/python.exe -B -m pip check
```

Os testes existentes ajustam estimadores sintéticos em memória e gravam cópias em temporários do pytest. Não retreinam nem sobrescrevem os modelos do projeto. AppTest também usa temporários. No Windows restrito, temporários/pipes podem exigir execução fora do sandbox; falhas de permissão devem ser distinguidas de regressões do código.
