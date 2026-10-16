# Apresentação técnica

**Demonstração de cinco minutos:** abra Visão geral para apresentar a coorte real; filtre o mercado por CEP; consulte o estimador com área construída cadastral; compare baselines e modelos na página Performance; termine em Diagnóstico mostrando os erros acima de R$ 5 milhões. Explique que a seleção antecede o holdout e que o holdout histórico já foi examinado no projeto anterior.

**Texto sugerido para LinkedIn (rascunho, não publicado):**

Concluí o SmartPrice AI, um projeto de Data Science com transações reais de ITBI de São Paulo. O pipeline reúne processamento reproduzível, dois baselines simples, Ridge, RandomForest e HistGradientBoosting, validação temporal e uma demonstração em Streamlit.

A coorte possui 63.164 transações de apartamentos em 2025. O HistGradientBoosting com target em log foi escolhido em três janelas anteriores ao holdout. No período final, apresentou MAE de aproximadamente R$ 212 mil e R² de 0,716, contra MAE de R$ 289 mil do Ridge original.

Os resultados não contam toda a história: acima de R$ 5 milhões, o MAE ainda é de R$ 3,63 milhões e há subestimação em 71,63% dos casos. Essas limitações aparecem no dashboard junto com a comparação completa e uma faixa baseada no erro histórico, sem promessas de precisão.

Código, metodologia e screenshots: https://github.com/TenorioDevfullStack/smartprice-ai

Modelo experimental; não substitui avaliação imobiliária profissional. Próximo passo: validar em uma nova coorte de transações.
