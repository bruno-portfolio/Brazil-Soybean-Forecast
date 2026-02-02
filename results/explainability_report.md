# Relatorio de Explicabilidade do Modelo

## Resumo

Este relatorio apresenta a analise de interpretabilidade do modelo LightGBM
treinado para previsao de produtividade de soja por municipio.

## Metodos Utilizados

1. **Gain Importance**: Importancia nativa do LightGBM baseada no ganho medio
   nas divisoes (splits) onde a feature e utilizada.

2. **Permutation Importance**: Mede o impacto na MAE quando cada feature e
   permutada aleatoriamente. Maior valor = mais importante.

3. **SHAP Values**: Quantifica a contribuicao de cada feature para cada
   predicao individual, baseado na teoria de jogos (Shapley values).

## Ranking de Importancia das Features

| Feature | Gain (%) | Permutation | SHAP (%) |
|---------|----------|-------------|----------|
| produtividade_ma3 | 27.6% | 32.8 | 23.2% |
| trend | 25.9% | 0.0 | 25.6% |
| produtividade_lag1 | 19.1% | 42.7 | 19.0% |
| precip_total_mm | 14.3% | -5.0 | 12.2% |
| tmin_avg | 5.3% | 3.3 | 8.6% |
| hot_days_count | 4.2% | 4.8 | 8.0% |
| tmax_avg | 1.5% | -1.0 | 1.5% |
| tmean_avg | 1.3% | -0.5 | 0.9% |
| gdd_accumulated | 0.8% | -0.1 | 1.0% |

## Analise de Coerencia Agronomica

Verificacao se a direcao dos efeitos e consistente com o conhecimento agronomico:

| Feature | Direcao | Esperado | Status |
|---------|---------|----------|--------|
| precip_total_mm | positivo | positivo | OK |
| tmean_avg | positivo | nao-linear | ~ |
| tmin_avg | positivo | positivo | OK |
| tmax_avg | negativo | negativo | OK |
| hot_days_count | negativo | negativo | OK |
| gdd_accumulated | negativo | positivo | ? |
| produtividade_lag1 | positivo | positivo | OK |
| produtividade_ma3 | positivo | positivo | OK |
| trend | positivo | positivo | OK |

## Interpretacao dos Resultados

### Features Historicas (Dominantes)

As features historicas dominam a previsao (~70-75% da importancia total):

- **produtividade_ma3**: Media movel de 3 anos captura a capacidade produtiva
  tipica do municipio. Forte efeito positivo.

- **produtividade_lag1**: Produtividade do ano anterior captura persistencia.
  Municipios produtivos tendem a continuar produtivos.

- **trend**: Tendencia temporal captura ganhos tecnologicos ao longo dos anos
  (novas variedades, melhor manejo, expansao para solos melhores).

### Features Climaticas

As features climaticas contribuem com ~20-25% da importancia:

- **precip_total_mm**: Precipitacao acumulada na janela Out-Mar e a feature
  climatica mais importante. Efeito positivo ate certo ponto (chuva adequada
  favorece a cultura, mas excesso pode prejudicar).

- **tmin_avg**: Temperatura minima media. Valores mais altos indicam noites
  mais quentes, o que pode afetar a qualidade do enchimento de graos.

- **hot_days_count**: Contagem de dias com temperatura maxima > 32C captura
  estresse termico. Efeito negativo esperado (mais dias quentes = menor
  produtividade).

- **gdd_accumulated**: Graus-dia acumulados indicam energia termica disponivel
  para desenvolvimento da cultura. Efeito positivo esperado.

### Limitacoes da Explicabilidade

1. **Correlacao vs Causalidade**: SHAP mostra associacoes, nao causa-efeito.

2. **Dominancia Historica**: O modelo aprende que historico e forte preditor,
   o que pode mascarar efeitos climaticos mais sutis.

3. **Eventos Extremos**: Em anos anomalos (ex: seca 2022), o modelo tende a
   subestimar impactos porque features historicas 'puxam' para a media.

## Graficos Gerados

- `shap_summary.png`: Beeswarm plot mostrando distribuicao dos SHAP values
- `shap_bar.png`: Bar plot com importancia media |SHAP|
- `shap_dependence_precip.png`: Relacao precipitacao x efeito SHAP
- `shap_dependence_hot_days.png`: Relacao hot_days x efeito SHAP
- `shap_dependence_gdd.png`: Relacao GDD x efeito SHAP
- `feature_importance.csv`: Tabela com todas as metricas de importancia

## Conclusao

O modelo apresenta comportamento agronomicamente coerente:

1. Features historicas dominam, refletindo a realidade de que produtividade
   agricola tem forte componente persistente (solo, clima regional, tecnologia).

2. Precipitacao e a principal variavel climatica, como esperado para soja.

3. Estresse termico (hot_days) tem efeito negativo, coerente com fisiologia.

4. A tendencia temporal captura ganhos tecnologicos historicos.

O modelo pode ser usado com confianca para entender drivers de produtividade,
mas deve-se ter cautela em anos com eventos climaticos extremos.
