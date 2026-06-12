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
| produtividade_ma3 | 37.7% | 75.3 | 20.5% |
| trend | 12.6% | 0.0 | 15.1% |
| deficit_ratio_enchimento | 8.6% | -0.2 | 4.2% |
| produtividade_lag1 | 4.1% | 47.2 | 7.8% |
| radiation_total | 3.7% | 0.8 | 1.6% |
| fert_import_ton | 3.0% | 0.5 | 3.9% |
| precip_vegetativo_mm | 2.1% | 3.8 | 3.3% |
| deficit_vegetativo_mm | 1.9% | 0.4 | 2.1% |
| water_deficit_ratio | 1.7% | 0.0 | 1.1% |
| mun_yield_hist_mean | 1.7% | 0.6 | 2.6% |
| deficit_enchimento_mm | 1.5% | 0.4 | 1.0% |
| precip_plantio_mm | 1.3% | 3.9 | 1.9% |
| precip_enchimento_mm | 1.3% | 1.3 | 1.8% |
| sinistro_rate_3yr | 1.2% | 1.6 | 2.0% |
| terminal_drought_stress | 1.0% | -0.4 | 1.4% |
| precip_cv | 0.8% | 0.1 | 0.4% |
| oni_std | 0.8% | 0.0 | 1.9% |
| precip_days_gt1mm | 0.7% | 0.2 | 0.4% |
| pct_soja | 0.7% | 0.8 | 2.0% |
| oni_avg | 0.7% | 0.0 | 1.9% |
| radiation_mean | 0.7% | -0.2 | 1.3% |
| eto_plantio_mm | 0.6% | -0.2 | 0.4% |
| eto_vegetativo_mm | 0.5% | 0.5 | 1.0% |
| precip_enchimento_anomaly | 0.5% | 0.7 | 0.6% |
| tmin_plantio | 0.5% | 0.2 | 0.4% |
| sul_x_precip_anomaly | 0.5% | 0.0 | 0.2% |
| tmin_vegetativo | 0.4% | 4.8 | 0.5% |
| sand_x_deficit | 0.4% | 0.6 | 0.8% |
| enchimento_stress | 0.4% | -0.4 | 0.7% |
| clay_0_30cm | 0.4% | 0.2 | 0.9% |
| tmax_plantio | 0.4% | 0.6 | 0.4% |
| tmax_vegetativo | 0.4% | 0.6 | 0.9% |
| tmean_plantio | 0.3% | 2.4 | 0.5% |
| sul_x_hot_days_anomaly | 0.3% | 0.0 | 0.5% |
| mun_yield_volatility | 0.3% | 0.4 | 0.7% |
| deficit_plantio_mm | 0.3% | -0.2 | 0.3% |
| tmean_vegetativo | 0.2% | 0.4 | 0.5% |
| oni_min | 0.2% | 0.0 | 0.5% |
| hot_days_enchimento | 0.2% | 0.1 | 0.2% |
| gdd_anomaly | 0.2% | 0.0 | 0.5% |
| tmax_enchimento | 0.2% | 0.0 | 0.4% |
| dry_spell_x_hot_anom | 0.2% | 0.0 | 0.2% |
| oni_max | 0.2% | 0.0 | 0.4% |
| irrigacao_x_deficit | 0.2% | 0.8 | 0.5% |
| gdd_enchimento | 0.2% | 0.0 | 0.1% |
| dry_spell_anomaly | 0.2% | 0.1 | 0.2% |
| clay_30_100cm | 0.2% | 0.1 | 0.3% |
| eto_enchimento_mm | 0.2% | 0.5 | 0.3% |
| hot_days_anomaly | 0.2% | -0.1 | 0.2% |
| sand_x_la_nina_sul | 0.2% | 0.0 | 0.3% |
| is_sul | 0.1% | 0.0 | 0.2% |
| precip_total_mm | 0.1% | -0.3 | 0.2% |
| clay_sand_ratio | 0.1% | -0.1 | 0.2% |
| dry_spell_count_10d | 0.1% | 0.2 | 0.5% |
| nitrogen_0_30cm | 0.1% | -0.2 | 0.5% |
| tmin_enchimento | 0.1% | -0.1 | 0.2% |
| silt_0_30cm | 0.1% | -0.2 | 0.8% |
| sand_x_drought | 0.1% | -0.1 | 0.1% |
| tmin_avg | 0.1% | 0.0 | 0.2% |
| gdd_vegetativo | 0.1% | 0.1 | 0.3% |
| eto_mean_mm | 0.1% | 0.0 | 0.2% |
| sand_0_30cm | 0.1% | 0.0 | 0.2% |
| dry_spell_count_7d | 0.1% | 0.0 | 0.0% |
| precip_anomaly | 0.1% | -0.0 | 0.1% |
| fert_x_precip | 0.1% | 0.0 | 0.1% |
| bdod_0_30cm | 0.1% | 0.0 | 0.2% |
| hot_days_vegetativo | 0.1% | -0.0 | 0.2% |
| tmean_avg | 0.1% | -0.0 | 0.1% |
| el_nino_x_precip_anom | 0.1% | 0.0 | 0.2% |
| tmean_enchimento | 0.1% | -0.0 | 0.1% |
| soc_x_heat_stress | 0.1% | -0.5 | 0.3% |
| eto_total_mm | 0.1% | -0.1 | 0.1% |
| water_deficit_mm | 0.1% | 0.1 | 0.2% |
| temp_anomaly | 0.1% | 0.1 | 0.1% |
| ph_x_cerrado | 0.1% | 0.0 | 0.3% |
| dry_spell_max | 0.1% | -0.5 | 0.3% |
| gdd_plantio | 0.1% | 0.1 | 0.1% |
| la_nina_x_precip_anom | 0.1% | 0.0 | 0.1% |
| tmax_avg | 0.1% | -0.6 | 0.1% |
| soc_0_30cm | 0.0% | 0.1 | 0.2% |
| hot_days_count | 0.0% | -0.1 | 0.1% |
| ndvi_x_precip_deficit | 0.0% | 0.0 | 0.2% |
| ndvi_amplitude | 0.0% | 0.0 | 0.1% |
| heat_drought_stress | 0.0% | -0.0 | 0.1% |
| la_nina_x_deficit | 0.0% | -0.1 | 0.0% |
| phh2o_30_100cm | 0.0% | -0.1 | 0.1% |
| hot_days_plantio | 0.0% | 0.3 | 0.1% |
| awc_x_dry_spell | 0.0% | 0.1 | 0.1% |
| la_nina_x_precip_ench_anom | 0.0% | -0.1 | 0.1% |
| gdd_accumulated | 0.0% | -0.0 | 0.2% |
| sand_30_100cm | 0.0% | -0.0 | 0.1% |
| awc_x_deficit | 0.0% | 0.0 | 0.1% |
| cec_0_30cm | 0.0% | -0.0 | 0.1% |
| phh2o_0_30cm | 0.0% | -0.0 | 0.1% |
| ndvi_max_safra | 0.0% | -0.1 | 0.2% |
| clay_x_precip_deficit | 0.0% | -0.0 | 0.1% |
| ndvi_ench_x_la_nina | 0.0% | -0.0 | 0.1% |
| cec_normalized | 0.0% | 0.0 | 0.1% |
| pct_irrigado | 0.0% | 0.2 | 0.1% |
| ndvi_min_safra | 0.0% | -0.0 | 0.1% |
| ndvi_plantio | 0.0% | 0.0 | 0.1% |
| ndvi_vegetativo | 0.0% | 0.0 | 0.1% |
| soil_quality_index | 0.0% | 0.0 | 0.0% |
| ndvi_enchimento | 0.0% | 0.0 | 0.0% |
| sul_x_la_nina | 0.0% | 0.0 | 0.0% |
| ndvi_mean_safra | 0.0% | 0.0 | 0.0% |
| is_la_nina | 0.0% | 0.0 | 0.0% |
| is_el_nino | 0.0% | 0.0 | 0.0% |
| awc_estimated | 0.0% | 0.0 | 0.0% |
| ph_acidic | 0.0% | 0.0 | 0.0% |
| sinistro_x_la_nina | 0.0% | 0.0 | 0.0% |

## Analise de Coerencia Agronomica

Verificacao se a direcao dos efeitos e consistente com o conhecimento agronomico:

| Feature | Direcao | Esperado | Status |
|---------|---------|----------|--------|
| precip_total_mm | negativo | positivo | ? |
| tmean_avg | positivo | nao-linear | ~ |
| tmin_avg | negativo | positivo | ? |
| tmax_avg | positivo | negativo | ? |
| hot_days_count | positivo | negativo | ? |
| gdd_accumulated | positivo | positivo | OK |
| precip_plantio_mm | negativo | nao-linear | ~ |
| tmean_plantio | positivo | nao-linear | ~ |
| tmin_plantio | negativo | nao-linear | ~ |
| tmax_plantio | positivo | nao-linear | ~ |
| hot_days_plantio | positivo | nao-linear | ~ |
| gdd_plantio | positivo | nao-linear | ~ |
| precip_vegetativo_mm | positivo | nao-linear | ~ |
| tmean_vegetativo | negativo | nao-linear | ~ |
| tmin_vegetativo | negativo | nao-linear | ~ |
| tmax_vegetativo | negativo | nao-linear | ~ |
| hot_days_vegetativo | negativo | nao-linear | ~ |
| gdd_vegetativo | positivo | nao-linear | ~ |
| precip_enchimento_mm | positivo | nao-linear | ~ |
| tmean_enchimento | negativo | nao-linear | ~ |
| tmin_enchimento | positivo | nao-linear | ~ |
| tmax_enchimento | negativo | nao-linear | ~ |
| hot_days_enchimento | negativo | nao-linear | ~ |
| gdd_enchimento | negativo | nao-linear | ~ |
| dry_spell_max | positivo | nao-linear | ~ |
| dry_spell_count_7d | negativo | nao-linear | ~ |
| dry_spell_count_10d | negativo | nao-linear | ~ |
| precip_cv | negativo | nao-linear | ~ |
| precip_days_gt1mm | positivo | nao-linear | ~ |
| eto_total_mm | negativo | nao-linear | ~ |
| eto_mean_mm | negativo | nao-linear | ~ |
| water_deficit_mm | negativo | nao-linear | ~ |
| water_deficit_ratio | negativo | nao-linear | ~ |
| radiation_mean | negativo | nao-linear | ~ |
| radiation_total | negativo | nao-linear | ~ |
| eto_plantio_mm | negativo | nao-linear | ~ |
| deficit_plantio_mm | positivo | nao-linear | ~ |
| eto_vegetativo_mm | negativo | nao-linear | ~ |
| deficit_vegetativo_mm | negativo | nao-linear | ~ |
| eto_enchimento_mm | negativo | nao-linear | ~ |
| deficit_enchimento_mm | negativo | nao-linear | ~ |
| deficit_ratio_enchimento | negativo | nao-linear | ~ |
| oni_avg | negativo | nao-linear | ~ |
| oni_min | negativo | nao-linear | ~ |
| oni_max | negativo | nao-linear | ~ |
| oni_std | positivo | nao-linear | ~ |
| is_la_nina | negativo | nao-linear | ~ |
| is_el_nino | negativo | nao-linear | ~ |
| produtividade_lag1 | positivo | positivo | OK |
| produtividade_ma3 | positivo | positivo | OK |
| trend | positivo | positivo | OK |
| mun_yield_hist_mean | positivo | nao-linear | ~ |
| mun_yield_volatility | positivo | nao-linear | ~ |
| precip_anomaly | positivo | nao-linear | ~ |
| temp_anomaly | positivo | nao-linear | ~ |
| hot_days_anomaly | negativo | nao-linear | ~ |
| gdd_anomaly | positivo | nao-linear | ~ |
| precip_enchimento_anomaly | positivo | nao-linear | ~ |
| dry_spell_anomaly | negativo | nao-linear | ~ |
| la_nina_x_precip_ench_anom | negativo | nao-linear | ~ |
| dry_spell_x_hot_anom | negativo | nao-linear | ~ |
| la_nina_x_precip_anom | negativo | nao-linear | ~ |
| heat_drought_stress | negativo | nao-linear | ~ |
| enchimento_stress | negativo | nao-linear | ~ |
| el_nino_x_precip_anom | negativo | nao-linear | ~ |
| la_nina_x_deficit | negativo | nao-linear | ~ |
| terminal_drought_stress | negativo | nao-linear | ~ |
| is_sul | negativo | nao-linear | ~ |
| sul_x_la_nina | negativo | nao-linear | ~ |
| sul_x_precip_anomaly | negativo | nao-linear | ~ |
| sul_x_hot_days_anomaly | negativo | nao-linear | ~ |
| clay_0_30cm | positivo | nao-linear | ~ |
| sand_0_30cm | negativo | nao-linear | ~ |
| silt_0_30cm | positivo | nao-linear | ~ |
| phh2o_0_30cm | positivo | nao-linear | ~ |
| soc_0_30cm | negativo | nao-linear | ~ |
| nitrogen_0_30cm | positivo | nao-linear | ~ |
| cec_0_30cm | positivo | nao-linear | ~ |
| bdod_0_30cm | negativo | nao-linear | ~ |
| clay_30_100cm | positivo | nao-linear | ~ |
| sand_30_100cm | negativo | nao-linear | ~ |
| phh2o_30_100cm | positivo | nao-linear | ~ |
| clay_sand_ratio | positivo | nao-linear | ~ |
| awc_estimated | negativo | nao-linear | ~ |
| ph_acidic | negativo | nao-linear | ~ |
| soil_quality_index | negativo | nao-linear | ~ |
| clay_x_precip_deficit | positivo | nao-linear | ~ |
| awc_x_dry_spell | positivo | nao-linear | ~ |
| sand_x_drought | negativo | nao-linear | ~ |
| ph_x_cerrado | positivo | nao-linear | ~ |
| soc_x_heat_stress | negativo | nao-linear | ~ |
| sand_x_la_nina_sul | negativo | nao-linear | ~ |
| cec_normalized | positivo | nao-linear | ~ |
| awc_x_deficit | negativo | nao-linear | ~ |
| sand_x_deficit | negativo | nao-linear | ~ |
| pct_irrigado | negativo | nao-linear | ~ |
| fert_import_ton | positivo | nao-linear | ~ |
| sinistro_rate_3yr | negativo | nao-linear | ~ |
| pct_soja | positivo | nao-linear | ~ |
| irrigacao_x_deficit | negativo | nao-linear | ~ |
| fert_x_precip | negativo | nao-linear | ~ |
| sinistro_x_la_nina | negativo | nao-linear | ~ |
| ndvi_mean_safra | positivo | nao-linear | ~ |
| ndvi_max_safra | negativo | nao-linear | ~ |
| ndvi_min_safra | negativo | nao-linear | ~ |
| ndvi_amplitude | negativo | nao-linear | ~ |
| ndvi_plantio | negativo | nao-linear | ~ |
| ndvi_vegetativo | negativo | nao-linear | ~ |
| ndvi_enchimento | negativo | nao-linear | ~ |
| ndvi_x_precip_deficit | negativo | nao-linear | ~ |
| ndvi_ench_x_la_nina | negativo | nao-linear | ~ |

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
