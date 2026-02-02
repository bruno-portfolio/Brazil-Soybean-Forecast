# Relatorio de Analise de Drift

```
======================================================================
ANALISE DE DRIFT: TREINO vs TESTE
======================================================================

Data da analise: 2026-01-24
Drift detectado: SIM

----------------------------------------------------------------------
DRIFT POR FEATURE (ordenado por PSI)
----------------------------------------------------------------------

Feature                             PSI       KS    p-value          Status
----------------------------------------------------------------------
oni_std                           9.626    0.589     0.0000       [!] DRIFT
oni_avg                           2.739    0.355     0.0000       [!] DRIFT
produtividade_ma3                 0.990    0.415     0.0000       [!] DRIFT
produtividade_lag1                0.701    0.377     0.0000       [!] DRIFT
precip_vegetativo_mm              0.318    0.252     0.0000       [!] DRIFT
precip_total_mm                   0.267    0.207     0.0000       [!] DRIFT
hot_days_count                    0.246    0.204     0.0000       [!] DRIFT
precip_plantio_mm                 0.207    0.165     0.0000       [!] DRIFT
tmax_avg                          0.179    0.168     0.0000    [~] Moderado
tmean_avg                         0.096    0.117     0.0000    [~] Moderado
gdd_accumulated                   0.094    0.116     0.0000    [~] Moderado
tmin_avg                          0.048    0.071     0.0000    [~] Moderado
precip_enchimento_mm              0.043    0.094     0.0000    [~] Moderado
precip_cv                         0.014    0.043     0.0000    [~] Moderado
dry_spell_max                     0.005    0.038     0.0000    [~] Moderado

----------------------------------------------------------------------
FEATURES COM DRIFT DETECTADO
----------------------------------------------------------------------

  - oni_std: media mudou -33.5% (treino: 0.23, teste: 0.15)
  - oni_avg: media mudou +329.2% (treino: -0.07, teste: -0.29)
  - produtividade_ma3: media mudou +20.8% (treino: 2578.92, teste: 3114.78)
  - produtividade_lag1: media mudou +19.2% (treino: 2586.54, teste: 3083.80)
  - precip_vegetativo_mm: media mudou -17.0% (treino: 386.61, teste: 320.81)

----------------------------------------------------------------------
DEGRADACAO DO MODELO (MAE por ano)
----------------------------------------------------------------------

Ano            MAE (kg/ha)     vs Primeiro
----------------------------------------
2019                 399.9           +0.0%
2020                 459.2          +14.8%
2021                 341.8          -14.5%
2022                 679.7          +70.0%
2023                 425.8           +6.5%

----------------------------------------------------------------------
RECOMENDACOES
----------------------------------------------------------------------

  -> DRIFT SIGNIFICATIVO detectado em: oni_std, oni_avg, produtividade_ma3. Considerar retreino do modelo com dados mais recentes.

======================================================================
```
