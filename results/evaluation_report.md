# Relatorio de Avaliacao do Modelo

Data: 2026-01-25 12:01

## 1. Resumo Executivo

**SUCESSO**: O modelo LightGBM superou o baseline_ma3 em 5.0% no conjunto de teste.

- MAE Teste: **532.2 kg/ha** (8.87 sacas/ha)
- MAPE Teste: **36.1%**
- Melhor iteracao: 47

## 2. Comparacao com Baselines

| Modelo | Split | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | vs Baseline |
|--------|-------|-------------|-------------|----------|-------------|
| baseline_lag1 | Validacao | 509.0 | 8.48 | 19.9 | - |
| baseline_lag1 | Teste | 678.8 | 11.31 | 42.0 | - |
| baseline_ma3 | Validacao | 421.2 | 7.02 | 17.2 | - |
| baseline_ma3 | Teste | 560.1 | 9.34 | 36.1 | - |
| **LightGBM** | Validacao | **396.2** | **6.60** | **16.4** | +5.9% |
| **LightGBM** | Teste | **532.2** | **8.87** | **36.1** | +5.0% |

## 3. Analise de Erro por UF (Top 10 piores)

| UF | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |
|-----|-------------|-------------|----------|------|
| MS | 886.0 | 14.77 | 73.5 | 151 |
| RS | 880.0 | 14.67 | 88.1 | 818 |
| PR | 831.7 | 13.86 | 64.3 | 750 |
| AM | 693.0 | 11.55 | 32.2 | 2 |
| CE | 641.4 | 10.69 | 16.0 | 2 |
| SC | 612.9 | 10.21 | 29.5 | 369 |
| RR | 418.1 | 6.97 | 11.8 | 14 |
| BA | 401.4 | 6.69 | 20.3 | 39 |
| SP | 371.0 | 6.18 | 12.0 | 824 |
| PI | 299.6 | 4.99 | 9.3 | 54 |

## 4. Analise de Erro por Ano

| Ano | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |
|-----|-------------|-------------|----------|------|
| 2019 | 386.4 | 6.44 | 16.3 | 2061.0 |
| 2020 | 457.9 | 7.63 | 22.0 | 2111.0 |
| 2021 | 345.5 | 5.76 | 11.2 | 2172.0 |
| 2022 | 667.5 | 11.13 | 57.5 | 2245.0 |
| 2023 | 400.3 | 6.67 | 15.3 | 2303.0 |

## 5. Analise de Erro por Faixa de Produtividade

| Faixa (kg/ha) | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |
|---------------|-------------|-------------|----------|------|
| 0-1500 | 1685.3 | 28.09 | 223.9 | 483 |
| 1500-2500 | 744.4 | 12.41 | 38.3 | 644 |
| 3500+ | 412.0 | 6.87 | 10.5 | 1845 |
| 2500-3000 | 281.0 | 4.68 | 10.0 | 660 |
| 3000-3500 | 198.1 | 3.30 | 6.0 | 916 |

## 6. Importancia das Features

| Feature | Importancia |
|---------|-------------|
| produtividade_ma3 | 19943709974.00 |
| trend | 10354654074.00 |
| precip_vegetativo_mm | 9977177606.00 |
| produtividade_lag1 | 7574353830.00 |
| precip_cv | 3262274206.00 |
| precip_days_gt1mm | 2448759100.00 |
| sul_x_hot_days_anomaly | 2316813670.00 |
| hot_days_enchimento | 2218606704.00 |
| dry_spell_count_7d | 1422372572.00 |
| sul_x_precip_anomaly | 1326772356.00 |

## 7. Graficos

- `scatter_test.png`: Predicted vs Actual no conjunto de teste
- `error_by_year.png`: MAE por ano

## 8. Conclusoes e Proximos Passos

O modelo demonstra capacidade de aprender padroes alem da persistencia historica. As features climaticas contribuem para a previsao, especialmente em anos anomalos.