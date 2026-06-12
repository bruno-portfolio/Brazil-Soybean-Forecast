# Relatorio de Avaliacao do Modelo

Data: 2026-06-12 08:48

## 1. Resumo Executivo

**SUCESSO**: O modelo LightGBM superou o baseline_ma3 em 4.2% no conjunto de teste.

- MAE Teste: **419.6 kg/ha** (6.99 sacas/ha)
- MAPE Teste: **18.0%**
- Melhor iteracao: 116

## 2. Comparacao com Baselines

| Modelo | Split | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | vs Baseline |
|--------|-------|-------------|-------------|----------|-------------|
| baseline_lag1 | Validacao | 733.3 | 12.22 | 64.0 | - |
| baseline_lag1 | Teste | 625.6 | 10.43 | 20.5 | - |
| baseline_ma3 | Validacao | 680.7 | 11.34 | 57.0 | - |
| baseline_ma3 | Teste | 438.0 | 7.30 | 16.2 | - |
| **LightGBM** | Validacao | **629.2** | **10.49** | **53.7** | +7.6% |
| **LightGBM** | Teste | **419.6** | **6.99** | **18.0** | +4.2% |

## 3. Analise de Erro por UF (Top 10 piores)

| UF | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |
|-----|-------------|-------------|----------|------|
| ES | 1316.0 | 21.93 | 33.4 | 2 |
| PE | 1035.3 | 17.26 | 25.9 | 1 |
| RS | 739.9 | 12.33 | 54.8 | 435 |
| RR | 660.9 | 11.01 | 18.8 | 8 |
| AC | 629.8 | 10.50 | 16.2 | 8 |
| MS | 541.8 | 9.03 | 15.1 | 78 |
| BA | 515.0 | 8.58 | 19.5 | 23 |
| CE | 496.6 | 8.28 | 12.9 | 8 |
| SC | 481.9 | 8.03 | 13.7 | 218 |
| PR | 450.5 | 7.51 | 12.2 | 390 |

## 4. Analise de Erro por Ano

| Ano | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |
|-----|-------------|-------------|----------|------|
| 2022 | 629.2 | 10.49 | 53.7 | 2525.0 |
| 2023 | 419.6 | 6.99 | 18.0 | 2603.0 |

## 5. Analise de Erro por Faixa de Produtividade

| Faixa (kg/ha) | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |
|---------------|-------------|-------------|----------|------|
| 0-1500 | 1375.6 | 22.93 | 151.9 | 106 |
| 1500-2500 | 695.7 | 11.59 | 34.4 | 301 |
| 3500+ | 400.8 | 6.68 | 10.2 | 1270 |
| 2500-3000 | 269.9 | 4.50 | 9.6 | 401 |
| 3000-3500 | 228.4 | 3.81 | 6.9 | 525 |

## 6. Importancia das Features

| Feature | Importancia |
|---------|-------------|
| produtividade_ma3 | 58917150326.00 |
| trend | 19736643312.00 |
| deficit_ratio_enchimento | 13499875808.00 |
| produtividade_lag1 | 6430929676.00 |
| radiation_total | 5825242348.00 |
| fert_import_ton | 4764903876.00 |
| precip_vegetativo_mm | 3212713194.00 |
| deficit_vegetativo_mm | 2959614072.00 |
| water_deficit_ratio | 2707217082.00 |
| mun_yield_hist_mean | 2692995472.00 |

## 7. Graficos

- `scatter_test.png`: Predicted vs Actual no conjunto de teste
- `error_by_year.png`: MAE por ano

## 8. Conclusoes e Proximos Passos

O modelo demonstra capacidade de aprender padroes alem da persistencia historica. As features climaticas contribuem para a previsao, especialmente em anos anomalos.