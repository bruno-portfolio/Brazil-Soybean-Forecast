# Dicionario de Dados - Dataset Final v2.0

Este documento descreve todas as variaveis do dataset `data/processed/dataset_final.parquet`, usado para treinamento do modelo de previsao de produtividade de soja.

**Versao**: 2.0 (com Fase 1 de melhorias)

## Visao Geral

| Caracteristica      | Valor            |
|---------------------|------------------|
| Total de registros  | 48,147           |
| Municipios unicos   | 2,763            |
| Periodo             | 2000 - 2023      |
| Anos                | 24               |
| Total de features   | 38               |
| Granularidade       | Municipio x Ano  |
| Chave primaria      | (cod_ibge, ano)  |

## Variaveis

### Identificadores

| Coluna    | Tipo   | Descricao                                      | Exemplo    |
|-----------|--------|------------------------------------------------|------------|
| cod_ibge  | int64  | Codigo IBGE do municipio (7 digitos)           | 5200209    |
| ano       | int32  | Ano da safra (ano da colheita, conforme PAM)   | 2023       |

### Target (Variavel Alvo)

| Coluna              | Tipo    | Unidade | Descricao                                  | Range            |
|---------------------|---------|---------|--------------------------------------------|-----------------:|
| produtividade_kg_ha | float64 | kg/ha   | Rendimento medio da safra de soja          | [105, 8000]      |
| area_colhida_ha     | float64 | ha      | Area colhida no municipio                  | > 0              |
| producao_ton        | float64 | ton     | Producao total do municipio                | > 0              |

**Notas:**
- Fonte: PAM/IBGE (tabela SIDRA 5457)
- Conversao: 60 kg = 1 saca; portanto, `produtividade_kg_ha / 60` = sacas/ha
- Media historica: ~2750 kg/ha (~46 sacas/ha)

### Features Climaticas (Safra Completa)

Agregadas na **janela fenologica** de Outubro a Marco (periodo critico da soja).

**Mapeamento de safra:**
- A safra PAM de ano X corresponde ao clima de Out/(X-1) a Mar/X
- Exemplo: safra 2023 usa clima de Out/2022 a Mar/2023

| Coluna           | Tipo    | Unidade | Agregacao | Descricao                                        |
|------------------|---------|---------|-----------|--------------------------------------------------|
| precip_total_mm  | float64 | mm      | soma      | Precipitacao acumulada na janela fenologica      |
| tmean_avg        | float64 | C       | media     | Temperatura media do ar (media da janela)        |
| tmin_avg         | float64 | C       | media     | Temperatura minima do ar (media da janela)       |
| tmax_avg         | float64 | C       | media     | Temperatura maxima do ar (media da janela)       |
| hot_days_count   | int64   | dias    | contagem  | Dias com Tmax > 32C (estresse termico)           |
| gdd_accumulated  | float64 | C-dia   | soma      | Growing Degree Days acumulados (base 10C)        |

**Fonte:** NASA POWER (diario, por coordenada do centroide municipal)

### Features por Fase Fenologica (NOVO v2.0)

Mesmas metricas climaticas, mas agregadas por fase do ciclo da soja:

#### Fase Plantio (Outubro-Novembro)

| Coluna           | Tipo    | Unidade | Descricao                                        |
|------------------|---------|---------|--------------------------------------------------|
| precip_plantio_mm | float64 | mm      | Precipitacao acumulada no plantio               |
| tmean_plantio    | float64 | C       | Temperatura media no plantio                     |
| tmin_plantio     | float64 | C       | Temperatura minima no plantio                    |
| tmax_plantio     | float64 | C       | Temperatura maxima no plantio                    |
| hot_days_plantio | int64   | dias    | Dias quentes (>32C) no plantio                   |
| gdd_plantio      | float64 | C-dia   | GDD acumulado no plantio                         |

#### Fase Vegetativa (Dezembro-Janeiro)

| Coluna              | Tipo    | Unidade | Descricao                                     |
|---------------------|---------|---------|-----------------------------------------------|
| precip_vegetativo_mm | float64 | mm      | Precipitacao acumulada na fase vegetativa    |
| tmean_vegetativo    | float64 | C       | Temperatura media na fase vegetativa          |
| tmin_vegetativo     | float64 | C       | Temperatura minima na fase vegetativa         |
| tmax_vegetativo     | float64 | C       | Temperatura maxima na fase vegetativa         |
| hot_days_vegetativo | int64   | dias    | Dias quentes na fase vegetativa               |
| gdd_vegetativo      | float64 | C-dia   | GDD acumulado na fase vegetativa              |

#### Fase Enchimento (Fevereiro-Marco)

| Coluna              | Tipo    | Unidade | Descricao                                     |
|---------------------|---------|---------|-----------------------------------------------|
| precip_enchimento_mm | float64 | mm      | Precipitacao acumulada no enchimento         |
| tmean_enchimento    | float64 | C       | Temperatura media no enchimento               |
| tmin_enchimento     | float64 | C       | Temperatura minima no enchimento              |
| tmax_enchimento     | float64 | C       | Temperatura maxima no enchimento              |
| hot_days_enchimento | int64   | dias    | Dias quentes no enchimento                    |
| gdd_enchimento      | float64 | C-dia   | GDD acumulado no enchimento                   |

**Importancia das fases:**
- **Plantio**: Estabelecimento da cultura, sensivel a deficit hidrico
- **Vegetativa**: Crescimento, alta demanda de agua e nutrientes
- **Enchimento**: Fase critica para peso dos graos, sensivel a estresse

### Features de Veranico (NOVO v2.0)

Metricas que capturam a **distribuicao temporal** da precipitacao, nao apenas o total.

| Coluna             | Tipo    | Unidade | Descricao                                           |
|--------------------|---------|---------|-----------------------------------------------------|
| dry_spell_max      | int64   | dias    | Maior sequencia de dias secos (precip < 2mm)        |
| dry_spell_count_7d | int64   | -       | Numero de veranicos >= 7 dias consecutivos          |
| dry_spell_count_10d| int64   | -       | Numero de veranicos >= 10 dias consecutivos         |
| precip_cv          | float64 | -       | Coeficiente de variacao da precipitacao diaria      |
| precip_days_gt1mm  | int64   | dias    | Numero de dias com chuva significativa (>1mm)       |

**Por que importa:**
- 500mm bem distribuidos ≠ 500mm concentrados em poucos dias
- Veranicos (periodos secos) causam estresse mesmo com boa precipitacao total
- `dry_spell_max` maximo observado: 109 dias (seca extrema)

### Features ENSO (NOVO v2.0)

Indice ONI (Oceanic Nino Index) que indica El Nino/La Nina.

| Coluna     | Tipo    | Range        | Descricao                                      |
|------------|---------|--------------|------------------------------------------------|
| oni_avg    | float64 | [-1.4, 2.3]  | ONI medio da safra (Out-Mar)                   |
| oni_min    | float64 | [-1.7, 2.0]  | ONI minimo da safra                            |
| oni_max    | float64 | [-0.8, 2.6]  | ONI maximo da safra                            |
| oni_std    | float64 | [0, 0.6]     | Variabilidade do ONI na safra                  |
| is_la_nina | int64   | 0/1          | Flag: 1 se safra em La Nina (ONI < -0.5)       |
| is_el_nino | int64   | 0/1          | Flag: 1 se safra em El Nino (ONI > +0.5)       |

**Interpretacao:**
- **El Nino (ONI > +0.5)**: Chuva acima da media no Sul, seca no Norte/Nordeste
- **La Nina (ONI < -0.5)**: Seca no Sul, chuva acima da media no Norte
- **Neutro**: Condicoes normais

**Exemplos historicos:**
- 2022: La Nina forte (ONI = -0.95) → Seca severa no Sul
- 2016: El Nino forte (ONI = +2.33) → Recorde de chuva no Sul
- 2023: El Nino (ONI = +0.97) → Seca no Mato Grosso

### Features Historicas

Features que capturam comportamento passado da produtividade, sem usar informacao do ano atual.

| Coluna             | Tipo    | Descricao                                           | Missing   |
|--------------------|---------|-----------------------------------------------------|-----------|
| produtividade_lag1 | float64 | Produtividade do ano anterior (t-1)                 | 5.7%*     |
| produtividade_ma3  | float64 | Media movel dos 3 anos anteriores (t-1, t-2, t-3)   | ~0%       |
| trend              | float64 | Ano normalizado [0, 1] para capturar tendencia      | 0%        |

**Notas:**
- (*) Missing em lag1 ocorre no primeiro ano de cada municipio no dataset
- `trend = (ano - 2000) / (2023 - 2000)` = progresso temporal normalizado
- **Zero leakage garantido:** todas as features historicas usam apenas dados de anos anteriores

## Validacoes de Qualidade

1. **Unicidade de chave:** (cod_ibge, ano) e unica
2. **Range de produtividade:** [0, 10000] kg/ha
3. **Consistencia de temperatura:** tmin <= tmean <= tmax
4. **Precipitacao nao-negativa:** precip >= 0
5. **Zero leakage temporal:** features historicas nao usam dados do ano atual

## Cobertura Temporal

| Ano  | Municipios | Ano  | Municipios |
|------|------------|------|------------|
| 2000 | 1,663      | 2012 | 2,055      |
| 2001 | 1,702      | 2013 | 2,119      |
| 2002 | 1,739      | 2014 | 2,185      |
| 2003 | 1,816      | 2015 | 2,236      |
| 2004 | 1,876      | 2016 | 2,291      |
| 2005 | 1,893      | 2017 | 2,346      |
| 2006 | 1,848      | 2018 | 2,395      |
| 2007 | 1,855      | 2019 | 2,440      |
| 2008 | 1,895      | 2020 | 2,517      |
| 2009 | 1,910      | 2021 | 2,573      |
| 2010 | 1,958      | 2022 | 2,654      |
| 2011 | 2,010      | 2023 | 2,671      |

## Uso no Pipeline

```python
import pandas as pd

# Carregar dataset
df = pd.read_parquet("data/processed/dataset_final.parquet")

# Separar features e target
target_col = "produtividade_kg_ha"
exclude_cols = ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

print(f"Features: {len(feature_cols)}")  # 38 features
```

## Changelog

### v2.0 (2026-01-24) - Fase 1 de Melhorias
- Adicionadas 18 features de janela fenologica quebrada (6 por fase x 3 fases)
- Adicionadas 5 features de veranico (dry_spell, precip_cv, etc.)
- Adicionadas 6 features ENSO (ONI e flags)
- Total de features: 9 → 38
- **Resultado**: Modelo superou baseline em 5.6% (antes: 3.8%)

### v1.0 (2026-01-24)
- Versao inicial com 9 features
- Features climaticas agregadas (safra completa)
- Features historicas (lag1, ma3, trend)
