# Brazil Soybean Yield Forecast

![CI](https://github.com/bruno-portfolio/Brazil-Soybean-Forecast/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model MAE](https://img.shields.io/badge/MAE-409_kg%2Fha-brightgreen)
![Features](https://img.shields.io/badge/Features-111-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Regional-9cf)

## TL;DR

**Problem:** Agricultural cooperatives and credit banks need accurate municipality-level soybean yield estimates for crop insurance pricing, credit limits, and early warning of problematic harvests — but currently rely on subjective estimates or historical averages that ignore climate and regional variations.

**Solution:** End-to-end ML pipeline combining 9 public data sources (IBGE, NASA POWER, SoilGrids, ENSO, MapBiomas, and more) to predict yield in kg/ha with **MAE of 409 kg/ha (~6.8 bags/ha)**, validated across **8 independent harvest seasons** via temporal cross-validation, with calibrated confidence intervals and automatic translation to credit risk language.

**Key Features:**
- Regional LightGBM models (South vs Cerrado) with specialized regularization
- 111 engineered features: water balance, soil interactions, ENSO teleconnections, municipality history, NDVI
- Expanding-window temporal CV (2016-2023): **MAE 429 +/- 63 kg/ha across 8 folds**
- Conformal prediction intervals (95.6% coverage in South, 92.3% in Cerrado at 90% nominal)
- DVC-orchestrated pipeline with 12 reproducible stages

<img width="1572" height="970" alt="readme_model_comparison" src="https://github.com/user-attachments/assets/b41812e0-b53c-4ef4-9ef3-34fa278fe0dc" />
<img width="1573" height="970" alt="readme_regional_performance" src="https://github.com/user-attachments/assets/8595f484-ff28-400d-88d1-6e930c173966" />
<img width="1772" height="1371" alt="readme_feature_importance" src="https://github.com/user-attachments/assets/1dca3169-186d-4b34-9d13-17623aa60c66" />
<img width="2204" height="1127" alt="readme_scatter" src="https://github.com/user-attachments/assets/ddb310e1-dd30-4a07-b29d-20c01d703d6a" />
<img width="2373" height="970" alt="readme_error_by_year" src="https://github.com/user-attachments/assets/62f0cef3-7708-4444-a142-615c76d621dc" />

---


## Results

### Model Performance vs Baseline

| Model | MAE (kg/ha) | MAE (sacas/ha) | vs Baseline |
|-------|-------------|----------------|-------------|
| 3-Year Moving Average (MA3) | 438 | 7.3 | baseline |
| LightGBM (Global) | 417 | 6.9 | -4.9% |
| **Regional LightGBM** | **409** | **6.8** | **-6.7%** |

### Performance by Region

| Region | Baseline MAE | Model MAE | Improvement |
|--------|--------------|-----------|-------------|
| **South** (RS, PR, SC) | 582 kg/ha | 558 kg/ha | **-4.1%** |
| **Cerrado** (MT, GO, MS, etc.) | 340 kg/ha | 309 kg/ha | **-9.3%** |
| **Combined** | 438 kg/ha | 409 kg/ha | **-6.7%** |

*Test set: 2,603 municipality-year observations (2023 harvest season)*

### Temporal Cross-Validation (Expanding Window)

| Test Year | MAE (kg/ha) | MAPE (%) | n |
|-----------|-------------|----------|---|
| 2016 | 359 | 15.9 | 2,160 |
| 2017 | 448 | 14.5 | 2,275 |
| 2018 | 415 | 12.8 | 2,319 |
| 2019 | 408 | 17.1 | 2,369 |
| 2020 | 472 | 19.6 | 2,388 |
| 2021 | 356 | 11.6 | 2,472 |
| 2022 | 563 | 46.4 | 2,525 |
| 2023 | 409 | 17.1 | 2,603 |
| **Mean +/- Std** | **429 +/- 63** | **19.4** | |

*Each fold trains from scratch on years <= test_year - 2, validates on test_year - 1*

### Feature Importance

Top drivers: temporal trend, previous-year yield (`lag1`, `MA3`) and **municipality historical mean** (new in v3) account for ~55% of model gain. `mun_yield_hist_mean` captures the baseline productivity of each municipality (technology, soil management, farming maturity). Water deficit at grain fill (`deficit_ratio_enchimento`) is the strongest climate signal at 3.6%, followed by solar radiation and precipitation variability.

---

## Quick Start

### Option 1: Quick Run (uses existing processed data)
```bash
git clone https://github.com/bruno-portfolio/Brazil-Soybean-Forecast.git
cd Brazil-Soybean-Forecast

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

The pre-processed `dataset_final.parquet` is included for quick testing.

### Option 2: Full Pipeline (DVC)
```bash
# Run full pipeline: ingest -> features -> train -> evaluate
dvc repro

# Or run specific stages
dvc repro build_features
dvc repro train
dvc repro evaluate
```

**Note:** Large climate files (`climate_daily.parquet`, ~550MB) are not included in the repo. Run `dvc repro ingest_climate` to download from NASA POWER.

### Run Dashboard
```bash
streamlit run app/dashboard.py
```

---

## Architecture

```
Brazil-Soybean-Forecast/
├── dvc.yaml                       # Pipeline (12 stages)
├── configs/
│   ├── model.yaml                 # LightGBM hyperparameters
│   ├── split.yaml                 # Temporal train/val/test split
│   ├── features.yaml              # Feature engineering config
│   └── ...                        # climate, soil, target, geo, ndvi
├── data/
│   ├── raw/                       # Raw API downloads (cached)
│   └── processed/                 # Feature-engineered datasets
│       ├── dataset_final.parquet  # Main dataset (48K rows, 117 cols)
│       ├── climate_daily.parquet  # NASA POWER daily climate
│       ├── soil_properties.parquet
│       ├── target_soja.parquet
│       ├── fertilizante_uf.parquet
│       ├── pivos_irrigacao.parquet
│       ├── seguro_rural.parquet
│       └── mapbiomas_soja.parquet
├── src/
│   ├── ingest/                    # Data ingestion (9 sources)
│   │   ├── pam.py                 # IBGE PAM soybean yield
│   │   ├── climate_power.py       # NASA POWER daily climate
│   │   ├── soilgrids.py           # ISRIC SoilGrids soil properties
│   │   ├── pivos.py               # ANA irrigation pivots
│   │   ├── fertilizante.py        # ComexStat fertilizer imports
│   │   ├── seguro_rural.py        # MAPA PSR crop insurance
│   │   ├── mapbiomas_soja.py      # MapBiomas soybean land use
│   │   ├── ndvi_gee.py            # MODIS NDVI via Google Earth Engine
│   │   └── enso.py                # NOAA ONI climate indices
│   ├── features/
│   │   └── build_features.py      # Feature engineering (111 features)
│   ├── common/                    # Shared utilities
│   │   ├── water_balance.py       # ETo (Hargreaves/Penman-Monteith)
│   │   ├── phenology.py           # Regional phenological calendars
│   │   ├── conformal.py           # Conformal prediction calibrator
│   │   ├── climate_aggregation.py # DuckDB-accelerated aggregation
│   │   └── new_source_features.py # Irrigation, fertilizer, insurance
│   ├── modeling/
│   │   ├── train.py               # Global LightGBM training
│   │   ├── train_regional.py      # South + Cerrado models
│   │   ├── train_conformal.py     # Conformal prediction intervals
│   │   └── split.py               # Temporal split (no leakage)
│   ├── evaluation/                # Metrics, SHAP explainability
│   ├── inference/                 # Forecast generation
│   ├── monitoring/                # Drift detection
│   └── business/                  # Credit risk translation
├── models/                        # Trained model artifacts
│   ├── model_v2.pkl               # Global LightGBM
│   ├── model_sul.pkl              # South regional model
│   ├── model_cerrado.pkl          # Cerrado regional model
│   ├── conformal_sul.pkl          # Conformal calibrator (South)
│   └── conformal_cerrado.pkl      # Conformal calibrator (Cerrado)
├── results/                       # Evaluation outputs & plots
├── app/
│   └── dashboard.py               # Streamlit dashboard
└── tests/                         # Unit tests
```

---

## Features (111 total)

### Climate Features (by phenological phase)
| Category | Count | Examples |
|----------|-------|----------|
| Temperature | 12 | `tmean_plantio`, `tmax_vegetativo`, `hot_days_enchimento` |
| Precipitation | 11 | `precip_total_mm`, `dry_spell_max`, `precip_cv` |
| Evapotranspiration | 7 | `eto_total_mm`, `eto_plantio_mm`, `eto_enchimento_mm` |
| Water Deficit | 6 | `deficit_plantio_mm`, `deficit_ratio_enchimento`, `water_deficit_ratio` |
| Radiation | 2 | `radiation_total`, `radiation_mean` |
| GDD | 5 | `gdd_accumulated`, `gdd_plantio`, `gdd_vegetativo` |

### Soil Features (16)
| Feature | Description |
|---------|-------------|
| `clay_0_30cm`, `sand_0_30cm`, `silt_0_30cm` | Texture composition (0-30cm and 30-100cm) |
| `phh2o_0_30cm` | Soil acidity |
| `soc_0_30cm`, `nitrogen_0_30cm` | Organic matter |
| `awc_estimated` | Available Water Capacity |
| `cec_0_30cm` | Cation Exchange Capacity |
| `soil_quality_index` | Composite soil quality |

### ENSO & Climate Indices (6)
| Feature | Description |
|---------|-------------|
| `oni_avg`, `oni_std` | Oceanic Nino Index (mean, variability) |
| `is_la_nina`, `is_el_nino` | Binary ENSO flags |

### Historical & Municipality Identity (5)
| Feature | Description |
|---------|-------------|
| `produtividade_lag1` | Previous year yield |
| `produtividade_ma3` | 3-year moving average |
| `trend` | Temporal trend (technological progress) |
| `mun_yield_hist_mean` | Expanding historical mean yield per municipality (no leakage) |
| `mun_yield_volatility` | Historical yield coefficient of variation per municipality |

### NDVI Vegetation Index (9)
| Feature | Description |
|---------|-------------|
| `ndvi_mean_safra` | Mean NDVI for entire growing season |
| `ndvi_max_safra`, `ndvi_min_safra` | Peak and minimum vegetation vigor |
| `ndvi_amplitude` | Seasonal NDVI range |
| `ndvi_plantio`, `ndvi_vegetativo`, `ndvi_enchimento` | NDVI by phenological phase |
| `ndvi_x_precip_deficit` | NDVI x precipitation anomaly interaction |
| `ndvi_ench_x_la_nina` | Grain-fill NDVI under La Nina stress |

### New Data Sources (4)
| Feature | Source | Description |
|---------|--------|-------------|
| `pct_irrigado` | ANA | % irrigated area |
| `fert_import_ton` | ComexStat | Fertilizer imports by state |
| `sinistro_rate_3yr` | MAPA PSR | 3-year insurance loss rate |
| `pct_soja` | MapBiomas | Soybean land use fraction |

### Interactions & Anomalies (19)
Climate anomalies, ENSO interactions (`la_nina_x_deficit`, `terminal_drought_stress`), soil-climate interactions (`awc_x_deficit`, `sand_x_drought`), regional interactions (`sul_x_la_nina`), and source interactions (`irrigacao_x_deficit`, `fert_x_precip`).

---

## Model Details

### Regional LightGBM Architecture

```
                    +------------------+
                    |   Input Data     |
                    |  (111 features)  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
    +------------------+          +------------------+
    |   South Model    |          |  Cerrado Model   |
    |  (PR, SC, RS)    |          | (MT, GO, MS...)  |
    |                  |          |                  |
    |   LightGBM       |          |   LightGBM       |
    |                  |          |                  |
    |  Higher L2 reg   |          |  Standard params |
    |  min_leaf = 10   |          |  min_leaf = 20   |
    +--------+---------+          +--------+---------+
             |                             |
             +-------------+---------------+
                           v
                  +------------------+
                  |   Predictions    |
                  |    (kg/ha)       |
                  +------------------+
                           |
                           v
                  +------------------+
                  |    Conformal     |
                  |   Calibration    |
                  | (80%/90% bands)  |
                  +------------------+
```

### Why Regional Models?

1. **Climate Variability**: South Brazil has higher inter-annual variability due to La Nina/El Nino
2. **Different Yield Distributions**: Cerrado has higher, more stable yields; South has wider range
3. **Specialized Regularization**: South model uses stronger regularization to handle extreme event outliers

---

## Data Sources

| Source | Data | Granularity | Auth |
|--------|------|-------------|------|
| [IBGE/SIDRA](https://sidra.ibge.gov.br/) | Soybean yield & area | Municipality x Year | No |
| [NASA POWER](https://power.larc.nasa.gov/) | Daily climate (temp, precip, radiation, wind) | Point x Day | No |
| [ISRIC SoilGrids](https://soilgrids.org/) | Soil properties (300+ attributes) | 250m raster | No |
| [NOAA ONI](https://www.cpc.ncep.noaa.gov/) | El Nino/La Nina index | Monthly | No |
| [ANA](https://dadosabertos.ana.gov.br/) | Irrigation pivot locations | Point | No |
| [ComexStat](http://comexstat.mdic.gov.br/) | Fertilizer imports by state | State x Month | No |
| [MAPA PSR](https://www.gov.br/agricultura/) | Crop insurance claims | Municipality x Year | No |
| [MapBiomas](https://mapbiomas.org/) | Soybean land use extent | 30m raster | No |
| [MODIS/GEE](https://earthengine.google.com/) | NDVI vegetation index | Pixel x 16-day | GEE account |

---

## Train/Test Split

| Set | Years | Samples | Purpose |
|-----|-------|---------|---------|
| **Train** | 2000-2021 | 43,019 | Model training |
| **Validation** | 2022 | 2,525 | Early stopping (La Nina stress test) |
| **Test** | 2023 | 2,603 | Final evaluation |

*Strictly temporal split — no future information leaks into training*

Additionally, **expanding-window temporal CV** validates the model across 8 independent test years (2016-2023), training from scratch for each fold. This provides confidence that results are not specific to a single test year.

---

## Known Limitations

1. **Annual PAM Data Lag**: IBGE publishes ~18 months after harvest, preventing real-time validation

2. **NASA POWER is Satellite-Derived**: Climate data is interpolated to municipality centroid; large municipalities may have representativeness errors

3. **Technological Drift**: New cultivars, farming practices, and area expansion cause drift not captured by the model. Annual retraining recommended

4. **2022 La Nina**: Validation set (2022) includes extreme La Nina year with historic yield losses in RS. Model uses this as calibration, not as final evaluation

---

## Changelog

### v3.0 (Current)
- 111 features (up from 100): municipality identity, NDVI vegetation index
- `mun_yield_hist_mean` (expanding historical mean per municipality) — top 4 feature importance
- NDVI features from MODIS satellite (7 direct + 2 interaction features)
- Expanding-window temporal cross-validation across 8 harvest seasons (2016-2023)
- LightGBM native NaN handling (no more row dropping for sparse features)

### v2.0
- Regional LightGBM models (South vs Cerrado) replacing single global model
- 100 features (up from 76): water balance, soil interactions, new data sources
- 5 new data sources: SoilGrids, ANA pivots, ComexStat, MAPA PSR, MapBiomas
- Conformal prediction for calibrated uncertainty intervals
- DVC pipeline with 12 reproducible stages
- CI/CD with GitHub Actions (lint, test, build)

### v1.0
- Single LightGBM model (76 features)
- Climate + ENSO + historical features
- Streamlit dashboard

---

## Installation

### Requirements
- Python 3.10+
- 4GB RAM minimum
- ~2GB disk space for data

### Core Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
duckdb>=1.0.0
pyarrow>=14.0.0
pyyaml>=6.0
requests>=2.31.0
dvc>=3.30.0
```

### Optional (dashboard & explainability)
```
streamlit>=1.28.0
plotly>=5.18.0
shap>=0.44.0
matplotlib>=3.7.0
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

Bruno - [GitHub](https://github.com/bruno-portfolio) | [LinkedIn](https://www.linkedin.com/in/bruno-escalhao-32a41934b/)

---

## Citation

```bibtex
@software{brazil_soybean_forecast,
  author = {Bruno},
  title = {Brazil Soybean Yield Forecast},
  year = {2026},
  url = {https://github.com/bruno-portfolio/Brazil-Soybean-Forecast}
}
```
