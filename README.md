# Brazil Soybean Yield Forecast

![CI](https://github.com/bruno-portfolio/Brazil-Soybean-Forecast/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Model MAE](https://img.shields.io/badge/MAE-410_kg%2Fha-brightgreen)
![Features](https://img.shields.io/badge/Features-100-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Regional-9cf)

## TL;DR

**Problem:** Agricultural cooperatives and credit banks need accurate municipality-level soybean yield estimates for crop insurance pricing, credit limits, and early warning of problematic harvests — but currently rely on subjective estimates or historical averages that ignore climate and regional variations.

**Solution:** End-to-end ML pipeline combining 9 public data sources (IBGE, NASA POWER, SoilGrids, ENSO, MapBiomas, and more) to predict yield in kg/ha with **MAE of 410 kg/ha (~6.8 bags/ha)**, calibrated confidence intervals via conformal prediction, and automatic translation to credit risk language.

**Key Features:**
- Regional LightGBM models (South vs Cerrado) with specialized regularization
- 100 engineered features: water balance, soil interactions, ENSO teleconnections, phenological phases
- Conformal prediction intervals (95.6% coverage in South, 92.3% in Cerrado at 90% nominal)
- DVC-orchestrated pipeline with 12 reproducible stages

<img width="1572" height="970" alt="readme_model_comparison" src="https://github.com/user-attachments/assets/fc6ab88a-092a-4d93-bb27-7aaf220a37c2" />
<img width="1573" height="970" alt="readme_regional_performance" src="https://github.com/user-attachments/assets/0f916b93-ffe2-4e69-96dc-edd81e4ce30e" />
<img width="1772" height="1371" alt="readme_feature_importance" src="https://github.com/user-attachments/assets/044bf2e8-b881-49ed-88c5-a22fd1e7d845" />
<img width="2204" height="1127" alt="readme_scatter" src="https://github.com/user-attachments/assets/f6b08a43-52dd-4f9f-9011-c56f1f2d9531" />
<img width="2373" height="970" alt="readme_error_by_year" src="https://github.com/user-attachments/assets/425dc596-9078-4b12-a71f-038f9981fec1" />

---

## Results

### Model Performance vs Baseline

| Model | MAE (kg/ha) | MAE (sacas/ha) | vs Baseline |
|-------|-------------|----------------|-------------|
| 3-Year Moving Average (MA3) | 439 | 7.3 | baseline |
| LightGBM (Global) | 417 | 6.9 | -5.0% |
| **Regional LightGBM** | **410** | **6.8** | **-6.6%** |

### Performance by Region

| Region | Baseline MAE | Model MAE | Improvement |
|--------|--------------|-----------|-------------|
| **South** (RS, PR, SC) | 587 kg/ha | 561 kg/ha | **-4.4%** |
| **Cerrado** (MT, GO, MS, etc.) | 330 kg/ha | 299 kg/ha | **-9.5%** |
| **Combined** | 439 kg/ha | 410 kg/ha | **-6.6%** |

*Test set: 2,303 municipality-year observations (2023 harvest season)*

### Feature Importance

Top drivers: historical yield momentum (lag1, MA3, trend) accounts for ~50% of model gain. Water deficit at grain fill (`deficit_ratio_enchimento`) is the strongest climate signal at 6%, followed by solar radiation and precipitation variability.

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
│       ├── dataset_final.parquet  # Main dataset (48K rows, 106 cols)
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
│   │   └── build_features.py      # Feature engineering (100 features)
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

## Features (100 total)

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

### Historical & Temporal (3)
| Feature | Description |
|---------|-------------|
| `produtividade_lag1` | Previous year yield |
| `produtividade_ma3` | 3-year moving average |
| `trend` | Temporal trend (technological progress) |

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
                    |  (100 features)  |
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

---

## Known Limitations

1. **Annual PAM Data Lag**: IBGE publishes ~18 months after harvest, preventing real-time validation

2. **NASA POWER is Satellite-Derived**: Climate data is interpolated to municipality centroid; large municipalities may have representativeness errors

3. **Technological Drift**: New cultivars, farming practices, and area expansion cause drift not captured by the model. Annual retraining recommended

4. **2022 La Nina**: Validation set (2022) includes extreme La Nina year with historic yield losses in RS. Model uses this as calibration, not as final evaluation

---

## Changelog

### v2.0 (Current)
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
