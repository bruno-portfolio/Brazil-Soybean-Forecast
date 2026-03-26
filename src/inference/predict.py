import hashlib
import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from src.common.conformal import ConformalCalibrator  # noqa: F401
from src.common.constants import REGION_SUL
from src.common.features import (
    add_enso_features,
    add_enso_interactions,
    add_regional_features,
    add_soil_climate_interactions,
    add_soil_features,
    calculate_climate_anomalies,
)
from src.common.io import PROJECT_ROOT, load_config, load_municipalities
from src.common.phenology import (
    filter_phenology_window_regional,
    get_default_phenology,
    get_regional_phenology,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v2")
MODEL_PATH = PROJECT_ROOT / "models" / f"model_{MODEL_VERSION}.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "results" / f"training_result_{MODEL_VERSION}.json"

# Fallback para v1 se v2 nao existe
if not MODEL_PATH.exists():
    _fallback = PROJECT_ROOT / "models" / "model_v1.pkl"
    if _fallback.exists():
        MODEL_PATH = _fallback
        MODEL_METADATA_PATH = PROJECT_ROOT / "results" / "training_result.json"
CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
ENSO_PATH = PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet"
OUTPUT_PATH = PROJECT_ROOT / "results" / "predictions_2024_2025.parquet"
OUTPUT_JSON_PATH = PROJECT_ROOT / "results" / "predictions_metadata.json"

SOIL_PATH = PROJECT_ROOT / "data" / "processed" / "soil_properties.parquet"
MODEL_SUL_PATH = PROJECT_ROOT / "models" / "model_sul.pkl"
MODEL_CERRADO_PATH = PROJECT_ROOT / "models" / "model_cerrado.pkl"
REGIONAL_METADATA_PATH = PROJECT_ROOT / "results" / "regional_training_result.json"

CONFORMAL_SUL_PATH = PROJECT_ROOT / "models" / "conformal_sul.pkl"
CONFORMAL_CERRADO_PATH = PROJECT_ROOT / "models" / "conformal_cerrado.pkl"


QUANTILE_MODELS_PATH = {
    0.05: PROJECT_ROOT / "models" / "quantile_p05.pkl",
    0.10: PROJECT_ROOT / "models" / "quantile_p10.pkl",
    0.50: PROJECT_ROOT / "models" / "quantile_p50.pkl",
    0.90: PROJECT_ROOT / "models" / "quantile_p90.pkl",
    0.95: PROJECT_ROOT / "models" / "quantile_p95.pkl",
}


def load_model():
    """Carrega o modelo treinado (fallback para modelo unico)."""
    logger.info("Carregando modelo...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"  Modelo carregado de: {MODEL_PATH}")
    return model


def load_regional_models() -> tuple:
    """Carrega modelos regionais (Sul e Cerrado)."""
    logger.info("Carregando modelos regionais...")

    model_sul = None
    model_cerrado = None

    if MODEL_SUL_PATH.exists():
        with open(MODEL_SUL_PATH, "rb") as f:
            model_sul = pickle.load(f)
        logger.info(f"  Modelo Sul carregado de: {MODEL_SUL_PATH}")

    if MODEL_CERRADO_PATH.exists():
        with open(MODEL_CERRADO_PATH, "rb") as f:
            model_cerrado = pickle.load(f)
        logger.info(f"  Modelo Cerrado carregado de: {MODEL_CERRADO_PATH}")

    if model_sul is None or model_cerrado is None:
        logger.warning("  Modelos regionais nao encontrados! Execute train_regional.py primeiro.")
        return None, None

    return model_sul, model_cerrado


def load_conformal_calibrators() -> tuple:
    """Carrega calibradores conformal para intervalos."""
    logger.info("Carregando calibradores conformal...")

    calibrator_sul = None
    calibrator_cerrado = None

    if CONFORMAL_SUL_PATH.exists():
        with open(CONFORMAL_SUL_PATH, "rb") as f:
            calibrator_sul = pickle.load(f)
        logger.info(f"  Calibrador Sul carregado de: {CONFORMAL_SUL_PATH}")

    if CONFORMAL_CERRADO_PATH.exists():
        with open(CONFORMAL_CERRADO_PATH, "rb") as f:
            calibrator_cerrado = pickle.load(f)
        logger.info(f"  Calibrador Cerrado carregado de: {CONFORMAL_CERRADO_PATH}")

    if calibrator_sul is None or calibrator_cerrado is None:
        logger.warning(
            "  Calibradores conformal nao encontrados! Execute train_conformal.py primeiro."
        )
        return None, None

    return calibrator_sul, calibrator_cerrado


def load_quantile_models() -> dict:
    """Carrega modelos quantilicos para intervalos de confianca."""
    logger.info("Carregando modelos quantilicos...")

    models = {}
    for quantile, path in QUANTILE_MODELS_PATH.items():
        if path.exists():
            with open(path, "rb") as f:
                models[quantile] = pickle.load(f)
            logger.info(f"  p{int(quantile * 100):02d} carregado de: {path}")
        else:
            logger.warning(f"  p{int(quantile * 100):02d} NAO encontrado: {path}")

    if len(models) == 0:
        logger.warning("  Nenhum modelo quantilico encontrado!")
        return None

    return models


def load_model_metadata() -> dict:
    """Carrega metadados do modelo."""
    with open(MODEL_METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_climate_data() -> pd.DataFrame:
    """Carrega dados de clima diario."""
    logger.info("Carregando dados de clima...")
    df = pd.read_parquet(CLIMATE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  Registros de clima: {len(df):,}")
    return df


def load_target_data() -> pd.DataFrame:
    """Carrega dados de produtividade (target) - para features historicas."""
    logger.info("Carregando dados de target...")
    df = pd.read_parquet(TARGET_PATH)
    logger.info(f"  Registros de target: {len(df):,}")
    return df


def load_enso_data() -> pd.DataFrame:
    """Carrega dados ENSO (ONI)."""
    logger.info("Carregando dados ENSO...")
    df = pd.read_parquet(ENSO_PATH)
    logger.info(f"  Registros ENSO: {len(df):,}")
    return df


def get_soy_producing_municipalities(df_target: pd.DataFrame, min_years: int = 3) -> list:
    """Retorna lista de municipios produtores de soja."""
    recent_years = df_target[df_target["ano"] >= 2020]
    counts = recent_years.groupby("cod_ibge").size()
    producers = counts[counts >= min_years].index.tolist()
    logger.info(f"  Municipios produtores (>= {min_years} anos desde 2020): {len(producers):,}")
    return producers


ANOMALY_COLS = [
    "precip_anomaly",
    "temp_anomaly",
    "hot_days_anomaly",
    "gdd_anomaly",
    "precip_enchimento_anomaly",
    "dry_spell_anomaly",
]


def _fill_missing_anomalies(df: pd.DataFrame) -> None:
    """Preenche anomalias faltantes com 0.0.

    NaN ocorre em municipios com <5 anos de historico (min_years do expanding window).
    Anomalia=0 significa "sem desvio da normal", proxy aceitavel para inferencia
    quando nao ha dados suficientes para calcular o z-score real.
    """
    for col in ANOMALY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    n_nonzero = sum((df[col] != 0.0).sum() for col in ANOMALY_COLS)
    logger.info(f"  Anomalias: {n_nonzero:,} valores nao-zero")


def calculate_historical_features(
    df: pd.DataFrame,
    df_target: pd.DataFrame,
    years_to_predict: list,
    trend_ref_min: int = 2000,
    trend_ref_max: int = 2025,
) -> pd.DataFrame:
    """Calcula features historicas (lag1, ma3, trend) para previsao."""
    logger.info("Calculando features historicas...")

    df = df.copy()

    df_hist = df_target.sort_values(["cod_ibge", "ano"])

    historical_features = []

    for cod_ibge in df["cod_ibge"].unique():
        mun_hist = df_hist[df_hist["cod_ibge"] == cod_ibge].sort_values("ano")

        if len(mun_hist) == 0:
            continue

        for year in years_to_predict:
            past_data = mun_hist[mun_hist["ano"] < year]["produtividade_kg_ha"].values

            if len(past_data) == 0:
                continue

            lag1 = past_data[-1]

            ma3 = past_data[-3:].mean() if len(past_data) >= 1 else np.nan

            historical_features.append(
                {
                    "cod_ibge": cod_ibge,
                    "ano": year,
                    "produtividade_lag1": lag1,
                    "produtividade_ma3": ma3,
                }
            )

    df_hist_features = pd.DataFrame(historical_features)

    df = df.merge(df_hist_features, on=["cod_ibge", "ano"], how="inner")

    df["trend"] = (df["ano"] - trend_ref_min) / (trend_ref_max - trend_ref_min)

    logger.info(f"  Registros com features historicas: {len(df):,}")

    return df


def generate_predictions(
    model, df: pd.DataFrame, feature_names: list, quantile_models: dict = None
) -> pd.DataFrame:
    """Gera previsoes usando o modelo principal e modelos quantilicos."""
    logger.info("Gerando previsoes (modelo unico)...")

    X = df[feature_names].copy()

    missing = X.isnull().sum()
    if missing.sum() > 0:
        logger.warning("  Valores faltantes encontrados:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"    {col}: {count}")

        X = X.fillna(X.median())

    predictions = model.predict(X)

    df = df.copy()
    df["pred_produtividade_kg_ha"] = predictions
    df["pred_produtividade_sacas_ha"] = predictions / 60

    logger.info(f"  Previsoes ponto: {len(predictions):,}")

    if quantile_models is not None and len(quantile_models) > 0:
        logger.info("  Gerando intervalos de confianca...")

        for quantile, q_model in quantile_models.items():
            q_pred = q_model.predict(X)
            q_str = f"p{int(quantile * 100):02d}"
            df[f"pred_{q_str}_kg_ha"] = q_pred
            df[f"pred_{q_str}_sacas_ha"] = q_pred / 60

        if 0.10 in quantile_models and 0.90 in quantile_models:
            df["intervalo_80_largura"] = df["pred_p90_kg_ha"] - df["pred_p10_kg_ha"]

        logger.info(f"    Quantis gerados: {list(quantile_models.keys())}")

    return df


def generate_predictions_regional(
    model_sul,
    model_cerrado,
    calibrator_sul,
    calibrator_cerrado,
    df: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    """Gera previsoes usando modelos regionais e intervalos conformal."""
    logger.info("Gerando previsoes (modelos regionais + conformal)...")

    df = df.copy()

    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)
    df["is_sul"] = df["uf_cod"].isin(REGION_SUL).astype(int)

    X = df[feature_names].copy()

    missing = X.isnull().sum()
    if missing.sum() > 0:
        logger.warning("  Valores faltantes encontrados:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"    {col}: {count}")
        X = X.fillna(X.median())

    n = len(df)
    pred_point = np.zeros(n)
    pred_lower_80 = np.zeros(n)
    pred_upper_80 = np.zeros(n)
    pred_lower_90 = np.zeros(n)
    pred_upper_90 = np.zeros(n)

    idx_sul = df["is_sul"] == 1
    idx_cerrado = df["is_sul"] == 0

    if idx_sul.sum() > 0:
        X_sul = X[idx_sul].values
        pred_sul = model_sul.predict(X_sul)
        pred_point[idx_sul] = pred_sul

        if calibrator_sul is not None:
            lower_80, upper_80 = calibrator_sul.predict_interval(pred_sul, alpha=0.20)
            lower_90, upper_90 = calibrator_sul.predict_interval(pred_sul, alpha=0.10)
            pred_lower_80[idx_sul] = lower_80
            pred_upper_80[idx_sul] = upper_80
            pred_lower_90[idx_sul] = lower_90
            pred_upper_90[idx_sul] = upper_90

        logger.info(f"  Sul: {idx_sul.sum():,} previsoes")

    if idx_cerrado.sum() > 0:
        X_cerrado = X[idx_cerrado].values
        pred_cerrado = model_cerrado.predict(X_cerrado)
        pred_point[idx_cerrado] = pred_cerrado

        if calibrator_cerrado is not None:
            lower_80, upper_80 = calibrator_cerrado.predict_interval(pred_cerrado, alpha=0.20)
            lower_90, upper_90 = calibrator_cerrado.predict_interval(pred_cerrado, alpha=0.10)
            pred_lower_80[idx_cerrado] = lower_80
            pred_upper_80[idx_cerrado] = upper_80
            pred_lower_90[idx_cerrado] = lower_90
            pred_upper_90[idx_cerrado] = upper_90

        logger.info(f"  Cerrado: {idx_cerrado.sum():,} previsoes")

    df["pred_produtividade_kg_ha"] = pred_point
    df["pred_produtividade_sacas_ha"] = pred_point / 60

    df["pred_p10_kg_ha"] = pred_lower_80
    df["pred_p50_kg_ha"] = pred_point
    df["pred_p90_kg_ha"] = pred_upper_80
    df["pred_p05_kg_ha"] = pred_lower_90
    df["pred_p95_kg_ha"] = pred_upper_90

    df["intervalo_80_largura"] = df["pred_p90_kg_ha"] - df["pred_p10_kg_ha"]
    df["intervalo_90_largura"] = df["pred_p95_kg_ha"] - df["pred_p05_kg_ha"]

    df = df.drop(columns=["uf_cod"])

    logger.info(f"  Total: {len(df):,} previsoes com intervalos conformal")

    return df


def add_municipality_info(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona informacoes dos municipios (nome, UF)."""
    df_mun = load_municipalities()
    df_mun = df_mun[["cod_ibge", "nome", "uf"]]

    df = df.merge(df_mun, on="cod_ibge", how="left")

    return df


def calculate_dataset_hash(df: pd.DataFrame) -> str:
    """Calcula hash do dataset para rastreabilidade."""
    data_str = df.to_json()
    return hashlib.md5(data_str.encode()).hexdigest()[:12]


def save_predictions(df: pd.DataFrame, model_metadata: dict, years: list) -> None:
    """Salva previsoes e metadados."""
    logger.info("Salvando previsoes...")

    output_cols = [
        "cod_ibge",
        "nome",
        "uf",
        "ano",
        "pred_produtividade_kg_ha",
        "pred_produtividade_sacas_ha",
        "pred_p05_kg_ha",
        "pred_p10_kg_ha",
        "pred_p50_kg_ha",
        "pred_p90_kg_ha",
        "pred_p95_kg_ha",
        "intervalo_80_largura",
        "produtividade_lag1",
        "produtividade_ma3",
        "precip_total_mm",
        "hot_days_count",
        "oni_avg",
    ]

    output_cols = [c for c in output_cols if c in df.columns]

    df_output = df[output_cols].copy()

    df_output = df_output.sort_values(["ano", "uf", "nome"])

    df_output.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"  Previsoes salvas em: {OUTPUT_PATH}")

    metadata = {
        "inference_date": datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "prediction_type": "ex-post",
        "prediction_description": "Previsoes usando clima observado (nao e forecast real)",
        "years_predicted": years,
        "n_municipalities": df["cod_ibge"].nunique(),
        "n_predictions": len(df),
        "dataset_hash": calculate_dataset_hash(df),
        "model_path": str(MODEL_PATH),
        "feature_names": model_metadata["feature_names"],
        "model_test_metrics": model_metadata.get("test_metrics", {}),
        "statistics_by_year": {},
        "statistics_by_uf": {},
    }

    for year in years:
        df_year = df[df["ano"] == year]
        metadata["statistics_by_year"][str(year)] = {
            "n_municipalities": len(df_year),
            "pred_mean_kg_ha": round(df_year["pred_produtividade_kg_ha"].mean(), 1),
            "pred_median_kg_ha": round(df_year["pred_produtividade_kg_ha"].median(), 1),
            "pred_std_kg_ha": round(df_year["pred_produtividade_kg_ha"].std(), 1),
            "pred_min_kg_ha": round(df_year["pred_produtividade_kg_ha"].min(), 1),
            "pred_max_kg_ha": round(df_year["pred_produtividade_kg_ha"].max(), 1),
        }

    if "uf" in df.columns:
        uf_stats = df.groupby("uf").agg({"pred_produtividade_kg_ha": ["mean", "count"]}).round(1)
        uf_stats.columns = ["pred_mean_kg_ha", "n_municipalities"]
        uf_stats = uf_stats.sort_values("n_municipalities", ascending=False)

        for uf in uf_stats.head(10).index:
            metadata["statistics_by_uf"][uf] = {
                "n_municipalities": int(uf_stats.loc[uf, "n_municipalities"]),
                "pred_mean_kg_ha": float(uf_stats.loc[uf, "pred_mean_kg_ha"]),
            }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  Metadados salvos em: {OUTPUT_JSON_PATH}")


def main():
    """Pipeline principal de inferencia."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 70)
    logger.info("INFERENCIA DE PRODUTIVIDADE DE SOJA 2024-2025")
    logger.info("Modalidade: Ex-post (clima observado)")
    logger.info("Fase 3: Modelos regionais + Conformal Prediction")
    logger.info("=" * 70)

    years_to_predict = [2024, 2025]

    model_sul, model_cerrado = load_regional_models()
    calibrator_sul, calibrator_cerrado = load_conformal_calibrators()

    use_regional = model_sul is not None and model_cerrado is not None

    if use_regional:
        logger.info("\n[OK] Usando modelos regionais com intervalos conformal")
        if REGIONAL_METADATA_PATH.exists():
            with open(REGIONAL_METADATA_PATH) as f:
                model_metadata = json.load(f)
            feature_names = model_metadata["feature_names"]
        else:
            model_metadata = load_model_metadata()
            feature_names = model_metadata["feature_names"]
    else:
        logger.warning("\n[!] Modelos regionais nao disponiveis, usando modelo unico")
        model = load_model()
        model_metadata = load_model_metadata()
        feature_names = model_metadata["feature_names"]
        quantile_models = load_quantile_models()

    logger.info(f"Features do modelo: {len(feature_names)}")

    features_config = load_config("features")
    base_temp = 10.0
    hot_threshold = 32.0
    for feat in features_config["features"]["climate_features"]:
        if feat["name"] == "gdd_accumulated":
            base_temp = feat.get("base_temp", 10.0)
        if feat["name"] == "hot_days_count":
            hot_threshold = feat.get("threshold", 32.0)

    trend_ref_min = features_config.get("features", {}).get("trend_ref_year_min", 2000)
    trend_ref_max = features_config.get("features", {}).get("trend_ref_year_max", 2025)

    df_climate = load_climate_data()
    df_target = load_target_data()
    df_enso = load_enso_data()

    df_mun = load_municipalities(columns=["cod_ibge", "lat"])
    lat_lookup = dict(zip(df_mun["cod_ibge"], df_mun["lat"]))

    municipalities = get_soy_producing_municipalities(df_target, min_years=3)

    from src.common.climate_aggregation import aggregate_climate_duckdb

    hist_start = min(years_to_predict) - 6
    all_years = list(range(hist_start, min(years_to_predict))) + years_to_predict

    df_filtered = df_climate[df_climate["cod_ibge"].isin(municipalities)].copy()
    df_filtered = filter_phenology_window_regional(
        df_filtered,
        get_regional_phenology(features_config),
        get_default_phenology(features_config),
    )
    df_filtered = df_filtered[df_filtered["crop_year"].isin(all_years)]

    df_all = aggregate_climate_duckdb(df_filtered, base_temp, hot_threshold, lat_lookup=lat_lookup)

    df_all = add_enso_features(df_all, df_enso)

    df_all = calculate_climate_anomalies(df_all, min_years=5)

    df = df_all[df_all["ano"].isin(years_to_predict)].copy()

    df = calculate_historical_features(
        df, df_target, years_to_predict, trend_ref_min, trend_ref_max
    )

    _fill_missing_anomalies(df)
    df = add_regional_features(df)

    df = add_enso_interactions(df)

    if SOIL_PATH.exists():
        logger.info("Carregando dados de solo...")
        df_soil = pd.read_parquet(SOIL_PATH)
        df = add_soil_features(df, df_soil)
        df = add_soil_climate_interactions(df)
    else:
        logger.warning("soil_properties.parquet nao encontrado, solo nao sera adicionado")

    from src.common.new_source_features import (
        add_fertilizante_features,
        add_irrigacao_features,
        add_new_source_interactions,
        add_sinistro_features,
        add_uso_solo_features,
    )

    df = add_irrigacao_features(df)
    df = add_fertilizante_features(df)
    df = add_sinistro_features(df)
    df = add_uso_solo_features(df)
    df = add_new_source_interactions(df)

    if use_regional:
        df = generate_predictions_regional(
            model_sul, model_cerrado, calibrator_sul, calibrator_cerrado, df, feature_names
        )
    else:
        df = generate_predictions(model, df, feature_names, quantile_models)

    df = add_municipality_info(df)

    logger.info("\n" + "=" * 70)
    logger.info("ESTATISTICAS DAS PREVISOES")
    logger.info("=" * 70)

    for year in years_to_predict:
        df_year = df[df["ano"] == year]
        logger.info(f"\nAno {year}:")
        logger.info(f"  Municipios: {len(df_year):,}")
        logger.info("  Produtividade prevista (ponto):")
        logger.info(
            f"    Media: {df_year['pred_produtividade_kg_ha'].mean():.1f} kg/ha ({df_year['pred_produtividade_kg_ha'].mean() / 60:.1f} sc/ha)"
        )
        logger.info(f"    Mediana: {df_year['pred_produtividade_kg_ha'].median():.1f} kg/ha")
        logger.info(f"    Min: {df_year['pred_produtividade_kg_ha'].min():.1f} kg/ha")
        logger.info(f"    Max: {df_year['pred_produtividade_kg_ha'].max():.1f} kg/ha")

        if "pred_p10_kg_ha" in df_year.columns:
            logger.info("  Intervalos de confianca 80% (p10-p90):")
            logger.info(f"    p10 medio: {df_year['pred_p10_kg_ha'].mean():.1f} kg/ha")
            logger.info(f"    p50 medio: {df_year['pred_p50_kg_ha'].mean():.1f} kg/ha")
            logger.info(f"    p90 medio: {df_year['pred_p90_kg_ha'].mean():.1f} kg/ha")
            logger.info(f"    Largura media: {df_year['intervalo_80_largura'].mean():.1f} kg/ha")

    logger.info("\nTop 5 UFs por numero de municipios:")
    uf_counts = df.groupby("uf").size().sort_values(ascending=False)
    for uf in uf_counts.head(5).index:
        df_uf = df[df["uf"] == uf]
        mean_pred = df_uf["pred_produtividade_kg_ha"].mean()
        logger.info(f"  {uf}: {len(df_uf) // 2} municipios, media {mean_pred:.0f} kg/ha")

    save_predictions(df, model_metadata, years_to_predict)

    logger.info("\n" + "=" * 70)
    logger.info("INFERENCIA CONCLUIDA!")
    logger.info("=" * 70)
    logger.info("\nArquivos gerados:")
    logger.info(f"  - {OUTPUT_PATH}")
    logger.info(f"  - {OUTPUT_JSON_PATH}")

    return df


if __name__ == "__main__":
    main()
