import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_v1.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "results" / "training_result.json"
CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
ENSO_PATH = PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet"
MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"
OUTPUT_PATH = PROJECT_ROOT / "results" / "predictions_2024_2025.parquet"
OUTPUT_JSON_PATH = PROJECT_ROOT / "results" / "predictions_metadata.json"

MODEL_SUL_PATH = PROJECT_ROOT / "models" / "model_sul.pkl"
MODEL_CERRADO_PATH = PROJECT_ROOT / "models" / "model_cerrado.pkl"
REGIONAL_METADATA_PATH = PROJECT_ROOT / "results" / "regional_training_result.json"

CONFORMAL_SUL_PATH = PROJECT_ROOT / "models" / "conformal_sul.pkl"
CONFORMAL_CERRADO_PATH = PROJECT_ROOT / "models" / "conformal_cerrado.pkl"

REGION_SUL = [41, 42, 43]

QUANTILE_MODELS_PATH = {
    0.05: PROJECT_ROOT / "models" / "quantile_p05.pkl",
    0.10: PROJECT_ROOT / "models" / "quantile_p10.pkl",
    0.50: PROJECT_ROOT / "models" / "quantile_p50.pkl",
    0.90: PROJECT_ROOT / "models" / "quantile_p90.pkl",
    0.95: PROJECT_ROOT / "models" / "quantile_p95.pkl",
}


class ConformalCalibrator:
    """Calibrador conformal simples para intervalos de predicao."""

    def __init__(self):
        self.conformity_scores = None
        self.n_calib = 0

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalCalibrator":
        """Calibra o predictor usando residuos absolutos."""
        self.conformity_scores = np.abs(y_true - y_pred)
        self.n_calib = len(y_true)
        return self

    def predict_interval(
        self, y_pred: np.ndarray, alpha: float = 0.20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera intervalos de predicao calibrados."""
        if self.conformity_scores is None:
            raise ValueError("Calibrador nao foi treinado.")

        n = self.n_calib
        adjusted_quantile = min(1.0, (1 - alpha) * (n + 1) / n)
        q = np.quantile(self.conformity_scores, adjusted_quantile)

        lower = y_pred - q
        upper = y_pred + q

        return lower, upper


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
            logger.info(f"  p{int(quantile*100):02d} carregado de: {path}")
        else:
            logger.warning(f"  p{int(quantile*100):02d} NAO encontrado: {path}")

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


def load_municipalities() -> pd.DataFrame:
    """Carrega lista de municipios."""
    logger.info("Carregando municipios...")
    df = pd.read_parquet(MUNICIPALITIES_PATH)
    logger.info(f"  Municipios: {len(df):,}")
    return df


def get_soy_producing_municipalities(df_target: pd.DataFrame, min_years: int = 3) -> list:
    """Retorna lista de municipios produtores de soja."""
    recent_years = df_target[df_target["ano"] >= 2020]
    counts = recent_years.groupby("cod_ibge").size()
    producers = counts[counts >= min_years].index.tolist()
    logger.info(f"  Municipios produtores (>= {min_years} anos desde 2020): {len(producers):,}")
    return producers


def assign_crop_year(date: pd.Timestamp, start_month: int = 10, end_month: int = 3) -> int:
    """Atribui o ano da safra para uma data."""
    month = date.month
    year = date.year

    if month >= start_month:
        return year + 1
    elif month <= end_month:
        return year
    else:
        return None


def assign_phenology_phase(date: pd.Timestamp) -> str:
    """Atribui a fase fenologica para uma data."""
    month = date.month

    if month in [10, 11]:
        return "plantio"
    elif month in [12, 1]:
        return "vegetativo"
    elif month in [2, 3]:
        return "enchimento"
    else:
        return None


def calculate_gdd(row: pd.Series, base_temp: float = 10.0) -> float:
    """Calcula Growing Degree Days (GDD) para um dia."""
    return max(0, row["tmean"] - base_temp)


def calculate_dry_spell_metrics(df_group: pd.DataFrame, threshold_mm: float = 2.0) -> dict:
    """Calcula metricas de veranico para um grupo (municipio/safra)."""
    if len(df_group) == 0:
        return {
            "dry_spell_max": 0,
            "dry_spell_count_7d": 0,
            "dry_spell_count_10d": 0,
        }

    df_sorted = df_group.sort_values("date")
    precip = df_sorted["precip"].values

    is_dry = precip < threshold_mm

    dry_spells = []
    current_spell = 0

    for dry in is_dry:
        if dry:
            current_spell += 1
        else:
            if current_spell > 0:
                dry_spells.append(current_spell)
            current_spell = 0

    if current_spell > 0:
        dry_spells.append(current_spell)

    return {
        "dry_spell_max": max(dry_spells) if dry_spells else 0,
        "dry_spell_count_7d": sum(1 for s in dry_spells if s >= 7),
        "dry_spell_count_10d": sum(1 for s in dry_spells if s >= 10),
    }


def calculate_precip_variability(df_group: pd.DataFrame) -> dict:
    """Calcula metricas de variabilidade da precipitacao."""
    if len(df_group) == 0:
        return {"precip_cv": 0, "precip_days_gt1mm": 0}

    precip = df_group["precip"].values

    mean_precip = precip.mean()
    cv = precip.std() / mean_precip if mean_precip > 0 else 0

    days_with_rain = (precip > 1.0).sum()

    return {
        "precip_cv": cv,
        "precip_days_gt1mm": int(days_with_rain),
    }


def prepare_climate_features(
    df_climate: pd.DataFrame,
    municipalities: list,
    years: list,
    base_temp: float = 10.0,
    hot_threshold: float = 32.0,
) -> pd.DataFrame:
    """Prepara features climaticas para os anos especificados."""
    logger.info(f"Preparando features climaticas para anos: {years}")

    df = df_climate[df_climate["cod_ibge"].isin(municipalities)].copy()

    df["month"] = df["date"].dt.month

    mask = (df["month"] >= 10) | (df["month"] <= 3)
    df = df[mask].copy()

    df["crop_year"] = df["date"].apply(assign_crop_year)
    df["phase"] = df["date"].apply(assign_phenology_phase)

    df = df[df["crop_year"].isin(years)]

    df["gdd"] = df.apply(lambda row: calculate_gdd(row, base_temp), axis=1)
    df["is_hot_day"] = (df["tmax"] > hot_threshold).astype(int)

    logger.info(f"  Registros de clima filtrados: {len(df):,}")

    phases = ["plantio", "vegetativo", "enchimento"]
    all_results = []

    for cod_ibge in df["cod_ibge"].unique():
        df_mun = df[df["cod_ibge"] == cod_ibge]

        for crop_year in df_mun["crop_year"].unique():
            if pd.isna(crop_year):
                continue

            df_season = df_mun[df_mun["crop_year"] == crop_year]

            result = {
                "cod_ibge": cod_ibge,
                "ano": int(crop_year),
            }

            for phase in phases:
                df_phase = df_season[df_season["phase"] == phase]

                if len(df_phase) > 0:
                    result[f"precip_{phase}_mm"] = df_phase["precip"].sum()
                    result[f"tmean_{phase}"] = df_phase["tmean"].mean()
                    result[f"tmin_{phase}"] = df_phase["tmin"].mean()
                    result[f"tmax_{phase}"] = df_phase["tmax"].mean()
                    result[f"hot_days_{phase}"] = df_phase["is_hot_day"].sum()
                    result[f"gdd_{phase}"] = df_phase["gdd"].sum()
                else:
                    result[f"precip_{phase}_mm"] = 0
                    result[f"tmean_{phase}"] = np.nan
                    result[f"tmin_{phase}"] = np.nan
                    result[f"tmax_{phase}"] = np.nan
                    result[f"hot_days_{phase}"] = 0
                    result[f"gdd_{phase}"] = 0

            result["precip_total_mm"] = df_season["precip"].sum()
            result["tmean_avg"] = df_season["tmean"].mean()
            result["tmin_avg"] = df_season["tmin"].mean()
            result["tmax_avg"] = df_season["tmax"].mean()
            result["hot_days_count"] = df_season["is_hot_day"].sum()
            result["gdd_accumulated"] = df_season["gdd"].sum()

            dry_metrics = calculate_dry_spell_metrics(df_season)
            result.update(dry_metrics)

            var_metrics = calculate_precip_variability(df_season)
            result.update(var_metrics)

            all_results.append(result)

    df_agg = pd.DataFrame(all_results)
    logger.info(f"  Features climaticas agregadas: {len(df_agg):,} registros")

    return df_agg


def add_enso_features(df: pd.DataFrame, df_enso: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features ENSO ao dataset."""
    logger.info("Adicionando features ENSO...")

    enso_cols = ["ano", "oni_avg", "oni_min", "oni_max", "oni_std"]
    df_enso_num = df_enso[enso_cols].copy()

    df_enso_num["is_la_nina"] = (df_enso["enso_phase"] == "nina").astype(int)
    df_enso_num["is_el_nino"] = (df_enso["enso_phase"] == "nino").astype(int)

    df = df.merge(df_enso_num, on="ano", how="left")

    return df


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de anomalia climatica para inferencia."""
    logger.info("Calculando features de anomalia...")

    df = df.copy()

    anomaly_cols = [
        "precip_anomaly",
        "temp_anomaly",
        "hot_days_anomaly",
        "gdd_anomaly",
        "precip_enchimento_anomaly",
        "dry_spell_anomaly",
    ]

    for col in anomaly_cols:
        if col not in df.columns:
            df[col] = 0.0

    logger.info("  Features de anomalia adicionadas (assumindo valores normais)")

    return df


def add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de tratamento regional para o Sul do Brasil."""
    logger.info("Adicionando features regionais...")

    df = df.copy()

    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)

    sul_ufs = [41, 42, 43]

    df["is_sul"] = df["uf_cod"].isin(sul_ufs).astype(int)

    if "is_la_nina" in df.columns:
        df["sul_x_la_nina"] = df["is_sul"] * df["is_la_nina"]

    if "precip_anomaly" in df.columns:
        df["sul_x_precip_anomaly"] = df["is_sul"] * df["precip_anomaly"].fillna(0)

    if "hot_days_anomaly" in df.columns:
        df["sul_x_hot_days_anomaly"] = df["is_sul"] * df["hot_days_anomaly"].fillna(0)

    df = df.drop(columns=["uf_cod"])

    n_sul = df["is_sul"].sum()
    logger.info(f"  Registros do Sul: {n_sul:,} ({n_sul/len(df)*100:.1f}%)")

    return df


def calculate_historical_features(
    df: pd.DataFrame, df_target: pd.DataFrame, years_to_predict: list
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

    ano_min = 2000
    ano_max = 2023
    df["trend"] = (df["ano"] - ano_min) / (ano_max - ano_min)

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
            q_str = f"p{int(quantile*100):02d}"
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
        "model_version": "v1",
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


def add_enso_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona interacoes ENSO para inferencia (Fase 3)."""
    df = df.copy()

    if "is_la_nina" in df.columns and "precip_enchimento_anomaly" in df.columns:
        df["la_nina_x_precip_ench_anom"] = df["is_la_nina"] * df[
            "precip_enchimento_anomaly"
        ].fillna(0)

    if "dry_spell_max" in df.columns and "hot_days_anomaly" in df.columns:
        dry_spell_norm = (
            df["dry_spell_max"] / df["dry_spell_max"].std()
            if df["dry_spell_max"].std() > 0
            else df["dry_spell_max"]
        )
        df["dry_spell_x_hot_anom"] = dry_spell_norm * df["hot_days_anomaly"].fillna(0)

    if "is_la_nina" in df.columns and "precip_anomaly" in df.columns:
        df["la_nina_x_precip_anom"] = df["is_la_nina"] * df["precip_anomaly"].fillna(0)

    if "temp_anomaly" in df.columns and "precip_anomaly" in df.columns:
        df["heat_drought_stress"] = df["temp_anomaly"].fillna(0) * (-df["precip_anomaly"].fillna(0))

    if "hot_days_enchimento" in df.columns and "precip_enchimento_mm" in df.columns:
        precip_ench_mean = df["precip_enchimento_mm"].mean()
        if precip_ench_mean > 0:
            precip_ench_ratio = df["precip_enchimento_mm"] / precip_ench_mean
            precip_deficit = 1 - precip_ench_ratio.clip(0, 2)
            df["enchimento_stress"] = df["hot_days_enchimento"] * precip_deficit
        else:
            df["enchimento_stress"] = 0

    if "is_el_nino" in df.columns and "precip_anomaly" in df.columns:
        df["el_nino_x_precip_anom"] = df["is_el_nino"] * df["precip_anomaly"].fillna(0)

    return df


def main():
    """Pipeline principal de inferencia."""
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

    df_climate = load_climate_data()
    df_target = load_target_data()
    df_enso = load_enso_data()

    municipalities = get_soy_producing_municipalities(df_target, min_years=3)

    df = prepare_climate_features(
        df_climate, municipalities, years_to_predict, base_temp=10.0, hot_threshold=32.0
    )

    df = add_enso_features(df, df_enso)

    df = calculate_historical_features(df, df_target, years_to_predict)

    df = add_anomaly_features(df)
    df = add_regional_features(df)

    df = add_enso_interactions(df)

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
            f"    Media: {df_year['pred_produtividade_kg_ha'].mean():.1f} kg/ha ({df_year['pred_produtividade_kg_ha'].mean()/60:.1f} sc/ha)"
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
        logger.info(f"  {uf}: {len(df_uf)//2} municipios, media {mean_pred:.0f} kg/ha")

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
