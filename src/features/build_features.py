import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"
CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"
CLIMATE_V2_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily_v2.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
ENSO_PATH = PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet"
SOIL_PATH = PROJECT_ROOT / "data" / "processed" / "soil_properties.parquet"
NDVI_PATH = PROJECT_ROOT / "data" / "processed" / "ndvi_safra.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"

LATITUDE_DEFAULT = -15.0


def load_config() -> dict:
    """Carrega configuracao de features do YAML."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_climate_data() -> pd.DataFrame:
    """Carrega dados de clima diario (v2 se disponivel)."""
    logger.info("Carregando dados de clima...")

    if CLIMATE_V2_PATH.exists():
        df = pd.read_parquet(CLIMATE_V2_PATH)
        logger.info("  Usando climate_daily_v2.parquet (com radiacao)")
    else:
        df = pd.read_parquet(CLIMATE_PATH)
        logger.info("  Usando climate_daily.parquet (sem radiacao)")

    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  Registros de clima: {len(df):,}")

    if "radiation" in df.columns:
        n_radiation = df["radiation"].notna().sum()
        logger.info(f"  Registros com radiacao: {n_radiation:,}")

    return df


def calculate_eto_hargreaves(row: pd.Series, lat: float = LATITUDE_DEFAULT) -> float:
    """Calcula ETo pelo metodo Hargreaves-Samani (mm/dia)."""
    tmin = row.get("tmin", row.get("tmean", 20) - 5)
    tmax = row.get("tmax", row.get("tmean", 20) + 5)
    tmean = row.get("tmean", (tmin + tmax) / 2)

    if pd.isna(tmin) or pd.isna(tmax):
        return np.nan

    day_of_year = row["date"].dayofyear

    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    lat_rad = lat * np.pi / 180
    ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta) +
        np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )

    eto = 0.0023 * (tmean + 17.8) * np.sqrt(max(0, tmax - tmin)) * Ra
    return max(0, eto)


def calculate_eto_with_radiation(row: pd.Series) -> float:
    """Calcula ETo usando radiacao solar (Penman-Monteith simplificado)."""
    radiation = row.get("radiation")
    tmean = row.get("tmean")
    rh = row.get("rh", 60)
    wind = row.get("wind_speed", 2.0)

    if pd.isna(radiation) or pd.isna(tmean):
        return calculate_eto_hargreaves(row)

    Rs = radiation

    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)

    gamma = 0.066

    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    ea = es * rh / 100

    Rn = 0.77 * Rs - 2.0

    eto = (0.408 * delta * Rn + gamma * (900 / (tmean + 273)) * wind * (es - ea)) / (
        delta + gamma * (1 + 0.34 * wind)
    )

    return max(0, eto)


def load_target_data() -> pd.DataFrame:
    """Carrega dados de produtividade (target)."""
    logger.info("Carregando dados de target...")
    df = pd.read_parquet(TARGET_PATH)
    logger.info(f"  Registros de target: {len(df):,}")
    return df


def load_enso_data() -> pd.DataFrame:
    """Carrega dados ENSO (ONI)."""
    logger.info("Carregando dados ENSO...")
    if ENSO_PATH.exists():
        df = pd.read_parquet(ENSO_PATH)
        logger.info(f"  Registros ENSO: {len(df):,}")
        return df
    else:
        logger.warning("  Arquivo ENSO nao encontrado. Execute src/ingest/enso.py primeiro.")
        return None


def load_soil_data() -> pd.DataFrame:
    """Carrega dados de solo (SoilGrids)."""
    logger.info("Carregando dados de solo...")
    if SOIL_PATH.exists():
        df = pd.read_parquet(SOIL_PATH)
        logger.info(f"  Municipios com dados de solo: {len(df):,}")
        return df
    else:
        logger.warning("  Arquivo de solo nao encontrado. Execute src/ingest/soilgrids.py primeiro.")
        return None


def load_ndvi_data() -> pd.DataFrame:
    """Carrega dados NDVI (AppEEARS/MODIS)."""
    logger.info("Carregando dados NDVI...")
    if NDVI_PATH.exists():
        df = pd.read_parquet(NDVI_PATH)
        logger.info(f"  Registros NDVI: {len(df):,}")
        return df
    else:
        logger.warning("  Arquivo NDVI nao encontrado. Execute src/ingest/ndvi.py primeiro.")
        return None


def add_ndvi_features(df: pd.DataFrame, df_ndvi: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features NDVI ao dataset."""
    if df_ndvi is None:
        logger.warning("Dados NDVI nao disponiveis. Pulando...")
        return df

    logger.info("Adicionando features NDVI...")

    df = df.merge(df_ndvi, on=["cod_ibge", "ano"], how="left")

    ndvi_cols = [c for c in df_ndvi.columns if c.startswith("ndvi_")]
    if ndvi_cols:
        n_with_ndvi = df[ndvi_cols[0]].notna().sum()
        logger.info(f"  Registros com NDVI: {n_with_ndvi:,} ({100*n_with_ndvi/len(df):.1f}%)")

    return df


def add_ndvi_climate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona interacoes NDVI x clima."""
    logger.info("Adicionando interacoes NDVI x clima...")

    if "ndvi_mean_safra" in df.columns and "precip_anomaly" in df.columns:
        df["ndvi_x_precip_deficit"] = df["ndvi_mean_safra"] * (-df["precip_anomaly"].fillna(0))
        logger.info("  ndvi_x_precip_deficit adicionada")

    if "ndvi_enchimento" in df.columns and "is_la_nina" in df.columns:
        df["ndvi_ench_x_la_nina"] = df["ndvi_enchimento"].fillna(0) * df["is_la_nina"]
        logger.info("  ndvi_ench_x_la_nina adicionada")

    return df


def calculate_gdd(row: pd.Series, base_temp: float) -> float:
    """Calcula Growing Degree Days (GDD) para um dia."""
    return max(0, row["tmean"] - base_temp)


def get_regional_phenology(config: dict) -> dict:
    """Carrega configuracao de janelas fenologicas regionais."""
    regional = config.get("features", {}).get("regional_phenology", {})

    result = {}
    for uf_cod, params in regional.items():
        result[int(uf_cod)] = params

    return result


def get_default_phenology(config: dict) -> dict:
    """Retorna janela fenologica default."""
    window = config.get("features", {}).get("phenology_window", {})
    return {
        "start_month": window.get("start_month", 10),
        "end_month": window.get("end_month", 3),
        "phases": {
            "plantio": [10, 11],
            "vegetativo": [12, 1],
            "enchimento": [2, 3],
        },
    }


def assign_crop_year(date: pd.Timestamp, start_month: int, end_month: int) -> int:
    """Atribui o ano da safra para uma data."""
    month = date.month
    year = date.year

    if month >= start_month:
        return year + 1
    elif month <= end_month:
        return year
    else:
        return None


def assign_phenology_phase_regional(date: pd.Timestamp, phases: dict) -> str:
    """Atribui a fase fenologica para uma data usando configuracao regional."""
    month = date.month

    for phase_name, months in phases.items():
        if month in months:
            return phase_name

    return None


def assign_phenology_phase(date: pd.Timestamp) -> str:
    """Atribui a fase fenologica para uma data (versao default)."""
    month = date.month

    if month in [10, 11]:
        return "plantio"
    elif month in [12, 1]:
        return "vegetativo"
    elif month in [2, 3]:
        return "enchimento"
    else:
        return None


def filter_phenology_window(df: pd.DataFrame, start_month: int, end_month: int) -> pd.DataFrame:
    """Filtra dados de clima para a janela fenologica (versao simples, default)."""
    logger.info(f"Filtrando janela fenologica (mes {start_month} a {end_month})...")

    df = df.copy()
    df["month"] = df["date"].dt.month

    if start_month > end_month:
        mask = (df["month"] >= start_month) | (df["month"] <= end_month)
    else:
        mask = (df["month"] >= start_month) & (df["month"] <= end_month)

    df_filtered = df[mask].copy()

    df_filtered["crop_year"] = df_filtered["date"].apply(
        lambda x: assign_crop_year(x, start_month, end_month)
    )

    df_filtered["phase"] = df_filtered["date"].apply(assign_phenology_phase)

    logger.info(f"  Registros na janela: {len(df_filtered):,}")

    return df_filtered


def filter_phenology_window_regional(
    df: pd.DataFrame,
    regional_config: dict,
    default_config: dict,
) -> pd.DataFrame:
    """Filtra dados de clima usando janelas fenologicas regionais."""
    logger.info("Filtrando janela fenologica por regiao...")

    df = df.copy()
    df["month"] = df["date"].dt.month
    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)

    all_filtered = []

    ufs = df["uf_cod"].unique()

    for uf in ufs:
        df_uf = df[df["uf_cod"] == uf].copy()

        config = regional_config.get(uf, default_config)

        start_month = config["start_month"]
        end_month = config["end_month"]
        phases = config.get(
            "phases",
            {
                "plantio": [10, 11],
                "vegetativo": [12, 1],
                "enchimento": [2, 3],
            },
        )

        if start_month > end_month:
            mask = (df_uf["month"] >= start_month) | (df_uf["month"] <= end_month)
        else:
            mask = (df_uf["month"] >= start_month) & (df_uf["month"] <= end_month)

        df_uf_filtered = df_uf[mask].copy()

        if len(df_uf_filtered) == 0:
            continue

        df_uf_filtered["crop_year"] = df_uf_filtered["date"].apply(
            lambda x, sm=start_month, em=end_month: assign_crop_year(x, sm, em)
        )

        df_uf_filtered["phase"] = df_uf_filtered["date"].apply(
            lambda x, ph=phases: assign_phenology_phase_regional(x, ph)
        )

        all_filtered.append(df_uf_filtered)

    df_result = pd.concat(all_filtered, ignore_index=True)

    df_result = df_result.drop(columns=["uf_cod"])

    logger.info(f"  Registros na janela regional: {len(df_result):,}")

    for uf in sorted(regional_config.keys()):
        if uf in regional_config:
            cfg = regional_config[uf]
            logger.info(f"    UF {uf}: meses {cfg['start_month']}-{cfg['end_month']}")

    return df_result


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


def calculate_water_balance_metrics(df_group: pd.DataFrame) -> dict:
    """Calcula metricas de balanco hidrico (ETo, deficit)."""
    result = {
        "eto_total_mm": 0,
        "eto_mean_mm": np.nan,
        "water_deficit_mm": 0,
        "water_deficit_ratio": np.nan,
        "radiation_mean": np.nan,
        "radiation_total": 0,
    }

    if len(df_group) == 0:
        return result

    has_radiation = "radiation" in df_group.columns and df_group["radiation"].notna().any()

    if has_radiation:
        df_group = df_group.copy()
        df_group["eto"] = df_group.apply(calculate_eto_with_radiation, axis=1)
        result["radiation_mean"] = df_group["radiation"].mean()
        result["radiation_total"] = df_group["radiation"].sum()
    else:
        df_group = df_group.copy()
        df_group["eto"] = df_group.apply(calculate_eto_hargreaves, axis=1)

    eto_values = df_group["eto"].dropna()
    if len(eto_values) > 0:
        result["eto_total_mm"] = eto_values.sum()
        result["eto_mean_mm"] = eto_values.mean()

    precip = df_group["precip"].fillna(0).values
    eto = df_group["eto"].fillna(0).values

    deficit = np.maximum(0, eto - precip)
    result["water_deficit_mm"] = deficit.sum()

    if result["eto_total_mm"] > 0:
        result["water_deficit_ratio"] = result["water_deficit_mm"] / result["eto_total_mm"]

    return result


def calculate_water_balance_by_phase(df_season: pd.DataFrame, phases: list[str]) -> dict:
    """Calcula balanco hidrico por fase fenologica."""
    result = {}

    for phase in phases:
        df_phase = df_season[df_season["phase"] == phase]
        wb = calculate_water_balance_metrics(df_phase)

        result[f"eto_{phase}_mm"] = wb["eto_total_mm"]
        result[f"deficit_{phase}_mm"] = wb["water_deficit_mm"]

        if phase == "enchimento":
            result["deficit_ratio_enchimento"] = wb["water_deficit_ratio"]

    return result


def aggregate_climate_features_by_phase(
    df: pd.DataFrame, base_temp: float = 10.0, hot_threshold: float = 32.0
) -> pd.DataFrame:
    """Agrega features climaticas por fase fenologica."""
    logger.info("Agregando features climaticas por fase fenologica...")

    df = df.copy()

    df["gdd"] = df.apply(lambda row: calculate_gdd(row, base_temp), axis=1)

    df["is_hot_day"] = (df["tmax"] > hot_threshold).astype(int)

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

            wb_metrics = calculate_water_balance_metrics(df_season)
            result["eto_total_mm"] = wb_metrics["eto_total_mm"]
            result["eto_mean_mm"] = wb_metrics["eto_mean_mm"]
            result["water_deficit_mm"] = wb_metrics["water_deficit_mm"]
            result["water_deficit_ratio"] = wb_metrics["water_deficit_ratio"]
            result["radiation_mean"] = wb_metrics["radiation_mean"]
            result["radiation_total"] = wb_metrics["radiation_total"]

            wb_phase = calculate_water_balance_by_phase(df_season, phases)
            result.update(wb_phase)

            all_results.append(result)

    df_agg = pd.DataFrame(all_results)
    logger.info(f"  Registros agregados: {len(df_agg):,}")

    return df_agg


def add_enso_features(df: pd.DataFrame, df_enso: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features ENSO ao dataset."""
    if df_enso is None:
        logger.warning("Dados ENSO nao disponiveis. Pulando...")
        return df

    logger.info("Adicionando features ENSO...")

    df = df.merge(df_enso, on="ano", how="left")

    df["is_la_nina"] = (df["enso_phase"] == "nina").astype(int)
    df["is_el_nino"] = (df["enso_phase"] == "nino").astype(int)

    df = df.drop(columns=["enso_phase", "enso_intensity"], errors="ignore")

    logger.info("  Features ENSO adicionadas")

    return df


def calculate_climate_anomalies(df: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Calcula features de anomalia climatica (desvios da normal historica)."""
    logger.info("Calculando features de anomalia climatica...")

    df = df.copy()
    df = df.sort_values(["cod_ibge", "ano"])

    anomaly_vars = [
        ("precip_total_mm", "precip_anomaly"),
        ("tmean_avg", "temp_anomaly"),
        ("hot_days_count", "hot_days_anomaly"),
        ("gdd_accumulated", "gdd_anomaly"),
        ("precip_enchimento_mm", "precip_enchimento_anomaly"),
        ("dry_spell_max", "dry_spell_anomaly"),
    ]

    for var_name, anomaly_name in anomaly_vars:
        if var_name not in df.columns:
            logger.warning(f"  Variavel {var_name} nao encontrada, pulando...")
            continue

        df[f"_mean_{var_name}"] = (
            df.groupby("cod_ibge")[var_name]
            .apply(lambda x: x.shift(1).expanding(min_periods=min_years).mean())
            .reset_index(level=0, drop=True)
        )

        df[f"_std_{var_name}"] = (
            df.groupby("cod_ibge")[var_name]
            .apply(lambda x: x.shift(1).expanding(min_periods=min_years).std())
            .reset_index(level=0, drop=True)
        )

        std_col = df[f"_std_{var_name}"]
        std_col = std_col.replace(0, np.nan)

        df[anomaly_name] = (df[var_name] - df[f"_mean_{var_name}"]) / std_col

        df[anomaly_name] = df[anomaly_name].clip(-4, 4)

        df = df.drop(columns=[f"_mean_{var_name}", f"_std_{var_name}"])

        n_valid = df[anomaly_name].notna().sum()
        if n_valid > 0:
            logger.info(
                f"  {anomaly_name}: range [{df[anomaly_name].min():.2f}, "
                f"{df[anomaly_name].max():.2f}], {n_valid:,} valores validos"
            )

    logger.info("  Features de anomalia calculadas")

    return df


def add_enso_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona interacoes nao-lineares com ENSO para capturar eventos extremos."""
    logger.info("Adicionando interacoes nao-lineares ENSO...")

    df = df.copy()

    if "is_la_nina" in df.columns and "precip_enchimento_anomaly" in df.columns:
        df["la_nina_x_precip_ench_anom"] = df["is_la_nina"] * df[
            "precip_enchimento_anomaly"
        ].fillna(0)
        logger.info(
            f"  la_nina_x_precip_ench_anom: range "
            f"[{df['la_nina_x_precip_ench_anom'].min():.2f}, "
            f"{df['la_nina_x_precip_ench_anom'].max():.2f}]"
        )

    if "dry_spell_max" in df.columns and "hot_days_anomaly" in df.columns:
        dry_spell_norm = df["dry_spell_max"] / df["dry_spell_max"].std()
        df["dry_spell_x_hot_anom"] = dry_spell_norm * df["hot_days_anomaly"].fillna(0)
        logger.info(
            f"  dry_spell_x_hot_anom: range "
            f"[{df['dry_spell_x_hot_anom'].min():.2f}, "
            f"{df['dry_spell_x_hot_anom'].max():.2f}]"
        )

    if "is_la_nina" in df.columns and "precip_anomaly" in df.columns:
        df["la_nina_x_precip_anom"] = df["is_la_nina"] * df["precip_anomaly"].fillna(0)

    if "temp_anomaly" in df.columns and "precip_anomaly" in df.columns:
        df["heat_drought_stress"] = df["temp_anomaly"].fillna(0) * (-df["precip_anomaly"].fillna(0))
        logger.info(
            f"  heat_drought_stress: range "
            f"[{df['heat_drought_stress'].min():.2f}, "
            f"{df['heat_drought_stress'].max():.2f}]"
        )

    if "hot_days_enchimento" in df.columns and "precip_enchimento_mm" in df.columns:
        precip_ench_mean = df["precip_enchimento_mm"].mean()
        precip_ench_ratio = df["precip_enchimento_mm"] / precip_ench_mean
        precip_deficit = 1 - precip_ench_ratio.clip(0, 2)
        df["enchimento_stress"] = df["hot_days_enchimento"] * precip_deficit
        logger.info(
            f"  enchimento_stress: range "
            f"[{df['enchimento_stress'].min():.2f}, "
            f"{df['enchimento_stress'].max():.2f}]"
        )

    if "is_el_nino" in df.columns and "precip_anomaly" in df.columns:
        df["el_nino_x_precip_anom"] = df["is_el_nino"] * df["precip_anomaly"].fillna(0)

    if "water_deficit_mm" in df.columns and "is_la_nina" in df.columns:
        deficit_norm = df["water_deficit_mm"] / (df["water_deficit_mm"].std() + 1e-8)
        df["la_nina_x_deficit"] = df["is_la_nina"] * deficit_norm
        logger.info(f"  la_nina_x_deficit: range [{df['la_nina_x_deficit'].min():.2f}, {df['la_nina_x_deficit'].max():.2f}]")

    if "deficit_enchimento_mm" in df.columns and "hot_days_enchimento" in df.columns:
        deficit_ench_norm = df["deficit_enchimento_mm"] / (df["deficit_enchimento_mm"].std() + 1e-8)
        hot_ench_norm = df["hot_days_enchimento"] / (df["hot_days_enchimento"].std() + 1e-8)
        df["terminal_drought_stress"] = deficit_ench_norm * hot_ench_norm
        logger.info(f"  terminal_drought_stress: range [{df['terminal_drought_stress'].min():.2f}, {df['terminal_drought_stress'].max():.2f}]")

    logger.info("  Interacoes ENSO adicionadas")

    return df


def add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de tratamento regional para o Sul do Brasil."""
    logger.info("Adicionando features de tratamento regional...")

    df = df.copy()

    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)

    sul_ufs = [41, 42, 43]

    df["is_sul"] = df["uf_cod"].isin(sul_ufs).astype(int)

    if "is_la_nina" in df.columns:
        df["sul_x_la_nina"] = df["is_sul"] * df["is_la_nina"]
        logger.info(f"  sul_x_la_nina: {df['sul_x_la_nina'].sum():,} casos")

    if "precip_anomaly" in df.columns:
        df["sul_x_precip_anomaly"] = df["is_sul"] * df["precip_anomaly"].fillna(0)
        logger.info(
            f"  sul_x_precip_anomaly: range [{df['sul_x_precip_anomaly'].min():.2f}, "
            f"{df['sul_x_precip_anomaly'].max():.2f}]"
        )

    if "hot_days_anomaly" in df.columns:
        df["sul_x_hot_days_anomaly"] = df["is_sul"] * df["hot_days_anomaly"].fillna(0)

    df = df.drop(columns=["uf_cod"])

    n_sul = df["is_sul"].sum()
    logger.info(f"  Registros do Sul: {n_sul:,} ({n_sul/len(df)*100:.1f}%)")

    return df


def add_soil_features(df: pd.DataFrame, df_soil: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de solo ao dataset."""
    if df_soil is None:
        logger.warning("Dados de solo nao disponiveis. Pulando...")
        return df

    logger.info("Adicionando features de solo...")

    df = df.merge(df_soil, on="cod_ibge", how="left")

    soil_cols = [c for c in df_soil.columns if c != "cod_ibge"]
    if soil_cols:
        n_with_soil = df[soil_cols[0]].notna().sum()
        n_total = len(df)
        logger.info(f"  Registros com dados de solo: {n_with_soil:,} ({n_with_soil/n_total*100:.1f}%)")

    logger.info(f"  Features de solo adicionadas: {len(soil_cols)}")
    for col in soil_cols[:5]:
        if col in df.columns:
            logger.info(f"    - {col}: mean={df[col].mean():.2f}, nulls={df[col].isna().sum()}")

    return df


def add_soil_climate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona interacoes entre features de solo e clima."""
    logger.info("Adicionando interacoes solo x clima...")

    df = df.copy()
    interactions_added = 0

    if "clay_0_30cm" in df.columns and "precip_anomaly" in df.columns:
        clay_norm = df["clay_0_30cm"] / 100
        df["clay_x_precip_deficit"] = clay_norm * (-df["precip_anomaly"].fillna(0))
        interactions_added += 1
        logger.info(
            f"  clay_x_precip_deficit: range "
            f"[{df['clay_x_precip_deficit'].min():.2f}, "
            f"{df['clay_x_precip_deficit'].max():.2f}]"
        )

    if "awc_estimated" in df.columns and "dry_spell_max" in df.columns:
        awc_norm = df["awc_estimated"] / df["awc_estimated"].max()
        dry_norm = df["dry_spell_max"] / df["dry_spell_max"].std()
        df["awc_x_dry_spell"] = (1 - awc_norm) * dry_norm
        interactions_added += 1
        logger.info(
            f"  awc_x_dry_spell: range "
            f"[{df['awc_x_dry_spell'].min():.2f}, "
            f"{df['awc_x_dry_spell'].max():.2f}]"
        )

    if "sand_0_30cm" in df.columns and "dry_spell_max" in df.columns:
        sand_norm = df["sand_0_30cm"] / 100
        dry_norm = df["dry_spell_max"] / df["dry_spell_max"].std()
        df["sand_x_drought"] = sand_norm * dry_norm
        interactions_added += 1

    if "phh2o_0_30cm" in df.columns and "is_sul" in df.columns:
        is_cerrado = 1 - df["is_sul"]
        ph_deficit = 6.0 - df["phh2o_0_30cm"].fillna(6.0)
        df["ph_x_cerrado"] = ph_deficit.clip(lower=0) * is_cerrado
        interactions_added += 1
        logger.info(
            f"  ph_x_cerrado: range "
            f"[{df['ph_x_cerrado'].min():.2f}, "
            f"{df['ph_x_cerrado'].max():.2f}]"
        )

    if "soc_0_30cm" in df.columns and "hot_days_anomaly" in df.columns:
        soc_norm = df["soc_0_30cm"] / df["soc_0_30cm"].max()
        df["soc_x_heat_stress"] = (1 - soc_norm) * df["hot_days_anomaly"].fillna(0)
        interactions_added += 1

    if "sand_0_30cm" in df.columns and "is_la_nina" in df.columns and "is_sul" in df.columns:
        sand_norm = df["sand_0_30cm"] / 100
        df["sand_x_la_nina_sul"] = sand_norm * df["is_la_nina"] * df["is_sul"]
        interactions_added += 1
        logger.info(
            f"  sand_x_la_nina_sul: {(df['sand_x_la_nina_sul'] > 0).sum():,} casos"
        )

    if "cec_0_30cm" in df.columns:
        cec_norm = (df["cec_0_30cm"] - df["cec_0_30cm"].min()) / (
            df["cec_0_30cm"].max() - df["cec_0_30cm"].min() + 0.001
        )
        df["cec_normalized"] = cec_norm
        interactions_added += 1

    if "awc_estimated" in df.columns and "water_deficit_mm" in df.columns:
        awc_norm = df["awc_estimated"] / (df["awc_estimated"].max() + 1e-8)
        deficit_norm = df["water_deficit_mm"] / (df["water_deficit_mm"].std() + 1e-8)
        df["awc_x_deficit"] = (1 - awc_norm) * deficit_norm
        interactions_added += 1
        logger.info(f"  awc_x_deficit: range [{df['awc_x_deficit'].min():.2f}, {df['awc_x_deficit'].max():.2f}]")

    if "sand_0_30cm" in df.columns and "water_deficit_mm" in df.columns:
        sand_norm = df["sand_0_30cm"] / 100
        deficit_norm = df["water_deficit_mm"] / (df["water_deficit_mm"].std() + 1e-8)
        df["sand_x_deficit"] = sand_norm * deficit_norm
        interactions_added += 1

    logger.info(f"  Total de interacoes solo x clima adicionadas: {interactions_added}")

    return df


def calculate_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features historicas de produtividade."""
    logger.info("Calculando features historicas...")

    df = df.copy()

    df = df.sort_values(["cod_ibge", "ano"])

    df["produtividade_lag1"] = df.groupby("cod_ibge")["produtividade_kg_ha"].shift(1)

    df["produtividade_ma3"] = (
        df.groupby("cod_ibge")["produtividade_kg_ha"]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ano_min = df["ano"].min()
    ano_max = df["ano"].max()
    df["trend"] = (df["ano"] - ano_min) / (ano_max - ano_min)

    logger.info("  Features historicas calculadas")

    return df


def merge_features_and_target(df_climate: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Junta features climaticas com target de produtividade."""
    logger.info("Juntando features climaticas com target...")

    df = pd.merge(df_target, df_climate, on=["cod_ibge", "ano"], how="inner")

    logger.info(f"  Registros apos merge: {len(df):,}")

    return df


def validate_no_leakage(df: pd.DataFrame) -> bool:
    """Valida que nao ha leakage temporal no dataset."""
    logger.info("Validando ausencia de leakage temporal...")

    issues = []

    first_year_by_mun = df.groupby("cod_ibge")["ano"].min()

    for cod_ibge, first_year in first_year_by_mun.items():
        mask = (df["cod_ibge"] == cod_ibge) & (df["ano"] == first_year)
        lag1_value = df.loc[mask, "produtividade_lag1"].values

        if len(lag1_value) > 0 and not pd.isna(lag1_value[0]):
            issues.append(f"Municipio {cod_ibge}: lag1 nao e NaN no primeiro ano")

    if issues:
        for issue in issues[:5]:
            logger.warning(f"  {issue}")
        logger.warning(f"  Total de problemas: {len(issues)}")
        return False

    logger.info("  [OK] Nenhum leakage detectado")
    return True


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcula estatisticas do dataset final."""
    stats = {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "anos": df["ano"].nunique(),
        "ano_min": int(df["ano"].min()),
        "ano_max": int(df["ano"].max()),
        "produtividade_media": df["produtividade_kg_ha"].mean(),
        "produtividade_mediana": df["produtividade_kg_ha"].median(),
        "n_features": len(
            [
                c
                for c in df.columns
                if c
                not in ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]
            ]
        ),
        "missing_por_coluna": df.isnull().sum().to_dict(),
    }
    return stats


def main():
    """Pipeline principal de feature engineering."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING v2.0 (com Fase 1 de melhorias)")
    logger.info("=" * 60)

    config = load_config()
    default_phenology = get_default_phenology(config)
    regional_phenology = get_regional_phenology(config)

    start_month = default_phenology["start_month"]
    end_month = default_phenology["end_month"]

    base_temp = 10.0
    hot_threshold = 32.0
    for feat in config["features"]["climate_features"]:
        if feat["name"] == "gdd_accumulated":
            base_temp = feat.get("base_temp", 10.0)
        if feat["name"] == "hot_days_count":
            hot_threshold = feat.get("threshold", 32.0)

    logger.info(f"Janela fenologica default: mes {start_month} a {end_month}")
    logger.info(f"Janelas regionais definidas: {len(regional_phenology)} UFs")
    logger.info(f"Temperatura base GDD: {base_temp}C")
    logger.info(f"Threshold dias quentes: {hot_threshold}C")

    df_climate = load_climate_data()
    df_target = load_target_data()
    df_enso = load_enso_data()
    df_soil = load_soil_data()
    df_ndvi = load_ndvi_data()

    if regional_phenology:
        df_climate_window = filter_phenology_window_regional(
            df_climate, regional_phenology, default_phenology
        )
    else:
        df_climate_window = filter_phenology_window(df_climate, start_month, end_month)

    df_climate_agg = aggregate_climate_features_by_phase(
        df_climate_window, base_temp, hot_threshold
    )

    df_climate_agg = add_enso_features(df_climate_agg, df_enso)

    df = merge_features_and_target(df_climate_agg, df_target)

    df = calculate_historical_features(df)

    df = calculate_climate_anomalies(df, min_years=5)

    df = add_enso_interactions(df)

    df = add_regional_features(df)

    df = add_soil_features(df, df_soil)

    df = add_soil_climate_interactions(df)

    df = add_ndvi_features(df, df_ndvi)
    df = add_ndvi_climate_interactions(df)

    validate_no_leakage(df)

    key_cols = ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]

    climate_agg_cols = [
        "precip_total_mm",
        "tmean_avg",
        "tmin_avg",
        "tmax_avg",
        "hot_days_count",
        "gdd_accumulated",
    ]

    phase_cols = []
    for phase in ["plantio", "vegetativo", "enchimento"]:
        phase_cols.extend(
            [
                f"precip_{phase}_mm",
                f"tmean_{phase}",
                f"tmin_{phase}",
                f"tmax_{phase}",
                f"hot_days_{phase}",
                f"gdd_{phase}",
            ]
        )

    drought_cols = [
        "dry_spell_max",
        "dry_spell_count_7d",
        "dry_spell_count_10d",
        "precip_cv",
        "precip_days_gt1mm",
    ]

    water_balance_cols = [
        "eto_total_mm",
        "eto_mean_mm",
        "water_deficit_mm",
        "water_deficit_ratio",
        "radiation_mean",
        "radiation_total",
        "eto_plantio_mm",
        "deficit_plantio_mm",
        "eto_vegetativo_mm",
        "deficit_vegetativo_mm",
        "eto_enchimento_mm",
        "deficit_enchimento_mm",
        "deficit_ratio_enchimento",
    ]

    enso_cols = ["oni_avg", "oni_min", "oni_max", "oni_std", "is_la_nina", "is_el_nino"]

    hist_cols = ["produtividade_lag1", "produtividade_ma3", "trend"]

    anomaly_cols = [
        "precip_anomaly",
        "temp_anomaly",
        "hot_days_anomaly",
        "gdd_anomaly",
        "precip_enchimento_anomaly",
        "dry_spell_anomaly",
    ]

    interaction_cols = [
        "la_nina_x_precip_ench_anom",
        "dry_spell_x_hot_anom",
        "la_nina_x_precip_anom",
        "heat_drought_stress",
        "enchimento_stress",
        "el_nino_x_precip_anom",
        "la_nina_x_deficit",
        "terminal_drought_stress",
    ]

    regional_cols = ["is_sul", "sul_x_la_nina", "sul_x_precip_anomaly", "sul_x_hot_days_anomaly"]

    soil_cols = [
        "clay_0_30cm",
        "sand_0_30cm",
        "silt_0_30cm",
        "phh2o_0_30cm",
        "soc_0_30cm",
        "nitrogen_0_30cm",
        "cec_0_30cm",
        "bdod_0_30cm",
        "clay_30_100cm",
        "sand_30_100cm",
        "phh2o_30_100cm",
        "clay_sand_ratio",
        "awc_estimated",
        "ph_acidic",
        "texture_class",
        "soil_quality_index",
    ]

    soil_interaction_cols = [
        "clay_x_precip_deficit",
        "awc_x_dry_spell",
        "sand_x_drought",
        "ph_x_cerrado",
        "soc_x_heat_stress",
        "sand_x_la_nina_sul",
        "cec_normalized",
        "awc_x_deficit",
        "sand_x_deficit",
    ]

    all_cols = (
        key_cols
        + climate_agg_cols
        + phase_cols
        + drought_cols
        + water_balance_cols
        + enso_cols
        + hist_cols
        + anomaly_cols
        + interaction_cols
        + regional_cols
        + soil_cols
        + soil_interaction_cols
    )
    cols_order = [c for c in all_cols if c in df.columns]
    df = df[cols_order]

    stats = calculate_statistics(df)
    logger.info("\n" + "=" * 60)
    logger.info("ESTATISTICAS DO DATASET FINAL v2.0")
    logger.info("=" * 60)
    logger.info(f"Total de registros: {stats['total_registros']:,}")
    logger.info(f"Municipios unicos: {stats['municipios_unicos']:,}")
    logger.info(f"Anos: {stats['anos']} ({stats['ano_min']} - {stats['ano_max']})")
    logger.info(f"Numero de features: {stats['n_features']}")
    logger.info(f"Produtividade media: {stats['produtividade_media']:.1f} kg/ha")
    logger.info(f"Produtividade mediana: {stats['produtividade_mediana']:.1f} kg/ha")

    logger.info("\nMissing por coluna (top 10):")
    missing_sorted = sorted(stats["missing_por_coluna"].items(), key=lambda x: x[1], reverse=True)
    for col, missing in missing_sorted[:10]:
        if missing > 0:
            pct = missing / len(df) * 100
            logger.info(f"  {col}: {missing:,} ({pct:.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("NOVAS FEATURES (Fase 1)")
    logger.info("=" * 60)
    logger.info("Janelas fenologicas quebradas:")
    logger.info("  - precip_plantio_mm, precip_vegetativo_mm, precip_enchimento_mm")
    logger.info("  - tmean/tmin/tmax por fase")
    logger.info("  - hot_days e GDD por fase")
    logger.info("\nMetricas de veranico:")
    logger.info(f"  - dry_spell_max: max={df['dry_spell_max'].max()} dias")
    logger.info(f"  - dry_spell_count_7d: media={df['dry_spell_count_7d'].mean():.1f}")
    logger.info(f"  - precip_cv: media={df['precip_cv'].mean():.2f}")
    logger.info("\nFeatures ENSO:")
    if "oni_avg" in df.columns:
        logger.info(f"  - oni_avg: range [{df['oni_avg'].min():.2f}, {df['oni_avg'].max():.2f}]")
        logger.info(f"  - Anos La Nina: {df['is_la_nina'].sum():,}")
        logger.info(f"  - Anos El Nino: {df['is_el_nino'].sum():,}")

    logger.info("\n" + "=" * 60)
    logger.info("NOVAS FEATURES (Fase 2)")
    logger.info("=" * 60)

    logger.info("Features de anomalia climatica:")
    for col in anomaly_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                logger.info(
                    f"  - {col}: range [{df[col].min():.2f}, {df[col].max():.2f}], "
                    f"{valid:,} valores validos"
                )

    logger.info("\nFeatures regionais Sul:")
    if "is_sul" in df.columns:
        n_sul = df["is_sul"].sum()
        logger.info(f"  - is_sul: {n_sul:,} registros ({n_sul/len(df)*100:.1f}%)")
    if "sul_x_la_nina" in df.columns:
        logger.info(f"  - sul_x_la_nina: {df['sul_x_la_nina'].sum():,} casos")
    if "sul_x_precip_anomaly" in df.columns:
        valid = df["sul_x_precip_anomaly"].notna().sum()
        logger.info(f"  - sul_x_precip_anomaly: {valid:,} valores validos")

    logger.info("\n" + "=" * 60)
    logger.info("NOVAS FEATURES (Fase 3 - Interacoes ENSO)")
    logger.info("=" * 60)

    for col in interaction_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                logger.info(
                    f"  - {col}: range [{df[col].min():.2f}, {df[col].max():.2f}], "
                    f"{valid:,} valores"
                )

    logger.info("\n" + "=" * 60)
    logger.info("NOVAS FEATURES (Fase 4 - Solo SoilGrids)")
    logger.info("=" * 60)

    logger.info("Features de solo diretas:")
    for col in ["clay_0_30cm", "sand_0_30cm", "phh2o_0_30cm", "soc_0_30cm", "cec_0_30cm"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                logger.info(
                    f"  - {col}: mean={df[col].mean():.2f}, "
                    f"range [{df[col].min():.2f}, {df[col].max():.2f}], "
                    f"{valid:,} valores"
                )

    logger.info("\nFeatures de solo derivadas:")
    for col in ["clay_sand_ratio", "awc_estimated", "soil_quality_index"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                logger.info(
                    f"  - {col}: mean={df[col].mean():.2f}, "
                    f"range [{df[col].min():.2f}, {df[col].max():.2f}]"
                )

    if "ph_acidic" in df.columns:
        n_acidic = df["ph_acidic"].sum()
        logger.info(f"  - ph_acidic: {n_acidic:,} municipios com solo acido")

    if "texture_class" in df.columns:
        logger.info("  - texture_class distribuicao:")
        for tex, count in df["texture_class"].value_counts().items():
            logger.info(f"      {tex}: {count:,} ({count/len(df)*100:.1f}%)")

    logger.info("\nInteracoes solo x clima:")
    for col in soil_interaction_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0:
                logger.info(
                    f"  - {col}: range [{df[col].min():.2f}, {df[col].max():.2f}], "
                    f"{valid:,} valores"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nDataset salvo em: {OUTPUT_PATH}")

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING v5.0 CONCLUIDO!")
    logger.info(f"Total de features: {stats['n_features']}")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    main()
