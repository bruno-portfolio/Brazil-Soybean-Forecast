import logging

import pandas as pd

from src.common.features import (
    add_enso_features,
    add_enso_interactions,
    add_regional_features,
    add_soil_climate_interactions,
    add_soil_features,
    calculate_climate_anomalies,
)
from src.common.io import PROJECT_ROOT, load_config
from src.common.phenology import (
    filter_phenology_window,
    filter_phenology_window_regional,
    get_default_phenology,
    get_regional_phenology,
)

logger = logging.getLogger(__name__)
CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"
CLIMATE_V2_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily_v2.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
ENSO_PATH = PROJECT_ROOT / "data" / "processed" / "oni_enso.parquet"
SOIL_PATH = PROJECT_ROOT / "data" / "processed" / "soil_properties.parquet"
NDVI_PATH = PROJECT_ROOT / "data" / "processed" / "ndvi_safra.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"


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
        logger.warning(
            "  Arquivo de solo nao encontrado. Execute src/ingest/soilgrids.py primeiro."
        )
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
        logger.info(f"  Registros com NDVI: {n_with_ndvi:,} ({100 * n_with_ndvi / len(df):.1f}%)")

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


def calculate_historical_features(
    df: pd.DataFrame, trend_ref_min: int = 2000, trend_ref_max: int = 2025
) -> pd.DataFrame:
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

    df["trend"] = (df["ano"] - trend_ref_min) / (trend_ref_max - trend_ref_min)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING v2.0 (com Fase 1 de melhorias)")
    logger.info("=" * 60)

    config = load_config("features")
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

    trend_ref_min = config.get("features", {}).get("trend_ref_year_min", 2000)
    trend_ref_max = config.get("features", {}).get("trend_ref_year_max", 2025)

    df_climate = load_climate_data()
    df_target = load_target_data()
    df_enso = load_enso_data()
    df_soil = load_soil_data()
    df_ndvi = load_ndvi_data()

    from src.common.io import load_municipalities

    df_mun = load_municipalities(columns=["cod_ibge", "lat"])
    lat_lookup = dict(zip(df_mun["cod_ibge"], df_mun["lat"]))
    logger.info(f"Lat lookup: {len(lat_lookup)} municipios")

    if regional_phenology:
        df_climate_window = filter_phenology_window_regional(
            df_climate, regional_phenology, default_phenology
        )
    else:
        df_climate_window = filter_phenology_window(df_climate, start_month, end_month)

    from src.common.climate_aggregation import aggregate_climate_duckdb

    df_climate_agg = aggregate_climate_duckdb(
        df_climate_window, base_temp, hot_threshold, lat_lookup=lat_lookup
    )

    df_climate_agg = add_enso_features(df_climate_agg, df_enso)

    df = merge_features_and_target(df_climate_agg, df_target)

    df = calculate_historical_features(df, trend_ref_min, trend_ref_max)

    df = calculate_climate_anomalies(df, min_years=5)

    df = add_enso_interactions(df)

    df = add_regional_features(df)

    df = add_soil_features(df, df_soil)

    df = add_soil_climate_interactions(df)

    df = add_ndvi_features(df, df_ndvi)
    df = add_ndvi_climate_interactions(df)

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

    new_source_cols = [
        "pct_irrigado",
        "fert_total_br_ton",
        "sinistro_rate_3yr",
        "pct_soja",
    ]

    new_source_interaction_cols = [
        "irrigacao_x_deficit",
        "fert_x_precip",
        "sinistro_x_la_nina",
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
        + new_source_cols
        + new_source_interaction_cols
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
        logger.info(f"  - is_sul: {n_sul:,} registros ({n_sul / len(df) * 100:.1f}%)")
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
            logger.info(f"      {tex}: {count:,} ({count / len(df) * 100:.1f}%)")

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
