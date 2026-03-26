"""Agregacao climatica vetorizada com DuckDB.

Substitui o loop Python sobre cod_ibge x crop_year por queries SQL.
Inclui: features por fase, totais, dry spell, variabilidade, water balance (ETo + deficit).
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

from src.common.water_balance import LATITUDE_DEFAULT, compute_eto_column

logger = logging.getLogger(__name__)


def aggregate_climate_duckdb(
    df: pd.DataFrame,
    base_temp: float = 10.0,
    hot_threshold: float = 32.0,
    lat_lookup: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Agrega features climaticas por (cod_ibge, crop_year) usando DuckDB.

    Substitui aggregate_climate_features_by_phase (build_features.py) e
    prepare_climate_features (predict.py) com performance ~100x melhor.
    """
    logger.info("Agregando features climaticas (DuckDB)...")

    df = df.copy()

    # 1. Colunas derivadas vetorizadas (numpy)
    df["gdd"] = np.maximum(0, df["tmean"].values - base_temp)
    df["is_hot_day"] = (df["tmax"] > hot_threshold).astype(int)
    df["is_dry"] = (df["precip"] < 2.0).astype(int)

    # Garantir colunas opcionais existam pra SQL
    if "radiation" not in df.columns:
        df["radiation"] = np.nan
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    # 2. ETo vetorizado per-municipio (numpy, batch por lat)
    _lat_lookup = lat_lookup or {}
    if _lat_lookup:
        unique_lats = {}
        for cod in df["cod_ibge"].unique():
            lat = _lat_lookup.get(cod, LATITUDE_DEFAULT)
            unique_lats.setdefault(lat, []).append(cod)

        eto_series = pd.Series(np.nan, index=df.index)
        for lat, codes in unique_lats.items():
            mask = df["cod_ibge"].isin(codes)
            eto_series[mask] = compute_eto_column(df[mask], lat).values
        df["eto"] = eto_series
    else:
        df["eto"] = compute_eto_column(df, LATITUDE_DEFAULT)

    df["deficit_day"] = np.maximum(0, df["eto"].fillna(0).values - df["precip"].fillna(0).values)

    logger.info(f"  Colunas derivadas calculadas para {len(df):,} registros")

    # 3. Agregacao com DuckDB
    con = duckdb.connect()
    con.register("climate", df)

    # 3a. Features por fase
    phase_sql = """
    SELECT
        cod_ibge, crop_year AS ano,
        -- Por fase
        SUM(CASE WHEN phase='plantio' THEN precip ELSE 0 END) AS precip_plantio_mm,
        AVG(CASE WHEN phase='plantio' THEN tmean END) AS tmean_plantio,
        AVG(CASE WHEN phase='plantio' THEN tmin END) AS tmin_plantio,
        AVG(CASE WHEN phase='plantio' THEN tmax END) AS tmax_plantio,
        SUM(CASE WHEN phase='plantio' THEN is_hot_day ELSE 0 END) AS hot_days_plantio,
        SUM(CASE WHEN phase='plantio' THEN gdd ELSE 0 END) AS gdd_plantio,

        SUM(CASE WHEN phase='vegetativo' THEN precip ELSE 0 END) AS precip_vegetativo_mm,
        AVG(CASE WHEN phase='vegetativo' THEN tmean END) AS tmean_vegetativo,
        AVG(CASE WHEN phase='vegetativo' THEN tmin END) AS tmin_vegetativo,
        AVG(CASE WHEN phase='vegetativo' THEN tmax END) AS tmax_vegetativo,
        SUM(CASE WHEN phase='vegetativo' THEN is_hot_day ELSE 0 END) AS hot_days_vegetativo,
        SUM(CASE WHEN phase='vegetativo' THEN gdd ELSE 0 END) AS gdd_vegetativo,

        SUM(CASE WHEN phase='enchimento' THEN precip ELSE 0 END) AS precip_enchimento_mm,
        AVG(CASE WHEN phase='enchimento' THEN tmean END) AS tmean_enchimento,
        AVG(CASE WHEN phase='enchimento' THEN tmin END) AS tmin_enchimento,
        AVG(CASE WHEN phase='enchimento' THEN tmax END) AS tmax_enchimento,
        SUM(CASE WHEN phase='enchimento' THEN is_hot_day ELSE 0 END) AS hot_days_enchimento,
        SUM(CASE WHEN phase='enchimento' THEN gdd ELSE 0 END) AS gdd_enchimento,

        -- Totais safra
        SUM(precip) AS precip_total_mm,
        AVG(tmean) AS tmean_avg,
        AVG(tmin) AS tmin_avg,
        AVG(tmax) AS tmax_avg,
        SUM(is_hot_day) AS hot_days_count,
        SUM(gdd) AS gdd_accumulated,

        -- Variabilidade precipitacao
        CASE WHEN AVG(precip) > 0 THEN STDDEV(precip) / AVG(precip) ELSE 0 END AS precip_cv,
        SUM(CASE WHEN precip > 1.0 THEN 1 ELSE 0 END) AS precip_days_gt1mm,

        -- Water balance safra
        SUM(eto) AS eto_total_mm,
        AVG(eto) AS eto_mean_mm,
        SUM(deficit_day) AS water_deficit_mm,
        CASE WHEN SUM(eto) > 0 THEN SUM(deficit_day) / SUM(eto) ELSE NULL END AS water_deficit_ratio,

        -- Radiacao
        AVG(radiation) AS radiation_mean,
        SUM(COALESCE(radiation, 0)) AS radiation_total,

        -- Water balance por fase
        SUM(CASE WHEN phase='plantio' THEN eto ELSE 0 END) AS eto_plantio_mm,
        SUM(CASE WHEN phase='plantio' THEN deficit_day ELSE 0 END) AS deficit_plantio_mm,
        SUM(CASE WHEN phase='vegetativo' THEN eto ELSE 0 END) AS eto_vegetativo_mm,
        SUM(CASE WHEN phase='vegetativo' THEN deficit_day ELSE 0 END) AS deficit_vegetativo_mm,
        SUM(CASE WHEN phase='enchimento' THEN eto ELSE 0 END) AS eto_enchimento_mm,
        SUM(CASE WHEN phase='enchimento' THEN deficit_day ELSE 0 END) AS deficit_enchimento_mm,
        CASE WHEN SUM(CASE WHEN phase='enchimento' THEN eto ELSE 0 END) > 0
             THEN SUM(CASE WHEN phase='enchimento' THEN deficit_day ELSE 0 END)
                  / SUM(CASE WHEN phase='enchimento' THEN eto ELSE 0 END)
             ELSE NULL END AS deficit_ratio_enchimento

    FROM climate
    WHERE crop_year IS NOT NULL
    GROUP BY cod_ibge, crop_year
    ORDER BY cod_ibge, crop_year
    """

    df_agg = con.execute(phase_sql).fetchdf()
    df_agg["ano"] = df_agg["ano"].astype(int)

    logger.info(f"  Agregacao DuckDB: {len(df_agg):,} registros")

    # 4. Dry spells (requer window function para sequencias consecutivas)
    dry_sql = """
    WITH ordered AS (
        SELECT cod_ibge, crop_year, date, is_dry,
               ROW_NUMBER() OVER (PARTITION BY cod_ibge, crop_year ORDER BY date) AS rn,
               SUM(CASE WHEN is_dry = 0 THEN 1 ELSE 0 END)
                   OVER (PARTITION BY cod_ibge, crop_year ORDER BY date) AS grp
        FROM climate
        WHERE crop_year IS NOT NULL
    ),
    spells AS (
        SELECT cod_ibge, crop_year, grp, COUNT(*) AS spell_len
        FROM ordered
        WHERE is_dry = 1
        GROUP BY cod_ibge, crop_year, grp
    )
    SELECT
        cod_ibge, crop_year AS ano,
        COALESCE(MAX(spell_len), 0) AS dry_spell_max,
        COALESCE(SUM(CASE WHEN spell_len >= 7 THEN 1 ELSE 0 END), 0) AS dry_spell_count_7d,
        COALESCE(SUM(CASE WHEN spell_len >= 10 THEN 1 ELSE 0 END), 0) AS dry_spell_count_10d
    FROM spells
    GROUP BY cod_ibge, crop_year
    ORDER BY cod_ibge, crop_year
    """

    df_dry = con.execute(dry_sql).fetchdf()
    df_dry["ano"] = df_dry["ano"].astype(int)

    # 5. Merge dry spells
    df_agg = df_agg.merge(df_dry, on=["cod_ibge", "ano"], how="left")
    df_agg["dry_spell_max"] = df_agg["dry_spell_max"].fillna(0).astype(int)
    df_agg["dry_spell_count_7d"] = df_agg["dry_spell_count_7d"].fillna(0).astype(int)
    df_agg["dry_spell_count_10d"] = df_agg["dry_spell_count_10d"].fillna(0).astype(int)

    con.close()

    logger.info(f"  Features finais: {len(df_agg.columns)} colunas")

    return df_agg
