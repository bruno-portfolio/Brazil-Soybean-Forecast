from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.common.constants import REGION_SUL

logger = logging.getLogger(__name__)


def calculate_climate_anomalies(df: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Calcula features de anomalia climatica (desvios da normal historica).

    Para cada municipio, calcula z-score usando expanding window dos anos
    anteriores (shift(1) previne leakage). Requer min_years de historico.
    """
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


def add_enso_features(df: pd.DataFrame, df_enso: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features ENSO ao dataset (merge + flags la_nina/el_nino)."""
    if df_enso is None:
        logger.warning("Dados ENSO nao disponiveis. Pulando...")
        return df

    logger.info("Adicionando features ENSO...")

    enso_cols = ["ano", "oni_avg", "oni_min", "oni_max", "oni_std"]
    available_cols = [c for c in enso_cols if c in df_enso.columns]
    df_enso_num = df_enso[available_cols].copy()

    if "enso_phase" in df_enso.columns:
        df_enso_num["is_la_nina"] = (df_enso["enso_phase"] == "nina").astype(int)
        df_enso_num["is_el_nino"] = (df_enso["enso_phase"] == "nino").astype(int)

    df = df.merge(df_enso_num, on="ano", how="left")

    logger.info("  Features ENSO adicionadas")

    return df


def add_enso_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona interacoes nao-lineares com ENSO para capturar eventos extremos.

    Features condicionais (la_nina_x_deficit, terminal_drought_stress) so sao
    geradas quando water_deficit_mm/deficit_enchimento_mm existem no DataFrame
    (pipeline de treinamento com balanco hidrico).
    """
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
        std = df["dry_spell_max"].std()
        dry_spell_norm = df["dry_spell_max"] / std if std > 0 else df["dry_spell_max"]
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
        if precip_ench_mean > 0:
            precip_ench_ratio = df["precip_enchimento_mm"] / precip_ench_mean
            precip_deficit = 1 - precip_ench_ratio.clip(0, 2)
            df["enchimento_stress"] = df["hot_days_enchimento"] * precip_deficit
        else:
            df["enchimento_stress"] = 0
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
        logger.info(
            f"  la_nina_x_deficit: range "
            f"[{df['la_nina_x_deficit'].min():.2f}, "
            f"{df['la_nina_x_deficit'].max():.2f}]"
        )

    if "deficit_enchimento_mm" in df.columns and "hot_days_enchimento" in df.columns:
        deficit_ench_norm = df["deficit_enchimento_mm"] / (df["deficit_enchimento_mm"].std() + 1e-8)
        hot_ench_norm = df["hot_days_enchimento"] / (df["hot_days_enchimento"].std() + 1e-8)
        df["terminal_drought_stress"] = deficit_ench_norm * hot_ench_norm
        logger.info(
            f"  terminal_drought_stress: range "
            f"[{df['terminal_drought_stress'].min():.2f}, "
            f"{df['terminal_drought_stress'].max():.2f}]"
        )

    logger.info("  Interacoes ENSO adicionadas")

    return df


def add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de tratamento regional para o Sul do Brasil."""
    logger.info("Adicionando features de tratamento regional...")

    df = df.copy()

    df["uf_cod"] = df["cod_ibge"].astype(str).str[:2].astype(int)

    df["is_sul"] = df["uf_cod"].isin(REGION_SUL).astype(int)

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
    logger.info(f"  Registros do Sul: {n_sul:,} ({n_sul / len(df) * 100:.1f}%)")

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
        logger.info(
            f"  Registros com dados de solo: {n_with_soil:,} ({n_with_soil / n_total * 100:.1f}%)"
        )

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
        logger.info(f"  sand_x_la_nina_sul: {(df['sand_x_la_nina_sul'] > 0).sum():,} casos")

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
        logger.info(
            f"  awc_x_deficit: range [{df['awc_x_deficit'].min():.2f}, {df['awc_x_deficit'].max():.2f}]"
        )

    if "sand_0_30cm" in df.columns and "water_deficit_mm" in df.columns:
        sand_norm = df["sand_0_30cm"] / 100
        deficit_norm = df["water_deficit_mm"] / (df["water_deficit_mm"].std() + 1e-8)
        df["sand_x_deficit"] = sand_norm * deficit_norm
        interactions_added += 1

    logger.info(f"  Total de interacoes solo x clima adicionadas: {interactions_added}")

    return df
