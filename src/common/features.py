from __future__ import annotations

import logging

import pandas as pd

from src.common.constants import REGION_SUL

logger = logging.getLogger(__name__)


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
