from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PHASES = ["plantio", "vegetativo", "enchimento"]


def aggregate_season_climate(df_season: pd.DataFrame, phases: list[str] = PHASES) -> dict:
    """Agrega features climaticas base para uma safra de um municipio.

    Retorna dict com features por fase + totais + dry spell + variabilidade.
    Nao inclui water balance (adicionado separadamente em build_features).
    """
    result = {}

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

    return result


def assign_crop_year(date: pd.Timestamp, start_month: int = 10, end_month: int = 3) -> int | None:
    """Atribui o ano da safra (colheita) para uma data.

    Safra brasileira: plantio ~outubro, colheita ~marco.
    Data em outubro/2023 → safra 2024. Data em fevereiro/2024 → safra 2024.
    """
    if date.month >= start_month:
        return date.year + 1
    elif date.month <= end_month:
        return date.year
    return None


def assign_phenology_phase(date: pd.Timestamp) -> str | None:
    """Atribui fase fenologica para uma data (plantio/vegetativo/enchimento)."""
    month = date.month
    if month in [10, 11]:
        return "plantio"
    elif month in [12, 1]:
        return "vegetativo"
    elif month in [2, 3]:
        return "enchimento"
    return None


def calculate_gdd(row: pd.Series, base_temp: float = 10.0) -> float:
    """Calcula Growing Degree Days (GDD) para um dia."""
    return max(0, row["tmean"] - base_temp)


def calculate_dry_spell_metrics(df_group: pd.DataFrame, threshold_mm: float = 2.0) -> dict:
    """Calcula metricas de veranico (sequencia de dias secos) para um grupo."""
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


def assign_phenology_phase_regional(date: pd.Timestamp, phases: dict) -> str | None:
    """Atribui a fase fenologica para uma data usando configuracao regional."""
    month = date.month

    for phase_name, months in phases.items():
        if month in months:
            return phase_name

    return None


def filter_phenology_window(df: pd.DataFrame, start_month: int, end_month: int) -> pd.DataFrame:
    """Filtra dados de clima para a janela fenologica default."""
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
