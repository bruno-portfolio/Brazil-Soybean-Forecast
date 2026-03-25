from __future__ import annotations

import pandas as pd


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
