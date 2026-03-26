"""Balanco hidrico: ETo (Hargreaves-Samani / Penman-Monteith) e deficit."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LATITUDE_DEFAULT = -15.0


def calculate_eto_hargreaves(row: pd.Series, lat: float = LATITUDE_DEFAULT) -> float:
    """Calcula ETo pelo metodo Hargreaves-Samani (mm/dia).

    Parameters
    ----------
    row : pd.Series
        Deve conter 'tmin', 'tmax', 'tmean' e 'date'.
    lat : float
        Latitude do ponto (graus decimais, negativo para hemisferio sul).
    """
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
    Ra = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )

    eto = 0.0023 * (tmean + 17.8) * np.sqrt(max(0, tmax - tmin)) * Ra
    return max(0, eto)


def calculate_eto_with_radiation(row: pd.Series, lat: float = LATITUDE_DEFAULT) -> float:
    """Calcula ETo usando radiacao solar (Penman-Monteith simplificado).

    Fallback para Hargreaves-Samani se radiacao nao disponivel.
    """
    radiation = row.get("radiation")
    tmean = row.get("tmean")
    rh = row.get("rh", 60)
    wind = row.get("wind_speed", 2.0)

    if pd.isna(radiation) or pd.isna(tmean):
        return calculate_eto_hargreaves(row, lat)

    Rs = radiation

    delta_slope = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)

    gamma = 0.066

    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    ea = es * rh / 100

    Rn = 0.77 * Rs - 2.0

    eto = (0.408 * delta_slope * Rn + gamma * (900 / (tmean + 273)) * wind * (es - ea)) / (
        delta_slope + gamma * (1 + 0.34 * wind)
    )

    return max(0, eto)


def calculate_water_balance_metrics(df_group: pd.DataFrame, lat: float = LATITUDE_DEFAULT) -> dict:
    """Calcula metricas de balanco hidrico (ETo, deficit) para uma safra.

    Returns
    -------
    dict com chaves: eto_total_mm, eto_mean_mm, water_deficit_mm,
    water_deficit_ratio, radiation_mean, radiation_total.
    """
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

    df_group = df_group.copy()
    if has_radiation:
        df_group["eto"] = df_group.apply(lambda row: calculate_eto_with_radiation(row, lat), axis=1)
        result["radiation_mean"] = df_group["radiation"].mean()
        result["radiation_total"] = df_group["radiation"].sum()
    else:
        df_group["eto"] = df_group.apply(lambda row: calculate_eto_hargreaves(row, lat), axis=1)

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


def calculate_water_balance_by_phase(
    df_season: pd.DataFrame, phases: list[str], lat: float = LATITUDE_DEFAULT
) -> dict:
    """Calcula balanco hidrico por fase fenologica.

    Returns
    -------
    dict com chaves: eto_{phase}_mm, deficit_{phase}_mm para cada fase,
    e deficit_ratio_enchimento.
    """
    result = {}

    for phase in phases:
        df_phase = df_season[df_season["phase"] == phase]
        wb = calculate_water_balance_metrics(df_phase, lat)

        result[f"eto_{phase}_mm"] = wb["eto_total_mm"]
        result[f"deficit_{phase}_mm"] = wb["water_deficit_mm"]

        if phase == "enchimento":
            result["deficit_ratio_enchimento"] = wb["water_deficit_ratio"]

    return result
