"""Balanco hidrico: ETo (Hargreaves-Samani / Penman-Monteith) e deficit.

Versao vetorizada com numpy para performance em datasets grandes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LATITUDE_DEFAULT = -15.0


def _eto_hargreaves_vec(
    tmean: np.ndarray, tmin: np.ndarray, tmax: np.ndarray, day_of_year: np.ndarray, lat: float
) -> np.ndarray:
    """Hargreaves-Samani vetorizado (numpy arrays)."""
    lat_rad = lat * np.pi / 180
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)

    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1, 1))
    Ra = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )

    temp_range = np.maximum(0, tmax - tmin)
    eto = 0.0023 * (tmean + 17.8) * np.sqrt(temp_range) * Ra
    return np.maximum(0, eto)


def _eto_penman_vec(
    tmean: np.ndarray, radiation: np.ndarray, rh: np.ndarray, wind: np.ndarray
) -> np.ndarray:
    """Penman-Monteith simplificado vetorizado."""
    delta_slope = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / ((tmean + 237.3) ** 2)
    gamma = 0.066
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    ea = es * rh / 100
    Rn = 0.77 * radiation - 2.0

    eto = (0.408 * delta_slope * Rn + gamma * (900 / (tmean + 273)) * wind * (es - ea)) / (
        delta_slope + gamma * (1 + 0.34 * wind)
    )
    return np.maximum(0, eto)


def compute_eto_column(df: pd.DataFrame, lat: float = LATITUDE_DEFAULT) -> pd.Series:
    """Calcula ETo vetorizado para todo o DataFrame de uma vez.

    Usa Penman-Monteith quando radiacao disponivel, Hargreaves caso contrario.
    """
    tmean = df["tmean"].values.astype(float)
    tmin_raw = df["tmin"].values.astype(float)
    tmax_raw = df["tmax"].values.astype(float)
    tmin = np.where(np.isnan(tmin_raw), tmean - 5, tmin_raw)
    tmax = np.where(np.isnan(tmax_raw), tmean + 5, tmax_raw)
    day_of_year = df["date"].dt.dayofyear.values.astype(float)

    eto = _eto_hargreaves_vec(tmean, tmin, tmax, day_of_year, lat)

    has_radiation = "radiation" in df.columns and df["radiation"].notna().any()
    if has_radiation:
        rad = df["radiation"].values.astype(float)
        rh = df["rh"].fillna(60).values.astype(float)
        wind = (
            df["wind_speed"].fillna(2.0).values.astype(float)
            if "wind_speed" in df.columns
            else np.full(len(df), 2.0)
        )
        mask = df["radiation"].notna().values & np.isfinite(rad)
        if mask.any():
            eto[mask] = _eto_penman_vec(tmean[mask], rad[mask], rh[mask], wind[mask])

    eto[np.isnan(tmin) | np.isnan(tmax)] = np.nan
    return pd.Series(eto, index=df.index)


# --- Funcoes row-level mantidas para compatibilidade com testes ---


def calculate_eto_hargreaves(row: pd.Series, lat: float = LATITUDE_DEFAULT) -> float:
    """Calcula ETo Hargreaves-Samani para uma unica linha."""
    tmin = row.get("tmin", row.get("tmean", 20) - 5)
    tmax = row.get("tmax", row.get("tmean", 20) + 5)
    tmean = row.get("tmean", (tmin + tmax) / 2)

    if pd.isna(tmin) or pd.isna(tmax):
        return np.nan

    doy = np.array([row["date"].dayofyear], dtype=float)
    result = _eto_hargreaves_vec(np.array([tmean]), np.array([tmin]), np.array([tmax]), doy, lat)
    return float(result[0])


def calculate_eto_with_radiation(row: pd.Series, lat: float = LATITUDE_DEFAULT) -> float:
    """Calcula ETo com radiacao para uma unica linha."""
    radiation = row.get("radiation")
    tmean = row.get("tmean")

    if pd.isna(radiation) or pd.isna(tmean):
        return calculate_eto_hargreaves(row, lat)

    rh = row.get("rh", 60)
    wind = row.get("wind_speed", 2.0)
    result = _eto_penman_vec(
        np.array([tmean]), np.array([radiation]), np.array([rh]), np.array([wind])
    )
    return float(result[0])


def calculate_water_balance_metrics(df_group: pd.DataFrame, lat: float = LATITUDE_DEFAULT) -> dict:
    """Calcula metricas de balanco hidrico (ETo, deficit) para uma safra."""
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

    eto = compute_eto_column(df_group, lat)

    has_radiation = "radiation" in df_group.columns and df_group["radiation"].notna().any()
    if has_radiation:
        result["radiation_mean"] = float(df_group["radiation"].mean())
        result["radiation_total"] = float(df_group["radiation"].sum())

    eto_valid = eto.dropna()
    if len(eto_valid) > 0:
        result["eto_total_mm"] = float(eto_valid.sum())
        result["eto_mean_mm"] = float(eto_valid.mean())

    precip = df_group["precip"].fillna(0).values
    eto_arr = eto.fillna(0).values
    deficit = np.maximum(0, eto_arr - precip)
    result["water_deficit_mm"] = float(deficit.sum())

    if result["eto_total_mm"] > 0:
        result["water_deficit_ratio"] = result["water_deficit_mm"] / result["eto_total_mm"]

    return result


def calculate_water_balance_by_phase(
    df_season: pd.DataFrame, phases: list[str], lat: float = LATITUDE_DEFAULT
) -> dict:
    """Calcula balanco hidrico por fase fenologica."""
    result = {}

    for phase in phases:
        df_phase = df_season[df_season["phase"] == phase]
        wb = calculate_water_balance_metrics(df_phase, lat)

        result[f"eto_{phase}_mm"] = wb["eto_total_mm"]
        result[f"deficit_{phase}_mm"] = wb["water_deficit_mm"]

        if phase == "enchimento":
            result["deficit_ratio_enchimento"] = wb["water_deficit_ratio"]

    return result
