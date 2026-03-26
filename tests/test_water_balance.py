"""Testes para src/common/water_balance.py."""

import numpy as np
import pandas as pd

from src.common.water_balance import (
    LATITUDE_DEFAULT,
    calculate_eto_hargreaves,
    calculate_eto_with_radiation,
    calculate_water_balance_by_phase,
    calculate_water_balance_metrics,
)


def _make_row(tmean=25.0, tmin=20.0, tmax=30.0, date="2024-01-15", **kwargs):
    """Cria um pd.Series simulando uma linha de clima diario."""
    data = {"tmean": tmean, "tmin": tmin, "tmax": tmax, "date": pd.Timestamp(date)}
    data.update(kwargs)
    return pd.Series(data)


def _make_season_df(n_days=30, precip=5.0, tmean=25.0, start_date="2024-01-01"):
    """Cria DataFrame de uma safra com n_days de dados diarios."""
    dates = pd.date_range(start_date, periods=n_days)
    df = pd.DataFrame(
        {
            "date": dates,
            "tmean": tmean,
            "tmin": tmean - 5,
            "tmax": tmean + 5,
            "precip": precip,
            "phase": ["plantio"] * 10 + ["vegetativo"] * 10 + ["enchimento"] * 10
            if n_days == 30
            else ["plantio"] * n_days,
        }
    )
    return df


class TestEtoHargreaves:
    def test_returns_positive(self):
        row = _make_row()
        eto = calculate_eto_hargreaves(row)
        assert eto >= 0

    def test_returns_nan_when_tmin_nan(self):
        row = _make_row(tmin=np.nan)
        eto = calculate_eto_hargreaves(row)
        assert np.isnan(eto)

    def test_returns_nan_when_tmax_nan(self):
        row = _make_row(tmax=np.nan)
        eto = calculate_eto_hargreaves(row)
        assert np.isnan(eto)

    def test_latitude_varies_result(self):
        row = _make_row()
        eto_south = calculate_eto_hargreaves(row, lat=-30.0)
        eto_equator = calculate_eto_hargreaves(row, lat=-5.0)
        assert eto_south != eto_equator

    def test_default_latitude(self):
        row = _make_row()
        eto_default = calculate_eto_hargreaves(row)
        eto_explicit = calculate_eto_hargreaves(row, lat=LATITUDE_DEFAULT)
        assert eto_default == eto_explicit

    def test_higher_temp_higher_eto(self):
        row_cool = _make_row(tmean=15.0, tmin=10.0, tmax=20.0)
        row_hot = _make_row(tmean=35.0, tmin=30.0, tmax=40.0)
        assert calculate_eto_hargreaves(row_hot) > calculate_eto_hargreaves(row_cool)


class TestEtoWithRadiation:
    def test_fallback_to_hargreaves_when_no_radiation(self):
        row = _make_row(radiation=np.nan)
        eto_rad = calculate_eto_with_radiation(row)
        eto_harg = calculate_eto_hargreaves(row)
        assert eto_rad == eto_harg

    def test_uses_radiation_when_available(self):
        row = _make_row(radiation=20.0, rh=60, wind_speed=2.0)
        eto = calculate_eto_with_radiation(row)
        assert eto >= 0
        assert eto != calculate_eto_hargreaves(row)

    def test_returns_positive_with_radiation(self):
        row = _make_row(radiation=15.0, rh=70, wind_speed=1.5)
        assert calculate_eto_with_radiation(row) >= 0


class TestWaterBalanceMetrics:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["date", "tmean", "tmin", "tmax", "precip"])
        result = calculate_water_balance_metrics(df)
        assert result["eto_total_mm"] == 0
        assert result["water_deficit_mm"] == 0
        assert np.isnan(result["eto_mean_mm"])

    def test_deficit_when_eto_exceeds_precip(self):
        df = _make_season_df(n_days=10, precip=0.0, tmean=30.0)
        result = calculate_water_balance_metrics(df)
        assert result["water_deficit_mm"] > 0
        assert result["eto_total_mm"] > 0

    def test_no_deficit_when_precip_exceeds_eto(self):
        df = _make_season_df(n_days=10, precip=100.0, tmean=15.0)
        result = calculate_water_balance_metrics(df)
        assert result["water_deficit_mm"] == 0

    def test_lat_parameter_propagated(self):
        df = _make_season_df(n_days=10, precip=2.0)
        result_south = calculate_water_balance_metrics(df, lat=-30.0)
        result_equator = calculate_water_balance_metrics(df, lat=-5.0)
        assert result_south["eto_total_mm"] != result_equator["eto_total_mm"]

    def test_deficit_ratio_bounded(self):
        df = _make_season_df(n_days=10, precip=2.0)
        result = calculate_water_balance_metrics(df)
        if not np.isnan(result["water_deficit_ratio"]):
            assert 0 <= result["water_deficit_ratio"] <= 1


class TestWaterBalanceByPhase:
    def test_all_phases_present(self):
        df = _make_season_df(n_days=30)
        phases = ["plantio", "vegetativo", "enchimento"]
        result = calculate_water_balance_by_phase(df, phases)
        for phase in phases:
            assert f"eto_{phase}_mm" in result
            assert f"deficit_{phase}_mm" in result
        assert "deficit_ratio_enchimento" in result

    def test_lat_parameter_used(self):
        df = _make_season_df(n_days=30)
        phases = ["plantio", "vegetativo", "enchimento"]
        r1 = calculate_water_balance_by_phase(df, phases, lat=-30.0)
        r2 = calculate_water_balance_by_phase(df, phases, lat=-5.0)
        assert r1["eto_plantio_mm"] != r2["eto_plantio_mm"]

    def test_empty_phase(self):
        df = _make_season_df(n_days=10)
        df["phase"] = "plantio"
        phases = ["plantio", "vegetativo", "enchimento"]
        result = calculate_water_balance_by_phase(df, phases)
        assert result["eto_plantio_mm"] > 0
        assert result["eto_vegetativo_mm"] == 0
        assert result["eto_enchimento_mm"] == 0
