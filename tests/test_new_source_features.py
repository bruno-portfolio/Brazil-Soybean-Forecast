"""Testes para src/common/new_source_features.py."""

import numpy as np
import pandas as pd

from src.common.new_source_features import (
    add_irrigacao_features,
    add_new_source_interactions,
    add_sinistro_features,
    add_uso_solo_features,
)


def _make_df(n=10):
    """Cria DataFrame base para testes."""
    return pd.DataFrame(
        {
            "cod_ibge": [5107602] * n,
            "ano": list(range(2014, 2014 + n)),
            "produtividade_kg_ha": np.random.uniform(2500, 3500, n),
            "area_colhida_ha": np.random.uniform(5000, 50000, n),
            "precip_anomaly": np.random.normal(0, 1, n),
            "is_la_nina": np.random.choice([0, 1], n),
            "water_deficit_mm": np.random.uniform(0, 200, n),
        }
    )


class TestAddIrrigacao:
    def test_returns_unchanged_when_no_file(self, tmp_path, monkeypatch):
        import src.common.new_source_features as mod

        monkeypatch.setattr(mod, "IRRIGACAO_PATH", tmp_path / "nao_existe.parquet")
        df = _make_df()
        result = add_irrigacao_features(df)
        assert "pct_irrigado" not in result.columns
        assert len(result) == len(df)

    def test_adds_pct_irrigado(self, tmp_path, monkeypatch):
        import src.common.new_source_features as mod

        df_irr = pd.DataFrame({"cod_ibge": [5107602], "area_irrigada_ha": [10000.0]})
        path = tmp_path / "pivos.parquet"
        df_irr.to_parquet(path)
        monkeypatch.setattr(mod, "IRRIGACAO_PATH", path)

        df = _make_df(3)
        result = add_irrigacao_features(df)
        assert "pct_irrigado" in result.columns
        assert (result["pct_irrigado"] >= 0).all()
        assert (result["pct_irrigado"] <= 1).all()


class TestAddSinistro:
    def test_returns_unchanged_when_no_file(self, tmp_path, monkeypatch):
        import src.common.new_source_features as mod

        monkeypatch.setattr(mod, "SINISTRO_PATH", tmp_path / "nao_existe.parquet")
        df = _make_df()
        result = add_sinistro_features(df)
        assert "sinistro_rate_3yr" not in result.columns

    def test_adds_sinistro_and_fills_zero(self, tmp_path, monkeypatch):
        import src.common.new_source_features as mod

        df_sin = pd.DataFrame({"cod_ibge": [5107602], "ano": [2015], "sinistro_rate_3yr": [0.3]})
        path = tmp_path / "sinistro.parquet"
        df_sin.to_parquet(path)
        monkeypatch.setattr(mod, "SINISTRO_PATH", path)

        df = _make_df(3)
        result = add_sinistro_features(df)
        assert "sinistro_rate_3yr" in result.columns
        assert (result["sinistro_rate_3yr"] >= 0).all()


class TestAddUsoSolo:
    def test_returns_unchanged_when_no_file(self, tmp_path, monkeypatch):
        import src.common.new_source_features as mod

        monkeypatch.setattr(mod, "MAPBIOMAS_PATH", tmp_path / "nao_existe.parquet")
        df = _make_df()
        result = add_uso_solo_features(df)
        assert "pct_soja" not in result.columns


class TestNewSourceInteractions:
    def test_creates_irrigacao_x_deficit(self):
        df = _make_df(5)
        df["pct_irrigado"] = 0.5
        result = add_new_source_interactions(df)
        assert "irrigacao_x_deficit" in result.columns

    def test_creates_sinistro_x_la_nina(self):
        df = _make_df(5)
        df["sinistro_rate_3yr"] = 0.2
        result = add_new_source_interactions(df)
        assert "sinistro_x_la_nina" in result.columns

    def test_skips_when_columns_missing(self):
        df = pd.DataFrame({"cod_ibge": [1], "ano": [2020]})
        result = add_new_source_interactions(df)
        assert "irrigacao_x_deficit" not in result.columns
        assert "sinistro_x_la_nina" not in result.columns
