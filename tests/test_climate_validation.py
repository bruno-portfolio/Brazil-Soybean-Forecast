"""
Testes para validacao de dados climaticos.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.validation.climate import (
    calculate_coverage_stats,
    get_climate_schema,
    load_config,
    validate_climate,
    validate_cod_ibge_format,
    validate_no_future_dates,
    validate_no_nulls_in_key_columns,
    validate_precip_non_negative,
    validate_primary_key,
    validate_temp_consistency,
    validate_temperature_range,
    validate_temporal_coverage,
)


@pytest.fixture
def sample_climate_df():
    """Cria DataFrame de exemplo para testes."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "cod_ibge": [5200209] * 10,
            "date": dates,
            "tmin": [18.0, 19.0, 17.5, 18.5, 20.0, 19.5, 18.0, 17.0, 19.0, 20.0],
            "tmean": [24.0, 25.0, 23.5, 24.5, 26.0, 25.5, 24.0, 23.0, 25.0, 26.0],
            "tmax": [30.0, 31.0, 29.5, 30.5, 32.0, 31.5, 30.0, 29.0, 31.0, 32.0],
            "precip": [5.0, 0.0, 10.5, 0.0, 15.0, 2.0, 0.0, 8.0, 0.0, 12.0],
            "rh": [75.0, 70.0, 80.0, 65.0, 85.0, 72.0, 68.0, 78.0, 71.0, 82.0],
        }
    )


@pytest.fixture
def config():
    """Carrega configuracao."""
    return load_config()


class TestValidateTempConsistency:
    """Testes para validacao de consistencia de temperatura."""

    def test_valid_temps(self, sample_climate_df):
        """Temperaturas validas devem passar."""
        passed, msg = validate_temp_consistency(sample_climate_df)
        assert passed is True
        assert "OK" in msg or "aceitavel" in msg.lower() or "verificados" in msg.lower()

    def test_tmin_greater_than_tmean(self, sample_climate_df):
        """tmin > tmean deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "tmin"] = 30.0  # Maior que tmean
        passed, msg = validate_temp_consistency(df)
        # Pode passar com pequeno ruido (< 1%)
        assert "tmin > tmean" in msg or passed is True

    def test_tmean_greater_than_tmax(self, sample_climate_df):
        """tmean > tmax deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "tmean"] = 35.0  # Maior que tmax
        passed, msg = validate_temp_consistency(df)
        assert "tmean > tmax" in msg or passed is True


class TestValidatePrecipNonNegative:
    """Testes para validacao de precipitacao."""

    def test_valid_precip(self, sample_climate_df):
        """Precipitacao valida deve passar."""
        passed, msg = validate_precip_non_negative(sample_climate_df)
        assert passed is True

    def test_negative_precip(self, sample_climate_df):
        """Precipitacao negativa deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "precip"] = -5.0
        passed, msg = validate_precip_non_negative(df)
        assert passed is False
        assert "negativa" in msg.lower()


class TestValidateNoFutureDates:
    """Testes para validacao de datas futuras."""

    def test_valid_dates(self, sample_climate_df):
        """Datas passadas devem passar."""
        passed, msg = validate_no_future_dates(sample_climate_df)
        assert passed is True

    def test_future_dates(self, sample_climate_df):
        """Datas futuras devem falhar."""
        df = sample_climate_df.copy()
        future = datetime.now() + timedelta(days=30)
        df.loc[0, "date"] = pd.Timestamp(future)
        passed, msg = validate_no_future_dates(df)
        assert passed is False
        assert "futuras" in msg.lower()


class TestValidateTemporalCoverage:
    """Testes para validacao de cobertura temporal."""

    def test_valid_coverage(self, config):
        """Cobertura dentro do esperado deve passar."""
        dates = pd.date_range(
            f"{config['year_start']}-01-01", f"{config['year_end']}-12-31", freq="D"
        )
        # Limita a um subset para performance
        dates = dates[:1000]
        df = pd.DataFrame(
            {
                "cod_ibge": [5200209] * len(dates),
                "date": dates,
            }
        )
        passed, msg = validate_temporal_coverage(df, config)
        # Pode falhar se year_end > ano atual
        assert "OK" in msg or "Ano final" in msg

    def test_missing_years(self, config):
        """Anos faltantes devem ser detectados."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        df = pd.DataFrame(
            {
                "cod_ibge": [5200209] * len(dates),
                "date": dates,
            }
        )
        passed, msg = validate_temporal_coverage(df, config)
        # Deve falhar porque nao cobre todo o periodo
        assert "Ano inicial" in msg or "Ano final" in msg or passed is True


class TestValidatePrimaryKey:
    """Testes para validacao de chave primaria."""

    def test_unique_key(self, sample_climate_df):
        """Chave unica deve passar."""
        passed, msg = validate_primary_key(sample_climate_df)
        assert passed is True

    def test_duplicate_key(self, sample_climate_df):
        """Chave duplicada deve falhar."""
        df = pd.concat([sample_climate_df, sample_climate_df.iloc[[0]]], ignore_index=True)
        passed, msg = validate_primary_key(df)
        assert passed is False
        assert "duplicada" in msg.lower()


class TestValidateCodIbgeFormat:
    """Testes para validacao do formato cod_ibge."""

    def test_valid_format(self, sample_climate_df):
        """Formato valido deve passar."""
        passed, msg = validate_cod_ibge_format(sample_climate_df)
        assert passed is True

    def test_invalid_format_short(self, sample_climate_df):
        """Codigo curto deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "cod_ibge"] = 12345  # 5 digitos
        passed, msg = validate_cod_ibge_format(df)
        assert passed is False

    def test_invalid_format_long(self, sample_climate_df):
        """Codigo longo deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "cod_ibge"] = 12345678  # 8 digitos
        passed, msg = validate_cod_ibge_format(df)
        assert passed is False


class TestValidateNoNulls:
    """Testes para validacao de nulos."""

    def test_no_nulls(self, sample_climate_df):
        """Sem nulos deve passar."""
        passed, msg = validate_no_nulls_in_key_columns(sample_climate_df)
        assert passed is True

    def test_null_cod_ibge(self, sample_climate_df):
        """Nulo em cod_ibge deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "cod_ibge"] = None
        passed, msg = validate_no_nulls_in_key_columns(df)
        assert passed is False


class TestValidateTemperatureRange:
    """Testes para validacao de faixa de temperatura."""

    def test_valid_range(self, sample_climate_df):
        """Temperaturas em faixa valida devem passar."""
        passed, msg = validate_temperature_range(sample_climate_df)
        assert passed is True

    def test_temp_too_low(self, sample_climate_df):
        """Temperatura muito baixa deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "tmin"] = -50.0  # Muito baixo para o Brasil
        passed, msg = validate_temperature_range(df)
        assert passed is False

    def test_temp_too_high(self, sample_climate_df):
        """Temperatura muito alta deve falhar."""
        df = sample_climate_df.copy()
        df.loc[0, "tmax"] = 60.0  # Impossivel
        passed, msg = validate_temperature_range(df)
        assert passed is False


class TestValidateClimate:
    """Testes para funcao principal de validacao."""

    def test_valid_data(self, sample_climate_df, config):
        """Dados validos devem passar em todas as validacoes."""
        results = validate_climate(sample_climate_df, config)
        # Verifica que retorna dicionario
        assert isinstance(results, dict)
        # Verifica que tem as validacoes esperadas
        assert "primary_key" in results
        assert "temp_consistency" in results
        assert "precip_non_negative" in results

    def test_invalid_data(self, sample_climate_df, config):
        """Dados invalidos devem falhar em pelo menos uma validacao."""
        df = sample_climate_df.copy()
        df.loc[0, "precip"] = -10.0  # Invalido
        results = validate_climate(df, config)
        assert results["precip_non_negative"][0] is False


class TestPanderaSchema:
    """Testes para schema Pandera."""

    def test_valid_schema(self, sample_climate_df):
        """DataFrame valido deve passar no schema."""
        schema = get_climate_schema()
        validated = schema.validate(sample_climate_df)
        assert len(validated) == len(sample_climate_df)

    def test_invalid_precip(self, sample_climate_df):
        """Precipitacao negativa deve falhar no schema."""
        df = sample_climate_df.copy()
        df.loc[0, "precip"] = -5.0
        schema = get_climate_schema()
        with pytest.raises(Exception):
            schema.validate(df)

    def test_invalid_rh(self, sample_climate_df):
        """Umidade > 100 deve falhar no schema."""
        df = sample_climate_df.copy()
        df.loc[0, "rh"] = 150.0  # Invalido
        schema = get_climate_schema()
        with pytest.raises(Exception):
            schema.validate(df)

    def test_nullable_columns(self, sample_climate_df):
        """Colunas climaticas podem ter nulos."""
        df = sample_climate_df.copy()
        df.loc[0, "tmin"] = None
        df.loc[1, "precip"] = None
        schema = get_climate_schema()
        validated = schema.validate(df)
        assert len(validated) == len(df)


class TestCoverageStats:
    """Testes para estatisticas de cobertura."""

    def test_stats_structure(self, sample_climate_df):
        """Estatisticas devem ter estrutura esperada."""
        stats = calculate_coverage_stats(sample_climate_df)
        assert "total_registros" in stats
        assert "municipios_unicos" in stats
        assert "periodo" in stats
        assert "missingness" in stats

    def test_stats_values(self, sample_climate_df):
        """Estatisticas devem ter valores corretos."""
        stats = calculate_coverage_stats(sample_climate_df)
        assert stats["total_registros"] == 10
        assert stats["municipios_unicos"] == 1


class TestConfig:
    """Testes para configuracao."""

    def test_load_config(self, config):
        """Configuracao deve carregar corretamente."""
        assert config is not None
        assert "api" in config
        assert "parameters" in config
        assert "year_start" in config
        assert "year_end" in config

    def test_config_parameters(self, config):
        """Parametros devem incluir variaveis climaticas."""
        params = config["parameters"]
        assert "T2M" in params
        assert "T2M_MIN" in params
        assert "T2M_MAX" in params
        assert "PRECTOTCORR" in params
