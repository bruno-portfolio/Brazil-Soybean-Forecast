"""
Testes unitários para validação de dados geográficos.

Testa as funções de validação do módulo src/validation/geo.py
usando dados sintéticos e casos de borda.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.geo import (
    get_municipalities_schema,
    load_config,
    validate_cod_ibge_format,
    validate_lat_bounds,
    validate_lon_bounds,
    validate_municipalities,
    validate_municipality_count,
    validate_no_nulls,
    validate_ufs,
    validate_uniqueness,
)

# Fixtures


@pytest.fixture
def config():
    """Carrega configuração do YAML."""
    return load_config()


@pytest.fixture
def valid_df():
    """DataFrame válido com dados de exemplo."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, 3304557, 2927408, 4314902, 5300108],
            "nome": ["São Paulo", "Rio de Janeiro", "Salvador", "Porto Alegre", "Brasília"],
            "uf": ["SP", "RJ", "BA", "RS", "DF"],
            "lat": [-23.55, -22.91, -12.97, -30.03, -15.78],
            "lon": [-46.63, -43.17, -38.50, -51.23, -47.93],
        }
    )


@pytest.fixture
def sample_municipalities_df():
    """DataFrame com amostra maior de municípios válidos."""
    np.random.seed(42)
    n = 100

    # Gerar códigos IBGE válidos (simplificado)
    ufs_codes = [
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        35,
        41,
        42,
        43,
        50,
        51,
        52,
        53,
    ]
    uf_names = [
        "RO",
        "AC",
        "AM",
        "RR",
        "PA",
        "AP",
        "TO",
        "MA",
        "PI",
        "CE",
        "RN",
        "PB",
        "PE",
        "AL",
        "SE",
        "BA",
        "MG",
        "ES",
        "RJ",
        "SP",
        "PR",
        "SC",
        "RS",
        "MS",
        "MT",
        "GO",
        "DF",
    ]
    uf_map = dict(zip(ufs_codes, uf_names))

    codes = []
    ufs = []
    for i in range(n):
        uf_code = np.random.choice(ufs_codes)
        muni_code = int(uf_code * 100000 + np.random.randint(1, 9999))
        codes.append(muni_code)
        ufs.append(uf_map[uf_code])

    return pd.DataFrame(
        {
            "cod_ibge": codes,
            "nome": [f"Município {i}" for i in range(n)],
            "uf": ufs,
            "lat": np.random.uniform(-33.0, 5.0, n),
            "lon": np.random.uniform(-73.0, -35.0, n),
        }
    )


# Testes de validate_uniqueness


class TestValidateUniqueness:
    """Testes para validação de unicidade."""

    def test_unique_codes_pass(self, valid_df):
        """Códigos únicos devem passar."""
        result = validate_uniqueness(valid_df)
        assert result.valid is True
        assert "OK" in result.message

    def test_duplicate_codes_fail(self, valid_df):
        """Códigos duplicados devem falhar."""
        df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
        result = validate_uniqueness(df)
        assert result.valid is False
        assert "duplicados" in result.message.lower()
        assert result.details is not None


# Testes de validate_lat_bounds


class TestValidateLatBounds:
    """Testes para validação de limites de latitude."""

    def test_valid_latitudes_pass(self, valid_df, config):
        """Latitudes dentro do range devem passar."""
        result = validate_lat_bounds(valid_df, config)
        assert result.valid is True

    def test_lat_below_min_fail(self, valid_df, config):
        """Latitude abaixo do mínimo deve falhar."""
        df = valid_df.copy()
        df.loc[0, "lat"] = -40.0  # Fora do Brasil
        result = validate_lat_bounds(df, config)
        assert result.valid is False

    def test_lat_above_max_fail(self, valid_df, config):
        """Latitude acima do máximo deve falhar."""
        df = valid_df.copy()
        df.loc[0, "lat"] = 10.0  # Fora do Brasil
        result = validate_lat_bounds(df, config)
        assert result.valid is False


# Testes de validate_lon_bounds


class TestValidateLonBounds:
    """Testes para validação de limites de longitude."""

    def test_valid_longitudes_pass(self, valid_df, config):
        """Longitudes dentro do range devem passar."""
        result = validate_lon_bounds(valid_df, config)
        assert result.valid is True

    def test_lon_below_min_fail(self, valid_df, config):
        """Longitude abaixo do mínimo deve falhar."""
        df = valid_df.copy()
        df.loc[0, "lon"] = -80.0  # Fora do Brasil
        result = validate_lon_bounds(df, config)
        assert result.valid is False

    def test_lon_above_max_fail(self, valid_df, config):
        """Longitude acima do máximo deve falhar."""
        df = valid_df.copy()
        df.loc[0, "lon"] = -30.0  # Fora do Brasil
        result = validate_lon_bounds(df, config)
        assert result.valid is False


# Testes de validate_ufs


class TestValidateUfs:
    """Testes para validação de UFs."""

    def test_valid_ufs_pass(self, valid_df, config):
        """UFs válidas devem passar."""
        result = validate_ufs(valid_df, config)
        assert result.valid is True

    def test_invalid_uf_fail(self, valid_df, config):
        """UF inválida deve falhar."""
        df = valid_df.copy()
        df.loc[0, "uf"] = "XX"  # UF inválida
        result = validate_ufs(df, config)
        assert result.valid is False
        assert "XX" in str(result.details)


# Testes de validate_municipality_count


class TestValidateMunicipalityCount:
    """Testes para validação de contagem de municípios."""

    def test_count_within_tolerance_pass(self, config):
        """Contagem dentro da tolerância deve passar."""
        expected = config["expected_municipalities"]
        tolerance = config["tolerance"]

        df = pd.DataFrame(
            {
                "cod_ibge": range(expected),
                "nome": [""] * expected,
                "uf": ["SP"] * expected,
                "lat": [0.0] * expected,
                "lon": [0.0] * expected,
            }
        )

        result = validate_municipality_count(df, config)
        assert result.valid is True

    def test_count_below_tolerance_fail(self, config):
        """Contagem abaixo da tolerância deve falhar."""
        expected = config["expected_municipalities"]
        tolerance = config["tolerance"]
        count = expected - tolerance - 100  # Muito abaixo

        df = pd.DataFrame(
            {
                "cod_ibge": range(count),
            }
        )

        result = validate_municipality_count(df, config)
        assert result.valid is False


# Testes de validate_no_nulls


class TestValidateNoNulls:
    """Testes para validação de valores nulos."""

    def test_no_nulls_pass(self, valid_df):
        """DataFrame sem nulos deve passar."""
        result = validate_no_nulls(valid_df)
        assert result.valid is True

    def test_with_nulls_fail(self, valid_df):
        """DataFrame com nulos deve falhar."""
        df = valid_df.copy()
        df.loc[0, "nome"] = None
        result = validate_no_nulls(df)
        assert result.valid is False
        assert "nome" in str(result.details)


# Testes de validate_cod_ibge_format


class TestValidateCodIbgeFormat:
    """Testes para validação do formato do código IBGE."""

    def test_valid_codes_pass(self, valid_df):
        """Códigos válidos devem passar."""
        result = validate_cod_ibge_format(valid_df)
        assert result.valid is True

    def test_code_too_small_fail(self, valid_df):
        """Código muito pequeno deve falhar."""
        df = valid_df.copy()
        df.loc[0, "cod_ibge"] = 100000  # Muito pequeno
        result = validate_cod_ibge_format(df)
        assert result.valid is False

    def test_code_too_large_fail(self, valid_df):
        """Código muito grande deve falhar."""
        df = valid_df.copy()
        df.loc[0, "cod_ibge"] = 9999999  # Muito grande
        result = validate_cod_ibge_format(df)
        assert result.valid is False


# Testes de validate_municipalities (integração)


class TestValidateMunicipalities:
    """Testes de integração para validação completa."""

    def test_valid_df_passes_all(self, valid_df, config):
        """DataFrame válido deve passar todas as validações."""
        # Ajustar contagem para tolerância
        config_adjusted = config.copy()
        config_adjusted["expected_municipalities"] = len(valid_df)
        config_adjusted["tolerance"] = 1

        results = validate_municipalities(valid_df, config_adjusted)

        # Verificar que todas passaram (exceto contagem)
        for result in results[:-1]:  # Excluir contagem
            if "Contagem" not in result.message:
                assert result.valid is True

    def test_invalid_df_raises_on_error(self, valid_df, config):
        """DataFrame inválido deve levantar exceção com raise_on_error=True."""
        df = valid_df.copy()
        df.loc[0, "lat"] = 100.0  # Latitude inválida

        with pytest.raises(ValueError):
            validate_municipalities(df, config, raise_on_error=True)


# Testes do Schema Pandera


class TestPanderaSchema:
    """Testes para o schema Pandera."""

    def test_schema_creation(self, config):
        """Schema deve ser criado sem erros."""
        schema = get_municipalities_schema(config)
        assert schema is not None

    def test_schema_validates_correct_df(self, valid_df, config):
        """Schema deve validar DataFrame correto."""
        schema = get_municipalities_schema(config)
        validated = schema.validate(valid_df)
        assert len(validated) == len(valid_df)

    def test_schema_rejects_invalid_uf(self, valid_df, config):
        """Schema deve rejeitar UF inválida."""
        import pandera as pa

        schema = get_municipalities_schema(config)
        df = valid_df.copy()
        df.loc[0, "uf"] = "XX"

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_schema_rejects_duplicate_pk(self, valid_df, config):
        """Schema deve rejeitar chave primária duplicada."""
        import pandera as pa

        schema = get_municipalities_schema(config)
        df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)


# Testes de configuração


class TestConfig:
    """Testes para carregamento de configuração."""

    def test_load_config(self):
        """Configuração deve ser carregada corretamente."""
        config = load_config()
        assert "bounds" in config
        assert "valid_ufs" in config
        assert len(config["valid_ufs"]) == 27

    def test_config_bounds_valid(self):
        """Limites geográficos devem ser válidos."""
        config = load_config()
        bounds = config["bounds"]

        # Latitude
        assert bounds["lat_min"] < bounds["lat_max"]
        assert bounds["lat_min"] >= -90
        assert bounds["lat_max"] <= 90

        # Longitude
        assert bounds["lon_min"] < bounds["lon_max"]
        assert bounds["lon_min"] >= -180
        assert bounds["lon_max"] <= 180


# Testes com dados reais (se disponíveis)


class TestWithRealData:
    """Testes com dados reais do parquet (se disponível)."""

    @pytest.fixture
    def municipalities_parquet(self):
        """Carrega parquet de municípios se existir."""
        path = Path(__file__).parents[1] / "data" / "processed" / "municipalities.parquet"
        if path.exists():
            return pd.read_parquet(path)
        pytest.skip("Parquet de municípios não disponível")

    def test_real_data_validates(self, municipalities_parquet, config):
        """Dados reais devem passar nas validações."""
        results = validate_municipalities(municipalities_parquet, config)

        # Todas as validações devem passar
        for result in results:
            assert result.valid is True, f"Falhou: {result.message}"
