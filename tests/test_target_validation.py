"""
Testes unitarios para validacao do target (PAM - Produtividade de Soja).

Testa as funcoes de validacao em src/validation/target.py
"""

from pathlib import Path

import pandas as pd
import pytest

from src.validation.target import (
    calculate_coverage_stats,
    load_config,
    validate_cod_ibge_format,
    validate_cross_check,
    validate_no_nulls_in_key_columns,
    validate_primary_key,
    validate_productivity_range,
    validate_target,
    validate_year_range,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_df():
    """DataFrame valido para testes."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, 3550308, 4106902, 4106902, 5208707],
            "ano": [2020, 2021, 2020, 2021, 2020],
            "produtividade_kg_ha": [3200.0, 3400.0, 3100.0, 3300.0, 2900.0],
            "area_colhida_ha": [1000.0, 1100.0, 2000.0, 2200.0, 500.0],
            "producao_ton": [3200.0, 3740.0, 6200.0, 7260.0, 1450.0],
        }
    )


@pytest.fixture
def df_with_duplicates():
    """DataFrame com chaves duplicadas."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, 3550308, 3550308],
            "ano": [2020, 2020, 2021],  # 2020 duplicado
            "produtividade_kg_ha": [3200.0, 3300.0, 3400.0],
        }
    )


@pytest.fixture
def df_productivity_out_of_range():
    """DataFrame com produtividade fora do range."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, 4106902, 5208707],
            "ano": [2020, 2020, 2020],
            "produtividade_kg_ha": [-100.0, 3200.0, 7000.0],  # -100 e 7000 invalidos
        }
    )


@pytest.fixture
def df_years_out_of_range():
    """DataFrame com anos fora do range."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, 4106902, 5208707],
            "ano": [1999, 2020, 2025],  # 1999 e 2025 fora do range 2000-2023
            "produtividade_kg_ha": [3200.0, 3300.0, 3400.0],
        }
    )


@pytest.fixture
def df_cross_check_fail():
    """DataFrame com inconsistencia producao/area vs produtividade."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308],
            "ano": [2020],
            "produtividade_kg_ha": [3200.0],
            "area_colhida_ha": [1000.0],
            "producao_ton": [5000.0],  # 5000*1000/1000 = 5000, muito diferente de 3200
        }
    )


@pytest.fixture
def df_with_nulls():
    """DataFrame com valores nulos nas colunas-chave."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, None, 5208707],
            "ano": [2020, 2021, None],
            "produtividade_kg_ha": [3200.0, None, 3400.0],
        }
    )


@pytest.fixture
def df_invalid_cod_ibge():
    """DataFrame com cod_ibge em formato invalido."""
    return pd.DataFrame(
        {
            "cod_ibge": [3550308, 12345, 35503081234],  # 12345 (5 dig) e 35503081234 (11 dig)
            "ano": [2020, 2020, 2020],
            "produtividade_kg_ha": [3200.0, 3300.0, 3400.0],
        }
    )


# ============================================================================
# Testes: validate_primary_key
# ============================================================================


class TestValidatePrimaryKey:
    """Testes para validacao de chave primaria."""

    def test_valid_pk(self, valid_df):
        """Chave primaria unica deve passar."""
        success, msg = validate_primary_key(valid_df)
        assert success is True
        assert "unica" in msg.lower()

    def test_duplicate_pk(self, df_with_duplicates):
        """Chave primaria duplicada deve falhar."""
        success, msg = validate_primary_key(df_with_duplicates)
        assert success is False
        assert "duplicadas" in msg.lower()


# ============================================================================
# Testes: validate_productivity_range
# ============================================================================


class TestValidateProductivityRange:
    """Testes para validacao de range de produtividade."""

    def test_valid_productivity(self, valid_df):
        """Produtividade dentro do range deve passar."""
        success, msg = validate_productivity_range(valid_df, 0, 6000)
        assert success is True

    def test_productivity_below_min(self, df_productivity_out_of_range):
        """Produtividade negativa deve falhar."""
        success, msg = validate_productivity_range(df_productivity_out_of_range, 0, 6000)
        assert success is False
        assert "abaixo" in msg.lower()

    def test_productivity_above_max(self, df_productivity_out_of_range):
        """Produtividade acima do maximo deve falhar."""
        success, msg = validate_productivity_range(df_productivity_out_of_range, 0, 6000)
        assert success is False
        assert "acima" in msg.lower()

    def test_custom_range(self, valid_df):
        """Range customizado deve funcionar."""
        success, msg = validate_productivity_range(valid_df, 0, 3000)
        assert success is False  # 3400 esta acima de 3000


# ============================================================================
# Testes: validate_year_range
# ============================================================================


class TestValidateYearRange:
    """Testes para validacao de range de anos."""

    def test_valid_years(self, valid_df):
        """Anos dentro do range devem passar."""
        success, msg = validate_year_range(valid_df, 2000, 2023)
        assert success is True

    def test_year_before_start(self, df_years_out_of_range):
        """Ano antes do inicio deve falhar."""
        success, msg = validate_year_range(df_years_out_of_range, 2000, 2023)
        assert success is False
        assert "antes" in msg.lower()

    def test_year_after_end(self, df_years_out_of_range):
        """Ano depois do fim deve falhar."""
        success, msg = validate_year_range(df_years_out_of_range, 2000, 2023)
        assert success is False
        assert "depois" in msg.lower()


# ============================================================================
# Testes: validate_cross_check
# ============================================================================


class TestValidateCrossCheck:
    """Testes para validacao de consistencia producao/area."""

    def test_valid_cross_check(self, valid_df):
        """Dados consistentes devem passar."""
        success, msg = validate_cross_check(valid_df, tolerance=0.05)
        assert success is True

    def test_cross_check_fail(self, df_cross_check_fail):
        """Dados inconsistentes devem falhar."""
        success, msg = validate_cross_check(df_cross_check_fail, tolerance=0.05)
        # Como temos apenas 1 registro e ele viola, deve falhar
        assert success is False

    def test_cross_check_without_columns(self, valid_df):
        """Sem colunas producao/area deve passar (ignorado)."""
        df_no_cols = valid_df[["cod_ibge", "ano", "produtividade_kg_ha"]].copy()
        success, msg = validate_cross_check(df_no_cols)
        assert success is True
        assert "ignorado" in msg.lower()

    def test_cross_check_with_tolerance(self):
        """Diferenca dentro da tolerancia deve passar."""
        df = pd.DataFrame(
            {
                "cod_ibge": [3550308],
                "ano": [2020],
                "produtividade_kg_ha": [3200.0],
                "area_colhida_ha": [1000.0],
                "producao_ton": [3200.0],  # Exato
            }
        )
        success, msg = validate_cross_check(df, tolerance=0.05)
        assert success is True


# ============================================================================
# Testes: validate_no_nulls_in_key_columns
# ============================================================================


class TestValidateNoNulls:
    """Testes para validacao de nulos."""

    def test_no_nulls(self, valid_df):
        """Sem nulos deve passar."""
        success, msg = validate_no_nulls_in_key_columns(valid_df)
        assert success is True

    def test_with_nulls(self, df_with_nulls):
        """Com nulos deve falhar."""
        success, msg = validate_no_nulls_in_key_columns(df_with_nulls)
        assert success is False
        assert "nulos" in msg.lower()


# ============================================================================
# Testes: validate_cod_ibge_format
# ============================================================================


class TestValidateCodIbgeFormat:
    """Testes para validacao de formato do cod_ibge."""

    def test_valid_format(self, valid_df):
        """Formato valido (7 digitos) deve passar."""
        success, msg = validate_cod_ibge_format(valid_df)
        assert success is True

    def test_invalid_format_short(self, df_invalid_cod_ibge):
        """Codigo curto demais deve falhar."""
        success, msg = validate_cod_ibge_format(df_invalid_cod_ibge)
        assert success is False
        assert "invalido" in msg.lower()

    def test_invalid_format_long(self, df_invalid_cod_ibge):
        """Codigo longo demais deve falhar."""
        success, msg = validate_cod_ibge_format(df_invalid_cod_ibge)
        assert success is False


# ============================================================================
# Testes: validate_target (integrado)
# ============================================================================


class TestValidateTarget:
    """Testes para validacao integrada."""

    def test_all_validations_pass(self, valid_df):
        """DataFrame valido deve passar todas as validacoes."""
        config = {
            "validation": {
                "productivity_min": 0,
                "productivity_max": 6000,
                "cross_check_tolerance": 0.05,
            },
            "year_start": 2000,
            "year_end": 2023,
        }
        results = validate_target(valid_df, config)
        assert results["all_passed"] is True

    def test_fails_with_invalid_data(self, df_with_duplicates):
        """DataFrame invalido deve falhar."""
        config = {
            "validation": {
                "productivity_min": 0,
                "productivity_max": 6000,
                "cross_check_tolerance": 0.05,
            },
            "year_start": 2000,
            "year_end": 2023,
        }
        results = validate_target(df_with_duplicates, config)
        assert results["all_passed"] is False


# ============================================================================
# Testes: calculate_coverage_stats
# ============================================================================


class TestCalculateCoverageStats:
    """Testes para estatisticas de cobertura."""

    def test_coverage_stats(self, valid_df):
        """Deve calcular estatisticas corretamente."""
        stats = calculate_coverage_stats(valid_df)

        assert stats["total_registros"] == 5
        assert stats["municipios_unicos"] == 3
        assert stats["anos_unicos"] == 2
        assert "missingness" in stats
        assert "cobertura_por_ano" in stats

    def test_missingness_calculation(self):
        """Deve calcular missingness corretamente."""
        df = pd.DataFrame(
            {
                "cod_ibge": [3550308, 4106902, 5208707],
                "ano": [2020, 2021, 2022],
                "produtividade_kg_ha": [3200.0, None, 3400.0],
            }
        )
        stats = calculate_coverage_stats(df)

        assert stats["missingness"]["produtividade_kg_ha"]["nulls"] == 1
        assert stats["missingness"]["produtividade_kg_ha"]["pct"] == pytest.approx(33.33, rel=0.1)


# ============================================================================
# Testes: load_config
# ============================================================================


class TestConfig:
    """Testes para carregamento de configuracao."""

    def test_load_config(self):
        """Deve carregar configuracao do arquivo."""
        config = load_config()
        assert "crop" in config
        assert config["crop"] == "soja"
        assert "year_start" in config
        assert "year_end" in config
        assert "validation" in config

    def test_config_validation_params(self):
        """Configuracao deve ter parametros de validacao."""
        config = load_config()
        validation = config["validation"]
        assert "productivity_min" in validation
        assert "productivity_max" in validation
        assert "cross_check_tolerance" in validation


# ============================================================================
# Testes com dados reais (se existirem)
# ============================================================================


class TestWithRealData:
    """Testes com dados reais (executa apenas se o arquivo existir)."""

    @pytest.fixture
    def real_data_path(self):
        """Caminho para dados reais."""
        return Path(__file__).parent.parent / "data" / "processed" / "target_soja.parquet"

    def test_real_data_validation(self, real_data_path):
        """Se dados reais existirem, devem passar nas validacoes."""
        if not real_data_path.exists():
            pytest.skip("Arquivo target_soja.parquet nao existe ainda")

        df = pd.read_parquet(real_data_path)
        config = load_config()
        results = validate_target(df, config)

        # Todas as validacoes devem passar
        assert results["all_passed"] is True, f"Validacoes falharam: {results}"
