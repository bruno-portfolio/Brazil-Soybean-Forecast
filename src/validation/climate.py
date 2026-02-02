import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "climate.yaml"
CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"


def load_config() -> dict[str, Any]:
    """Carrega configuracao do climate.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["climate"]


def validate_temp_consistency(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida consistencia das temperaturas: tmin <= tmean <= tmax."""
    mask = df[["tmin", "tmean", "tmax"]].notna().all(axis=1)
    df_valid = df[mask]

    if len(df_valid) == 0:
        return True, "Nenhum registro valido para verificar"

    violations_min = df_valid[df_valid["tmin"] > df_valid["tmean"]]
    violations_max = df_valid[df_valid["tmean"] > df_valid["tmax"]]

    total_violations = len(violations_min) + len(violations_max)

    if total_violations > 0:
        pct = 100 * total_violations / len(df_valid)
        msg = (
            f"Violacoes de consistencia: {total_violations:,} registros ({pct:.2f}%)\n"
            f"  - tmin > tmean: {len(violations_min):,}\n"
            f"  - tmean > tmax: {len(violations_max):,}"
        )
        if pct <= 1.0:
            return True, f"Pequeno ruido aceitavel: {msg}"
        return False, msg

    return True, f"Consistencia OK: {len(df_valid):,} registros verificados"


def validate_precip_non_negative(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida que precipitacao nao e negativa."""
    if "precip" not in df.columns:
        return True, "Coluna precip nao encontrada"

    mask = df["precip"].notna()
    df_valid = df[mask]

    negative = df_valid[df_valid["precip"] < 0]

    if len(negative) > 0:
        return False, f"Precipitacao negativa: {len(negative):,} registros"

    return True, f"Precipitacao OK: {len(df_valid):,} registros >= 0"


def validate_no_future_dates(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida que nao ha datas futuras."""
    if "date" not in df.columns:
        return False, "Coluna date nao encontrada"

    today = pd.Timestamp.now().normalize()
    future = df[df["date"] > today]

    if len(future) > 0:
        return False, f"Datas futuras encontradas: {len(future):,} registros"

    max_date = df["date"].max()
    return True, f"Datas OK: max = {max_date.date()}"


def validate_temporal_coverage(df: pd.DataFrame, config: dict) -> tuple[bool, str]:
    """Valida cobertura temporal conforme configuracao."""
    if "date" not in df.columns:
        return False, "Coluna date nao encontrada"

    year_start = config["year_start"]
    year_end = config["year_end"]

    df_years = df["date"].dt.year.unique()
    min_year = df_years.min()
    max_year = df_years.max()

    issues = []

    if min_year > year_start:
        issues.append(f"Ano inicial {min_year} > esperado {year_start}")

    current_year = datetime.now().year
    expected_max = min(year_end, current_year)

    if max_year < expected_max - 1:
        issues.append(f"Ano final {max_year} < esperado {expected_max}")

    if issues:
        return False, "; ".join(issues)

    return True, f"Cobertura temporal OK: {min_year} - {max_year}"


def validate_primary_key(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida unicidade da chave primaria (cod_ibge, date)."""
    duplicates = df.duplicated(subset=["cod_ibge", "date"])
    n_duplicates = duplicates.sum()

    if n_duplicates > 0:
        return False, f"Chave duplicada: {n_duplicates:,} registros"

    return True, f"Chave unica OK: {len(df):,} registros"


def validate_cod_ibge_format(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida formato do codigo IBGE (7 digitos)."""
    if "cod_ibge" not in df.columns:
        return False, "Coluna cod_ibge nao encontrada"

    invalid = df[(df["cod_ibge"] < 1000000) | (df["cod_ibge"] > 9999999)]

    if len(invalid) > 0:
        return False, f"Formato invalido: {len(invalid):,} registros"

    return True, "Formato cod_ibge OK"


def validate_no_nulls_in_key_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida ausencia de nulos em colunas-chave."""
    key_cols = ["cod_ibge", "date"]
    nulls = {}

    for col in key_cols:
        if col in df.columns:
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                nulls[col] = n_nulls

    if nulls:
        msg = ", ".join(f"{k}: {v}" for k, v in nulls.items())
        return False, f"Nulos em colunas-chave: {msg}"

    return True, "Sem nulos em colunas-chave"


def validate_temperature_range(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida que temperaturas estao em faixas plausiveis."""
    issues = []

    temp_min = -10
    temp_max = 50

    for col in ["tmin", "tmean", "tmax"]:
        if col in df.columns:
            mask = df[col].notna()
            df_valid = df[mask]

            below = df_valid[df_valid[col] < temp_min]
            above = df_valid[df_valid[col] > temp_max]

            if len(below) > 0:
                issues.append(f"{col} < {temp_min}: {len(below)} registros")
            if len(above) > 0:
                issues.append(f"{col} > {temp_max}: {len(above)} registros")

    if issues:
        return False, "; ".join(issues)

    return True, "Temperaturas em faixa plausivel"


def validate_climate(df: pd.DataFrame, config: dict | None = None) -> dict[str, tuple[bool, str]]:
    """Executa todas as validacoes do dataset climatico."""
    if config is None:
        config = load_config()

    results = {}

    results["primary_key"] = validate_primary_key(df)
    results["no_nulls_key"] = validate_no_nulls_in_key_columns(df)
    results["cod_ibge_format"] = validate_cod_ibge_format(df)
    results["temp_consistency"] = validate_temp_consistency(df)
    results["precip_non_negative"] = validate_precip_non_negative(df)
    results["no_future_dates"] = validate_no_future_dates(df)
    results["temporal_coverage"] = validate_temporal_coverage(df, config)
    results["temperature_range"] = validate_temperature_range(df)

    return results


def get_climate_schema() -> pa.DataFrameSchema:
    """Cria schema Pandera para validacao do dataset climatico."""
    schema = pa.DataFrameSchema(
        {
            "cod_ibge": pa.Column(
                int,
                checks=[
                    pa.Check.ge(1000000, error="cod_ibge deve ter 7 digitos"),
                    pa.Check.le(9999999, error="cod_ibge deve ter 7 digitos"),
                ],
                nullable=False,
            ),
            "date": pa.Column(
                "datetime64[ns]",
                nullable=False,
            ),
            "tmin": pa.Column(
                float,
                checks=[
                    pa.Check.ge(-20, error="tmin muito baixa"),
                    pa.Check.le(50, error="tmin muito alta"),
                ],
                nullable=True,
            ),
            "tmean": pa.Column(
                float,
                checks=[
                    pa.Check.ge(-15, error="tmean muito baixa"),
                    pa.Check.le(50, error="tmean muito alta"),
                ],
                nullable=True,
            ),
            "tmax": pa.Column(
                float,
                checks=[
                    pa.Check.ge(-10, error="tmax muito baixa"),
                    pa.Check.le(55, error="tmax muito alta"),
                ],
                nullable=True,
            ),
            "precip": pa.Column(
                float,
                checks=[
                    pa.Check.ge(0, error="precip nao pode ser negativa"),
                    pa.Check.le(500, error="precip diaria muito alta"),
                ],
                nullable=True,
            ),
            "rh": pa.Column(
                float,
                checks=[
                    pa.Check.ge(0, error="umidade relativa deve ser >= 0"),
                    pa.Check.le(100, error="umidade relativa deve ser <= 100"),
                ],
                nullable=True,
            ),
        },
        index=pa.Index(int),
        strict=False,
        coerce=True,
    )
    return schema


def calculate_coverage_stats(df: pd.DataFrame) -> dict:
    """Calcula estatisticas de cobertura do dataset."""
    stats = {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "periodo": {
            "min": str(df["date"].min().date()) if "date" in df.columns else None,
            "max": str(df["date"].max().date()) if "date" in df.columns else None,
        },
        "missingness": {},
    }

    for col in ["tmin", "tmean", "tmax", "precip", "rh"]:
        if col in df.columns:
            n_nulls = df[col].isna().sum()
            pct = 100 * n_nulls / len(df) if len(df) > 0 else 0
            stats["missingness"][col] = {
                "nulls": int(n_nulls),
                "pct": round(pct, 2),
            }

    records_per_mun = df.groupby("cod_ibge").size()
    stats["records_per_municipality"] = {
        "min": int(records_per_mun.min()),
        "max": int(records_per_mun.max()),
        "mean": round(records_per_mun.mean(), 1),
    }

    return stats


def main():
    """Executa validacao do dataset climatico."""
    logger.info("=" * 60)
    logger.info("VALIDACAO CLIMA")
    logger.info("=" * 60)

    if not CLIMATE_PATH.exists():
        logger.error(f"Arquivo nao encontrado: {CLIMATE_PATH}")
        logger.error("Execute primeiro: python -m src.ingest.climate_power")
        return

    df = pd.read_parquet(CLIMATE_PATH)
    logger.info(f"Registros carregados: {len(df):,}")

    config = load_config()

    results = validate_climate(df, config)

    logger.info("\n" + "-" * 40)
    logger.info("RESULTADOS DAS VALIDACOES")
    logger.info("-" * 40)

    all_passed = True
    for name, (passed, msg) in results.items():
        status = "[OK]" if passed else "[FALHA]"
        logger.info(f"{status} {name}: {msg}")
        if not passed:
            all_passed = False

    stats = calculate_coverage_stats(df)

    logger.info("\n" + "-" * 40)
    logger.info("ESTATISTICAS DE COBERTURA")
    logger.info("-" * 40)
    logger.info(f"Total de registros: {stats['total_registros']:,}")
    logger.info(f"Municipios unicos: {stats['municipios_unicos']:,}")
    logger.info(f"Periodo: {stats['periodo']['min']} a {stats['periodo']['max']}")

    logger.info("\nMissingness por coluna:")
    for col, info in stats["missingness"].items():
        logger.info(f"  {col}: {info['nulls']:,} ({info['pct']:.2f}%)")

    logger.info(
        f"\nRegistros por municipio: min={stats['records_per_municipality']['min']}, "
        f"max={stats['records_per_municipality']['max']}, "
        f"media={stats['records_per_municipality']['mean']}"
    )

    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("TODAS AS VALIDACOES PASSARAM")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("ALGUMAS VALIDACOES FALHARAM")
        logger.error("=" * 60)

    return results, stats


if __name__ == "__main__":
    main()
