from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pandera as pa
import yaml
from loguru import logger
from pandera import Check, Column, DataFrameSchema


class ValidationResult(NamedTuple):
    """Resultado de uma validacao."""

    valid: bool
    message: str
    details: dict | None = None


def load_config(config_path: Path | None = None) -> dict:
    """Carrega configuracao geografica do arquivo YAML."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "geo.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config["geo"]


def get_municipalities_schema(config: dict | None = None) -> DataFrameSchema:
    """Cria schema Pandera para validacao de municipios."""
    if config is None:
        config = load_config()

    bounds = config["bounds"]
    valid_ufs = config["valid_ufs"]

    schema = DataFrameSchema(
        columns={
            "cod_ibge": Column(
                int,
                checks=[
                    Check.ge(1100015),
                    Check.le(5300108),
                ],
                unique=True,
                nullable=False,
                coerce=True,
            ),
            "nome": Column(
                str,
                checks=[Check.str_length(min_value=2, max_value=100)],
                nullable=False,
            ),
            "uf": Column(
                str,
                checks=[
                    Check.str_length(min_value=2, max_value=2),
                    Check.isin(valid_ufs),
                ],
                nullable=False,
            ),
            "lat": Column(
                float,
                checks=[Check.in_range(bounds["lat_min"], bounds["lat_max"])],
                nullable=False,
            ),
            "lon": Column(
                float,
                checks=[Check.in_range(bounds["lon_min"], bounds["lon_max"])],
                nullable=False,
            ),
        },
        strict=True,
        coerce=True,
    )

    return schema


def validate_uniqueness(df: pd.DataFrame) -> ValidationResult:
    """Valida unicidade do codigo IBGE."""
    duplicates = df[df.duplicated(subset=["cod_ibge"], keep=False)]

    if len(duplicates) == 0:
        return ValidationResult(
            valid=True,
            message="Unicidade de cod_ibge: OK",
        )

    return ValidationResult(
        valid=False,
        message=f"Encontrados {len(duplicates)} registros duplicados",
        details={"duplicates": duplicates["cod_ibge"].unique().tolist()[:10]},
    )


def validate_lat_bounds(df: pd.DataFrame, config: dict | None = None) -> ValidationResult:
    """Valida se latitudes estao dentro do envelope brasileiro."""
    if config is None:
        config = load_config()

    bounds = config["bounds"]
    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]

    out_of_bounds = df[(df["lat"] < lat_min) | (df["lat"] > lat_max)]

    if len(out_of_bounds) == 0:
        return ValidationResult(
            valid=True,
            message=f"Latitude [{lat_min}, {lat_max}]: OK",
        )

    return ValidationResult(
        valid=False,
        message=f"{len(out_of_bounds)} municipios com latitude fora do range",
        details={
            "expected_range": [lat_min, lat_max],
            "actual_range": [df["lat"].min(), df["lat"].max()],
            "examples": out_of_bounds[["cod_ibge", "nome", "lat"]].head(5).to_dict("records"),
        },
    )


def validate_lon_bounds(df: pd.DataFrame, config: dict | None = None) -> ValidationResult:
    """Valida se longitudes estao dentro do envelope brasileiro."""
    if config is None:
        config = load_config()

    bounds = config["bounds"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]

    out_of_bounds = df[(df["lon"] < lon_min) | (df["lon"] > lon_max)]

    if len(out_of_bounds) == 0:
        return ValidationResult(
            valid=True,
            message=f"Longitude [{lon_min}, {lon_max}]: OK",
        )

    return ValidationResult(
        valid=False,
        message=f"{len(out_of_bounds)} municipios com longitude fora do range",
        details={
            "expected_range": [lon_min, lon_max],
            "actual_range": [df["lon"].min(), df["lon"].max()],
            "examples": out_of_bounds[["cod_ibge", "nome", "lon"]].head(5).to_dict("records"),
        },
    )


def validate_ufs(df: pd.DataFrame, config: dict | None = None) -> ValidationResult:
    """Valida se todas as UFs sao validas."""
    if config is None:
        config = load_config()

    valid_ufs = set(config["valid_ufs"])
    found_ufs = set(df["uf"].unique())

    invalid_ufs = found_ufs - valid_ufs
    missing_ufs = valid_ufs - found_ufs

    if not invalid_ufs and not missing_ufs:
        return ValidationResult(
            valid=True,
            message=f"UFs validas ({len(found_ufs)}/27): OK",
        )

    details = {}
    if invalid_ufs:
        details["invalid_ufs"] = list(invalid_ufs)
    if missing_ufs:
        details["missing_ufs"] = list(missing_ufs)

    return ValidationResult(
        valid=len(invalid_ufs) == 0,
        message=f"UFs invalidas: {invalid_ufs}" if invalid_ufs else f"UFs ausentes: {missing_ufs}",
        details=details,
    )


def validate_municipality_count(df: pd.DataFrame, config: dict | None = None) -> ValidationResult:
    """Valida se o numero de municipios esta dentro da tolerancia esperada."""
    if config is None:
        config = load_config()

    expected = config["expected_municipalities"]
    tolerance = config["tolerance"]
    actual = len(df)

    if expected - tolerance <= actual <= expected + tolerance:
        return ValidationResult(
            valid=True,
            message=f"Contagem de municipios: {actual} (esperado: {expected} +/- {tolerance}): OK",
        )

    return ValidationResult(
        valid=False,
        message=f"Contagem fora da tolerancia: {actual} (esperado: {expected} +/- {tolerance})",
        details={
            "expected": expected,
            "tolerance": tolerance,
            "actual": actual,
            "difference": actual - expected,
        },
    )


def validate_no_nulls(df: pd.DataFrame) -> ValidationResult:
    """Valida ausencia de valores nulos."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) == 0:
        return ValidationResult(
            valid=True,
            message="Sem valores nulos: OK",
        )

    return ValidationResult(
        valid=False,
        message=f"Valores nulos encontrados em {len(cols_with_nulls)} colunas",
        details={"null_counts": cols_with_nulls.to_dict()},
    )


def validate_cod_ibge_format(df: pd.DataFrame) -> ValidationResult:
    """Valida formato do codigo IBGE (7 digitos, comeca com codigo da UF)."""
    invalid = df[
        (df["cod_ibge"] < 1100000)
        | (df["cod_ibge"] > 5399999)
    ]

    if len(invalid) == 0:
        return ValidationResult(
            valid=True,
            message="Formato cod_ibge (7 digitos): OK",
        )

    return ValidationResult(
        valid=False,
        message=f"{len(invalid)} codigos IBGE com formato invalido",
        details={"examples": invalid["cod_ibge"].head(10).tolist()},
    )


def validate_municipalities(
    df: pd.DataFrame,
    config: dict | None = None,
    raise_on_error: bool = False,
) -> list[ValidationResult]:
    """Executa todas as validacoes no DataFrame de municipios."""
    if config is None:
        config = load_config()

    logger.info("=== Validando dados de municipios ===")

    validations = [
        ("Unicidade", validate_uniqueness(df)),
        ("Nulos", validate_no_nulls(df)),
        ("Formato cod_ibge", validate_cod_ibge_format(df)),
        ("Latitude", validate_lat_bounds(df, config)),
        ("Longitude", validate_lon_bounds(df, config)),
        ("UFs", validate_ufs(df, config)),
        ("Contagem", validate_municipality_count(df, config)),
    ]

    results = []
    all_valid = True

    for name, result in validations:
        status = "+" if result.valid else "x"
        logger.info(f"  [{status}] {name}: {result.message}")

        if not result.valid:
            all_valid = False
            if result.details:
                logger.debug(f"      Detalhes: {result.details}")

        results.append(result)

    if not all_valid and raise_on_error:
        failed = [r for r in results if not r.valid]
        raise ValueError(f"Validacao falhou: {[r.message for r in failed]}")

    logger.info(f"=== Validacao {'PASSOU' if all_valid else 'FALHOU'} ===")

    return results


def validate_municipalities_with_pandera(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Valida DataFrame usando schema Pandera."""
    schema = get_municipalities_schema(config)

    try:
        validated_df = schema.validate(df)
        logger.info("Validacao Pandera: OK")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"Validacao Pandera falhou: {e}")
        raise


def main():
    """Valida arquivo de municipios existente."""
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parquet_path = Path(__file__).parents[2] / "data" / "processed" / "municipalities.parquet"

    if not parquet_path.exists():
        logger.error(f"Arquivo nao encontrado: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path)
    logger.info(f"Carregados {len(df)} municipios de {parquet_path}")

    results = validate_municipalities(df)

    all_valid = all(r.valid for r in results)
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())
