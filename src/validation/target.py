import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandera import Check, Column, DataFrameSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "target.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"


def load_config() -> dict[str, Any]:
    """Carrega configuracao do target.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["target"]


def validate_primary_key(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida unicidade da chave primaria (cod_ibge, ano)."""
    duplicates = df.duplicated(subset=["cod_ibge", "ano"], keep=False)
    n_duplicates = duplicates.sum()

    if n_duplicates > 0:
        dup_examples = df[duplicates].head(5)[["cod_ibge", "ano"]].to_dict("records")
        return False, f"Encontradas {n_duplicates} linhas duplicadas. Exemplos: {dup_examples}"

    return True, f"Chave primaria (cod_ibge, ano) unica: {len(df):,} registros"


def validate_productivity_range(
    df: pd.DataFrame,
    min_val: float = 0,
    max_val: float = 6000,
) -> tuple[bool, str]:
    """Valida produtividade dentro do range plausivel."""
    prod = df["produtividade_kg_ha"].dropna()

    below_min = (prod < min_val).sum()
    above_max = (prod > max_val).sum()

    if below_min > 0 or above_max > 0:
        details = []
        if below_min > 0:
            details.append(f"{below_min} abaixo de {min_val}")
        if above_max > 0:
            details.append(f"{above_max} acima de {max_val}")
        return False, f"Produtividade fora do range [{min_val}, {max_val}]: {', '.join(details)}"

    return True, f"Produtividade dentro do range [{min_val}, {max_val}] kg/ha"


def validate_year_range(
    df: pd.DataFrame,
    year_start: int = 2000,
    year_end: int = 2023,
) -> tuple[bool, str]:
    """Valida anos dentro do periodo definido."""
    years = df["ano"]

    below_start = (years < year_start).sum()
    above_end = (years > year_end).sum()

    if below_start > 0 or above_end > 0:
        details = []
        if below_start > 0:
            details.append(f"{below_start} antes de {year_start}")
        if above_end > 0:
            details.append(f"{above_end} depois de {year_end}")
        return False, f"Anos fora do periodo [{year_start}, {year_end}]: {', '.join(details)}"

    actual_min = int(years.min())
    actual_max = int(years.max())
    return True, f"Anos dentro do periodo esperado: {actual_min} - {actual_max}"


def validate_cross_check(
    df: pd.DataFrame,
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """Valida consistencia: produtividade deve aproximar producao/area."""
    if "producao_ton" not in df.columns or "area_colhida_ha" not in df.columns:
        return True, "Cross-check ignorado: colunas producao/area nao disponiveis"

    mask = (
        df["produtividade_kg_ha"].notna()
        & df["producao_ton"].notna()
        & df["area_colhida_ha"].notna()
        & (df["area_colhida_ha"] > 0)
    )

    df_check = df[mask].copy()

    if len(df_check) == 0:
        return True, "Cross-check ignorado: sem registros com todos os valores"

    df_check["prod_calculada"] = (df_check["producao_ton"] * 1000) / df_check["area_colhida_ha"]

    df_check["diff_relativa"] = (
        abs(df_check["prod_calculada"] - df_check["produtividade_kg_ha"])
        / df_check["produtividade_kg_ha"]
    )

    violacoes = (df_check["diff_relativa"] > tolerance).sum()
    pct_violacoes = 100 * violacoes / len(df_check)

    if pct_violacoes > 1.0:
        return False, (
            f"Cross-check falhou: {violacoes:,} registros ({pct_violacoes:.1f}%) "
            f"com diferenca > {tolerance*100:.0f}% entre produtividade reportada e calculada"
        )

    return True, (
        f"Cross-check OK: {len(df_check):,} registros verificados, "
        f"{violacoes:,} ({pct_violacoes:.2f}%) com diferenca > {tolerance*100:.0f}%"
    )


def validate_no_nulls_in_key_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida ausencia de nulos nas colunas-chave."""
    key_cols = ["cod_ibge", "ano", "produtividade_kg_ha"]
    null_counts = {col: df[col].isna().sum() for col in key_cols}

    any_nulls = any(v > 0 for v in null_counts.values())

    if any_nulls:
        nulls_str = ", ".join(f"{k}: {v}" for k, v in null_counts.items() if v > 0)
        return False, f"Valores nulos encontrados: {nulls_str}"

    return True, "Sem valores nulos nas colunas-chave"


def validate_cod_ibge_format(df: pd.DataFrame) -> tuple[bool, str]:
    """Valida formato do codigo IBGE (7 digitos)."""
    cod_str = df["cod_ibge"].astype(str)

    invalid_length = cod_str.str.len() != 7
    n_invalid = invalid_length.sum()

    if n_invalid > 0:
        examples = df[invalid_length]["cod_ibge"].head(5).tolist()
        return False, f"Codigos IBGE com formato invalido: {n_invalid}. Exemplos: {examples}"

    return True, "Formato do cod_ibge valido (7 digitos)"


def validate_target(df: pd.DataFrame, config: dict | None = None) -> dict:
    """Executa todas as validacoes do target."""
    if config is None:
        config = load_config()

    validation_config = config.get("validation", {})
    prod_min = validation_config.get("productivity_min", 0)
    prod_max = validation_config.get("productivity_max", 6000)
    cross_tolerance = validation_config.get("cross_check_tolerance", 0.05)

    year_start = config.get("year_start", 2000)
    year_end = config.get("year_end", 2023)

    results = {}

    validations = [
        ("primary_key", lambda: validate_primary_key(df)),
        ("no_nulls", lambda: validate_no_nulls_in_key_columns(df)),
        ("cod_ibge_format", lambda: validate_cod_ibge_format(df)),
        ("productivity_range", lambda: validate_productivity_range(df, prod_min, prod_max)),
        ("year_range", lambda: validate_year_range(df, year_start, year_end)),
        ("cross_check", lambda: validate_cross_check(df, cross_tolerance)),
    ]

    all_passed = True
    for name, validator in validations:
        success, message = validator()
        results[name] = {"passed": success, "message": message}
        status = "OK" if success else "FALHOU"
        logger.info(f"[{status}] {name}: {message}")
        if not success:
            all_passed = False

    results["all_passed"] = all_passed

    return results


def get_target_schema(config: dict | None = None) -> DataFrameSchema:
    """Cria schema Pandera para validacao do target."""
    if config is None:
        config = load_config()

    validation_config = config.get("validation", {})
    prod_min = validation_config.get("productivity_min", 0)
    prod_max = validation_config.get("productivity_max", 6000)
    year_start = config.get("year_start", 2000)
    year_end = config.get("year_end", 2023)

    schema = DataFrameSchema(
        columns={
            "cod_ibge": Column(
                int,
                Check.str_length(7, 7, lambda x: str(x)),
                nullable=False,
            ),
            "ano": Column(
                int,
                Check.in_range(year_start, year_end),
                nullable=False,
            ),
            "produtividade_kg_ha": Column(
                float,
                Check.in_range(prod_min, prod_max),
                nullable=False,
            ),
            "area_colhida_ha": Column(
                float,
                Check.greater_than_or_equal_to(0),
                nullable=True,
                required=False,
            ),
            "producao_ton": Column(
                float,
                Check.greater_than_or_equal_to(0),
                nullable=True,
                required=False,
            ),
        },
        unique=["cod_ibge", "ano"],
        coerce=True,
    )

    return schema


def calculate_coverage_stats(df: pd.DataFrame) -> dict:
    """Calcula estatisticas de cobertura e missingness."""
    stats = {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "anos_unicos": df["ano"].nunique(),
        "periodo": f"{df['ano'].min()} - {df['ano'].max()}",
    }

    coverage_year = (
        df.groupby("ano")
        .agg(
            municipios=("cod_ibge", "nunique"),
            prod_mean=("produtividade_kg_ha", "mean"),
            prod_median=("produtividade_kg_ha", "median"),
        )
        .round(1)
    )
    stats["cobertura_por_ano"] = coverage_year.to_dict("index")

    df_temp = df.copy()
    df_temp["uf_code"] = df_temp["cod_ibge"].astype(str).str[:2]
    coverage_uf = df_temp.groupby("uf_code").agg(
        municipios=("cod_ibge", "nunique"),
        registros=("cod_ibge", "count"),
    )
    stats["cobertura_por_uf"] = coverage_uf.to_dict("index")

    missingness = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = 100 * null_count / len(df)
        missingness[col] = {"nulls": int(null_count), "pct": round(null_pct, 2)}
    stats["missingness"] = missingness

    return stats


def main():
    """Executa validacao do arquivo de target."""
    logger.info("=" * 60)
    logger.info("VALIDACAO DO TARGET - PRODUTIVIDADE DE SOJA")
    logger.info("=" * 60)

    if not DATA_PATH.exists():
        logger.error(f"Arquivo nao encontrado: {DATA_PATH}")
        logger.error("Execute primeiro: python -m src.ingest.pam")
        return

    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Dados carregados: {len(df):,} registros")

    config = load_config()

    results = validate_target(df, config)

    coverage = calculate_coverage_stats(df)

    logger.info("\n" + "=" * 60)
    logger.info("ESTATISTICAS DE COBERTURA")
    logger.info("=" * 60)
    logger.info(f"Total registros: {coverage['total_registros']:,}")
    logger.info(f"Municipios unicos: {coverage['municipios_unicos']:,}")
    logger.info(f"Anos unicos: {coverage['anos_unicos']}")
    logger.info(f"Periodo: {coverage['periodo']}")

    logger.info("\nMissingness por coluna:")
    for col, info in coverage["missingness"].items():
        logger.info(f"  {col}: {info['nulls']:,} ({info['pct']:.1f}%)")

    logger.info("\n" + "=" * 60)
    if results["all_passed"]:
        logger.info("TODAS AS VALIDACOES PASSARAM!")
    else:
        logger.error("ALGUMAS VALIDACOES FALHARAM!")
    logger.info("=" * 60)

    return results, coverage


if __name__ == "__main__":
    main()
