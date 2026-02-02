import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "target.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"

SIDRA_BASE_URL = "https://apisidra.ibge.gov.br/values"


def load_config() -> dict[str, Any]:
    """Carrega configuracao do target.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["target"]


def build_sidra_url(
    table: int,
    variables: list[int],
    year: int,
    product_code: int = 40124,
) -> str:
    """Constroi URL para a API SIDRA (um ano por vez)."""
    vars_str = ",".join(str(v) for v in variables)

    url = (
        f"{SIDRA_BASE_URL}"
        f"/t/{table}"
        f"/n6/all"
        f"/v/{vars_str}"
        f"/p/{year}"
        f"/c782/{product_code}"
    )

    return url


def download_pam_data_year(url: str, year: int) -> pd.DataFrame | None:
    """Baixa dados da API SIDRA para um ano especifico."""
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        data = response.json()

        if len(data) < 2:
            logger.warning(f"Ano {year}: sem dados")
            return None

        rows = data[1:]
        df = pd.DataFrame(rows)

        logger.info(f"Ano {year}: {len(df):,} registros")
        return df

    except requests.exceptions.RequestException as e:
        logger.warning(f"Ano {year}: erro na requisicao - {e}")
        return None


def download_pam_data(
    table: int,
    variables: list[int],
    year_start: int,
    year_end: int,
    product_code: int = 40124,
) -> pd.DataFrame:
    """Baixa dados da API SIDRA para todos os anos."""
    logger.info("Baixando dados da PAM via SIDRA...")
    logger.info(f"Periodo: {year_start} - {year_end}")

    all_dfs = []

    for year in range(year_start, year_end + 1):
        url = build_sidra_url(table, variables, year, product_code)

        df_year = download_pam_data_year(url, year)
        if df_year is not None:
            all_dfs.append(df_year)

        time.sleep(0.5)

    if not all_dfs:
        raise ValueError("Nenhum dado retornado da API SIDRA")

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total de registros brutos: {len(df):,}")

    return df


def process_pam_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processa dados brutos da PAM para formato padronizado."""
    logger.info("Processando dados da PAM...")

    col_mapping = {
        "NC": "nivel_codigo",
        "NN": "nivel_nome",
        "MC": "unidade_medida_codigo",
        "MN": "unidade_medida",
        "D1C": "cod_ibge",
        "D1N": "municipio_nome",
        "D2C": "variavel_codigo",
        "D2N": "variavel_nome",
        "D3C": "ano_codigo",
        "D3N": "ano",
        "D4C": "produto_codigo",
        "D4N": "produto_nome",
        "V": "valor",
    }

    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    logger.info(f"Colunas apos renomeacao: {list(df.columns)}")

    if "nivel_codigo" in df.columns:
        df = df[df["nivel_codigo"] == "6"].copy()

    df["cod_ibge"] = pd.to_numeric(df["cod_ibge"], errors="coerce")
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["variavel_codigo"] = pd.to_numeric(df["variavel_codigo"], errors="coerce")

    df = df.dropna(subset=["cod_ibge", "ano"])

    df_pivot = df.pivot_table(
        index=["cod_ibge", "ano"],
        columns="variavel_codigo",
        values="valor",
        aggfunc="first",
    ).reset_index()

    col_rename = {
        216: "area_colhida_ha",
        214: "producao_ton",
        112: "produtividade_kg_ha",
    }

    df_pivot = df_pivot.rename(columns=col_rename)

    df_pivot["cod_ibge"] = df_pivot["cod_ibge"].astype("int64")
    df_pivot["ano"] = df_pivot["ano"].astype("int32")

    df_pivot = df_pivot.sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    cols_final = ["cod_ibge", "ano", "produtividade_kg_ha"]
    if "area_colhida_ha" in df_pivot.columns:
        cols_final.append("area_colhida_ha")
    if "producao_ton" in df_pivot.columns:
        cols_final.append("producao_ton")

    df_final = df_pivot[cols_final].copy()

    if "area_colhida_ha" in df_final.columns:
        n_before = len(df_final)
        df_final = df_final[df_final["area_colhida_ha"] > 0].copy()
        n_removed = n_before - len(df_final)
        if n_removed > 0:
            logger.info(f"Removidos {n_removed} registros com area_colhida = 0")

    n_before = len(df_final)
    df_final = df_final.dropna(subset=["produtividade_kg_ha"])
    n_removed = n_before - len(df_final)
    if n_removed > 0:
        logger.info(f"Removidos {n_removed} registros com produtividade nula")

    OUTLIER_THRESHOLD = 10000
    n_before = len(df_final)
    outliers = df_final[df_final["produtividade_kg_ha"] > OUTLIER_THRESHOLD]
    if len(outliers) > 0:
        logger.warning(f"Outliers extremos removidos (> {OUTLIER_THRESHOLD} kg/ha):")
        for _, row in outliers.iterrows():
            logger.warning(
                f"  cod_ibge={row['cod_ibge']}, ano={row['ano']}, prod={row['produtividade_kg_ha']:.0f} kg/ha"
            )
        df_final = df_final[df_final["produtividade_kg_ha"] <= OUTLIER_THRESHOLD].copy()
        logger.info(
            f"Removidos {n_before - len(df_final)} registros com produtividade > {OUTLIER_THRESHOLD} kg/ha"
        )

    logger.info(f"Registros processados: {len(df_final):,}")
    logger.info(f"Municipios unicos: {df_final['cod_ibge'].nunique():,}")
    logger.info(f"Anos: {df_final['ano'].min()} - {df_final['ano'].max()}")

    return df_final


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcula estatisticas do dataset para documentacao."""
    stats = {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "anos": {
            "min": int(df["ano"].min()),
            "max": int(df["ano"].max()),
            "total": df["ano"].nunique(),
        },
        "produtividade_kg_ha": {
            "min": float(df["produtividade_kg_ha"].min()),
            "max": float(df["produtividade_kg_ha"].max()),
            "mean": float(df["produtividade_kg_ha"].mean()),
            "median": float(df["produtividade_kg_ha"].median()),
            "nulls": int(df["produtividade_kg_ha"].isna().sum()),
        },
    }

    coverage_by_year = df.groupby("ano")["cod_ibge"].nunique().to_dict()
    stats["cobertura_por_ano"] = {int(k): v for k, v in coverage_by_year.items()}

    df_temp = df.copy()
    df_temp["uf_code"] = df_temp["cod_ibge"].astype(str).str[:2]
    coverage_by_uf = df_temp.groupby("uf_code")["cod_ibge"].nunique().to_dict()
    stats["cobertura_por_uf"] = coverage_by_uf

    if "area_colhida_ha" in df.columns:
        stats["area_colhida_nulls"] = int(df["area_colhida_ha"].isna().sum())
    if "producao_ton" in df.columns:
        stats["producao_ton_nulls"] = int(df["producao_ton"].isna().sum())

    return stats


def main():
    """Pipeline principal de ingestao da PAM."""
    logger.info("=" * 60)
    logger.info("INGESTAO PAM - PRODUTIVIDADE DE SOJA")
    logger.info("=" * 60)

    config = load_config()
    logger.info(
        f"Configuracao carregada: {config['crop']}, {config['year_start']}-{config['year_end']}"
    )

    df_raw = download_pam_data(
        table=config["sidra_table"],
        variables=[112, 214, 216],
        year_start=config["year_start"],
        year_end=config["year_end"],
    )

    df_processed = process_pam_data(df_raw)

    stats = calculate_statistics(df_processed)

    logger.info("\n" + "=" * 60)
    logger.info("ESTATISTICAS DO DATASET")
    logger.info("=" * 60)
    logger.info(f"Total de registros: {stats['total_registros']:,}")
    logger.info(f"Municipios unicos: {stats['municipios_unicos']:,}")
    logger.info(f"Periodo: {stats['anos']['min']} - {stats['anos']['max']}")
    logger.info(f"Produtividade media: {stats['produtividade_kg_ha']['mean']:.1f} kg/ha")
    logger.info(f"Produtividade mediana: {stats['produtividade_kg_ha']['median']:.1f} kg/ha")
    logger.info(f"Produtividade min: {stats['produtividade_kg_ha']['min']:.1f} kg/ha")
    logger.info(f"Produtividade max: {stats['produtividade_kg_ha']['max']:.1f} kg/ha")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nArquivo salvo: {OUTPUT_PATH}")

    return df_processed, stats


if __name__ == "__main__":
    main()
