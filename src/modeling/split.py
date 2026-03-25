from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    pass

from src.common.io import PROJECT_ROOT

logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "configs" / "split.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "processed"


@dataclass
class SplitConfig:
    """Configuracao do split temporal."""

    train_end_year: int
    val_start_year: int
    val_end_year: int
    test_start_year: int
    test_end_year: int
    inference_years: list[int]


def load_config() -> SplitConfig:
    """Carrega configuracao do split.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    split_cfg = config["split"]
    return SplitConfig(
        train_end_year=split_cfg["train"]["end_year"],
        val_start_year=split_cfg["validation"]["start_year"],
        val_end_year=split_cfg["validation"]["end_year"],
        test_start_year=split_cfg["test"]["start_year"],
        test_end_year=split_cfg["test"]["end_year"],
        inference_years=split_cfg["inference"]["years"],
    )


def load_dataset() -> pd.DataFrame:
    """Carrega dataset final."""
    path = DATA_PATH / "dataset_final.parquet"
    return pd.read_parquet(path)


@dataclass
class TemporalSplit:
    """Resultado do split temporal."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    config: SplitConfig

    def summary(self) -> dict:
        """Retorna resumo do split."""
        return {
            "train": {
                "n_samples": len(self.train),
                "n_municipalities": self.train["cod_ibge"].nunique(),
                "years": sorted(self.train["ano"].unique().tolist()),
                "year_range": f"<= {self.config.train_end_year}",
            },
            "validation": {
                "n_samples": len(self.validation),
                "n_municipalities": self.validation["cod_ibge"].nunique(),
                "years": sorted(self.validation["ano"].unique().tolist()),
                "year_range": f"{self.config.val_start_year}-{self.config.val_end_year}",
            },
            "test": {
                "n_samples": len(self.test),
                "n_municipalities": self.test["cod_ibge"].nunique(),
                "years": sorted(self.test["ano"].unique().tolist()),
                "year_range": f"{self.config.test_start_year}-{self.config.test_end_year}",
            },
        }


def validate_no_leakage(split: TemporalSplit) -> bool:
    """Valida ausencia de leakage temporal."""
    cfg = split.config

    train_max_year = split.train["ano"].max()
    if train_max_year > cfg.train_end_year:
        raise ValueError(
            f"Leakage detectado: treino contem ano {train_max_year} > {cfg.train_end_year}"
        )

    val_years = split.validation["ano"].unique()
    for year in val_years:
        if year < cfg.val_start_year or year > cfg.val_end_year:
            raise ValueError(
                f"Leakage detectado: validacao contem ano {year} fora do range "
                f"[{cfg.val_start_year}, {cfg.val_end_year}]"
            )

    test_years = split.test["ano"].unique()
    for year in test_years:
        if year < cfg.test_start_year or year > cfg.test_end_year:
            raise ValueError(
                f"Leakage detectado: teste contem ano {year} fora do range "
                f"[{cfg.test_start_year}, {cfg.test_end_year}]"
            )

    if split.train["ano"].max() >= split.validation["ano"].min():
        raise ValueError("Leakage: treino se sobrepoe com validacao")

    if split.validation["ano"].max() >= split.test["ano"].min():
        raise ValueError("Leakage: validacao se sobrepoe com teste")

    return True


def create_temporal_split(df: pd.DataFrame | None = None) -> TemporalSplit:
    """Cria split temporal dos dados."""
    if df is None:
        df = load_dataset()

    config = load_config()

    train = df[df["ano"] <= config.train_end_year].copy()
    validation = df[
        (df["ano"] >= config.val_start_year) & (df["ano"] <= config.val_end_year)
    ].copy()
    test = df[(df["ano"] >= config.test_start_year) & (df["ano"] <= config.test_end_year)].copy()

    split = TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        config=config,
    )

    validate_no_leakage(split)

    return split


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna colunas de features (exclui target e identificadores)."""
    exclude = ["cod_ibge", "ano", "produtividade_kg_ha", "area_colhida_ha", "producao_ton"]
    categorical_string_cols = ["texture_class"]
    exclude = exclude + categorical_string_cols
    return [col for col in df.columns if col not in exclude]


def main() -> None:
    """Pipeline principal de split."""
    logger.info("=" * 60)
    logger.info("SPLIT TEMPORAL DOS DADOS")
    logger.info("=" * 60)

    config = load_config()
    logger.info("Configuracao:")
    logger.info(f"  Treino: <= {config.train_end_year}")
    logger.info(f"  Validacao: {config.val_start_year}-{config.val_end_year}")
    logger.info(f"  Teste: {config.test_start_year}-{config.test_end_year}")

    logger.info("Criando split temporal...")
    split = create_temporal_split()

    summary = split.summary()
    logger.info("Resumo do split:")
    for name, stats in summary.items():
        logger.info(f"  {name.upper()}:")
        logger.info(f"    Amostras: {stats['n_samples']:,}")
        logger.info(f"    Municipios: {stats['n_municipalities']:,}")
        logger.info(f"    Anos: {stats['year_range']}")
        logger.info(f"    Anos especificos: {stats['years']}")

    logger.info("=" * 60)
    logger.info("VALIDACAO DE LEAKAGE")
    logger.info("=" * 60)
    try:
        validate_no_leakage(split)
        logger.info("[OK] Nenhum leakage temporal detectado")
        logger.info(f"[OK] Treino: max ano = {split.train['ano'].max()} <= {config.train_end_year}")
        logger.info(f"[OK] Validacao: anos = {sorted(split.validation['ano'].unique().tolist())}")
        logger.info(f"[OK] Teste: anos = {sorted(split.test['ano'].unique().tolist())}")
    except ValueError as e:
        logger.warning(f"[ERRO] {e}")

    logger.info("=" * 60)
    logger.info("ESTATISTICAS DO TARGET POR SPLIT")
    logger.info("=" * 60)
    target_col = "produtividade_kg_ha"
    for name, data in [
        ("Treino", split.train),
        ("Validacao", split.validation),
        ("Teste", split.test),
    ]:
        stats = data[target_col].describe()
        logger.info(f"{name}:")
        logger.info(f"  Media: {stats['mean']:.1f} kg/ha ({stats['mean'] / 60:.1f} sacas/ha)")
        logger.info(f"  Mediana: {stats['50%']:.1f} kg/ha")
        logger.info(f"  Std: {stats['std']:.1f} kg/ha")
        logger.info(f"  Min-Max: {stats['min']:.0f} - {stats['max']:.0f} kg/ha")


if __name__ == "__main__":
    main()
