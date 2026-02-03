from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ndvi_safra.parquet"

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("earthengine-api nao instalado. Execute: pip install earthengine-api")


GEE_PROJECT = "teste-483217"


def initialize_gee(project_id: str | None = None) -> bool:
    """Inicializa Google Earth Engine."""
    if not GEE_AVAILABLE:
        return False

    project = project_id or GEE_PROJECT
    try:
        ee.Initialize(project=project)
        logger.info(f"GEE inicializado com projeto: {project}")
        return True
    except Exception as e:
        logger.info(f"GEE nao inicializado: {e}")
        logger.info("Para usar NDVI, configure GEE: https://earthengine.google.com/signup/")
        return False


def load_municipalities() -> pd.DataFrame:
    """Carrega municipios."""
    return pd.read_parquet(MUNICIPALITIES_PATH)


def load_target_municipalities() -> set[int]:
    """Carrega lista de municipios produtores de soja."""
    df = pd.read_parquet(TARGET_PATH)
    return set(df["cod_ibge"].unique())


def get_ndvi_for_point(lon: float, lat: float, start_date: str, end_date: str) -> dict[str, Any]:
    """Extrai NDVI medio para um ponto usando MODIS MOD13Q1."""
    point = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start_date, end_date)
        .filterBounds(point)
        .select("NDVI")
    )

    def extract_value(image):
        value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=250
        ).get("NDVI")
        return image.set("ndvi_value", value)

    with_values = collection.map(extract_value)

    values = with_values.aggregate_array("ndvi_value").getInfo()

    valid_values = [v * 0.0001 for v in values if v is not None]

    if not valid_values:
        return {
            "ndvi_mean": None,
            "ndvi_max": None,
            "ndvi_min": None,
            "ndvi_count": 0
        }

    return {
        "ndvi_mean": sum(valid_values) / len(valid_values),
        "ndvi_max": max(valid_values),
        "ndvi_min": min(valid_values),
        "ndvi_count": len(valid_values)
    }


def get_ndvi_by_phase(
    lon: float, lat: float, year: int
) -> dict[str, float | None]:
    """Extrai NDVI por fase fenologica para uma safra."""
    phases = {
        "plantio": (f"{year-1}-10-01", f"{year-1}-11-30"),
        "vegetativo": (f"{year-1}-12-01", f"{year}-01-31"),
        "enchimento": (f"{year}-02-01", f"{year}-03-31"),
    }

    results = {}

    safra_start = f"{year-1}-10-01"
    safra_end = f"{year}-03-31"
    safra_ndvi = get_ndvi_for_point(lon, lat, safra_start, safra_end)

    results["ndvi_mean_safra"] = safra_ndvi["ndvi_mean"]
    results["ndvi_max_safra"] = safra_ndvi["ndvi_max"]
    results["ndvi_min_safra"] = safra_ndvi["ndvi_min"]

    if safra_ndvi["ndvi_max"] and safra_ndvi["ndvi_min"]:
        results["ndvi_amplitude"] = safra_ndvi["ndvi_max"] - safra_ndvi["ndvi_min"]
    else:
        results["ndvi_amplitude"] = None

    for phase, (start, end) in phases.items():
        phase_ndvi = get_ndvi_for_point(lon, lat, start, end)
        results[f"ndvi_{phase}"] = phase_ndvi["ndvi_mean"]

    return results


def process_all_municipalities(
    years: list[int] | None = None,
    max_municipalities: int | None = None,
    batch_size: int = 100
) -> pd.DataFrame:
    """Processa NDVI para todos os municipios."""
    if not initialize_gee():
        raise RuntimeError("GEE nao disponivel")

    if years is None:
        years = list(range(2000, 2026))

    df_mun = load_municipalities()
    soy_producers = load_target_municipalities()
    df_mun = df_mun[df_mun["cod_ibge"].isin(soy_producers)]

    if max_municipalities:
        df_mun = df_mun.head(max_municipalities)

    logger.info(f"Processando {len(df_mun)} municipios x {len(years)} anos")

    cache_path = PROJECT_ROOT / "data" / "raw" / "ndvi_gee"
    cache_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    processed = 0

    for _, row in df_mun.iterrows():
        cod_ibge = row["cod_ibge"]
        lat = row["lat"]
        lon = row["lon"]

        cache_file = cache_path / f"{cod_ibge}.parquet"

        if cache_file.exists():
            df_cached = pd.read_parquet(cache_file)
            all_results.append(df_cached)
            processed += 1
            continue

        mun_results = []

        for year in years:
            try:
                ndvi_data = get_ndvi_by_phase(lon, lat, year)
                ndvi_data["cod_ibge"] = cod_ibge
                ndvi_data["ano"] = year
                mun_results.append(ndvi_data)
            except Exception as e:
                logger.warning(f"Erro cod_ibge={cod_ibge}, ano={year}: {e}")

        if mun_results:
            df_mun_result = pd.DataFrame(mun_results)
            df_mun_result.to_parquet(cache_file, index=False)
            all_results.append(df_mun_result)

        processed += 1
        if processed % batch_size == 0:
            logger.info(f"Processados: {processed}/{len(df_mun)}")

    if not all_results:
        raise ValueError("Nenhum dado NDVI processado")

    df_final = pd.concat(all_results, ignore_index=True)

    df_final.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Salvo em: {OUTPUT_PATH}")
    logger.info(f"Total: {len(df_final)} registros")

    return df_final


def main():
    """Pipeline principal."""
    logger.info("=" * 60)
    logger.info("INGESTAO NDVI - Google Earth Engine")
    logger.info("=" * 60)

    df = process_all_municipalities(
        years=list(range(2000, 2026)),
        max_municipalities=None,
        batch_size=50
    )

    logger.info("\nEstatisticas:")
    logger.info(f"  Municipios: {df['cod_ibge'].nunique()}")
    logger.info(f"  Anos: {df['ano'].nunique()}")
    logger.info(f"  NDVI medio: {df['ndvi_mean_safra'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
