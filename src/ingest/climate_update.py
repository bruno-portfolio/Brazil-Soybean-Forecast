from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "climate.yaml"
MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "climate"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"

NEW_PARAMETERS = ["ALLSKY_SFC_SW_DWN", "WS2M"]


def load_config() -> dict:
    """Carrega configuracao do climate.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["climate"]


def load_municipalities() -> pd.DataFrame:
    """Carrega tabela de municipios."""
    return pd.read_parquet(MUNICIPALITIES_PATH)


def load_target_municipalities() -> set[int]:
    """Carrega lista de municipios que produzem soja."""
    df = pd.read_parquet(TARGET_PATH)
    return set(df["cod_ibge"].unique())


def download_new_params(cod_ibge: int, lat: float, lon: float, config: dict) -> pd.DataFrame | None:
    """Baixa apenas os novos parametros (radiacao e vento)."""
    start_date = f"{config['year_start']}0101"
    end_date = f"{config['year_end']}1231"

    params_str = ",".join(NEW_PARAMETERS)
    url = (
        f"{config['api']['base_url']}"
        f"?parameters={params_str}"
        f"&community={config['api']['community']}"
        f"&longitude={lon:.4f}"
        f"&latitude={lat:.4f}"
        f"&start={start_date}"
        f"&end={end_date}"
        f"&format={config['api']['format']}"
    )

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            return None

        params = data["properties"]["parameter"]
        records = []
        first_param = list(params.keys())[0]

        for date_str in params[first_param].keys():
            record = {"date": date_str}
            for param_name, param_values in params.items():
                value = param_values.get(date_str)
                if value == -999 or value == -999.0:
                    value = None
                record[param_name] = value
            records.append(record)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["cod_ibge"] = cod_ibge

        df = df.rename(columns={
            "ALLSKY_SFC_SW_DWN": "radiation",
            "WS2M": "wind_speed",
        })

        return df

    except Exception as e:
        logger.warning(f"cod_ibge={cod_ibge}: erro - {e}")
        return None


def update_climate_cache():
    """Atualiza cache com novos parametros."""
    logger.info("=" * 60)
    logger.info("ATUALIZACAO CLIMA - Radiacao e Vento")
    logger.info("=" * 60)

    config = load_config()
    df_mun = load_municipalities()
    soy_producers = load_target_municipalities()
    df_mun = df_mun[df_mun["cod_ibge"].isin(soy_producers)]

    logger.info(f"Municipios a processar: {len(df_mun):,}")

    new_cache_dir = PROJECT_ROOT / "data" / "raw" / "climate_extra"
    new_cache_dir.mkdir(parents=True, exist_ok=True)

    cached = {int(f.stem) for f in new_cache_dir.glob("*.parquet")}
    pending = df_mun[~df_mun["cod_ibge"].isin(cached)]

    logger.info(f"Ja em cache: {len(cached):,}")
    logger.info(f"Pendentes: {len(pending):,}")

    success = 0
    failed = 0
    delay = 60.0 / config["rate_limit"]["requests_per_minute"]

    for _, row in pending.iterrows():
        cod_ibge = row["cod_ibge"]
        lat = row["lat"]
        lon = row["lon"]

        df_new = download_new_params(cod_ibge, lat, lon, config)

        if df_new is not None:
            cache_path = new_cache_dir / f"{cod_ibge}.parquet"
            df_new.to_parquet(cache_path, index=False)
            success += 1
            if success % 50 == 0:
                logger.info(f"Processados: {success + failed}/{len(pending)}")
        else:
            failed += 1

        time.sleep(delay)

    logger.info(f"Concluido - Sucesso: {success}, Falha: {failed}")
    return success, failed


def merge_climate_data():
    """Merge dados originais com novos parametros."""
    logger.info("Consolidando dados de clima...")

    df_original = pd.read_parquet(OUTPUT_PATH)
    logger.info(f"Dados originais: {len(df_original):,} registros")

    new_cache_dir = PROJECT_ROOT / "data" / "raw" / "climate_extra"

    if not new_cache_dir.exists():
        logger.warning("Cache de dados extras nao encontrado")
        return df_original

    all_new = []
    for f in new_cache_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            all_new.append(df)
        except Exception as e:
            logger.warning(f"Erro ao ler {f}: {e}")

    if not all_new:
        logger.warning("Nenhum dado extra encontrado")
        return df_original

    df_new = pd.concat(all_new, ignore_index=True)
    logger.info(f"Dados novos: {len(df_new):,} registros")

    df_new["date"] = pd.to_datetime(df_new["date"])
    df_original["date"] = pd.to_datetime(df_original["date"])

    df_merged = df_original.merge(
        df_new[["cod_ibge", "date", "radiation", "wind_speed"]],
        on=["cod_ibge", "date"],
        how="left"
    )

    n_with_radiation = df_merged["radiation"].notna().sum()
    logger.info(f"Registros com radiacao: {n_with_radiation:,} ({100*n_with_radiation/len(df_merged):.1f}%)")

    output_merged = PROJECT_ROOT / "data" / "processed" / "climate_daily_v2.parquet"
    df_merged.to_parquet(output_merged, index=False)
    logger.info(f"Salvo em: {output_merged}")

    return df_merged


def main():
    """Pipeline principal."""
    update_climate_cache()
    merge_climate_data()


if __name__ == "__main__":
    main()
