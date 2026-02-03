from __future__ import annotations

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
CONFIG_PATH = PROJECT_ROOT / "configs" / "climate.yaml"
MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "climate"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "climate_daily.parquet"


def load_config() -> dict[str, Any]:
    """Carrega configuracao do climate.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["climate"]


def load_municipalities() -> pd.DataFrame:
    """Carrega tabela de municipios com coordenadas."""
    if not MUNICIPALITIES_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de municipios nao encontrado: {MUNICIPALITIES_PATH}\n"
            "Execute primeiro: python -m src.ingest.municipalities"
        )
    return pd.read_parquet(MUNICIPALITIES_PATH)


def load_target_municipalities() -> set[int]:
    """Carrega lista de municipios que produzem soja."""
    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de target nao encontrado: {TARGET_PATH}\n"
            "Execute primeiro: python -m src.ingest.pam"
        )
    df = pd.read_parquet(TARGET_PATH)
    return set(df["cod_ibge"].unique())


def build_power_url(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: list[str],
    config: dict,
) -> str:
    """Constroi URL para a API NASA POWER."""
    params_str = ",".join(parameters)
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
    return url


def download_climate_for_municipality(
    cod_ibge: int,
    lat: float,
    lon: float,
    config: dict,
) -> pd.DataFrame | None:
    """Baixa dados climaticos para um municipio."""
    start_date = f"{config['year_start']}0101"
    end_date = f"{config['year_end']}1231"

    url = build_power_url(lat, lon, start_date, end_date, config["parameters"], config)

    retry_attempts = config["rate_limit"]["retry_attempts"]
    backoff_factor = config["rate_limit"]["backoff_factor"]

    for attempt in range(retry_attempts):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            data = response.json()

            if "properties" not in data or "parameter" not in data["properties"]:
                logger.warning(f"cod_ibge={cod_ibge}: resposta sem dados")
                return None

            params = data["properties"]["parameter"]

            records = []
            first_param = list(params.keys())[0]
            for date_str, _ in params[first_param].items():
                record = {"date": date_str}
                for param_name, param_values in params.items():
                    value = param_values.get(date_str)
                    if value == -999 or value == -999.0:
                        value = None
                    record[param_name] = value
                records.append(record)

            df = pd.DataFrame(records)

            if len(df) == 0:
                logger.warning(f"cod_ibge={cod_ibge}: DataFrame vazio")
                return None

            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df["cod_ibge"] = cod_ibge

            col_rename = {
                "T2M": "tmean",
                "T2M_MIN": "tmin",
                "T2M_MAX": "tmax",
                "PRECTOTCORR": "precip",
                "RH2M": "rh",
            }
            df = df.rename(columns=col_rename)

            return df

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = backoff_factor ** (attempt + 1)
                logger.warning(f"Rate limit atingido, aguardando {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"cod_ibge={cod_ibge}: HTTP {response.status_code} - {e}")
                return None

        except requests.exceptions.RequestException as e:
            wait_time = backoff_factor ** (attempt + 1)
            logger.warning(
                f"cod_ibge={cod_ibge}: erro na requisicao (tentativa {attempt + 1}) - {e}"
            )
            if attempt < retry_attempts - 1:
                time.sleep(wait_time)

        except Exception as e:
            logger.warning(f"cod_ibge={cod_ibge}: erro inesperado - {e}")
            return None

    return None


def get_cached_municipalities() -> set[int]:
    """Retorna conjunto de municipios ja presentes no cache."""
    if not CACHE_DIR.exists():
        return set()

    cached = set()
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            cod_ibge = int(f.stem)
            cached.add(cod_ibge)
        except ValueError:
            continue

    return cached


def save_to_cache(df: pd.DataFrame, cod_ibge: int) -> None:
    """Salva dados de um municipio no cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cod_ibge}.parquet"
    df.to_parquet(cache_path, index=False)


def load_from_cache(cod_ibge: int) -> pd.DataFrame | None:
    """Carrega dados de um municipio do cache."""
    cache_path = CACHE_DIR / f"{cod_ibge}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


def consolidate_cache() -> pd.DataFrame:
    """Consolida todos os arquivos de cache em um unico DataFrame."""
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Diretorio de cache nao encontrado: {CACHE_DIR}")

    all_dfs = []
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Erro ao ler {f}: {e}")

    if not all_dfs:
        raise ValueError("Nenhum arquivo de cache encontrado")

    df = pd.concat(all_dfs, ignore_index=True)
    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcula estatisticas do dataset climatico."""
    stats = {
        "total_registros": len(df),
        "municipios_unicos": df["cod_ibge"].nunique(),
        "periodo": {
            "min": str(df["date"].min().date()),
            "max": str(df["date"].max().date()),
        },
        "tmean": {
            "min": float(df["tmean"].min()) if "tmean" in df.columns else None,
            "max": float(df["tmean"].max()) if "tmean" in df.columns else None,
            "nulls": int(df["tmean"].isna().sum()) if "tmean" in df.columns else None,
        },
        "precip": {
            "min": float(df["precip"].min()) if "precip" in df.columns else None,
            "max": float(df["precip"].max()) if "precip" in df.columns else None,
            "nulls": int(df["precip"].isna().sum()) if "precip" in df.columns else None,
        },
    }
    return stats


def fetch_climate_for_municipalities(
    only_soy_producers: bool = True, max_municipalities: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Pipeline principal de ingestao de clima."""
    logger.info("=" * 60)
    logger.info("INGESTAO CLIMA - NASA POWER")
    logger.info("=" * 60)

    config = load_config()
    logger.info(f"Periodo: {config['year_start']} - {config['year_end']}")
    logger.info(f"Parametros: {config['parameters']}")

    df_mun = load_municipalities()
    logger.info(f"Municipios disponiveis: {len(df_mun):,}")

    if only_soy_producers:
        soy_producers = load_target_municipalities()
        df_mun = df_mun[df_mun["cod_ibge"].isin(soy_producers)]
        logger.info(f"Municipios produtores de soja: {len(df_mun):,}")

    if max_municipalities is not None:
        df_mun = df_mun.head(max_municipalities)
        logger.info(f"Limitado a {max_municipalities} municipios para teste")

    cached = get_cached_municipalities()
    logger.info(f"Municipios ja em cache: {len(cached):,}")

    pending = df_mun[~df_mun["cod_ibge"].isin(cached)]
    logger.info(f"Municipios pendentes: {len(pending):,}")

    requests_per_minute = config["rate_limit"]["requests_per_minute"]
    delay = 60.0 / requests_per_minute

    success = 0
    failed = 0
    failed_list = []

    for _, row in pending.iterrows():
        cod_ibge = row["cod_ibge"]
        lat = row["lat"]
        lon = row["lon"]

        df_climate = download_climate_for_municipality(cod_ibge, lat, lon, config)

        if df_climate is not None:
            save_to_cache(df_climate, cod_ibge)
            success += 1
            logger.info(
                f"[{success + failed}/{len(pending)}] cod_ibge={cod_ibge}: OK ({len(df_climate)} dias)"
            )
        else:
            failed += 1
            failed_list.append(cod_ibge)
            logger.warning(f"[{success + failed}/{len(pending)}] cod_ibge={cod_ibge}: FALHOU")

        time.sleep(delay)

    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD CONCLUIDO")
    logger.info("=" * 60)
    logger.info(f"Sucesso: {success:,}")
    logger.info(f"Falha: {failed:,}")

    if failed_list:
        logger.info(f"Municipios com falha: {failed_list[:20]}...")

    logger.info("\nConsolidando cache...")
    df_consolidated = consolidate_cache()

    df_consolidated = df_consolidated.sort_values(["cod_ibge", "date"]).reset_index(drop=True)

    cols = ["cod_ibge", "date", "tmin", "tmean", "tmax", "precip", "rh"]
    cols = [c for c in cols if c in df_consolidated.columns]
    df_consolidated = df_consolidated[cols]

    stats = calculate_statistics(df_consolidated)

    logger.info("\n" + "=" * 60)
    logger.info("ESTATISTICAS DO DATASET")
    logger.info("=" * 60)
    logger.info(f"Total de registros: {stats['total_registros']:,}")
    logger.info(f"Municipios unicos: {stats['municipios_unicos']:,}")
    logger.info(f"Periodo: {stats['periodo']['min']} a {stats['periodo']['max']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_consolidated.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nArquivo salvo: {OUTPUT_PATH}")

    total_expected = len(df_mun)
    total_cached = df_consolidated["cod_ibge"].nunique()
    success_rate = 100 * total_cached / total_expected if total_expected > 0 else 0

    logger.info(f"\nTaxa de cobertura: {success_rate:.1f}% ({total_cached}/{total_expected})")

    return df_consolidated, stats


def main(only_soy_producers: bool = True, max_municipalities: int | None = None):
    """Pipeline principal de ingestao de clima."""
    return fetch_climate_for_municipalities(only_soy_producers, max_municipalities)


if __name__ == "__main__":
    main()
