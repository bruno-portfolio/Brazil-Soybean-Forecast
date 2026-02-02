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
CONFIG_PATH = PROJECT_ROOT / "configs" / "soil.yaml"
MUNICIPALITIES_PATH = PROJECT_ROOT / "data" / "processed" / "municipalities.parquet"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target_soja.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "soil"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "soil_properties.parquet"


def load_config() -> dict[str, Any]:
    """Carrega configuracao do soil.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["soil"]


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


def build_soilgrids_url(
    lat: float,
    lon: float,
    properties: list[str],
    depths: list[str],
    values: list[str],
    config: dict,
) -> str:
    """Constroi URL para a API SoilGrids."""
    base_url = config["api"]["base_url"]

    params = [
        f"lon={lon:.6f}",
        f"lat={lat:.6f}",
    ]

    for prop in properties:
        params.append(f"property={prop}")

    for depth in depths:
        params.append(f"depth={depth}")

    for value in values:
        params.append(f"value={value}")

    url = f"{base_url}?{'&'.join(params)}"
    return url


def parse_soilgrids_response(
    data: dict,
    cod_ibge: int,
    config: dict,
) -> dict[str, Any] | None:
    """Parseia resposta da API SoilGrids."""
    if "properties" not in data or "layers" not in data["properties"]:
        return None

    layers = data["properties"]["layers"]
    result = {"cod_ibge": cod_ibge}

    for layer in layers:
        prop_name = layer.get("name")
        if prop_name is None:
            continue

        depths_data = layer.get("depths", [])
        for depth_info in depths_data:
            depth_label = depth_info.get("label")
            values = depth_info.get("values", {})

            mean_value = values.get("mean")

            if mean_value is not None:
                col_name = f"{prop_name}_{depth_label}"
                result[col_name] = mean_value

    return result


def validate_soil_data(record: dict, config: dict) -> dict:
    """Valida e limpa dados de solo."""
    validation = config.get("validation", {})
    cleaned = {"cod_ibge": record["cod_ibge"]}

    for key, value in record.items():
        if key == "cod_ibge":
            continue

        prop_name = key.split("_")[0]

        if prop_name in validation and value is not None:
            limits = validation[prop_name]
            min_val = limits.get("min")
            max_val = limits.get("max")

            if min_val is not None and value < min_val:
                value = None
            if max_val is not None and value > max_val:
                value = None

        cleaned[key] = value

    return cleaned


def download_soil_for_municipality(
    cod_ibge: int,
    lat: float,
    lon: float,
    config: dict,
) -> dict[str, Any] | None:
    """Baixa dados de solo para um municipio."""
    properties = config["properties"]
    depths = config["depths"]
    values = config["values"]

    url = build_soilgrids_url(lat, lon, properties, depths, values, config)

    retry_attempts = config["rate_limit"]["retry_attempts"]
    backoff_factor = config["rate_limit"]["backoff_factor"]
    timeout = config["rate_limit"]["timeout_seconds"]

    for attempt in range(retry_attempts):
        try:
            response = requests.get(url, timeout=timeout)

            if response.status_code == 400:
                logger.warning(f"cod_ibge={cod_ibge}: ponto fora da cobertura SoilGrids")
                return None

            response.raise_for_status()
            data = response.json()

            record = parse_soilgrids_response(data, cod_ibge, config)

            if record is None or len(record) <= 1:
                logger.warning(f"cod_ibge={cod_ibge}: resposta sem dados de solo")
                return None

            record = validate_soil_data(record, config)

            return record

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = backoff_factor ** (attempt + 1)
                logger.warning(f"Rate limit atingido, aguardando {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 503:
                wait_time = backoff_factor ** (attempt + 1) * 5
                logger.warning(f"Servico indisponivel, aguardando {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"cod_ibge={cod_ibge}: HTTP {response.status_code} - {e}")
                return None

        except requests.exceptions.Timeout:
            wait_time = backoff_factor ** (attempt + 1)
            logger.warning(
                f"cod_ibge={cod_ibge}: timeout (tentativa {attempt + 1}), aguardando {wait_time}s..."
            )
            time.sleep(wait_time)

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


def save_to_cache(record: dict, cod_ibge: int) -> None:
    """Salva dados de um municipio no cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cod_ibge}.parquet"
    df = pd.DataFrame([record])
    df.to_parquet(cache_path, index=False)


def load_from_cache(cod_ibge: int) -> dict | None:
    """Carrega dados de um municipio do cache."""
    cache_path = CACHE_DIR / f"{cod_ibge}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df.iloc[0].to_dict()
    return None


def consolidate_cache() -> pd.DataFrame:
    """Consolida todos os arquivos de cache em um unico DataFrame."""
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Diretorio de cache nao encontrado: {CACHE_DIR}")

    all_records = []
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            all_records.append(df)
        except Exception as e:
            logger.warning(f"Erro ao ler {f}: {e}")

    if not all_records:
        raise ValueError("Nenhum arquivo de cache encontrado")

    df = pd.concat(all_records, ignore_index=True)
    return df


def aggregate_depths(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Agrega propriedades do solo por camadas (superficial e subsuperficial)."""
    aggregation = config["aggregation"]
    conversions = config.get("conversions", {})

    result = {"cod_ibge": df["cod_ibge"].values}

    properties = config["properties"]

    for prop in properties:
        surface_cfg = aggregation["surface"]
        surface_depths = surface_cfg["depths"]
        surface_label = surface_cfg["label"]

        surface_cols = [f"{prop}_{d}" for d in surface_depths if f"{prop}_{d}" in df.columns]
        if surface_cols:
            weights = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
            total_weight = sum(weights.get(d.split("_")[-1], 1) for d in surface_depths if f"{prop}_{d.split('_')[-1]}" in surface_cols)

            weighted_sum = 0
            for col in surface_cols:
                depth = col.replace(f"{prop}_", "")
                w = weights.get(depth, 1)
                weighted_sum += df[col].fillna(0) * w

            result[f"{prop}_{surface_label}"] = weighted_sum / total_weight if total_weight > 0 else None

        subsurface_cfg = aggregation["subsurface"]
        subsurface_depths = subsurface_cfg["depths"]
        subsurface_label = subsurface_cfg["label"]

        subsurface_cols = [f"{prop}_{d}" for d in subsurface_depths if f"{prop}_{d}" in df.columns]
        if subsurface_cols:
            weights = {"30-60cm": 30, "60-100cm": 40}
            total_weight = sum(weights.get(d.split("_")[-1], 1) for d in subsurface_depths if f"{prop}_{d.split('_')[-1]}" in subsurface_cols)

            weighted_sum = 0
            for col in subsurface_cols:
                depth = col.replace(f"{prop}_", "")
                w = weights.get(depth, 1)
                weighted_sum += df[col].fillna(0) * w

            result[f"{prop}_{subsurface_label}"] = weighted_sum / total_weight if total_weight > 0 else None

    df_agg = pd.DataFrame(result)

    for col in df_agg.columns:
        if col == "cod_ibge":
            continue

        prop = col.split("_")[0]

        if prop == "clay" and "clay_to_percent" in conversions:
            df_agg[col] = df_agg[col] * conversions["clay_to_percent"]
        elif prop == "sand" and "sand_to_percent" in conversions:
            df_agg[col] = df_agg[col] * conversions["sand_to_percent"]
        elif prop == "silt" and "silt_to_percent" in conversions:
            df_agg[col] = df_agg[col] * conversions["silt_to_percent"]
        elif prop == "phh2o" and "ph_to_real" in conversions:
            df_agg[col] = df_agg[col] * conversions["ph_to_real"]
        elif prop == "soc" and "soc_to_gkg" in conversions:
            df_agg[col] = df_agg[col] * conversions["soc_to_gkg"]
        elif prop == "nitrogen" and "nitrogen_to_gkg" in conversions:
            df_agg[col] = df_agg[col] * conversions["nitrogen_to_gkg"]
        elif prop == "bdod" and "bdod_to_gcm3" in conversions:
            df_agg[col] = df_agg[col] * conversions["bdod_to_gcm3"]

    return df_agg


def calculate_derived_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calcula features derivadas de solo."""
    df = df.copy()

    if "clay_0_30cm" in df.columns and "sand_0_30cm" in df.columns:
        df["clay_sand_ratio"] = df["clay_0_30cm"] / (df["sand_0_30cm"] + 0.1)

    if "clay_0_30cm" in df.columns and "sand_0_30cm" in df.columns:
        clay_pct = df["clay_0_30cm"]
        sand_pct = df["sand_0_30cm"]
        df["awc_estimated"] = (0.76 * clay_pct - 0.4 * sand_pct + 25).clip(lower=5, upper=25)

    if "phh2o_0_30cm" in df.columns:
        df["ph_acidic"] = (df["phh2o_0_30cm"] < 5.5).astype(int)

    if "clay_0_30cm" in df.columns and "sand_0_30cm" in df.columns:
        conditions = [
            df["clay_0_30cm"] >= 35,
            df["sand_0_30cm"] >= 70,
        ]
        choices = ["clayey", "sandy"]
        df["texture_class"] = pd.Series(
            ["loamy"] * len(df),
            index=df.index
        )
        for cond, choice in zip(conditions, choices):
            df.loc[cond, "texture_class"] = choice

    if all(col in df.columns for col in ["soc_0_30cm", "phh2o_0_30cm", "cec_0_30cm"]):
        soc_norm = (df["soc_0_30cm"] - df["soc_0_30cm"].min()) / (df["soc_0_30cm"].max() - df["soc_0_30cm"].min() + 0.001)
        ph_norm = 1 - abs(df["phh2o_0_30cm"] - 6.2) / 3.0
        ph_norm = ph_norm.clip(lower=0, upper=1)
        cec_norm = (df["cec_0_30cm"] - df["cec_0_30cm"].min()) / (df["cec_0_30cm"].max() - df["cec_0_30cm"].min() + 0.001)

        df["soil_quality_index"] = 0.4 * soc_norm + 0.3 * ph_norm + 0.3 * cec_norm

    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcula estatisticas do dataset de solo."""
    stats = {
        "total_municipios": len(df),
        "colunas": list(df.columns),
    }

    for col in df.columns:
        if col in ["cod_ibge", "texture_class"]:
            continue

        if df[col].dtype in ["float64", "int64"]:
            stats[col] = {
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                "nulls": int(df[col].isna().sum()),
            }

    return stats


def main(only_soy_producers: bool = True, max_municipalities: int | None = None):
    """Pipeline principal de ingestao de dados de solo."""
    logger.info("=" * 60)
    logger.info("INGESTAO SOLO - SOILGRIDS")
    logger.info("=" * 60)

    config = load_config()
    logger.info(f"Propriedades: {config['properties']}")
    logger.info(f"Profundidades: {config['depths']}")

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

    for idx, row in pending.iterrows():
        cod_ibge = row["cod_ibge"]
        lat = row["lat"]
        lon = row["lon"]

        record = download_soil_for_municipality(cod_ibge, lat, lon, config)

        if record is not None:
            save_to_cache(record, cod_ibge)
            success += 1
            n_props = len([k for k in record.keys() if k != "cod_ibge"])
            logger.info(
                f"[{success + failed}/{len(pending)}] cod_ibge={cod_ibge}: OK ({n_props} propriedades)"
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
        logger.info(f"Municipios com falha (primeiros 20): {failed_list[:20]}")

    logger.info("\nConsolidando cache...")
    df_raw = consolidate_cache()
    logger.info(f"Registros consolidados: {len(df_raw):,}")

    logger.info("\nAgregando profundidades...")
    df_agg = aggregate_depths(df_raw, config)

    logger.info("Calculando features derivadas...")
    df_final = calculate_derived_features(df_agg, config)

    df_final = df_final.sort_values("cod_ibge").reset_index(drop=True)

    stats = calculate_statistics(df_final)

    logger.info("\n" + "=" * 60)
    logger.info("ESTATISTICAS DO DATASET")
    logger.info("=" * 60)
    logger.info(f"Total de municipios: {stats['total_municipios']:,}")
    logger.info(f"Colunas: {len(stats['colunas'])}")

    for col in ["clay_0_30cm", "sand_0_30cm", "phh2o_0_30cm", "soc_0_30cm"]:
        if col in stats:
            s = stats[col]
            logger.info(f"{col}: min={s['min']:.1f}, max={s['max']:.1f}, mean={s['mean']:.1f}, nulls={s['nulls']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nArquivo salvo: {OUTPUT_PATH}")

    total_expected = len(df_mun)
    total_success = len(df_final)
    success_rate = 100 * total_success / total_expected if total_expected > 0 else 0

    logger.info(f"\nTaxa de cobertura: {success_rate:.1f}% ({total_success}/{total_expected})")

    return df_final, stats


if __name__ == "__main__":
    main()
