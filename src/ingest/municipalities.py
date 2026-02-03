from pathlib import Path

import pandas as pd
import yaml
from loguru import logger


def load_config(config_path: Path | None = None) -> dict:
    """Carrega configuracao geografica do arquivo YAML."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "geo.yaml"

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_municipalities_from_ibge_api() -> pd.DataFrame:
    """Baixa lista de municipios via API de localidades do IBGE."""
    import requests

    logger.info("Baixando lista de municípios da API do IBGE...")

    url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        municipios = response.json()

        logger.info(f"Recebidos {len(municipios)} municípios da API")

        data = []
        for m in municipios:
            try:
                uf = m["microrregiao"]["mesorregiao"]["UF"]["sigla"]
            except (KeyError, TypeError):
                cod = int(m["id"])
                uf_code = cod // 100000
                uf_map = {
                    11: "RO",
                    12: "AC",
                    13: "AM",
                    14: "RR",
                    15: "PA",
                    16: "AP",
                    17: "TO",
                    21: "MA",
                    22: "PI",
                    23: "CE",
                    24: "RN",
                    25: "PB",
                    26: "PE",
                    27: "AL",
                    28: "SE",
                    29: "BA",
                    31: "MG",
                    32: "ES",
                    33: "RJ",
                    35: "SP",
                    41: "PR",
                    42: "SC",
                    43: "RS",
                    50: "MS",
                    51: "MT",
                    52: "GO",
                    53: "DF",
                }
                uf = uf_map.get(uf_code, "XX")

            data.append(
                {
                    "cod_ibge": int(m["id"]),
                    "nome": m["nome"],
                    "uf": uf,
                }
            )

        df = pd.DataFrame(data)
        logger.info(f"DataFrame criado com {len(df)} municípios")

        return df

    except Exception as e:
        logger.error(f"Erro ao baixar municípios: {e}")
        raise


def download_coordinates_from_ibge_cidades() -> dict[int, tuple[float, float]]:
    """Baixa coordenadas das sedes municipais do IBGE Cidades."""
    import requests

    logger.info("Baixando coordenadas das sedes municipais...")

    uf_to_code = {
        "RO": 11,
        "AC": 12,
        "AM": 13,
        "RR": 14,
        "PA": 15,
        "AP": 16,
        "TO": 17,
        "MA": 21,
        "PI": 22,
        "CE": 23,
        "RN": 24,
        "PB": 25,
        "PE": 26,
        "AL": 27,
        "SE": 28,
        "BA": 29,
        "MG": 31,
        "ES": 32,
        "RJ": 33,
        "SP": 35,
        "PR": 41,
        "SC": 42,
        "RS": 43,
        "MS": 50,
        "MT": 51,
        "GO": 52,
        "DF": 53,
    }

    coords = {}

    for uf, uf_code in uf_to_code.items():
        logger.info(f"Baixando coordenadas de {uf}...")

        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf_code}?formato=application/vnd.geo+json&intrarregiao=municipio&qualidade=minima"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            geojson = response.json()

            for feature in geojson.get("features", []):
                cod_ibge = int(feature["properties"]["codarea"])

                geom = feature["geometry"]
                if geom["type"] == "Polygon":
                    coords_list = geom["coordinates"][0]
                elif geom["type"] == "MultiPolygon":
                    largest = max(geom["coordinates"], key=lambda x: len(x[0]))
                    coords_list = largest[0]
                else:
                    continue

                lons = [c[0] for c in coords_list]
                lats = [c[1] for c in coords_list]
                centroid_lon = sum(lons) / len(lons)
                centroid_lat = sum(lats) / len(lats)

                coords[cod_ibge] = (centroid_lat, centroid_lon)

        except Exception as e:
            logger.warning(f"Erro ao baixar coordenadas de {uf}: {e}")
            continue

    logger.info(f"Coordenadas obtidas para {len(coords)} municípios")
    return coords


def process_municipalities(output_path: Path | None = None) -> pd.DataFrame:
    """Pipeline completo de processamento de municipios."""
    if output_path is None:
        output_path = Path(__file__).parents[2] / "data" / "processed" / "municipalities.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Iniciando ingestão de municípios ===")

    df = download_municipalities_from_ibge_api()

    coords = download_coordinates_from_ibge_cidades()

    df["lat"] = df["cod_ibge"].map(lambda x: coords.get(x, (None, None))[0])
    df["lon"] = df["cod_ibge"].map(lambda x: coords.get(x, (None, None))[1])

    sem_coords = df[df["lat"].isna()]
    if len(sem_coords) > 0:
        logger.warning(f"{len(sem_coords)} municípios sem coordenadas")
        df = df.dropna(subset=["lat", "lon"])

    df = df[["cod_ibge", "nome", "uf", "lat", "lon"]]

    df = df.sort_values("cod_ibge").reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    logger.info(f"Dados salvos em: {output_path}")
    logger.info(f"Total de municípios: {len(df)}")

    logger.info(f"UFs presentes: {df['uf'].nunique()}")
    logger.info(f"Lat range: [{df['lat'].min():.2f}, {df['lat'].max():.2f}]")
    logger.info(f"Lon range: [{df['lon'].min():.2f}, {df['lon'].max():.2f}]")

    return df


def main():
    """Ponto de entrada para execucao via CLI."""
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    try:
        process_municipalities()
        logger.info("=== Ingestão concluída com sucesso ===")
        return 0
    except Exception as e:
        logger.error(f"Falha na ingestão: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
