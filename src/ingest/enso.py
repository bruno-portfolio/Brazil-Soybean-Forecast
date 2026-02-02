from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

ONI_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"

ONI_TABLE_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"


def download_oni_data() -> pd.DataFrame:
    """Baixa dados do ONI da NOAA."""
    try:
        return _download_oni_ascii()
    except Exception as e:
        print(f"Falha no download ASCII: {e}")
        print("Tentando parse da tabela HTML...")
        return _download_oni_html()


def _download_oni_ascii() -> pd.DataFrame:
    response = requests.get(ONI_URL, timeout=30)
    response.raise_for_status()

    lines = response.text.strip().split("\n")

    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            year = int(parts[0])
            month = int(parts[1])
            oni = float(parts[3])
            data.append({"year": year, "month": month, "oni": oni})

    return pd.DataFrame(data)


def _download_oni_html() -> pd.DataFrame:
    response = requests.get(ONI_TABLE_URL, timeout=30)
    response.raise_for_status()

    html = response.text

    pattern = r"(\d{4})\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"

    matches = re.findall(pattern, html)

    trimester_to_month = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: 11,
        11: 12,
    }

    data = []
    for match in matches:
        year = int(match[0])
        for i, oni_str in enumerate(match[1:]):
            try:
                oni = float(oni_str)
                month = trimester_to_month[i]
                data.append({"year": year, "month": month, "oni": oni})
            except ValueError:
                continue

    return pd.DataFrame(data)


def create_oni_from_hardcoded() -> pd.DataFrame:
    """Cria DataFrame com dados ONI hardcoded (fallback se APIs falharem)."""
    oni_data = {
        1999: [-1.5, -1.3, -1.0, -0.9, -0.9, -1.0, -1.0, -1.1, -1.1, -1.2, -1.4, -1.7],
        2000: [-1.7, -1.4, -1.0, -0.7, -0.6, -0.5, -0.4, -0.4, -0.4, -0.5, -0.7, -0.7],
        2001: [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.1, -0.1, -0.2, -0.3, -0.3, -0.3],
        2002: [-0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.1],
        2003: [0.9, 0.6, 0.4, 0.0, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.4, 0.3],
        2004: [0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7],
        2005: [0.6, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1, 0.0, -0.2, -0.5, -0.7],
        2006: [-0.7, -0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9],
        2007: [0.7, 0.3, -0.1, -0.2, -0.3, -0.3, -0.4, -0.6, -0.9, -1.1, -1.3, -1.4],
        2008: [-1.5, -1.5, -1.2, -0.9, -0.7, -0.5, -0.4, -0.3, -0.3, -0.4, -0.6, -0.7],
        2009: [-0.8, -0.7, -0.5, -0.2, 0.1, 0.4, 0.5, 0.6, 0.7, 1.0, 1.3, 1.6],
        2010: [1.5, 1.3, 0.9, 0.4, -0.1, -0.6, -1.0, -1.4, -1.6, -1.7, -1.7, -1.6],
        2011: [-1.4, -1.1, -0.8, -0.6, -0.5, -0.4, -0.5, -0.7, -0.9, -1.1, -1.1, -1.0],
        2012: [-0.8, -0.6, -0.5, -0.4, -0.2, 0.1, 0.3, 0.4, 0.4, 0.4, 0.2, -0.2],
        2013: [-0.4, -0.3, -0.2, -0.2, -0.3, -0.3, -0.4, -0.4, -0.3, -0.2, -0.2, -0.3],
        2014: [-0.4, -0.4, -0.2, 0.1, 0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.6, 0.7],
        2015: [0.6, 0.6, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.6, 2.6],
        2016: [2.5, 2.2, 1.7, 1.0, 0.5, 0.0, -0.3, -0.6, -0.7, -0.7, -0.7, -0.6],
        2017: [-0.3, -0.1, 0.1, 0.3, 0.4, 0.4, 0.2, -0.1, -0.4, -0.7, -0.9, -1.0],
        2018: [-0.9, -0.8, -0.6, -0.4, -0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 0.9, 0.8],
        2019: [0.8, 0.8, 0.8, 0.8, 0.6, 0.5, 0.3, 0.1, 0.2, 0.3, 0.5, 0.5],
        2020: [0.5, 0.5, 0.5, 0.4, 0.2, -0.1, -0.4, -0.6, -0.9, -1.2, -1.3, -1.2],
        2021: [-1.0, -0.9, -0.7, -0.5, -0.3, -0.2, -0.4, -0.5, -0.7, -0.8, -1.0, -1.0],
        2022: [-1.0, -0.9, -1.0, -1.0, -1.0, -0.9, -0.8, -0.8, -0.9, -0.9, -0.9, -0.8],
        2023: [-0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.4, 1.6, 1.8, 2.0, 2.0],
        2024: [2.0, 1.8, 1.5, 1.1, 0.7, 0.4, 0.2, 0.0, -0.2, -0.4, -0.5, -0.6],
        2025: [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    data = []
    for year, monthly_values in oni_data.items():
        for month, oni in enumerate(monthly_values, start=1):
            data.append({"year": year, "month": month, "oni": oni})

    return pd.DataFrame(data)


def classify_enso_phase(oni: float) -> str:
    """Classifica a fase do ENSO baseado no ONI."""
    if oni >= 0.5:
        return "nino"
    elif oni <= -0.5:
        return "nina"
    else:
        return "neutro"


def classify_enso_intensity(oni: float) -> str:
    """Classifica a intensidade do ENSO."""
    abs_oni = abs(oni)
    if abs_oni >= 1.5:
        return "forte"
    elif abs_oni >= 1.0:
        return "moderado"
    elif abs_oni >= 0.5:
        return "fraco"
    else:
        return "neutro"


def calculate_oni_for_crop_year(
    oni_df: pd.DataFrame,
    crop_year: int,
    start_month: int = 10,
    end_month: int = 3,
) -> dict:
    """Calcula metricas ONI para uma safra."""
    prev_year_months = oni_df[(oni_df["year"] == crop_year - 1) & (oni_df["month"] >= start_month)]
    curr_year_months = oni_df[(oni_df["year"] == crop_year) & (oni_df["month"] <= end_month)]

    season_data = pd.concat([prev_year_months, curr_year_months])

    if len(season_data) == 0:
        return {
            "oni_avg": None,
            "oni_min": None,
            "oni_max": None,
            "oni_std": None,
            "enso_phase": None,
            "enso_intensity": None,
        }

    oni_avg = season_data["oni"].mean()

    return {
        "oni_avg": round(oni_avg, 2),
        "oni_min": round(season_data["oni"].min(), 2),
        "oni_max": round(season_data["oni"].max(), 2),
        "oni_std": round(season_data["oni"].std(), 2),
        "enso_phase": classify_enso_phase(oni_avg),
        "enso_intensity": classify_enso_intensity(oni_avg),
    }


def process_oni_data() -> pd.DataFrame:
    """Processa dados ONI e gera DataFrame por ano de safra."""
    print("Baixando dados ONI da NOAA...")

    try:
        oni_df = download_oni_data()
        print(f"  Download OK: {len(oni_df)} registros")
    except Exception as e:
        print(f"  Falha no download: {e}")
        print("  Usando dados hardcoded...")
        oni_df = create_oni_from_hardcoded()

    crop_years = range(2000, 2026)
    results = []

    for year in crop_years:
        metrics = calculate_oni_for_crop_year(oni_df, year)
        metrics["ano"] = year
        results.append(metrics)

    df = pd.DataFrame(results)

    cols = ["ano", "oni_avg", "oni_min", "oni_max", "oni_std", "enso_phase", "enso_intensity"]
    df = df[cols]

    return df


def main() -> None:
    """Pipeline principal de ingestao do ONI."""
    print("=" * 60)
    print("INGESTAO DO INDICE ONI (ENSO)")
    print("=" * 60)

    df = process_oni_data()

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PATH / "oni_enso.parquet"
    df.to_parquet(output_path, index=False)

    print(f"\nArquivo salvo: {output_path}")
    print(f"Safras: {df['ano'].min()} - {df['ano'].max()}")

    print("\nResumo por fase ENSO:")
    print(df.groupby("enso_phase").size())

    print("\nAnos com ENSO mais intenso:")
    print(df.nlargest(5, "oni_avg")[["ano", "oni_avg", "enso_phase"]])
    print(df.nsmallest(5, "oni_avg")[["ano", "oni_avg", "enso_phase"]])


if __name__ == "__main__":
    main()
