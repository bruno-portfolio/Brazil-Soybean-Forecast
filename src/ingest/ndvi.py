import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "ndvi.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ndvi_safra.parquet"


def load_config() -> dict:
    """Carrega configuracao."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["ndvi"]


def assign_crop_year(date: pd.Timestamp, start_month: int = 10, end_month: int = 3) -> int:
    """Atribui ano da safra (ano da colheita)."""
    if date.month >= start_month:
        return date.year + 1
    elif date.month <= end_month:
        return date.year
    return None


def assign_phase(month: int, phases: dict) -> str:
    """Atribui fase fenologica."""
    for phase, months in phases.items():
        if month in months:
            return phase
    return None


def process_appeears_csv(input_path: str, config: dict) -> pd.DataFrame:
    """Processa CSV do AppEEARS."""
    logger.info(f"Carregando {input_path}...")
    df = pd.read_csv(input_path)

    ndvi_col = [c for c in df.columns if 'NDVI' in c.upper()][0]

    df = df.rename(columns={
        'ID': 'cod_ibge',
        'Date': 'date',
        ndvi_col: 'ndvi_raw'
    })

    df['date'] = pd.to_datetime(df['date'])
    df['cod_ibge'] = df['cod_ibge'].astype(int)

    scale = config.get('scale', {}).get('factor', 0.0001)
    df['ndvi'] = df['ndvi_raw'] * scale

    df = df[(df['ndvi'] >= -1) & (df['ndvi'] <= 1)]

    logger.info(f"Registros validos: {len(df):,}")
    return df


def aggregate_ndvi_by_safra(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Agrega NDVI por municipio e safra."""
    logger.info("Agregando NDVI por safra...")

    agg = config['aggregation']
    start_month = agg['start_month']
    end_month = agg['end_month']
    phases = agg['phases']

    df['month'] = df['date'].dt.month
    mask = (df['month'] >= start_month) | (df['month'] <= end_month)
    df = df[mask].copy()

    df['ano'] = df['date'].apply(lambda x: assign_crop_year(x, start_month, end_month))
    df['phase'] = df['month'].apply(lambda x: assign_phase(x, phases))
    df = df.dropna(subset=['ano'])
    df['ano'] = df['ano'].astype(int)

    results = []
    for (cod_ibge, ano), group in df.groupby(['cod_ibge', 'ano']):
        record = {
            'cod_ibge': cod_ibge,
            'ano': ano,
            'ndvi_mean_safra': group['ndvi'].mean(),
            'ndvi_max_safra': group['ndvi'].max(),
            'ndvi_min_safra': group['ndvi'].min(),
            'ndvi_amplitude': group['ndvi'].max() - group['ndvi'].min(),
        }

        for phase in ['vegetativo', 'enchimento']:
            phase_data = group[group['phase'] == phase]
            if len(phase_data) > 0:
                record[f'ndvi_{phase}'] = phase_data['ndvi'].mean()
            else:
                record[f'ndvi_{phase}'] = None

        results.append(record)

    df_agg = pd.DataFrame(results)
    logger.info(f"Registros agregados: {len(df_agg):,}")
    return df_agg


def main(input_file: str = None):
    """Pipeline principal."""
    logger.info("=" * 60)
    logger.info("PROCESSAMENTO NDVI - AppEEARS")
    logger.info("=" * 60)

    config = load_config()

    if input_file is None:
        input_file = PROJECT_ROOT / config['input']['file']

    if not Path(input_file).exists():
        logger.error(f"Arquivo nao encontrado: {input_file}")
        logger.info("Baixe o CSV do AppEEARS e coloque em data/raw/ndvi/")
        return None

    df = process_appeears_csv(input_file, config)
    df_agg = aggregate_ndvi_by_safra(df, config)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Salvo: {OUTPUT_PATH}")

    logger.info(f"\nMunicipios: {df_agg['cod_ibge'].nunique():,}")
    logger.info(f"Anos: {df_agg['ano'].min()}-{df_agg['ano'].max()}")
    logger.info(f"NDVI medio: {df_agg['ndvi_mean_safra'].mean():.3f}")

    return df_agg


if __name__ == "__main__":
    main()
