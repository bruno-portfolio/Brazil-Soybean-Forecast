"""
Prepara dados de demonstração para validação rápida do projeto.

Cria um subset pequeno de dados (~100 municípios) para que recrutadores
e avaliadores possam testar o dashboard sem precisar baixar dados completos via DVC.

Uso:
    python scripts/prepare_demo.py

Ou via Makefile:
    make demo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_demo_data() -> None:
    """Cria dataset de demonstração com subset de municípios."""

    project_root = Path(__file__).parent.parent
    demo_dir = project_root / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Verificar se dados completos existem
    dataset_path = project_root / "data" / "processed" / "dataset_final.parquet"
    predictions_path = project_root / "results" / "predictions_2024_2025.parquet"

    if dataset_path.exists() and predictions_path.exists():
        print("Criando subset de dados reais...")
        create_from_real_data(dataset_path, predictions_path, demo_dir)
    else:
        print("Dados completos não encontrados. Criando dados sintéticos para demo...")
        create_synthetic_demo(demo_dir)

    print(f"Dados de demonstração salvos em: {demo_dir}")


def create_from_real_data(dataset_path: Path, predictions_path: Path, demo_dir: Path) -> None:
    """Cria demo a partir de dados reais (subset)."""

    # Carregar dados
    df = pd.read_parquet(dataset_path)
    pred = pd.read_parquet(predictions_path)

    # Selecionar municípios representativos (maiores produtores de cada UF)
    top_munis = df.groupby("cod_ibge")["producao_ton"].sum().nlargest(100).index.tolist()

    # Filtrar
    df_demo = df[df["cod_ibge"].isin(top_munis)].copy()
    pred_demo = pred[pred["cod_ibge"].isin(top_munis)].copy()

    # Salvar
    df_demo.to_parquet(demo_dir / "dataset_demo.parquet", index=False)
    pred_demo.to_parquet(demo_dir / "predictions_demo.parquet", index=False)

    print(f"  - Dataset: {len(df_demo):,} registros ({df_demo['cod_ibge'].nunique()} municípios)")
    print(f"  - Previsões: {len(pred_demo):,} registros")


def create_synthetic_demo(demo_dir: Path) -> None:
    """Cria dados sintéticos para demonstração quando dados reais não existem."""

    np.random.seed(42)

    # Municípios fictícios representativos
    municipios = [
        (4113700, "Londrina", "PR", 0),  # Sul
        (4314902, "Passo Fundo", "RS", 1),  # Sul
        (5103403, "Sorriso", "MT", 0),  # Cerrado
        (5208707, "Rio Verde", "GO", 0),  # Cerrado
        (2109106, "Balsas", "MA", 0),  # MATOPIBA
    ]

    anos = list(range(2015, 2024))

    records = []
    for cod_ibge, nome, uf, is_sul in municipios:
        for ano in anos:
            # Produtividade base + tendência + ruído
            base = 3000 if is_sul == 0 else 2800
            trend = (ano - 2015) * 50
            noise = np.random.normal(0, 300)
            prod = max(1500, base + trend + noise)

            records.append(
                {
                    "cod_ibge": cod_ibge,
                    "nome_municipio": nome,
                    "uf": uf,
                    "ano": ano,
                    "produtividade_kg_ha": prod,
                    "is_sul": is_sul,
                    "precip_total_mm": np.random.normal(800, 150),
                    "tmean_avg": np.random.normal(25, 2),
                    "oni_avg": np.random.normal(0, 0.8),
                }
            )

    df_demo = pd.DataFrame(records)

    # Criar previsões sintéticas para 2024-2025
    pred_records = []
    for cod_ibge, nome, uf, is_sul in municipios:
        for ano in [2024, 2025]:
            pred_point = np.random.normal(3200, 400)
            pred_records.append(
                {
                    "cod_ibge": cod_ibge,
                    "nome_municipio": nome,
                    "uf": uf,
                    "ano": ano,
                    "pred_kg_ha": pred_point,
                    "pred_lower_80": pred_point - 600,
                    "pred_upper_80": pred_point + 600,
                    "is_sul": is_sul,
                }
            )

    pred_demo = pd.DataFrame(pred_records)

    # Salvar
    df_demo.to_parquet(demo_dir / "dataset_demo.parquet", index=False)
    pred_demo.to_parquet(demo_dir / "predictions_demo.parquet", index=False)

    print(f"  - Dataset sintético: {len(df_demo)} registros")
    print(f"  - Previsões sintéticas: {len(pred_demo)} registros")
    print("  - NOTA: Dados sintéticos para demonstração. Use 'make dvc-pull' para dados reais.")


if __name__ == "__main__":
    create_demo_data()
