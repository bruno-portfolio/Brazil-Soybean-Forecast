from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.modeling.split import create_temporal_split, get_feature_columns

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data" / "processed"


def load_model(version: str = "v1"):
    """Carrega modelo treinado."""
    model_path = MODELS_PATH / f"model_{version}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_municipalities() -> pd.DataFrame:
    """Carrega dados de municipios para ter UF."""
    path = DATA_PATH / "municipalities.parquet"
    return pd.read_parquet(path)[["cod_ibge", "uf"]]


def load_baselines() -> dict[str, Any]:
    """Carrega resultados dos baselines."""
    path = RESULTS_PATH / "baselines.json"
    with open(path) as f:
        return json.load(f)


def add_predictions(df: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    """Adiciona predicoes ao DataFrame."""
    df = df.copy()

    mask = df[feature_cols].notna().all(axis=1)
    df_valid = df[mask].copy()

    X = df_valid[feature_cols].values
    df_valid["pred"] = model.predict(X)

    return df_valid


def calculate_error_by_group(
    df: pd.DataFrame,
    group_col: str,
    actual_col: str = "produtividade_kg_ha",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Calcula erro por grupo."""
    results = []
    for group, group_df in df.groupby(group_col):
        metrics = compute_all_metrics(
            group_df[actual_col].values,
            group_df[pred_col].values,
        )
        results.append(
            {
                group_col: group,
                "mae_kg_ha": metrics["mae_kg_ha"],
                "mae_sacas_ha": metrics["mae_sacas_ha"],
                "mape_percent": metrics["mape_percent"],
                "n_samples": metrics["n_samples"],
                "mean_actual": group_df[actual_col].mean(),
            }
        )

    return pd.DataFrame(results).sort_values("mae_kg_ha", ascending=False)


def calculate_error_by_productivity_range(
    df: pd.DataFrame,
    actual_col: str = "produtividade_kg_ha",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Calcula erro por faixa de produtividade."""
    bins = [0, 1500, 2500, 3000, 3500, 10000]
    labels = ["0-1500", "1500-2500", "2500-3000", "3000-3500", "3500+"]

    df = df.copy()
    df["faixa"] = pd.cut(df[actual_col], bins=bins, labels=labels, include_lowest=True)

    return calculate_error_by_group(df, "faixa", actual_col, pred_col)


def create_scatter_plot(
    df: pd.DataFrame,
    actual_col: str = "produtividade_kg_ha",
    pred_col: str = "pred",
    title: str = "Predicted vs Actual",
    output_path: Path | None = None,
) -> None:
    """Cria scatter plot de predicted vs actual."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        df[actual_col],
        df[pred_col],
        alpha=0.3,
        s=10,
        c="steelblue",
    )

    min_val = min(df[actual_col].min(), df[pred_col].min())
    max_val = max(df[actual_col].max(), df[pred_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")

    metrics = compute_all_metrics(df[actual_col].values, df[pred_col].values)

    ax.set_xlabel("Produtividade Real (kg/ha)", fontsize=12)
    ax.set_ylabel("Produtividade Predita (kg/ha)", fontsize=12)
    ax.set_title(
        f"{title}\nMAE: {metrics['mae_kg_ha']:.1f} kg/ha ({metrics['mae_sacas_ha']:.2f} sc/ha) | "
        f"MAPE: {metrics['mape_percent']:.1f}%",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_error_by_year_plot(
    error_by_year: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Cria grafico de erro por ano."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years = error_by_year["ano"].values
    mae_values = error_by_year["mae_kg_ha"].values

    bars = ax.bar(years, mae_values, color="steelblue", alpha=0.7)

    for bar, val in zip(bars, mae_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("MAE (kg/ha)", fontsize=12)
    ax.set_title("Erro (MAE) por Ano", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    ax.axvline(x=2018.5, color="red", linestyle="--", label="Fim do treino")
    ax.axvline(x=2021.5, color="orange", linestyle="--", label="Fim da validacao")
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def generate_comparison_table(
    model_metrics: dict[str, dict],
    baselines: dict[str, Any],
) -> str:
    """Gera tabela comparativa entre modelo e baselines."""
    lines = []
    lines.append("| Modelo | Split | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | vs Baseline |")
    lines.append("|--------|-------|-------------|-------------|----------|-------------|")

    bl_lag1_val = baselines["baseline_lag1"]["validation_metrics"]
    bl_lag1_test = baselines["baseline_lag1"]["test_metrics"]
    lines.append(
        f"| baseline_lag1 | Validacao | {bl_lag1_val['mae_kg_ha']:.1f} | "
        f"{bl_lag1_val['mae_sacas_ha']:.2f} | {bl_lag1_val['mape_percent']:.1f} | - |"
    )
    lines.append(
        f"| baseline_lag1 | Teste | {bl_lag1_test['mae_kg_ha']:.1f} | "
        f"{bl_lag1_test['mae_sacas_ha']:.2f} | {bl_lag1_test['mape_percent']:.1f} | - |"
    )

    bl_ma3_val = baselines["baseline_ma3"]["validation_metrics"]
    bl_ma3_test = baselines["baseline_ma3"]["test_metrics"]
    lines.append(
        f"| baseline_ma3 | Validacao | {bl_ma3_val['mae_kg_ha']:.1f} | "
        f"{bl_ma3_val['mae_sacas_ha']:.2f} | {bl_ma3_val['mape_percent']:.1f} | - |"
    )
    lines.append(
        f"| baseline_ma3 | Teste | {bl_ma3_test['mae_kg_ha']:.1f} | "
        f"{bl_ma3_test['mae_sacas_ha']:.2f} | {bl_ma3_test['mape_percent']:.1f} | - |"
    )

    val_m = model_metrics["validation"]
    test_m = model_metrics["test"]

    val_gain = ((bl_ma3_val["mae_kg_ha"] - val_m["mae_kg_ha"]) / bl_ma3_val["mae_kg_ha"]) * 100
    test_gain = ((bl_ma3_test["mae_kg_ha"] - test_m["mae_kg_ha"]) / bl_ma3_test["mae_kg_ha"]) * 100

    val_status = f"+{val_gain:.1f}%" if val_gain > 0 else f"{val_gain:.1f}%"
    test_status = f"+{test_gain:.1f}%" if test_gain > 0 else f"{test_gain:.1f}%"

    lines.append(
        f"| **LightGBM** | Validacao | **{val_m['mae_kg_ha']:.1f}** | "
        f"**{val_m['mae_sacas_ha']:.2f}** | **{val_m['mape_percent']:.1f}** | {val_status} |"
    )
    lines.append(
        f"| **LightGBM** | Teste | **{test_m['mae_kg_ha']:.1f}** | "
        f"**{test_m['mae_sacas_ha']:.2f}** | **{test_m['mape_percent']:.1f}** | {test_status} |"
    )

    return "\n".join(lines)


def generate_evaluation_report(
    model_metrics: dict[str, dict],
    baselines: dict[str, Any],
    error_by_uf: pd.DataFrame,
    error_by_year: pd.DataFrame,
    error_by_range: pd.DataFrame,
    feature_importance: dict[str, float],
    best_iteration: int,
) -> str:
    """Gera relatorio de avaliacao em Markdown."""
    report = []
    report.append("# Relatorio de Avaliacao do Modelo")
    report.append(f"\nData: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("\n## 1. Resumo Executivo")

    test_m = model_metrics["test"]
    bl_ma3_test = baselines["baseline_ma3"]["test_metrics"]
    beat_baseline = test_m["mae_kg_ha"] < bl_ma3_test["mae_kg_ha"]

    if beat_baseline:
        gain = ((bl_ma3_test["mae_kg_ha"] - test_m["mae_kg_ha"]) / bl_ma3_test["mae_kg_ha"]) * 100
        report.append(
            f"\n**SUCESSO**: O modelo LightGBM superou o baseline_ma3 em {gain:.1f}% no conjunto de teste."
        )
    else:
        report.append(
            "\n**ATENCAO**: O modelo LightGBM NAO superou o baseline_ma3 no conjunto de teste. "
            "Investigar features climaticas e janela fenologica."
        )

    report.append(
        f"\n- MAE Teste: **{test_m['mae_kg_ha']:.1f} kg/ha** ({test_m['mae_sacas_ha']:.2f} sacas/ha)"
    )
    report.append(f"- MAPE Teste: **{test_m['mape_percent']:.1f}%**")
    report.append(f"- Melhor iteracao: {best_iteration}")

    report.append("\n## 2. Comparacao com Baselines")
    report.append("\n" + generate_comparison_table(model_metrics, baselines))

    report.append("\n## 3. Analise de Erro por UF (Top 10 piores)")
    report.append("\n| UF | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |")
    report.append("|-----|-------------|-------------|----------|------|")
    for _, row in error_by_uf.head(10).iterrows():
        report.append(
            f"| {row['uf']} | {row['mae_kg_ha']:.1f} | {row['mae_sacas_ha']:.2f} | "
            f"{row['mape_percent']:.1f} | {row['n_samples']} |"
        )

    report.append("\n## 4. Analise de Erro por Ano")
    report.append("\n| Ano | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |")
    report.append("|-----|-------------|-------------|----------|------|")
    for _, row in error_by_year.iterrows():
        report.append(
            f"| {int(row['ano'])} | {row['mae_kg_ha']:.1f} | {row['mae_sacas_ha']:.2f} | "
            f"{row['mape_percent']:.1f} | {row['n_samples']} |"
        )

    report.append("\n## 5. Analise de Erro por Faixa de Produtividade")
    report.append("\n| Faixa (kg/ha) | MAE (kg/ha) | MAE (sc/ha) | MAPE (%) | N |")
    report.append("|---------------|-------------|-------------|----------|------|")
    for _, row in error_by_range.iterrows():
        report.append(
            f"| {row['faixa']} | {row['mae_kg_ha']:.1f} | {row['mae_sacas_ha']:.2f} | "
            f"{row['mape_percent']:.1f} | {row['n_samples']} |"
        )

    report.append("\n## 6. Importancia das Features")
    report.append("\n| Feature | Importancia |")
    report.append("|---------|-------------|")
    for feat, imp in list(feature_importance.items())[:10]:
        report.append(f"| {feat} | {imp:.2f} |")

    report.append("\n## 7. Graficos")
    report.append("\n- `scatter_test.png`: Predicted vs Actual no conjunto de teste")
    report.append("- `error_by_year.png`: MAE por ano")

    report.append("\n## 8. Conclusoes e Proximos Passos")
    if beat_baseline:
        report.append(
            "\nO modelo demonstra capacidade de aprender padroes alem da persistencia historica. "
            "As features climaticas contribuem para a previsao, especialmente em anos anomalos."
        )
    else:
        report.append(
            "\nO modelo nao superou o baseline. Possíveis acoes:\n"
            "1. Revisar janela fenologica (Out-Mar pode nao ser ideal para todas as regioes)\n"
            "2. Adicionar features de interacao clima x historico\n"
            "3. Treinar modelos regionais por UF\n"
            "4. Investigar qualidade dos dados climaticos para municipios com alto erro"
        )

    return "\n".join(report)


def run_full_evaluation(model_version: str = "v1") -> dict[str, Any]:
    """Executa avaliacao completa do modelo."""
    print("=" * 60)
    print("AVALIACAO COMPLETA DO MODELO")
    print("=" * 60)

    print("\nCarregando modelo e dados...")
    model = load_model(model_version)
    split = create_temporal_split()
    feature_cols = get_feature_columns(split.train)
    municipalities = load_municipalities()
    baselines = load_baselines()

    metadata_path = MODELS_PATH / f"model_{model_version}_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    print("\nGerando predicoes...")
    val_with_pred = add_predictions(split.validation, model, feature_cols)
    test_with_pred = add_predictions(split.test, model, feature_cols)

    val_with_pred = val_with_pred.merge(municipalities, on="cod_ibge", how="left")
    test_with_pred = test_with_pred.merge(municipalities, on="cod_ibge", how="left")

    print("\nCalculando metricas...")
    val_metrics = compute_all_metrics(
        val_with_pred["produtividade_kg_ha"].values,
        val_with_pred["pred"].values,
    )
    test_metrics = compute_all_metrics(
        test_with_pred["produtividade_kg_ha"].values,
        test_with_pred["pred"].values,
    )

    model_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
    }

    print("\nAnalisando erro por UF...")
    error_by_uf = calculate_error_by_group(test_with_pred, "uf")

    print("\nAnalisando erro por ano...")
    all_with_pred = pd.concat([val_with_pred, test_with_pred])
    error_by_year = calculate_error_by_group(all_with_pred, "ano")
    error_by_year = error_by_year.sort_values("ano")

    print("\nAnalisando erro por faixa de produtividade...")
    error_by_range = calculate_error_by_productivity_range(test_with_pred)

    print("\nGerando graficos...")
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    create_scatter_plot(
        test_with_pred,
        title="Predicted vs Actual (Teste 2022-2023)",
        output_path=RESULTS_PATH / "scatter_test.png",
    )

    create_error_by_year_plot(
        error_by_year,
        output_path=RESULTS_PATH / "error_by_year.png",
    )

    print("\nGerando relatorio...")
    report = generate_evaluation_report(
        model_metrics=model_metrics,
        baselines=baselines,
        error_by_uf=error_by_uf,
        error_by_year=error_by_year,
        error_by_range=error_by_range,
        feature_importance=metadata["feature_importance"],
        best_iteration=metadata["best_iteration"],
    )

    report_path = RESULTS_PATH / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    results = {
        "model_version": model_version,
        "model_metrics": model_metrics,
        "error_by_uf": error_by_uf.to_dict(orient="records"),
        "error_by_year": error_by_year.to_dict(orient="records"),
        "error_by_range": error_by_range.to_dict(orient="records"),
        "beat_baseline_ma3": test_metrics["mae_kg_ha"]
        < baselines["baseline_ma3"]["test_metrics"]["mae_kg_ha"],
    }

    results_path = RESULTS_PATH / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("RESUMO DA AVALIACAO")
    print("=" * 60)

    print("\nComparacao com Baselines (TESTE):")
    print("-" * 50)
    print(f"{'Modelo':<20} {'MAE (kg/ha)':<15} {'MAE (sc/ha)':<15}")
    print("-" * 50)
    print(
        f"{'baseline_lag1':<20} {baselines['baseline_lag1']['test_metrics']['mae_kg_ha']:<15.1f} "
        f"{baselines['baseline_lag1']['test_metrics']['mae_sacas_ha']:<15.2f}"
    )
    print(
        f"{'baseline_ma3':<20} {baselines['baseline_ma3']['test_metrics']['mae_kg_ha']:<15.1f} "
        f"{baselines['baseline_ma3']['test_metrics']['mae_sacas_ha']:<15.2f}"
    )
    print(
        f"{'LightGBM':<20} {test_metrics['mae_kg_ha']:<15.1f} {test_metrics['mae_sacas_ha']:<15.2f}"
    )

    if results["beat_baseline_ma3"]:
        gain = (
            (baselines["baseline_ma3"]["test_metrics"]["mae_kg_ha"] - test_metrics["mae_kg_ha"])
            / baselines["baseline_ma3"]["test_metrics"]["mae_kg_ha"]
        ) * 100
        print(f"\n[OK] Modelo superou baseline_ma3 em {gain:.1f}%")
    else:
        print("\n[ATENCAO] Modelo NAO superou baseline_ma3")

    print(f"\nRelatorio salvo em: {report_path}")
    print(f"Graficos salvos em: {RESULTS_PATH}")

    return results


def main() -> None:
    """Pipeline principal de avaliacao."""
    run_full_evaluation(model_version="v1")


if __name__ == "__main__":
    main()
