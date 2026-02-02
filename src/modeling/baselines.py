from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.metrics import compute_all_metrics
from src.modeling.split import TemporalSplit, create_temporal_split

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results"


@dataclass
class BaselineResults:
    """Resultados de um baseline."""

    name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_coverage: float
    test_coverage: float


def evaluate_baseline_lag1(split: TemporalSplit) -> BaselineResults:
    """Avalia baseline usando produtividade do ano anterior."""
    target_col = "produtividade_kg_ha"
    pred_col = "produtividade_lag1"

    val_data = split.validation.dropna(subset=[pred_col, target_col])
    val_coverage = len(val_data) / len(split.validation) * 100
    val_metrics = compute_all_metrics(val_data[target_col].values, val_data[pred_col].values)

    test_data = split.test.dropna(subset=[pred_col, target_col])
    test_coverage = len(test_data) / len(split.test) * 100
    test_metrics = compute_all_metrics(test_data[target_col].values, test_data[pred_col].values)

    return BaselineResults(
        name="baseline_lag1",
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        validation_coverage=round(val_coverage, 1),
        test_coverage=round(test_coverage, 1),
    )


def evaluate_baseline_ma3(split: TemporalSplit) -> BaselineResults:
    """Avalia baseline usando media movel de 3 anos."""
    target_col = "produtividade_kg_ha"
    pred_col = "produtividade_ma3"

    val_data = split.validation.dropna(subset=[pred_col, target_col])
    val_coverage = len(val_data) / len(split.validation) * 100
    val_metrics = compute_all_metrics(val_data[target_col].values, val_data[pred_col].values)

    test_data = split.test.dropna(subset=[pred_col, target_col])
    test_coverage = len(test_data) / len(split.test) * 100
    test_metrics = compute_all_metrics(test_data[target_col].values, test_data[pred_col].values)

    return BaselineResults(
        name="baseline_ma3",
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        validation_coverage=round(val_coverage, 1),
        test_coverage=round(test_coverage, 1),
    )


def evaluate_all_baselines(
    split: TemporalSplit | None = None,
) -> dict[str, BaselineResults]:
    """Avalia todos os baselines."""
    if split is None:
        split = create_temporal_split()

    return {
        "baseline_lag1": evaluate_baseline_lag1(split),
        "baseline_ma3": evaluate_baseline_ma3(split),
    }


def save_results(results: dict[str, BaselineResults], path: Path | None = None) -> Path:
    """Salva resultados em JSON."""
    if path is None:
        path = RESULTS_PATH / "baselines.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    output = {}
    for name, result in results.items():
        output[name] = asdict(result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return path


def print_comparison_table(results: dict[str, BaselineResults]) -> None:
    """Imprime tabela comparativa dos baselines."""
    print("\n" + "=" * 80)
    print("COMPARACAO DE BASELINES")
    print("=" * 80)

    print(
        f"\n{'Baseline':<15} | {'Split':<10} | "
        f"{'MAE (kg/ha)':<12} | {'MAE (sc/ha)':<12} | "
        f"{'RMSE (kg/ha)':<12} | {'MAPE (%)':<10} | {'N':<8}"
    )
    print("-" * 95)

    for name, result in results.items():
        vm = result.validation_metrics
        print(
            f"{name:<15} | {'Validacao':<10} | "
            f"{vm['mae_kg_ha']:>11.1f} | {vm['mae_sacas_ha']:>11.2f} | "
            f"{vm['rmse_kg_ha']:>11.1f} | {vm['mape_percent']:>9.2f} | "
            f"{vm['n_samples']:>7,}"
        )

        tm = result.test_metrics
        print(
            f"{'':<15} | {'Teste':<10} | "
            f"{tm['mae_kg_ha']:>11.1f} | {tm['mae_sacas_ha']:>11.2f} | "
            f"{tm['rmse_kg_ha']:>11.1f} | {tm['mape_percent']:>9.2f} | "
            f"{tm['n_samples']:>7,}"
        )
        print("-" * 95)


def main() -> None:
    """Pipeline principal de avaliacao de baselines."""
    print("=" * 60)
    print("AVALIACAO DE BASELINES")
    print("=" * 60)

    print("\nCarregando dados e criando split temporal...")
    split = create_temporal_split()

    summary = split.summary()
    print("\nSplit temporal:")
    for name, stats in summary.items():
        print(f"  {name}: {stats['n_samples']:,} amostras ({stats['year_range']})")

    print("\nAvaliando baselines...")
    results = evaluate_all_baselines(split)

    print_comparison_table(results)

    print("\n" + "=" * 60)
    print("ANALISE DETALHADA")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Cobertura validacao: {result.validation_coverage}%")
        print(f"  Cobertura teste: {result.test_coverage}%")

    best_val = min(results.items(), key=lambda x: x[1].validation_metrics["mae_kg_ha"])
    best_test = min(results.items(), key=lambda x: x[1].test_metrics["mae_kg_ha"])

    print("\n" + "=" * 60)
    print("MELHOR BASELINE")
    print("=" * 60)
    print(
        f"\n  Por validacao: {best_val[0]} (MAE = {best_val[1].validation_metrics['mae_kg_ha']:.1f} kg/ha)"
    )
    print(f"  Por teste: {best_test[0]} (MAE = {best_test[1].test_metrics['mae_kg_ha']:.1f} kg/ha)")

    print("\n" + "=" * 60)
    print("SALVANDO RESULTADOS")
    print("=" * 60)
    output_path = save_results(results)
    print(f"\n  Arquivo salvo: {output_path}")

    print("\n" + "=" * 60)
    print("META PARA O MODELO PRINCIPAL")
    print("=" * 60)
    best_mae = best_test[1].test_metrics["mae_kg_ha"]
    print(f"\n  O modelo principal deve superar MAE < {best_mae:.1f} kg/ha no teste")
    print(f"  (equivalente a < {best_mae/60:.2f} sacas/ha)")


if __name__ == "__main__":
    main()
