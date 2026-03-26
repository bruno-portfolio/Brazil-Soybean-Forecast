"""Gera plots para o README com numeros validados."""

from __future__ import annotations

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ===== SETUP =====
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C_BLUE = "#2563eb"
C_ORANGE = "#f59e0b"
C_GREEN = "#10b981"
C_RED = "#ef4444"
C_GRAY = "#6b7280"

REGION_SUL = [41, 42, 43]
OUT_DIR = "results"

# ===== CARREGAR DADOS =====
df = pd.read_parquet("data/processed/dataset_final.parquet")

with open("models/model_v2.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/model_sul.pkl", "rb") as f:
    model_sul = pickle.load(f)
with open("models/model_cerrado.pkl", "rb") as f:
    model_cerrado = pickle.load(f)

model_features = model.feature_name()

test = df[df["ano"] == 2023]
y_test = test["produtividade_kg_ha"].values
y_pred_global = model.predict(test[model_features].values)

test_c = test.copy()
test_c["uf_cod"] = test_c["cod_ibge"].astype(str).str[:2].astype(int)
test_c["is_sul"] = test_c["uf_cod"].isin(REGION_SUL).astype(int)

t_sul = test_c[test_c["is_sul"] == 1].dropna(subset=list(model_features) + ["produtividade_kg_ha"])
t_cer = test_c[test_c["is_sul"] == 0].dropna(subset=list(model_features) + ["produtividade_kg_ha"])

y_sul = t_sul["produtividade_kg_ha"].values
y_cer = t_cer["produtividade_kg_ha"].values
y_pred_sul = model_sul.predict(t_sul[model_features].values)
y_pred_cer = model_cerrado.predict(t_cer[model_features].values)

# Baselines
all_data = df[["cod_ibge", "ano", "produtividade_kg_ha"]].sort_values(["cod_ibge", "ano"])
ma3_preds = []
for _, row in test.iterrows():
    mun, ano = row["cod_ibge"], row["ano"]
    hist = all_data[(all_data["cod_ibge"] == mun) & (all_data["ano"] < ano)].tail(3)
    if len(hist) >= 1:
        ma3_preds.append(
            {
                "cod_ibge": mun,
                "actual": row["produtividade_kg_ha"],
                "pred_ma3": hist["produtividade_kg_ha"].mean(),
            }
        )
ma3_df = pd.DataFrame(ma3_preds).dropna()
mae_ma3 = np.mean(np.abs(ma3_df["actual"] - ma3_df["pred_ma3"]))

mae_global = np.mean(np.abs(y_test - y_pred_global))
y_comb = np.concatenate([y_sul, y_cer])
y_pred_comb = np.concatenate([y_pred_sul, y_pred_cer])
mae_reg = np.mean(np.abs(y_comb - y_pred_comb))
mae_sul_v = np.mean(np.abs(y_sul - y_pred_sul))
mae_cer_v = np.mean(np.abs(y_cer - y_pred_cer))

ma3_c = ma3_df.copy()
ma3_c["uf_cod"] = ma3_c["cod_ibge"].astype(str).str[:2].astype(int)
ma3_c["is_sul"] = ma3_c["uf_cod"].isin(REGION_SUL).astype(int)
mae_ma3_sul = np.mean(
    np.abs(ma3_c[ma3_c["is_sul"] == 1]["actual"] - ma3_c[ma3_c["is_sul"] == 1]["pred_ma3"])
)
mae_ma3_cer = np.mean(
    np.abs(ma3_c[ma3_c["is_sul"] == 0]["actual"] - ma3_c[ma3_c["is_sul"] == 0]["pred_ma3"])
)

print("Data loaded. Generating plots...")

# ===================================================================
# PLOT 1: MODEL COMPARISON BAR CHART
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 5))

models_names = ["3-Year MA\n(Baseline)", "LightGBM\n(Global)", "Regional\nLightGBM"]
maes = [mae_ma3, mae_global, mae_reg]
sacas = [m / 60 for m in maes]
colors = [C_GRAY, C_BLUE, C_GREEN]

bars = ax.bar(models_names, maes, color=colors, width=0.5, edgecolor="white", linewidth=1.5)

for bar, mae, saca in zip(bars, maes, sacas):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{mae:.0f} kg/ha\n({saca:.1f} sacas)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_ylabel("MAE (kg/ha)")
ax.set_title("Model Performance — Test Set (2023 Harvest)")
ax.set_ylim(0, max(maes) * 1.25)
ax.axhline(y=mae_ma3, color=C_GRAY, linestyle="--", alpha=0.4, linewidth=1)

imp = (mae_ma3 - mae_reg) / mae_ma3 * 100
ax.annotate(
    f"-{imp:.0f}% vs baseline",
    xy=(2, mae_reg),
    xytext=(2.4, mae_reg + 60),
    fontsize=10,
    color=C_GREEN,
    fontweight="bold",
    arrowprops={"arrowstyle": "->", "color": C_GREEN, "lw": 1.5},
)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/readme_model_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  1/5 readme_model_comparison.png")

# ===================================================================
# PLOT 2: REGIONAL PERFORMANCE
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(2)
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    [mae_ma3_sul, mae_ma3_cer],
    width,
    label="3-Year MA (Baseline)",
    color=C_GRAY,
    alpha=0.7,
)
bars2 = ax.bar(
    x + width / 2,
    [mae_sul_v, mae_cer_v],
    width,
    label="Regional LightGBM",
    color=[C_ORANGE, C_GREEN],
)

for bar, val in zip(bars1, [mae_ma3_sul, mae_ma3_cer]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{val:.0f}",
        ha="center",
        fontsize=10,
        color=C_GRAY,
    )
for bar, val in zip(bars2, [mae_sul_v, mae_cer_v]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{val:.0f}",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )

imp_s = (mae_ma3_sul - mae_sul_v) / mae_ma3_sul * 100
imp_c = (mae_ma3_cer - mae_cer_v) / mae_ma3_cer * 100
ax.text(
    0,
    max(mae_ma3_sul, mae_sul_v) + 45,
    f"-{imp_s:.0f}%",
    ha="center",
    fontsize=11,
    color=C_ORANGE,
    fontweight="bold",
)
ax.text(
    1,
    max(mae_ma3_cer, mae_cer_v) + 45,
    f"-{imp_c:.0f}%",
    ha="center",
    fontsize=11,
    color=C_GREEN,
    fontweight="bold",
)

ax.set_ylabel("MAE (kg/ha)")
ax.set_title("Regional Model Performance — Test Set (2023)")
ax.set_xticks(x)
ax.set_xticklabels(["South\n(RS, PR, SC)", "Cerrado\n(MT, GO, MS, ...)"])
ax.legend(loc="upper right")
ax.set_ylim(0, mae_ma3_sul * 1.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/readme_regional_performance.png", dpi=200, bbox_inches="tight")
plt.close()
print("  2/5 readme_regional_performance.png")

# ===================================================================
# PLOT 3: SCATTER PREDICTED VS ACTUAL (regional)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, y_true, y_p, title, color in [
    (axes[0], y_sul, y_pred_sul, "South", C_ORANGE),
    (axes[1], y_cer, y_pred_cer, "Cerrado", C_GREEN),
]:
    ax.scatter(y_true, y_p, alpha=0.35, s=12, c=color, edgecolors="none")

    lims = [min(y_true.min(), y_p.min()) - 100, max(y_true.max(), y_p.max()) + 100]
    ax.plot(lims, lims, "--", color=C_RED, lw=1.5, alpha=0.6, label="y = x")

    mae = np.mean(np.abs(y_true - y_p))
    r2 = 1 - np.sum((y_true - y_p) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    ax.text(
        0.05,
        0.92,
        f"MAE = {mae:.0f} kg/ha\nR\u00b2 = {r2:.3f}\nn = {len(y_true):,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xlabel("Actual Yield (kg/ha)")
    ax.set_ylabel("Predicted Yield (kg/ha)")
    ax.set_title(f"{title} Region")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)

fig.suptitle(
    "Predicted vs Actual \u2014 Test Set (2023 Harvest)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/readme_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("  3/5 readme_scatter.png")

# ===================================================================
# PLOT 4: FEATURE IMPORTANCE (Top 20)
# ===================================================================
with open("results/training_result.json") as f:
    tr = json.load(f)

fi = tr["feature_importance"]
top_n = 20
feats = list(fi.keys())[:top_n]
vals = [fi[f] for f in feats]
total = sum(fi.values())
pcts = [v / total * 100 for v in vals]

pretty = {
    "produtividade_lag1": "Previous Year Yield",
    "produtividade_ma3": "3-Year Moving Avg",
    "trend": "Temporal Trend",
    "deficit_ratio_enchimento": "Grain Fill Drought Ratio",
    "radiation_total": "Solar Radiation (total)",
    "deficit_enchimento_mm": "Grain Fill Water Deficit",
    "precip_cv": "Precipitation Variability",
    "precip_vegetativo_mm": "Vegetative Precip",
    "water_deficit_ratio": "Overall Water Deficit",
    "deficit_vegetativo_mm": "Vegetative Water Deficit",
    "precip_enchimento_anomaly": "Grain Fill Precip Anomaly",
    "precip_plantio_mm": "Planting Precip",
    "precip_enchimento_mm": "Grain Fill Precip",
    "pct_soja": "Soybean Land Use %",
    "fert_import_ton": "Fertilizer Imports",
    "oni_std": "ENSO Variability",
    "sinistro_rate_3yr": "Insurance Loss Rate",
    "radiation_mean": "Solar Radiation (mean)",
    "oni_avg": "ENSO Index (avg)",
    "tmean_plantio": "Planting Temperature",
}

labels = [pretty.get(f, f) for f in feats]

cat_colors = []
for f in feats:
    if f.startswith("produtividade") or f == "trend":
        cat_colors.append(C_BLUE)
    elif any(k in f for k in ["deficit", "water", "precip", "eto", "dry"]):
        cat_colors.append(C_GREEN)
    elif any(k in f for k in ["oni", "la_nina", "el_nino"]):
        cat_colors.append(C_ORANGE)
    elif any(k in f for k in ["radiation", "temp", "hot", "gdd", "tmean", "tmin", "tmax"]):
        cat_colors.append(C_RED)
    else:
        cat_colors.append(C_GRAY)

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = list(range(len(feats) - 1, -1, -1))

bars = ax.barh(y_pos, pcts, color=cat_colors, height=0.7, edgecolor="white", linewidth=0.5)

for bar, pct in zip(bars, pcts):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%",
        va="center",
        fontsize=9,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Relative Importance (%)")
ax.set_title(f"Top {top_n} Feature Importance (LightGBM Gain)")

legend_elements = [
    Patch(facecolor=C_BLUE, label="Historical"),
    Patch(facecolor=C_GREEN, label="Water Balance"),
    Patch(facecolor=C_RED, label="Temperature / Radiation"),
    Patch(facecolor=C_ORANGE, label="ENSO"),
    Patch(facecolor=C_GRAY, label="Other"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/readme_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print("  4/5 readme_feature_importance.png")

# ===================================================================
# PLOT 5: ERROR BY YEAR
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 5))

year_errors = []
for ano in sorted(df["ano"].unique()):
    year_data = df[df["ano"] == ano]
    X = year_data[model_features].values
    y = year_data["produtividade_kg_ha"].values
    y_p = model.predict(X)
    mae = np.mean(np.abs(y - y_p))
    year_errors.append({"ano": int(ano), "mae": mae, "n": len(year_data)})

ye_df = pd.DataFrame(year_errors)

colors_yr = []
for _, row in ye_df.iterrows():
    if row["ano"] <= 2021:
        colors_yr.append(C_BLUE)
    elif row["ano"] == 2022:
        colors_yr.append(C_ORANGE)
    else:
        colors_yr.append(C_GREEN)

bars = ax.bar(ye_df["ano"], ye_df["mae"], color=colors_yr, width=0.7, alpha=0.8, edgecolor="white")

for _, row in ye_df.iterrows():
    if row["ano"] in [2005, 2012, 2022, 2023]:
        ax.text(
            row["ano"],
            row["mae"] + 15,
            f"{row['mae']:.0f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

ax.axvline(x=2021.5, color=C_ORANGE, linestyle="--", alpha=0.7, linewidth=1.2)
ax.axvline(x=2022.5, color=C_GREEN, linestyle="--", alpha=0.7, linewidth=1.2)

legend_elements = [
    Patch(facecolor=C_BLUE, alpha=0.8, label="Train (2000-2021)"),
    Patch(facecolor=C_ORANGE, alpha=0.8, label="Validation (2022)"),
    Patch(facecolor=C_GREEN, alpha=0.8, label="Test (2023)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

ax.set_xlabel("Harvest Year")
ax.set_ylabel("MAE (kg/ha)")
ax.set_title("Model Error (MAE) by Harvest Year")
ax.set_xticks(ye_df["ano"])
ax.set_xticklabels([str(y) for y in ye_df["ano"]], rotation=45, fontsize=8.5)
ax.set_ylim(0, ye_df["mae"].max() * 1.15)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/readme_error_by_year.png", dpi=200, bbox_inches="tight")
plt.close()
print("  5/5 readme_error_by_year.png")

# ===================================================================
# RESUMO FINAL PARA VALIDACAO
# ===================================================================
print("\n" + "=" * 60)
print("NUMEROS VALIDADOS PARA O README")
print("=" * 60)
print(f"Features: {len(model_features)}")
print(f"Dataset: {len(df):,} rows, {df['cod_ibge'].nunique()} municipios")
print(
    f"Split: train<=2021 ({len(df[df['ano'] <= 2021]):,}), val=2022 ({len(df[df['ano'] == 2022]):,}), test=2023 ({len(df[df['ano'] == 2023]):,})"
)
print("")
print(f"Global model  test MAE: {mae_global:.1f} kg/ha ({mae_global / 60:.1f} sacas)")
print(f"Regional      test MAE: {mae_reg:.1f} kg/ha ({mae_reg / 60:.1f} sacas)")
print(f"  Sul:        {mae_sul_v:.1f} kg/ha ({mae_sul_v / 60:.1f} sacas), n={len(t_sul)}")
print(f"  Cerrado:    {mae_cer_v:.1f} kg/ha ({mae_cer_v / 60:.1f} sacas), n={len(t_cer)}")
print("")
print(f"Baseline MA3  test MAE: {mae_ma3:.1f} kg/ha ({mae_ma3 / 60:.1f} sacas)")
print(f"  MA3 Sul:    {mae_ma3_sul:.1f} kg/ha")
print(f"  MA3 Cerrado:{mae_ma3_cer:.1f} kg/ha")
print("")
print(f"Improvement global:   {(mae_ma3 - mae_global) / mae_ma3 * 100:+.1f}%")
print(f"Improvement regional: {(mae_ma3 - mae_reg) / mae_ma3 * 100:+.1f}%")
print(f"Improvement Sul:      {(mae_ma3_sul - mae_sul_v) / mae_ma3_sul * 100:+.1f}%")
print(f"Improvement Cerrado:  {(mae_ma3_cer - mae_cer_v) / mae_ma3_cer * 100:+.1f}%")
print("")
print("Data sources: IBGE PAM, NASA POWER, ISRIC SoilGrids, NOAA ONI, MODIS NDVI,")
print("              ANA Pivos, ComexStat, MAPA PSR, MapBiomas")
print("=" * 60)
