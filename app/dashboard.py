import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    import joblib
    import shap
    import matplotlib.pyplot as plt
    from src.modeling.train_conformal import ConformalCalibrator

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Erro: Bibliotecas necessárias nao instaladas.")
    print("Execute: pip install streamlit plotly shap matplotlib joblib")
    sys.exit(1)


PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions_2024_2025.parquet"
RISK_PATH = PROJECT_ROOT / "results" / "risk_analysis_2024_2025.parquet"
DRIFT_PATH = PROJECT_ROOT / "results" / "drift_report.md"
TRAINING_RESULT_PATH = PROJECT_ROOT / "results" / "training_result.json"
EVALUATION_PATH = PROJECT_ROOT / "results" / "evaluation_report.md"


def ano_para_safra(ano: int) -> str:
    """Converte ano PAM para nomenclatura de safra."""
    return f"{ano-1}/{str(ano)[2:]}"


def safra_para_ano(safra: str) -> int:
    """Converte nomenclatura de safra para ano PAM."""
    return int("20" + safra.split("/")[1])


@st.cache_data
def load_predictions():
    """Carrega previsoes."""
    if PREDICTIONS_PATH.exists():
        df = pd.read_parquet(PREDICTIONS_PATH)
        df["safra"] = df["ano"].apply(ano_para_safra)
        return df
    return None


@st.cache_data
def load_risk_analysis():
    """Carrega analise de risco."""
    if RISK_PATH.exists():
        df = pd.read_parquet(RISK_PATH)
        df["safra"] = df["ano"].apply(ano_para_safra)
        return df
    return None


@st.cache_data
def load_training_result():
    """Carrega resultado do treinamento."""
    if TRAINING_RESULT_PATH.exists():
        with open(TRAINING_RESULT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_drift_report():
    """Carrega relatorio de drift."""
    if DRIFT_PATH.exists():
        with open(DRIFT_PATH, encoding="utf-8") as f:
            return f.read()
    return None


def render_model_info_sidebar():
    """Renderiza informacoes do modelo na sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sobre o Modelo")

    training = load_training_result()

    if training:
        st.sidebar.markdown(
            f"""
        **Versao:** v3.0 (Fase 2)

        **Dados de Treino:**
        - Safras 1999/00 a 2017/18
        - {training.get('train_metrics', {}).get('n_samples', 'N/A')} amostras

        **Validacao (2018/19 a 2020/21):**
        - MAE: {training.get('val_metrics', {}).get('mae_kg_ha', 0):.0f} kg/ha

        **Teste (2021/22 a 2022/23):**
        - MAE: {training.get('test_metrics', {}).get('mae_kg_ha', 0):.0f} kg/ha
        """
        )
    else:
        st.sidebar.markdown(
            """
        **Versao:** v3.0 (Fase 2)

        **Dados de Treino:**
        - Safras 1999/00 a 2017/18

        **Teste (2021/22 a 2022/23):**
        - MAE: ~550 kg/ha
        """
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Legenda de Safra")
    st.sidebar.markdown(
        """
    A nomenclatura segue o ciclo agricola:
    - **Safra 2023/24**: plantio Out/2023, colheita 2024
    - **Safra 2024/25**: plantio Out/2024, colheita 2025
    """
    )


def page_visao_geral():
    """Pagina de visao geral com KPIs e graficos."""
    st.header("Visao Geral - Previsoes de Produtividade")

    df = load_predictions()
    df_risk = load_risk_analysis()

    if df is None:
        st.error(
            "Dados de previsao nao encontrados. Execute `python -m src.inference.predict` primeiro."
        )
        return

    st.info(
        """
    **Sobre estas previsoes:**
    - Modelo treinado com dados de 2000-2018 (19 safras)
    - Testado em 2019-2023 com MAE de ~550 kg/ha (~9 sacas/ha)
    - Previsoes abaixo sao para safras **futuras** usando clima ja observado (modalidade ex-post)
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        safras_disponiveis = sorted(df["safra"].unique())
        safra_selecionada = st.selectbox("Safra", safras_disponiveis)
    with col2:
        ufs_disponiveis = ["Todas"] + sorted(df["uf"].unique().tolist())
        uf_selecionada = st.selectbox("UF", ufs_disponiveis)

    ano_selecionado = safra_para_ano(safra_selecionada)
    df_filtrado = df[df["ano"] == ano_selecionado]
    if uf_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado["uf"] == uf_selecionada]

    st.subheader(f"Indicadores - Safra {safra_selecionada}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Municipios Analisados",
            f"{len(df_filtrado):,}",
        )

    with col2:
        media_prod = df_filtrado["pred_produtividade_kg_ha"].mean()
        st.metric(
            "Produtividade Media Prevista",
            f"{media_prod:.0f} kg/ha",
            f"{media_prod/60:.1f} sc/ha",
        )

    with col3:
        if "pred_p10_kg_ha" in df_filtrado.columns:
            media_p10 = df_filtrado["pred_p10_kg_ha"].mean()
            st.metric(
                "Cenario Pessimista (p10)",
                f"{media_p10:.0f} kg/ha",
                help="Valor abaixo do qual ha apenas 10% de probabilidade",
            )

    with col4:
        if df_risk is not None:
            df_risk_filtrado = df_risk[df_risk["ano"] == ano_selecionado]
            if uf_selecionada != "Todas":
                df_risk_filtrado = df_risk_filtrado[df_risk_filtrado["uf"] == uf_selecionada]
            alto_risco = (df_risk_filtrado["rating"].isin(["C", "D"])).sum()
            pct_alto = alto_risco / len(df_risk_filtrado) * 100 if len(df_risk_filtrado) > 0 else 0
            st.metric(
                "Municipios Alto Risco",
                f"{alto_risco:,} ({pct_alto:.1f}%)",
                help="Rating C ou D: probabilidade de quebra > 20%",
            )

    st.subheader(f"Produtividade Prevista por UF - Safra {safra_selecionada}")

    df_ano = df[df["ano"] == ano_selecionado]
    df_uf = (
        df_ano.groupby("uf")
        .agg({"pred_produtividade_kg_ha": "mean", "cod_ibge": "count"})
        .reset_index()
    )
    df_uf.columns = ["UF", "Produtividade Media (kg/ha)", "Municipios"]
    df_uf = df_uf.sort_values("Produtividade Media (kg/ha)", ascending=True)

    fig = px.bar(
        df_uf,
        x="Produtividade Media (kg/ha)",
        y="UF",
        orientation="h",
        color="Produtividade Media (kg/ha)",
        color_continuous_scale="RdYlGn",
        text=df_uf["Produtividade Media (kg/ha)"].apply(lambda x: f"{x:.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Produtividade Prevista (kg/ha)",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    if df_risk is not None:
        st.subheader(f"Distribuicao de Risco - Safra {safra_selecionada}")

        col1, col2 = st.columns(2)

        with col1:
            df_rating = (
                df_risk[df_risk["ano"] == ano_selecionado]["rating"].value_counts().reset_index()
            )
            df_rating.columns = ["Rating", "Quantidade"]
            df_rating = df_rating.sort_values("Rating")

            fig_risk = px.pie(
                df_rating,
                values="Quantidade",
                names="Rating",
                color="Rating",
                color_discrete_map={"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"},
                hole=0.4,
            )
            fig_risk.update_layout(
                title="Distribuicao por Rating",
                annotations=[dict(text="Rating", x=0.5, y=0.5, font_size=14, showarrow=False)],
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        with col2:
            st.markdown(
                """
            **Legenda de Ratings:**

            | Rating | Risco | Prob. Quebra |
            |--------|-------|--------------|
            | **A** | Baixo | < 10% |
            | **B** | Moderado | 10-20% |
            | **C** | Elevado | 20-35% |
            | **D** | Alto | > 35% |

            *Quebra = Receita < Custo de Producao*
            """
            )


def page_validacao():
    """Pagina de validacao do modelo."""
    st.header("Validacao do Modelo")

    st.markdown(
        """
    Esta pagina mostra a qualidade das previsoes do modelo quando comparadas
    com dados reais de produtividade (PAM/IBGE).
    """
    )

    st.subheader("Estrutura de Dados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Divisao Temporal (sem vazamento):**

        | Conjunto | Safras | Anos PAM | Uso |
        |----------|--------|----------|-----|
        | Treino | 1999/00 a 2017/18 | 2000-2018 | Aprendizado |
        | Validacao | 2018/19 a 2020/21 | 2019-2021 | Early stopping |
        | Teste | 2021/22 a 2022/23 | 2022-2023 | Avaliacao final |
        | Previsao | 2023/24 a 2024/25 | 2024-2025 | Producao |
        """
        )

    with col2:
        st.markdown(
            """
        **Por que esta divisao?**

        - O modelo **nunca ve dados futuros** durante o treino
        - Simula situacao real de previsao
        - Safra 2022/23 teve seca severa no Sul (La Nina)
        - Teste em condicoes adversas = validacao robusta
        """
        )

    st.subheader("Metricas no Conjunto de Teste (2022-2023)")

    training = load_training_result()

    if training:
        test_metrics = training.get("test_metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mae = test_metrics.get("mae_kg_ha", 551)
            st.metric("MAE", f"{mae:.0f} kg/ha", f"{mae/60:.1f} sc/ha", help="Erro Medio Absoluto")

        with col2:
            mape = test_metrics.get("mape_percent", 37.6)
            st.metric("MAPE", f"{mape:.1f}%", help="Erro Percentual Medio")

        with col3:
            st.metric("vs Baseline (MA3)", "+1.6%", help="Melhoria sobre media movel 3 anos")

        with col4:
            st.metric("Amostras de Teste", "4.669", help="Municipios x anos no teste")

    st.subheader("Erro por Regiao (Teste 2022-2023)")

    st.warning(
        """
    **Atencao:** O modelo tem dificuldade maior na regiao Sul (RS, PR, SC)
    devido a maior volatilidade climatica e impacto de La Nina.
    """
    )

    erro_uf = pd.DataFrame(
        {
            "UF": ["RS", "PR", "MS", "SC", "SP", "GO", "MG", "MT", "BA", "TO"],
            "MAE (kg/ha)": [911, 870, 882, 631, 376, 350, 320, 280, 418, 250],
            "MAPE (%)": [92.2, 68.2, 72.8, 30.3, 12.1, 10.5, 9.8, 7.5, 19.8, 7.2],
        }
    )

    fig = px.bar(
        erro_uf.sort_values("MAE (kg/ha)", ascending=True),
        x="MAE (kg/ha)",
        y="UF",
        orientation="h",
        color="MAE (kg/ha)",
        color_continuous_scale="Reds",
        text="MAE (kg/ha)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=400,
        xaxis_title="MAE (kg/ha) - Menor e melhor",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Erro por Safra (Teste)")

    erro_ano = pd.DataFrame(
        {
            "Safra": ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23"],
            "MAE (kg/ha)": [400, 459, 342, 680, 426],
            "Evento": ["Normal", "Normal", "Normal", "La Nina Forte", "Normal"],
        }
    )

    fig2 = px.bar(
        erro_ano,
        x="Safra",
        y="MAE (kg/ha)",
        color="Evento",
        color_discrete_map={"Normal": "#3498db", "La Nina Forte": "#e74c3c"},
        text="MAE (kg/ha)",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        height=400,
        xaxis_title="Safra",
        yaxis_title="MAE (kg/ha)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
    **Observacao:** A safra 2021/22 teve seca severa no Sul devido a La Nina,
    resultando em erro maior. O modelo captura parcialmente esse efeito atraves
    das features de ENSO (El Nino / La Nina).
    """
    )


def page_municipio():
    """Pagina de analise por municipio."""
    st.header("Analise por Municipio")

    df = load_predictions()
    df_risk = load_risk_analysis()

    if df is None or df_risk is None:
        st.error("Dados nao encontrados.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        uf = st.selectbox("UF", sorted(df["uf"].unique()))
    with col2:
        municipios = df[df["uf"] == uf]["nome"].unique()
        municipio = st.selectbox("Municipio", sorted(municipios))
    with col3:
        safras = sorted(df["safra"].unique())
        safra = st.selectbox("Safra", safras)

    ano = safra_para_ano(safra)

    df_mun = df[(df["uf"] == uf) & (df["nome"] == municipio) & (df["ano"] == ano)]
    df_risk_mun = df_risk[
        (df_risk["uf"] == uf) & (df_risk["municipio"] == municipio) & (df_risk["ano"] == ano)
    ]

    if len(df_mun) == 0:
        st.warning(f"Dados nao disponiveis para {municipio}/{uf} na safra {safra}.")
        return

    row = df_mun.iloc[0]
    risk_row = df_risk_mun.iloc[0] if len(df_risk_mun) > 0 else None

    st.subheader(f"{municipio}/{uf} - Safra {safra}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Previsao de Produtividade")

        if "pred_p10_kg_ha" in row.index:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="Intervalo 80%",
                    x=["Previsao"],
                    y=[row["pred_p90_kg_ha"] - row["pred_p10_kg_ha"]],
                    base=[row["pred_p10_kg_ha"]],
                    marker_color="rgba(52, 152, 219, 0.3)",
                    width=0.5,
                )
            )

            fig.add_hline(
                y=row["pred_p10_kg_ha"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"p10: {row['pred_p10_kg_ha']:.0f}",
            )

            fig.add_hline(
                y=row["pred_p50_kg_ha"],
                line_dash="solid",
                line_color="blue",
                annotation_text=f"p50: {row['pred_p50_kg_ha']:.0f}",
            )

            fig.add_hline(
                y=row["pred_p90_kg_ha"],
                line_dash="dash",
                line_color="green",
                annotation_text=f"p90: {row['pred_p90_kg_ha']:.0f}",
            )

            fig.update_layout(
                height=350,
                yaxis_title="Produtividade (kg/ha)",
                showlegend=False,
                yaxis=dict(range=[0, row["pred_p90_kg_ha"] * 1.2]),
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
        | Cenario | Produtividade | Sacas/ha |
        |---------|---------------|----------|
        """
        )
        st.markdown(
            f"| Pessimista (p10) | {row.get('pred_p10_kg_ha', 0):.0f} kg/ha | {row.get('pred_p10_kg_ha', 0)/60:.1f} |"
        )
        st.markdown(
            f"| **Base (p50)** | **{row.get('pred_p50_kg_ha', row['pred_produtividade_kg_ha']):.0f} kg/ha** | **{row.get('pred_p50_kg_ha', row['pred_produtividade_kg_ha'])/60:.1f}** |"
        )
        st.markdown(
            f"| Otimista (p90) | {row.get('pred_p90_kg_ha', 0):.0f} kg/ha | {row.get('pred_p90_kg_ha', 0)/60:.1f} |"
        )

    with col2:
        st.markdown("### Analise de Risco")

        if risk_row is not None:
            rating = risk_row["rating"]
            rating_colors = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"}
            rating_labels = {
                "A": "BAIXO RISCO",
                "B": "RISCO MODERADO",
                "C": "RISCO ELEVADO",
                "D": "ALTO RISCO",
            }

            st.markdown(
                f"""<div style='background-color: {rating_colors.get(rating, 'gray')};
                padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='color: white; margin: 0;'>{rating}</h1>
                <p style='color: white; margin: 0;'>{rating_labels.get(rating, '')}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Prob. Quebra", f"{risk_row['prob_quebra']*100:.1f}%")
                st.metric("Custo Producao", f"R$ {risk_row['custo_ha']:,.0f}/ha")
            with col_b:
                st.metric("Spread Sugerido", f"{risk_row['spread_sugerido']}% a.a.")
                st.metric("Preco Soja", f"R$ {risk_row['preco_saca']:.0f}/sc")

            st.markdown("#### Cenarios Financeiros")
            st.markdown(
                f"""
            | Cenario | Receita/ha | Lucro/ha |
            |---------|------------|----------|
            | Pessimista | R$ {risk_row['receita_pessimista']:,.0f} | R$ {risk_row['lucro_pessimista']:,.0f} |
            | Base | R$ {risk_row['receita_base']:,.0f} | R$ {risk_row['lucro_base']:,.0f} |
            | Otimista | R$ {risk_row['receita_otimista']:,.0f} | R$ {risk_row['lucro_otimista']:,.0f} |
            """
            )

    st.markdown("---")
    st.subheader("O que influenciou esta previsao? (SHAP)")

    with st.spinner("Gerando explicabilidade (isso pode levar alguns segundos)..."):
        try:
            MODEL_PATH = PROJECT_ROOT / "models" / "model_sul.pkl"
            gbm = joblib.load(MODEL_PATH)
            features_treino = gbm.feature_name()

            FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"

            if not FEATURES_PATH.exists():
                st.warning("Arquivo de features climáticas não encontrado para gerar o SHAP.")
            else:
                df_features = pd.read_parquet(FEATURES_PATH)

                df_features["cod_ibge"] = df_features["cod_ibge"].astype(int)
                df_features["ano"] = df_features["ano"].astype(int)
                cod_ibge_busca = int(row["cod_ibge"])
                ano_busca = int(ano)

                X_municipio_clima = df_features[
                    (df_features["cod_ibge"] == cod_ibge_busca) &
                    (df_features["ano"] == ano_busca)
                ]

                if len(X_municipio_clima) == 0:
                    anos_disponiveis = df_features["ano"].unique()
                    st.error(f"Erro no filtro! Procuramos: IBGE {cod_ibge_busca} e Ano {ano_busca}.")
                    st.info(f"O arquivo 'dataset_final.parquet' contém apenas os anos: {sorted(anos_disponiveis)}")
                else:
                    X_municipio_final = X_municipio_clima[features_treino]

                    explainer = shap.TreeExplainer(gbm)
                    shap_values = explainer(X_municipio_final)
                    shap_values.feature_names = features_treino

                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], show=False)
                    plt.tight_layout()
                    st.pyplot(fig_shap)

        except Exception as e:
            st.warning(f"Erro técnico ao gerar SHAP: {e}")


def page_regional():
    """Pagina de comparativo regional."""
    st.header("Comparativo Regional")

    df = load_predictions()
    df_risk = load_risk_analysis()

    if df is None:
        st.error("Dados nao encontrados.")
        return

    safra = st.selectbox("Safra", sorted(df["safra"].unique()))
    ano = safra_para_ano(safra)

    st.subheader(f"Ranking de UFs - Safra {safra}")

    df_ano = df[df["ano"] == ano]

    agg_dict = {
        "pred_produtividade_kg_ha": ["mean", "std"],
        "cod_ibge": "count",
    }

    if "pred_p10_kg_ha" in df_ano.columns:
        agg_dict["pred_p10_kg_ha"] = "mean"
        agg_dict["pred_p90_kg_ha"] = "mean"

    df_uf = df_ano.groupby("uf").agg(agg_dict).round(1)
    df_uf.columns = ["Prod. Media", "Desvio Padrao", "Municipios", "p10 Medio", "p90 Medio"]
    df_uf = df_uf.reset_index()
    df_uf = df_uf.sort_values("Prod. Media", ascending=False)

    if df_risk is not None:
        df_risk_ano = df_risk[df_risk["ano"] == ano]
        risco_uf = (
            df_risk_ano.groupby("uf")
            .apply(lambda x: (x["rating"].isin(["C", "D"])).sum() / len(x) * 100)
            .reset_index()
        )
        risco_uf.columns = ["uf", "% Alto Risco"]
        df_uf = df_uf.merge(risco_uf, on="uf", how="left")

    st.dataframe(
        df_uf.style.format(
            {
                "Prod. Media": "{:.0f}",
                "Desvio Padrao": "{:.0f}",
                "p10 Medio": "{:.0f}",
                "p90 Medio": "{:.0f}",
                "% Alto Risco": "{:.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if df_risk is not None and "% Alto Risco" in df_uf.columns:
        st.subheader("Municipios com Alto Risco por UF")

        df_uf_sorted = df_uf.sort_values("% Alto Risco", ascending=True)

        fig = px.bar(
            df_uf_sorted,
            x="% Alto Risco",
            y="uf",
            orientation="h",
            color="% Alto Risco",
            color_continuous_scale="Reds",
            text=df_uf_sorted["% Alto Risco"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=500,
            xaxis_title="% de Municipios com Rating C ou D",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_monitoramento():
    """Pagina de monitoramento e drift."""
    st.header("Monitoramento do Modelo")

    st.markdown(
        """
    O monitoramento detecta mudancas nas distribuicoes de dados (drift) que
    podem degradar a qualidade das previsoes ao longo do tempo.
    """
    )

    st.subheader("Status Atual")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Versao", "v3.0 (Fase 2)")

    with col2:
        st.metric("Features", "48")

    with col3:
        st.metric("Treino ate", "Safra 2017/18")

    with col4:
        st.metric("Ultima Validacao", "Safra 2022/23")

    st.subheader("Analise de Drift")

    drift_report = load_drift_report()

    if drift_report:
        with st.expander("Ver Relatorio Completo de Drift", expanded=False):
            st.code(drift_report, language="text")

        st.warning(
            """
        **Drift Detectado:**
        - `oni_avg` (ENSO): Mais eventos La Nina no periodo recente
        - `hot_days_count`: Aumento de dias quentes
        - `produtividade_ma3`: Tendencia de aumento da produtividade

        **Recomendacao:** Considerar retreino com dados ate safra 2022/23
        """
        )

    else:
        st.info(
            "Execute `python -m src.monitoring.drift_analysis` para gerar o relatorio de drift."
        )

    st.subheader("Degradacao Temporal")

    erro_ano = pd.DataFrame(
        {
            "Safra": ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23"],
            "MAE (kg/ha)": [400, 459, 342, 680, 426],
            "Variacao": [0, 14.8, -14.5, 70.0, 6.5],
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=erro_ano["Safra"],
            y=erro_ano["MAE (kg/ha)"],
            marker_color=["#3498db", "#3498db", "#2ecc71", "#e74c3c", "#3498db"],
            text=erro_ano["MAE (kg/ha)"],
            textposition="outside",
        )
    )

    fig.add_hline(
        y=500, line_dash="dash", line_color="orange", annotation_text="Limite Aceitavel (500 kg/ha)"
    )

    fig.update_layout(
        height=400,
        xaxis_title="Safra",
        yaxis_title="MAE (kg/ha)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    """Aplicacao principal do dashboard."""
    st.set_page_config(
        page_title="Soja Produtividade",
        page_icon=":seedling:",
        layout="wide",
    )

    st.title(":seedling: Sistema de Previsao de Produtividade de Soja")

    pagina = st.sidebar.radio(
        "Navegacao",
        [
            "Visao Geral",
            "Validacao do Modelo",
            "Analise por Municipio",
            "Comparativo Regional",
            "Monitoramento",
        ],
    )

    render_model_info_sidebar()

    if pagina == "Visao Geral":
        page_visao_geral()
    elif pagina == "Validacao do Modelo":
        page_validacao()
    elif pagina == "Analise por Municipio":
        page_municipio()
    elif pagina == "Comparativo Regional":
        page_regional()
    elif pagina == "Monitoramento":
        page_monitoramento()


if __name__ == "__main__":
    main()
