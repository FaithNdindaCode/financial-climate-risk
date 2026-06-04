import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(__file__))
from models.tcfd_model import load_and_enrich, scenario_analysis, compute_climate_var
from models.forecast_model import run_all_forecasts, FORECAST_YEARS, SCENARIO_TRAJECTORIES

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TCFD Climate Risk | SmartHaven Digital",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 600; }
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🌍 TCFD Climate Risk Dashboard")
st.markdown(
    "**SmartHaven Digital | P02 — Financial Engineering Portfolio**  \n"
    "TCFD-aligned physical & transition risk analysis for NSE-listed companies · Kenya market"
)
st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_enrich("data/climate_risk_data.csv")

df_base = get_data()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    scenario = st.selectbox(
        "TCFD Scenario",
        ["1.5C", "2C", "3C"],
        index=1,
        help="1.5C = aggressive transition | 2C = moderate | 3C = physical risk dominant"
    )

    sectors = st.multiselect(
        "Filter by sector",
        options=sorted(df_base["sector"].unique()),
        default=sorted(df_base["sector"].unique()),
    )

    risk_filter = st.select_slider(
        "Minimum risk category",
        options=["🟢 Low", "🟡 Medium", "🔴 High"],
        value="🟢 Low",
    )

    st.divider()
    st.markdown("**About this model**")
    st.caption(
        "Monte Carlo CVaR with 10,000 simulations. "
        "Scenario multipliers follow NGFS/TCFD guidance. "
        "Carbon cost based on scenario carbon price (USD/tonne CO₂e)."
    )
    st.caption("📁 [GitHub repo](https://github.com/FaithNdindaCode/financial-engineering-portfolio)")

# ── Apply scenario + filters ──────────────────────────────────────────────────
df = scenario_analysis(df_base, scenario)
df = df[df["sector"].isin(sectors)]

risk_order = {"🟢 Low": 0, "🟡 Medium": 1, "🔴 High": 2}
min_risk = risk_order[risk_filter]
df = df[df["risk_category"].map(risk_order) >= min_risk]

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Companies analysed", len(df))
with col2:
    high_risk = (df["risk_category"] == "🔴 High").sum()
    st.metric("High risk companies", high_risk, delta=f"{high_risk/len(df)*100:.0f}% of portfolio")
with col3:
    avg_var = df["var_95"].mean()
    st.metric("Avg Climate VaR (95%)", f"{avg_var:.1f}%", help="Revenue at risk under selected scenario")
with col4:
    total_carbon_cost = df["carbon_cost_exposure_usd_m"].sum()
    st.metric("Total carbon cost exposure", f"USD {total_carbon_cost:,.0f}M")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Overview", "🎯 Scenario Analysis", "📉 Climate VaR",
    "🔮 Forecast 2025–2030", "📋 Data Table"
])

# ─── Tab 1: Risk Overview ──────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Physical vs Transition Risk")
        fig1 = px.scatter(
            df,
            x="adj_physical_risk",
            y="adj_transition_risk",
            size="revenue_usd_millions",
            color="risk_category",
            hover_name="company",
            hover_data={"sector": True, "carbon_intensity_tCO2e": True, "var_95": True},
            color_discrete_map={"🔴 High": "#e74c3c", "🟡 Medium": "#f39c12", "🟢 Low": "#27ae60"},
            labels={
                "adj_physical_risk": f"Physical Risk Score ({scenario})",
                "adj_transition_risk": f"Transition Risk Score ({scenario})",
                "revenue_usd_millions": "Revenue (USD M)",
            },
            title=f"Risk Matrix — {scenario} Scenario",
        )
        fig1.add_hline(y=7.5, line_dash="dash", line_color="red", opacity=0.4, annotation_text="High transition threshold")
        fig1.add_vline(x=7.5, line_dash="dash", line_color="orange", opacity=0.4, annotation_text="High physical threshold")
        fig1.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, width="stretch")

    with col_b:
        st.markdown("#### Carbon Intensity by Sector")
        sector_avg = df.groupby("sector")["carbon_intensity_tCO2e"].mean().reset_index().sort_values("carbon_intensity_tCO2e", ascending=True)
        fig2 = px.bar(
            sector_avg,
            x="carbon_intensity_tCO2e",
            y="sector",
            orientation="h",
            color="carbon_intensity_tCO2e",
            color_continuous_scale="RdYlGn_r",
            labels={"carbon_intensity_tCO2e": "Avg Carbon Intensity (tCO₂e/USD M rev)", "sector": ""},
            title="Carbon Intensity Heatbar by Sector",
        )
        fig2.update_layout(height=420, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, width="stretch")

    # TCFD disclosure vs risk
    st.markdown("#### TCFD Disclosure Score vs Risk Exposure")
    fig3 = px.scatter(
        df,
        x="tcfd_disclosure_score",
        y="var_95",
        color="sector",
        size="carbon_cost_exposure_usd_m",
        hover_name="company",
        trendline="ols",
        labels={
            "tcfd_disclosure_score": "TCFD Disclosure Score (0–100)",
            "var_95": "Climate VaR 95% (% revenue)",
            "carbon_cost_exposure_usd_m": "Carbon Cost Exposure (USD M)",
        },
        title="Does better disclosure correlate with lower climate risk?",
    )
    fig3.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, width="stretch")


# ─── Tab 2: Scenario Analysis ─────────────────────────────────────────────────
with tab2:
    st.markdown(f"#### Scenario Comparison — Carbon Cost Exposure")
    st.caption("Carbon cost exposure = carbon intensity × revenue × scenario carbon price (USD/tonne CO₂e)")

    scenarios_all = {}
    for sc in ["1.5C", "2C", "3C"]:
        sc_df = scenario_analysis(df_base[df_base["sector"].isin(sectors)], sc)
        sc_df["scenario"] = sc
        scenarios_all[sc] = sc_df

    combined = pd.concat(scenarios_all.values())

    fig4 = px.bar(
        combined,
        x="company",
        y="carbon_cost_exposure_usd_m",
        color="scenario",
        barmode="group",
        color_discrete_map={"1.5C": "#2ecc71", "2C": "#f39c12", "3C": "#e74c3c"},
        labels={"carbon_cost_exposure_usd_m": "Carbon Cost Exposure (USD M)", "company": ""},
        title="Carbon Cost Exposure Across TCFD Scenarios",
    )
    fig4.update_layout(height=420, xaxis_tickangle=-35, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, width="stretch")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### Scenario carbon price assumptions")
        scenario_info = pd.DataFrame({
            "Scenario": ["1.5C", "2C", "3C"],
            "Carbon Price (USD/tonne)": [85, 50, 25],
            "Physical Risk Multiplier": ["0.8×", "1.2×", "1.8×"],
            "Transition Risk Multiplier": ["1.6×", "1.2×", "0.7×"],
            "Dominant Risk": ["Transition", "Balanced", "Physical"],
        })
        st.dataframe(scenario_info, width="stretch", hide_index=True)

    with col_s2:
        st.markdown("#### Stranded asset exposure")
        stranded = df[df["stranded_asset_risk"] == "high"][["company", "sector", "carbon_intensity_tCO2e", "carbon_cost_exposure_usd_m"]]
        if stranded.empty:
            st.info("No high stranded asset risk companies in current filter.")
        else:
            st.dataframe(stranded.rename(columns={
                "carbon_intensity_tCO2e": "Carbon Intensity",
                "carbon_cost_exposure_usd_m": "Carbon Cost Exp. (USD M)"
            }), width="stretch", hide_index=True)


# ─── Tab 3: Climate VaR ───────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Climate Value-at-Risk (Monte Carlo, 10,000 simulations)")
    st.caption("VaR 95% = revenue % at risk in 95th percentile loss scenario | CVaR = expected loss beyond VaR")

    fig5 = go.Figure()
    df_sorted = df.sort_values("cvar_95", ascending=False)
    fig5.add_trace(go.Bar(
        name="CVaR 95%",
        x=df_sorted["company"],
        y=df_sorted["cvar_95"],
        marker_color="#e74c3c",
        opacity=0.85,
    ))
    fig5.add_trace(go.Bar(
        name="VaR 95%",
        x=df_sorted["company"],
        y=df_sorted["var_95"],
        marker_color="#f39c12",
        opacity=0.85,
    ))
    fig5.update_layout(
        barmode="overlay",
        xaxis_tickangle=-35,
        height=400,
        yaxis_title="% of Revenue at Risk",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig5, width="stretch")

    # Interactive single-company VaR
    st.markdown("#### 🔍 Single company deep-dive")
    selected_co = st.selectbox("Select company", df["company"].unique())
    row = df[df["company"] == selected_co].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("VaR 95%", f"{row['var_95']}%")
    c2.metric("CVaR 95%", f"{row['cvar_95']}%")
    c3.metric("Carbon Intensity", f"{row['carbon_intensity_tCO2e']} tCO₂e")

    # Distribution plot
    import numpy as np
    np.random.seed(42)
    n = 10000
    physical_shock = np.random.normal(row["physical_risk_score"] * 0.012, row["physical_risk_score"] * 0.008, n)
    transition_shock = np.random.normal(row["transition_risk_score"] * 0.015, row["transition_risk_score"] * 0.010, n)
    carbon_cost = (row["carbon_intensity_tCO2e"] / 500) * np.random.uniform(0.005, 0.025, n)
    losses = np.clip(physical_shock + transition_shock + carbon_cost, 0, None) * 100

    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(x=losses, nbinsx=80, name="Loss distribution", marker_color="#3498db", opacity=0.7))
    fig6.add_vline(x=row["var_95"], line_color="#f39c12", line_dash="dash", annotation_text=f"VaR 95% = {row['var_95']}%")
    fig6.add_vline(x=row["cvar_95"], line_color="#e74c3c", line_dash="dash", annotation_text=f"CVaR = {row['cvar_95']}%")
    fig6.update_layout(
        title=f"Loss Distribution — {selected_co}",
        xaxis_title="% Revenue Loss",
        yaxis_title="Frequency",
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig6, width="stretch")


# ─── Tab 4: Forecast 2025–2030 ───────────────────────────────────────────────
with tab4:
    st.markdown("#### 🔮 Climate Risk Forecast — 2025 to 2030")
    st.caption(
        "Forward-looking projections per company under each TCFD scenario. "
        "Includes carbon intensity trajectory, risk score evolution, "
        "Climate VaR trend, transition pathway classification, and early warning tipping points."
    )

    col_fc1, col_fc2 = st.columns([1, 2])
    with col_fc1:
        forecast_company = st.selectbox(
            "Select company", sorted(df["company"].unique()), key="fc_company"
        )
        forecast_scenario = st.selectbox(
            "Forecast scenario", ["1.5C", "2C", "3C"], index=1, key="fc_scenario"
        )

    company_row = df[df["company"] == forecast_company].iloc[0].to_dict()

    with st.spinner("Running forecast model..."):
        fc = run_all_forecasts(company_row, forecast_scenario)

    with col_fc2:
        pw_color = {"🟢 Early Mover": "#27ae60", "🔵 On Track": "#3498db",
                    "🟡 Laggard": "#f39c12", "🔴 Stranded Risk": "#e74c3c"}
        color = pw_color.get(fc["pathway"], "#888")
        st.markdown(
            f"<div style='background:rgba(0,0,0,0.05);border-left:4px solid {color};"
            f"padding:12px 16px;border-radius:6px;margin-top:8px'>"
            f"<b>Transition Pathway:</b> {fc['pathway']}<br>"
            f"<b>Pathway Score:</b> {fc['pathway_score']} / 100<br>"
            f"<b>Scenario:</b> {forecast_scenario} · "
            f"<b>Company:</b> {forecast_company}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Early warning tipping points ──────────────────────────────────────────
    if fc["tipping_points"]:
        st.markdown("#### ⚠️ Early Warning — Tipping Points Detected")
        for tp in fc["tipping_points"]:
            st.warning(
                f"**{tp['type']}** — crosses threshold in **{tp['year']}**  \n"
                f"Score: {tp['value']:.2f} (threshold: {tp['threshold']})  \n"
                f"*{tp['implication']}*"
            )
    else:
        st.success(
            f"✅ No critical tipping points detected for {forecast_company} "
            f"under the {forecast_scenario} scenario through 2030."
        )

    st.divider()

    # ── Charts row 1: Carbon intensity + Composite risk ───────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Carbon Intensity Trajectory (tCO₂e / USD M revenue)**")
        ci = fc["carbon"]
        fig_fc1 = go.Figure()
        fig_fc1.add_trace(go.Scatter(
            x=ci["year"], y=ci["upper_bound"],
            mode="lines", line=dict(width=0),
            showlegend=False, name="Upper bound"
        ))
        fig_fc1.add_trace(go.Scatter(
            x=ci["year"], y=ci["lower_bound"],
            fill="tonexty",
            fillcolor="rgba(52,152,219,0.15)",
            mode="lines", line=dict(width=0),
            name="Uncertainty band"
        ))
        fig_fc1.add_trace(go.Scatter(
            x=ci["year"], y=ci["carbon_intensity"],
            mode="lines+markers",
            line=dict(color="#3498db", width=2.5),
            marker=dict(size=7),
            name="Carbon intensity",
        ))
        # Add 2023 baseline
        fig_fc1.add_hline(
            y=company_row.get("carbon_intensity_tCO2e", 100),
            line_dash="dash", line_color="gray", opacity=0.5,
            annotation_text="2023 baseline",
        )
        fig_fc1.update_layout(
            height=340, xaxis_title="Year",
            yaxis_title="tCO₂e / USD M revenue",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_fc1, width="stretch")

    with c2:
        st.markdown("**Composite Climate Risk Score (0–10)**")
        rk = fc["risk"]
        fig_fc2 = go.Figure()
        fig_fc2.add_trace(go.Scatter(
            x=rk["year"], y=rk["physical_risk"],
            mode="lines+markers", name="Physical Risk",
            line=dict(color="#e67e22", width=2), marker=dict(size=6),
        ))
        fig_fc2.add_trace(go.Scatter(
            x=rk["year"], y=rk["transition_risk"],
            mode="lines+markers", name="Transition Risk",
            line=dict(color="#8e44ad", width=2), marker=dict(size=6),
        ))
        fig_fc2.add_trace(go.Scatter(
            x=rk["year"], y=rk["composite_risk"],
            mode="lines+markers", name="Composite Risk",
            line=dict(color="#e74c3c", width=3, dash="dot"), marker=dict(size=8),
        ))
        fig_fc2.add_hline(y=7.5, line_dash="dash", line_color="red",
                          opacity=0.4, annotation_text="High risk threshold")
        fig_fc2.update_layout(
            height=340, xaxis_title="Year", yaxis=dict(range=[0, 11]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_fc2, width="stretch")

    # ── Charts row 2: Climate VaR trend + Carbon cost path ────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Climate VaR Trend (% revenue at risk)**")
        vr = fc["var"]
        fig_fc3 = go.Figure()
        fig_fc3.add_trace(go.Scatter(
            x=vr["year"], y=vr["cvar_95"],
            fill="tozeroy", fillcolor="rgba(231,76,60,0.12)",
            mode="lines+markers", name="CVaR 95%",
            line=dict(color="#e74c3c", width=2.5), marker=dict(size=7),
        ))
        fig_fc3.add_trace(go.Scatter(
            x=vr["year"], y=vr["var_95"],
            mode="lines+markers", name="VaR 95%",
            line=dict(color="#f39c12", width=2), marker=dict(size=6),
        ))
        fig_fc3.add_trace(go.Scatter(
            x=vr["year"], y=vr["median_loss"],
            mode="lines", name="Median loss",
            line=dict(color="#95a5a6", width=1.5, dash="dot"),
        ))
        fig_fc3.update_layout(
            height=340, xaxis_title="Year",
            yaxis_title="% Revenue at Risk",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_fc3, width="stretch")

    with c4:
        st.markdown("**Carbon Cost Exposure Path (USD M)**")
        fig_fc4 = go.Figure()
        fig_fc4.add_trace(go.Bar(
            x=rk["year"], y=rk["carbon_cost_usd_m"],
            marker_color=rk["carbon_cost_usd_m"],
            marker_colorscale="YlOrRd",
            name="Carbon cost",
            text=rk["carbon_price"].apply(lambda p: f"${p}/t"),
            textposition="outside",
        ))
        fig_fc4.update_layout(
            height=340, xaxis_title="Year",
            yaxis_title="Carbon Cost (USD M)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_fc4, width="stretch")

    # ── Multi-company comparison ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Portfolio-Level Forecast Comparison")
    st.caption("Compare transition pathways and 2030 composite risk across all companies")

    all_pathways = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        fc_all = run_all_forecasts(row_dict, forecast_scenario)
        risk_2030 = fc_all["risk"].iloc[-1]
        var_2030 = fc_all["var"].iloc[-1]
        all_pathways.append({
            "Company": row["company"],
            "Sector": row["sector"],
            "Pathway": fc_all["pathway"],
            "Pathway Score": fc_all["pathway_score"],
            "2030 Composite Risk": risk_2030["composite_risk"],
            "2030 Physical Risk": risk_2030["physical_risk"],
            "2030 Transition Risk": risk_2030["transition_risk"],
            "2030 CVaR 95%": var_2030["cvar_95"],
            "2030 Carbon Cost (USD M)": risk_2030["carbon_cost_usd_m"],
            "Tipping Points": len(fc_all["tipping_points"]),
        })

    pathway_df = pd.DataFrame(all_pathways).sort_values("2030 Composite Risk", ascending=False)

    fig_fc5 = px.bar(
        pathway_df,
        x="Company", y="2030 Composite Risk",
        color="Pathway",
        color_discrete_map={
            "🟢 Early Mover": "#27ae60",
            "🔵 On Track": "#3498db",
            "🟡 Laggard": "#f39c12",
            "🔴 Stranded Risk": "#e74c3c",
        },
        hover_data=["Sector", "Pathway Score", "2030 CVaR 95%", "Tipping Points"],
        title=f"2030 Composite Risk by Transition Pathway — {forecast_scenario} Scenario",
        labels={"2030 Composite Risk": "Composite Risk Score (0-10)", "Company": ""},
    )
    fig_fc5.add_hline(y=7.5, line_dash="dash", line_color="red",
                      opacity=0.4, annotation_text="High risk threshold")
    fig_fc5.update_layout(
        height=400, xaxis_tickangle=-25,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_fc5, width="stretch")

    st.dataframe(
        pathway_df.style.apply(
            lambda col: [
                "color: #e74c3c" if "Stranded" in str(v)
                else "color: #f39c12" if "Laggard" in str(v)
                else "color: #27ae60" if "Early" in str(v)
                else "" for v in col
            ] if col.name == "Pathway" else [""] * len(col),
            axis=0
        ),
        width="stretch", hide_index=True
    )

    csv_fc = pathway_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download 2030 Forecast Report",
        csv_fc, "climate_risk_forecast_2030.csv", "text/csv"
    )


# ─── Tab 5: Data Table ────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### Full dataset")
    display_cols = [
        "company", "sector", "risk_category", "carbon_intensity_tCO2e",
        "physical_risk_score", "transition_risk_score",
        "adj_physical_risk", "adj_transition_risk",
        "var_95", "cvar_95", "carbon_cost_exposure_usd_m",
        "tcfd_disclosure_score", "green_revenue_pct", "credit_rating"
    ]
    st.dataframe(
        df[display_cols].rename(columns={
            "carbon_intensity_tCO2e": "Carbon Intensity",
            "physical_risk_score": "Physical Risk",
            "transition_risk_score": "Transition Risk",
            "adj_physical_risk": f"Adj Physical ({scenario})",
            "adj_transition_risk": f"Adj Transition ({scenario})",
            "var_95": "VaR 95%",
            "cvar_95": "CVaR 95%",
            "carbon_cost_exposure_usd_m": "Carbon Cost (USD M)",
            "tcfd_disclosure_score": "TCFD Score",
            "green_revenue_pct": "Green Rev %",
            "credit_rating": "Rating",
        }),
        width="stretch",
        hide_index=True,
    )

    csv_export = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv_export, "tcfd_climate_risk_export.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("SmartHaven Digital · TCFD Climate Risk Model P02 · Built with Python, Streamlit, Plotly · Faith Ndinda")

