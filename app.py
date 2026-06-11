import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from models.tcfd_model import load_and_enrich, scenario_analysis, compute_climate_var
from models.forecast_model import run_all_forecasts, FORECAST_YEARS, SCENARIO_TRAJECTORIES
from models.ai_insights import (
    build_analysis_prompt, compute_sector_stats,
    generate_insights_sync, generate_executive_summary
)

st.set_page_config(
    page_title="TCFD Climate Risk | SmartHaven Digital",
    page_icon="🌍", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --surface2:  #1c2333;
  --border:    #30363d;
  --gold:      #d4a853;
  --gold-dim:  #a07830;
  --red:       #f85149;
  --amber:     #e3b341;
  --green:     #3fb950;
  --blue:      #58a6ff;
  --text:      #e6edf3;
  --muted:     #8b949e;
  --serif:     'DM Serif Display', Georgia, serif;
  --sans:      'DM Sans', sans-serif;
  --mono:      'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--sans) !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
  border: 1px solid var(--border);
  border-bottom: 3px solid var(--gold);
  border-radius: 12px;
  padding: 2.5rem 3rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 240px; height: 240px;
  background: radial-gradient(circle, rgba(212,168,83,0.12) 0%, transparent 70%);
  border-radius: 50%;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -40px; left: 30%;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(63,185,80,0.06) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-eyebrow {
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.6rem;
}
.hero-title {
  font-family: var(--serif);
  font-size: 2.4rem;
  font-weight: 400;
  color: var(--text);
  line-height: 1.2;
  margin-bottom: 0.8rem;
}
.hero-title em { color: var(--gold); font-style: italic; }
.hero-sub {
  font-size: 0.95rem;
  color: var(--muted);
  line-height: 1.6;
  max-width: 680px;
}
.hero-badges {
  display: flex; gap: 0.5rem; flex-wrap: wrap;
  margin-top: 1.2rem;
}
.badge {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  padding: 3px 10px;
  border-radius: 20px;
  border: 1px solid;
}
.badge-gold  { color: var(--gold);  border-color: var(--gold-dim);  background: rgba(212,168,83,0.08); }
.badge-green { color: var(--green); border-color: #238636; background: rgba(63,185,80,0.08); }
.badge-blue  { color: var(--blue);  border-color: #1f6feb; background: rgba(88,166,255,0.08); }
.badge-red   { color: var(--red);   border-color: #b91c1c; background: rgba(248,81,73,0.08); }

/* ── KPI cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--gold-dim); }
.kpi-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
}
.kpi-red::after   { background: var(--red); }
.kpi-amber::after { background: var(--amber); }
.kpi-green::after { background: var(--green); }
.kpi-gold::after  { background: var(--gold); }
.kpi-blue::after  { background: var(--blue); }
.kpi-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.kpi-value {
  font-family: var(--serif);
  font-size: 2rem;
  font-weight: 400;
  color: var(--text);
  line-height: 1;
}
.kpi-delta {
  font-family: var(--mono);
  font-size: 0.7rem;
  margin-top: 0.3rem;
}
.delta-bad  { color: var(--red); }
.delta-warn { color: var(--amber); }
.delta-good { color: var(--green); }

/* ── Summary banner ── */
.summary-banner {
  background: linear-gradient(90deg, rgba(212,168,83,0.08) 0%, rgba(212,168,83,0.02) 100%);
  border: 1px solid rgba(212,168,83,0.25);
  border-left: 4px solid var(--gold);
  border-radius: 8px;
  padding: 1.1rem 1.4rem;
  margin-bottom: 1.5rem;
  font-size: 0.92rem;
  line-height: 1.7;
  color: var(--text);
}
.summary-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.4rem;
}

/* ── Section headers ── */
.section-header {
  font-family: var(--serif);
  font-size: 1.5rem;
  color: var(--text);
  margin: 1.8rem 0 0.3rem 0;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}
.section-sub {
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 1.2rem;
  line-height: 1.5;
}

/* ── Insight cards ── */
.icard {
  border-radius: 8px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  font-size: 0.88rem;
  line-height: 1.6;
}
.icard b { font-weight: 600; }
.icard-green { background: rgba(63,185,80,0.07);  border-left: 3px solid var(--green); }
.icard-red   { background: rgba(248,81,73,0.07);  border-left: 3px solid var(--red); }
.icard-amber { background: rgba(227,179,65,0.07); border-left: 3px solid var(--amber); }
.icard-blue  { background: rgba(88,166,255,0.07); border-left: 3px solid var(--blue); }
.icard-gold  { background: rgba(212,168,83,0.07); border-left: 3px solid var(--gold); }

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 8px;
  padding: 4px;
  gap: 2px;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  border-radius: 6px !important;
  padding: 6px 14px !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(212,168,83,0.15) !important;
  color: var(--gold) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
.sidebar-title {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 1rem;
}

/* ── Metric overrides ── */
[data-testid="stMetricValue"] { display: none; }

/* ── Table ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; }

/* ── Pathway badge ── */
.pathway-strip {
  display: flex; align-items: center; gap: 1rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.4rem;
  margin-bottom: 1rem;
}
.pathway-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
}
.pathway-value {
  font-family: var(--serif);
  font-size: 1.4rem;
}
.pathway-score {
  font-family: var(--mono);
  font-size: 0.8rem;
  color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_enrich("data/climate_risk_data.csv")

df_base = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Controls</div>', unsafe_allow_html=True)
    scenario = st.selectbox(
        "TCFD Scenario", ["1.5C", "2C", "3C"], index=1,
        help="1.5C = aggressive transition | 2C = moderate | 3C = physical dominant"
    )
    sectors = st.multiselect(
        "Sectors",
        options=sorted(df_base["sector"].unique()),
        default=sorted(df_base["sector"].unique()),
    )
    risk_filter = st.select_slider(
        "Minimum risk level",
        options=["🟢 Low", "🟡 Medium", "🔴 High"],
        value="🟢 Low",
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-title">Data Sources</div>', unsafe_allow_html=True)
    for src in ["📊 NSE fundamentals", "🛰️ Planetek Italia", "🌍 GEE / Copernicus", "🤖 Claude AI"]:
        st.markdown(f'<div style="font-size:0.78rem;color:var(--muted);padding:3px 0">{src}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:var(--muted);line-height:1.8">SmartHaven Digital<br>Nairobi, Kenya<br>Faith Ndinda</div>', unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
df = scenario_analysis(df_base, scenario)
df = df[df["sector"].isin(sectors)]
risk_order = {"🟢 Low": 0, "🟡 Medium": 1, "🔴 High": 2}
df = df[df["risk_category"].map(risk_order) >= risk_order[risk_filter]]

# ── Hero ──────────────────────────────────────────────────────────────────────
high_n = (df["risk_category"] == "🔴 High").sum()
scenario_desc = {
    "1.5C": "aggressive energy transition · carbon at $85/tonne by 2030",
    "2C":   "moderate transition · balanced physical & policy risk",
    "3C":   "physical risk dominant · severe weather intensification",
}
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">SmartHaven Digital &nbsp;·&nbsp; P02 Financial Engineering Portfolio</div>
  <div class="hero-title">Kenya Climate <em>Risk</em> Intelligence</div>
  <div class="hero-sub">
    TCFD-aligned physical &amp; transition risk analysis for {len(df)} NSE-listed companies.
    Under the <strong style="color:var(--gold)">{scenario}</strong> scenario
    ({scenario_desc.get(scenario, '')}),
    <strong style="color:var(--red)">{high_n} companies</strong> are in the high-risk category —
    with combined carbon cost exposure of
    <strong style="color:var(--amber)">USD {df['carbon_cost_exposure_usd_m'].sum():,.0f}M</strong>.
  </div>
  <div class="hero-badges">
    <span class="badge badge-gold">TCFD Aligned</span>
    <span class="badge badge-green">Monte Carlo VaR</span>
    <span class="badge badge-blue">2025–2030 Forecast</span>
    <span class="badge badge-blue">AI Insights</span>
    <span class="badge badge-red">Geospatial Risk Map</span>
    <span class="badge badge-gold">NSE Kenya</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Executive summary ─────────────────────────────────────────────────────────
from models.ai_insights import generate_executive_summary
summary = generate_executive_summary(df, scenario)
st.markdown(f"""
<div class="summary-banner">
  <div class="summary-label">📋 Executive Summary — {scenario} Scenario</div>
  {summary}
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
worst = df.nlargest(1, "var_95").iloc[0]
best_tcfd = df.nlargest(1, "tcfd_disclosure_score").iloc[0]
avg_var = df["var_95"].mean()
total_carbon = df["carbon_cost_exposure_usd_m"].sum()
avg_tcfd = df["tcfd_disclosure_score"].mean()

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-blue">
    <div class="kpi-label">Companies Analysed</div>
    <div class="kpi-value">{len(df)}</div>
    <div class="kpi-delta delta-warn">{len(sectors)} sectors</div>
  </div>
  <div class="kpi-card kpi-red">
    <div class="kpi-label">High Risk</div>
    <div class="kpi-value">{high_n}</div>
    <div class="kpi-delta delta-bad">{high_n/max(len(df),1)*100:.0f}% of portfolio</div>
  </div>
  <div class="kpi-card kpi-amber">
    <div class="kpi-label">Avg Climate VaR 95%</div>
    <div class="kpi-value">{avg_var:.1f}%</div>
    <div class="kpi-delta delta-bad">revenue at risk</div>
  </div>
  <div class="kpi-card kpi-gold">
    <div class="kpi-label">Carbon Cost Exposure</div>
    <div class="kpi-value">USD {total_carbon:,.0f}M</div>
    <div class="kpi-delta delta-warn">{scenario} scenario</div>
  </div>
  <div class="kpi-card kpi-green">
    <div class="kpi-label">Avg TCFD Score</div>
    <div class="kpi-value">{avg_tcfd:.0f}<span style="font-size:1rem;color:var(--muted)">/100</span></div>
    <div class="kpi-delta delta-good">disclosure quality</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor="#161b22",
    paper_bgcolor="#161b22",
    font=dict(family="DM Sans", color="#8b949e", size=11),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
    margin=dict(t=40, b=20, l=10, r=10),
)

COLOR_RISK = {"🔴 High": "#f85149", "🟡 Medium": "#e3b341", "🟢 Low": "#3fb950"}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Risk Overview", "🎯 Scenarios", "📉 Climate VaR",
    "🔮 Forecast 2030", "🗺️ Geo Map",
    "🤖 AI Insights", "📋 Data"
])


# ─── TAB 1: Risk Overview ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Risk Landscape</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Where does each company sit on the physical vs transition risk spectrum? Bubble size = revenue. Companies in the top-right quadrant face compounding risk from both dimensions.</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        fig1 = px.scatter(
            df, x="adj_physical_risk", y="adj_transition_risk",
            size="revenue_usd_millions", color="risk_category",
            hover_name="company",
            hover_data={"sector": True, "carbon_intensity_tCO2e": True, "var_95": True, "credit_rating": True},
            color_discrete_map=COLOR_RISK,
            labels={"adj_physical_risk": f"Physical Risk Score ({scenario})", "adj_transition_risk": f"Transition Risk Score ({scenario})"},
        )
        fig1.add_hline(y=7.5, line_dash="dash", line_color="#f85149", opacity=0.4, annotation_text="High transition threshold", annotation_font_color="#f85149")
        fig1.add_vline(x=7.5, line_dash="dash", line_color="#e3b341", opacity=0.4, annotation_text="High physical threshold", annotation_font_color="#e3b341")
        fig1.update_layout(**PLOT_LAYOUT, height=400, title=dict(text="Risk Matrix", font=dict(family="DM Serif Display", size=15, color="#e6edf3")))
        st.plotly_chart(fig1, use_container_width=True)

    with cb:
        fig2 = px.bar(
            df.sort_values("carbon_intensity_tCO2e", ascending=True),
            x="carbon_intensity_tCO2e", y="company", orientation="h",
            color="carbon_intensity_tCO2e",
            color_continuous_scale=[[0,"#3fb950"],[0.5,"#e3b341"],[1,"#f85149"]],
            labels={"carbon_intensity_tCO2e": "tCO₂e / USD M revenue", "company": ""},
        )
        fig2.update_layout(**PLOT_LAYOUT, height=400, coloraxis_showscale=False,
                           title=dict(text="Carbon Intensity Ranking", font=dict(family="DM Serif Display", size=15, color="#e6edf3")))
        st.plotly_chart(fig2, use_container_width=True)

    # 3 callout cards
    i1, i2, i3 = st.columns(3)
    worst_c = df.nlargest(1, "carbon_intensity_tCO2e").iloc[0]
    highest_var_c = df.nlargest(1, "var_95").iloc[0]
    with i1:
        st.markdown(f'<div class="icard icard-red">🏭 <b>Highest Carbon Intensity</b><br><b>{worst_c["company"]}</b> emits {worst_c["carbon_intensity_tCO2e"]:.0f} tCO₂e per USD M revenue. {worst_c["sector"]} faces the steepest carbon pricing exposure under any transition scenario.</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div class="icard icard-amber">📉 <b>Most Revenue at Risk</b><br><b>{highest_var_c["company"]}</b> has {highest_var_c["var_95"]:.1f}% of revenue at risk in a 95th-percentile climate loss scenario — highest in the portfolio.</div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div class="icard icard-green">✅ <b>Best Disclosure</b><br><b>{best_tcfd["company"]}</b> scores {best_tcfd["tcfd_disclosure_score"]}/100 on TCFD disclosure — best positioned to access green bonds and DFI concessional finance.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="font-size:1.1rem">Does transparency reduce risk?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A negative correlation here would confirm that companies investing in climate disclosure also manage risk better — a key argument for mandatory TCFD reporting in Kenya.</div>', unsafe_allow_html=True)
    fig3 = px.scatter(
        df, x="tcfd_disclosure_score", y="var_95",
        color="sector", size="carbon_intensity_tCO2e",
        hover_name="company", trendline="ols",
        labels={"tcfd_disclosure_score": "TCFD Disclosure Score (0–100)", "var_95": "Climate VaR 95% (% revenue)"},
        color_discrete_sequence=["#d4a853","#3fb950","#58a6ff","#f85149","#e3b341","#bc8cff","#39d353","#ffa657","#ff7b72"],
    )
    fig3.update_layout(**PLOT_LAYOUT, height=340, title=dict(text="Disclosure Score vs Climate VaR", font=dict(family="DM Serif Display", size=15, color="#e6edf3")))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Treemap ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="font-size:1.1rem">Portfolio Risk Composition</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Proportional view — box size = carbon cost exposure, color = Climate VaR. Instantly shows which companies and sectors dominate the risk landscape.</div>', unsafe_allow_html=True)
    df_tree = df.copy()
    df_tree["carbon_cost_exposure_usd_m"] = df_tree["carbon_cost_exposure_usd_m"].clip(lower=0.1)
    fig_tree = px.treemap(
        df_tree,
        path=[px.Constant("NSE Portfolio"), "sector", "company"],
        values="carbon_cost_exposure_usd_m",
        color="var_95",
        color_continuous_scale=[[0,"#3fb950"],[0.4,"#e3b341"],[1,"#f85149"]],
        hover_data={"carbon_intensity_tCO2e": True, "tcfd_disclosure_score": True, "credit_rating": True},
    )
    fig_tree.update_traces(
        texttemplate="<b>%{label}</b><br>VaR: %{color:.1f}%",
        hovertemplate="<b>%{label}</b><br>Carbon Cost: USD %{value:.1f}M<br>Climate VaR: %{color:.1f}%<extra></extra>",
        textfont=dict(family="DM Sans", size=12),
        marker=dict(line=dict(width=2, color="#0d1117")),
    )
    fig_tree.update_layout(
        **PLOT_LAYOUT, height=480,
        coloraxis_colorbar=dict(title="VaR 95%", tickfont=dict(color="#8b949e"), titlefont=dict(color="#8b949e")),
        title=dict(text="Carbon Cost Exposure Treemap — size = exposure, color = Climate VaR", font=dict(family="DM Serif Display", size=15, color="#e6edf3")),
    )
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown('<div class="icard icard-gold">💡 <b>How to read this</b> — larger boxes = more carbon cost exposure. Redder boxes = higher revenue loss under a climate shock. The worst companies appear both large <em>and</em> red.</div>', unsafe_allow_html=True)


# ─── TAB 2: Scenarios ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Scenario Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How does carbon cost exposure change depending on global climate policy ambition? The gap between 1.5°C and 3°C represents the financial spread of political decisions made today.</div>', unsafe_allow_html=True)

    sc_dfs = []
    for sc in ["1.5C", "2C", "3C"]:
        sc_df = scenario_analysis(df_base[df_base["sector"].isin(sectors)], sc).copy()
        sc_df["scenario"] = sc
        sc_dfs.append(sc_df)
    combined = pd.concat(sc_dfs)

    fig4 = px.bar(
        combined, x="company", y="carbon_cost_exposure_usd_m",
        color="scenario", barmode="group",
        color_discrete_map={"1.5C": "#3fb950", "2C": "#e3b341", "3C": "#f85149"},
        labels={"carbon_cost_exposure_usd_m": "Carbon Cost Exposure (USD M)", "company": ""},
    )
    fig4.update_layout(**PLOT_LAYOUT, height=400, xaxis_tickangle=-25,
                       title=dict(text="Carbon Cost Exposure Across All Scenarios", font=dict(family="DM Serif Display", size=15, color="#e6edf3")),
                       legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3")))
    st.plotly_chart(fig4, use_container_width=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown('<div class="section-sub" style="margin-top:1rem"><b style="color:#e6edf3">What each scenario assumes</b></div>', unsafe_allow_html=True)
        sc_table = pd.DataFrame({
            "Scenario": ["🟢 1.5C", "🟡 2C", "🔴 3C"],
            "Carbon Price 2030": ["$85/t", "$50/t", "$25/t"],
            "Dominant Risk": ["Transition", "Balanced", "Physical"],
            "Kenya Implication": [
                "Heavy carbon tax on Oil & Gas, Utilities",
                "Both risks grow at moderate pace",
                "Floods & drought devastate ASAL agriculture"
            ]
        })
        st.dataframe(sc_table, use_container_width=True, hide_index=True)

    with sc2:
        stranded = df[df["stranded_asset_risk"] == "high"]
        if not stranded.empty:
            st.markdown('<div class="section-sub" style="margin-top:1rem"><b style="color:#e6edf3">Stranded asset exposure</b></div>', unsafe_allow_html=True)
            st.dataframe(
                stranded[["company","sector","carbon_intensity_tCO2e","carbon_cost_exposure_usd_m"]],
                use_container_width=True, hide_index=True
            )
            st.markdown('<div class="icard icard-red">⚠️ Companies with <b>high stranded asset risk</b> hold fossil fuel infrastructure that may become uneconomic under carbon pricing — a critical flag for DFI lending decisions and green bond structuring.</div>', unsafe_allow_html=True)


# ─── TAB 3: Climate VaR ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Climate Value at Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How much revenue could each company lose in a severe climate event? VaR 95% = the loss exceeded only 5% of the time across 10,000 Monte Carlo simulations. CVaR = expected loss in that worst 5% tail.</div>', unsafe_allow_html=True)

    df_s = df.sort_values("cvar_95", ascending=False)
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(name="CVaR 95% (tail)", x=df_s["company"], y=df_s["cvar_95"], marker_color="#f85149", opacity=0.9))
    fig5.add_trace(go.Bar(name="VaR 95%", x=df_s["company"], y=df_s["var_95"], marker_color="#e3b341", opacity=0.9))
    fig5.update_layout(**PLOT_LAYOUT, barmode="overlay", height=380, xaxis_tickangle=-20,
                       yaxis_title="% Revenue at Risk",
                       title=dict(text="Climate VaR — Ranked Worst to Best", font=dict(family="DM Serif Display", size=15, color="#e6edf3")),
                       legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3")))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header" style="font-size:1.1rem">Company Deep-Dive</div>', unsafe_allow_html=True)
    sel_co = st.selectbox("Select company", df["company"].unique(), key="var_co")
    row = df[df["company"] == sel_co].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    for col, label, val, cls in [
        (d1, "VaR 95%", f"{row['var_95']}%", "red"),
        (d2, "CVaR 95%", f"{row['cvar_95']}%", "red"),
        (d3, "Carbon Intensity", f"{row['carbon_intensity_tCO2e']:.0f} tCO₂e", "amber"),
        (d4, "Credit Rating", row["credit_rating"], "blue"),
    ]:
        col.markdown(f'<div class="kpi-card kpi-{cls}"><div class="kpi-label">{label}</div><div class="kpi-value" style="font-size:1.4rem">{val}</div></div>', unsafe_allow_html=True)

    np.random.seed(42)
    n = 10000
    losses = np.clip(
        np.random.normal(row["physical_risk_score"]*0.012, row["physical_risk_score"]*0.008, n) +
        np.random.normal(row["transition_risk_score"]*0.015, row["transition_risk_score"]*0.010, n) +
        (row["carbon_intensity_tCO2e"]/500)*np.random.uniform(0.005, 0.025, n), 0, None
    ) * 100

    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(x=losses, nbinsx=80, marker_color="#58a6ff", opacity=0.7, name="Simulated losses"))
    fig6.add_vline(x=row["var_95"], line_color="#e3b341", line_dash="dash", annotation_text=f"VaR {row['var_95']}%", annotation_font_color="#e3b341")
    fig6.add_vline(x=row["cvar_95"], line_color="#f85149", line_dash="dash", annotation_text=f"CVaR {row['cvar_95']}%", annotation_font_color="#f85149")
    fig6.update_layout(**PLOT_LAYOUT, height=300,
                       title=dict(text=f"10,000-Scenario Loss Distribution — {sel_co}", font=dict(family="DM Serif Display", size=14, color="#e6edf3")),
                       xaxis_title="% Revenue Loss", yaxis_title="Frequency", showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

    if row["var_95"] > 8:
        st.markdown(f'<div class="icard icard-red">🔴 <b>Critical</b> — {sel_co} faces severe climate-driven revenue loss. Immediate board-level attention and transition strategy required.</div>', unsafe_allow_html=True)
    elif row["var_95"] > 5:
        st.markdown(f'<div class="icard icard-amber">🟡 <b>Elevated</b> — {sel_co} has meaningful climate exposure. Active risk management and green capex investment recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="icard icard-green">🟢 <b>Manageable</b> — {sel_co} climate VaR is within normal tolerance. Focus on disclosure improvement to access green finance.</div>', unsafe_allow_html=True)


# ─── TAB 4: Forecast ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">2025–2030 Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Forward projections under TCFD scenario trajectories. Carbon intensity decline adjusted per company based on TCFD score, green revenue %, and transition capex. Tipping points show when critical thresholds are breached.</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([1, 2])
    with fc1:
        fc_co = st.selectbox("Company", sorted(df["company"].unique()), key="fc_co2")
        fc_sc = st.selectbox("Scenario", ["1.5C", "2C", "3C"], index=1, key="fc_sc2")
    co_row = df[df["company"] == fc_co].iloc[0].to_dict()
    with st.spinner(""):
        fc = run_all_forecasts(co_row, fc_sc)

    pw_colors = {"🟢 Early Mover": "#3fb950", "🔵 On Track": "#58a6ff", "🟡 Laggard": "#e3b341", "🔴 Stranded Risk": "#f85149"}
    pw_c = pw_colors.get(fc["pathway"], "#8b949e")
    with fc2:
        st.markdown(f'<div class="pathway-strip"><div><div class="pathway-label">Transition Pathway</div><div class="pathway-value" style="color:{pw_c}">{fc["pathway"]}</div></div><div><div class="pathway-label">Pathway Score</div><div class="pathway-value">{fc["pathway_score"]}<span style="font-size:0.9rem;color:var(--muted)">/100</span></div></div><div style="font-size:0.78rem;color:var(--muted);max-width:300px">Scored on TCFD disclosure quality, green revenue share, and transition capex intensity.</div></div>', unsafe_allow_html=True)

    if fc["tipping_points"]:
        st.markdown('<div class="section-sub" style="color:#f85149"><b>⚠️ Tipping Points Detected — act before these years</b></div>', unsafe_allow_html=True)
        for tp in fc["tipping_points"]:
            st.markdown(f'<div class="icard icard-red"><b>{tp["type"]}</b> — breaches threshold in <b>{tp["year"]}</b> &nbsp;·&nbsp; score reaches {tp["value"]:.2f} (limit: {tp["threshold"]})<br><small>{tp["implication"]}</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="icard icard-green">✅ No critical tipping points for <b>{fc_co}</b> under {fc_sc} through 2030.</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    ci, rk = fc["carbon"], fc["risk"]
    with p1:
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=ci["year"], y=ci["upper_bound"], mode="lines", line=dict(width=0), showlegend=False))
        f1.add_trace(go.Scatter(x=ci["year"], y=ci["lower_bound"], fill="tonexty", fillcolor="rgba(88,166,255,0.1)", mode="lines", line=dict(width=0), name="Uncertainty"))
        f1.add_trace(go.Scatter(x=ci["year"], y=ci["carbon_intensity"], mode="lines+markers", line=dict(color="#58a6ff", width=2.5), marker=dict(size=7), name="Projection"))
        f1.add_hline(y=co_row.get("carbon_intensity_tCO2e",100), line_dash="dash", line_color="#8b949e", opacity=0.5, annotation_text="2023 baseline", annotation_font_color="#8b949e")
        f1.update_layout(**PLOT_LAYOUT, height=300, title=dict(text="Carbon Intensity to 2030", font=dict(family="DM Serif Display", size=14, color="#e6edf3")), yaxis_title="tCO₂e / USD M rev")
        st.plotly_chart(f1, use_container_width=True)
    with p2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=rk["year"], y=rk["physical_risk"], mode="lines+markers", name="Physical", line=dict(color="#e3b341", width=2)))
        f2.add_trace(go.Scatter(x=rk["year"], y=rk["transition_risk"], mode="lines+markers", name="Transition", line=dict(color="#bc8cff", width=2)))
        f2.add_trace(go.Scatter(x=rk["year"], y=rk["composite_risk"], mode="lines+markers", name="Composite", line=dict(color="#f85149", width=3, dash="dot")))
        f2.add_hline(y=7.5, line_dash="dash", line_color="#f85149", opacity=0.3, annotation_text="High threshold")
        f2.update_layout(**PLOT_LAYOUT, height=300,
                         title=dict(text="Risk Score Trajectory", font=dict(family="DM Serif Display", size=14, color="#e6edf3")),
                         legend=dict(orientation="h", y=-0.2, font=dict(color="#e6edf3")))
        f2.update_yaxes(range=[0, 11], gridcolor="#21262d")
        st.plotly_chart(f2, use_container_width=True)

    st.markdown('<div class="section-header" style="font-size:1.1rem;margin-top:2rem">Portfolio — Who is best positioned for 2030?</div>', unsafe_allow_html=True)
    pw_rows = []
    for _, r in df.iterrows():
        fc_r = run_all_forecasts(r.to_dict(), fc_sc)
        r30 = fc_r["risk"].iloc[-1]
        pw_rows.append({"Company": r["company"], "Sector": r["sector"],
                        "Pathway": fc_r["pathway"], "Score": fc_r["pathway_score"],
                        "2030 Risk": r30["composite_risk"],
                        "2030 CVaR": fc_r["var"].iloc[-1]["cvar_95"],
                        "Tipping Points": len(fc_r["tipping_points"])})
    pw_df = pd.DataFrame(pw_rows).sort_values("2030 Risk", ascending=False)
    f3 = px.bar(pw_df, x="Company", y="2030 Risk", color="Pathway",
                color_discrete_map=pw_colors,
                hover_data=["Sector","Score","2030 CVaR","Tipping Points"],
                labels={"2030 Risk": "Composite Risk (0–10)", "Company": ""})
    f3.add_hline(y=7.5, line_dash="dash", line_color="#f85149", opacity=0.3)
    f3.update_layout(**PLOT_LAYOUT, height=360, xaxis_tickangle=-20,
                     title=dict(text=f"2030 Portfolio Risk — {fc_sc} Scenario", font=dict(family="DM Serif Display", size=14, color="#e6edf3")),
                     legend=dict(orientation="h", y=1.1, font=dict(color="#e6edf3")))
    st.plotly_chart(f3, use_container_width=True)

    # ── Animated Scatter ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="font-size:1.1rem">Risk Evolution — Watch Companies Move 2024–2030</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Each company traces a path across the risk landscape year by year. Press play to watch the portfolio evolve under the selected scenario — laggards move toward the danger zone, early movers hold steady.</div>', unsafe_allow_html=True)
    anim_rows = []
    for _, r in df.iterrows():
        fc_anim = run_all_forecasts(r.to_dict(), fc_sc)
        for _, yr_row in fc_anim["risk"].iterrows():
            var_row = fc_anim["var"][fc_anim["var"]["year"] == yr_row["year"]].iloc[0]
            anim_rows.append({
                "company": r["company"], "sector": r["sector"],
                "year": str(int(yr_row["year"])),
                "physical_risk": yr_row["physical_risk"],
                "transition_risk": yr_row["transition_risk"],
                "composite_risk": yr_row["composite_risk"],
                "var_95": var_row["var_95"],
                "revenue": r["revenue_usd_millions"],
                "pathway": fc_anim["pathway"],
            })
    anim_df = pd.DataFrame(anim_rows)
    pw_colors_anim = {"Early Mover": "#3fb950", "On Track": "#58a6ff", "Laggard": "#e3b341", "Stranded Risk": "#f85149"}
    anim_df["pathway_clean"] = anim_df["pathway"].str.replace(r"^[^\w]+", "", regex=True)
    fig_anim = px.scatter(
        anim_df,
        x="physical_risk", y="transition_risk",
        size="var_95", color="pathway_clean",
        animation_frame="year",
        animation_group="company",
        hover_name="company",
        hover_data={"sector": True, "composite_risk": True, "var_95": True},
        color_discrete_map={"Early Mover": "#3fb950", "On Track": "#58a6ff", "Laggard": "#e3b341", "Stranded Risk": "#f85149"},
        size_max=40,
        range_x=[0, 12], range_y=[0, 12],
        labels={"physical_risk": "Physical Risk Score", "transition_risk": "Transition Risk Score", "pathway_clean": "Pathway"},
    )
    fig_anim.add_hline(y=7.5, line_dash="dash", line_color="#f85149", opacity=0.3)
    fig_anim.add_vline(x=7.5, line_dash="dash", line_color="#e3b341", opacity=0.3)
    fig_anim.update_layout(
        **PLOT_LAYOUT, height=500,
        title=dict(text=f"Risk Trajectory Animation 2024–2030 — {fc_sc} Scenario", font=dict(family="DM Serif Display", size=15, color="#e6edf3")),
        legend=dict(orientation="h", y=1.08, font=dict(color="#e6edf3")),
    )
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400
    st.plotly_chart(fig_anim, use_container_width=True)


# ─── TAB 5: Geo Map ──────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Geospatial Risk Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Climate risk is geographic. Each marker shows a company\'s primary operational county — bubble size = revenue. This is where Planetek Italia\'s satellite data feeds directly into the model, replacing estimated risk scores with live earth observation.</div>', unsafe_allow_html=True)

    LOCS = {
        "Safaricom": (-1.2921, 36.8219), "KenGen": (-0.3031, 36.08),
        "Equity Bank": (-1.2921, 36.82), "EABL": (-1.32, 36.81),
        "Nation Media": (-1.28, 36.83), "Bamburi Cement": (-4.02, 39.65),
        "KCB Group": (-1.275, 36.815), "KPLC": (-1.30, 36.80),
        "TotalEnergies Kenya": (-4.06, 39.67), "Britam": (-1.29, 36.825),
        "Co-op Bank": (-1.285, 36.82), "Mumias Sugar": (0.333, 34.483),
        "ARM Cement": (-1.31, 36.805),
    }
    map_rows = []
    for _, row in df.iterrows():
        lat, lon = LOCS.get(row["company"], (-1.29, 36.82))
        map_rows.append({
            "company": row["company"], "sector": row["sector"],
            "risk": row["risk_category"], "var_95": row["var_95"],
            "carbon": row["carbon_intensity_tCO2e"],
            "tcfd": row["tcfd_disclosure_score"],
            "rating": row["credit_rating"],
            "lat": lat + np.random.uniform(-0.05, 0.05),
            "lon": lon + np.random.uniform(-0.05, 0.05),
            "size": row["revenue_usd_millions"],
        })
    map_df = pd.DataFrame(map_rows)
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", color="risk", size="size",
        hover_name="company",
        hover_data={"sector": True, "var_95": True, "carbon": True, "tcfd": True, "rating": True, "lat": False, "lon": False, "size": False},
        color_discrete_map=COLOR_RISK,
        mapbox_style="carto-darkmatter",
        zoom=5.5, center={"lat": -0.5, "lon": 37.5}, size_max=45,
    )
    fig_map.update_layout(paper_bgcolor="#0d1117", height=500, margin={"r":0,"t":0,"l":0,"b":0},
                          legend=dict(font=dict(color="#e6edf3"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_map, use_container_width=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="icard icard-amber">🌵 <b>Drought Index (Planetek)</b><br>NDVI-derived drought stress per county replaces estimated physical risk scores — making Turkana, Garissa, and Marsabit risks precisely quantified rather than approximated.</div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="icard icard-blue">🌊 <b>Flood Mapping (Sentinel-1 SAR)</b><br>Real-time flood inundation updates KPLC, Bamburi, and coastal company risk scores after each major rainfall event — enabling dynamic risk repricing.</div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="icard icard-green">🌳 <b>Land Change (Hansen GFC)</b><br>Annual forest cover loss feeds carbon credit valuation for companies with land assets in Mau, Aberdares, and coastal forests — directly monetisable via REDD+.</div>', unsafe_allow_html=True)

    st.markdown('<div class="icard icard-gold" style="margin-top:0.5rem">🤖 <b>AI + Geospatial = The SmartHaven × Planetek Value Chain</b><br>Connecting Planetek\'s satellite APIs enables: (1) automatic risk score updates after each satellite pass, (2) county-level carbon credit valuation using verified land measurements, (3) flood early warnings triggering insurance payout thresholds for agricultural lenders. This bridges earth observation data directly to bankable climate finance instruments.</div>', unsafe_allow_html=True)


# ─── TAB 6: AI Insights ──────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">AI-Powered Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Claude reads your filtered dashboard data and generates a structured intelligence report — key risk flags, sector outlook, what satellite data would change, and specific investor actions. Updates with every scenario/filter change.</div>', unsafe_allow_html=True)

    st.markdown('<div class="icard icard-blue">Change the scenario or sector filters in the sidebar, then click Generate to get a fresh analysis focused on your selected slice of the portfolio.</div>', unsafe_allow_html=True)
    run_ai = st.button("🤖 Generate AI Analysis", type="primary", use_container_width=False)

    if run_ai:
        with st.spinner("Claude is analysing..."):
            from models.ai_insights import build_analysis_prompt, compute_sector_stats, generate_insights_sync
            prompt = build_analysis_prompt(df, scenario, df.nlargest(8, "var_95"), compute_sector_stats(df))
            result = generate_insights_sync(prompt)

        if "error" in result:
            st.error(f"API error: {result['error']}")
        else:
            st.markdown(f'<div class="summary-banner"><div class="summary-label">📋 Executive Summary</div>{result.get("executive_summary","")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="font-size:1.1rem">🚨 Top Risk Flags</div>', unsafe_allow_html=True)
            urgency_cls = {"Immediate": "red", "2026": "red", "2027": "amber", "2028": "amber", "2030": "blue"}
            for r in result.get("top_3_risks", []):
                cls = urgency_cls.get(r.get("urgency",""), "amber")
                st.markdown(f'<div class="icard icard-{cls}"><b>{r.get("risk","")}</b> — {r.get("company_or_sector","")} <span style="float:right;font-family:var(--mono);font-size:0.7rem;opacity:0.7">⏰ {r.get("urgency","")}</span><br>{r.get("detail","")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="font-size:1.1rem">🏭 Sector Outlook</div>', unsafe_allow_html=True)
            oc = st.columns(min(len(result.get("sector_outlook",[])), 4))
            for i, s in enumerate(result.get("sector_outlook",[])):
                ocls = {"Positive":"green","Cautious":"amber","Negative":"red"}.get(s.get("outlook",""),"amber")
                oc[i%4].markdown(f'<div class="icard icard-{ocls}"><b>{s.get("sector","")}</b><br>{s.get("outlook","")} · <small>{s.get("reasoning","")}</small></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="font-size:1.1rem">🛰️ What Satellite Data Would Change</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="icard icard-blue">{result.get("satellite_data_gaps","")}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="font-size:1.1rem">💼 Investor Actions</div>', unsafe_allow_html=True)
            for i, a in enumerate(result.get("investor_actions",[]), 1):
                st.markdown(f'<div class="icard icard-green"><b>{i}.</b> {a}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="font-size:1.1rem">🇰🇪 Kenya-Specific Insight</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="icard icard-gold">{result.get("kenya_specific_insight","")}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="icard icard-blue" style="text-align:center;padding:2rem">👆 Click <b>Generate AI Analysis</b> to run the intelligence report</div>', unsafe_allow_html=True)


# ─── TAB 7: Data ─────────────────────────────────────────────────────────────
with tab7:
    st.markdown('<div class="section-header">Full Dataset</div>', unsafe_allow_html=True)
    display_cols = ["company","sector","risk_category","carbon_intensity_tCO2e",
                    "physical_risk_score","transition_risk_score","adj_physical_risk","adj_transition_risk",
                    "var_95","cvar_95","carbon_cost_exposure_usd_m","tcfd_disclosure_score","green_revenue_pct","credit_rating"]
    st.dataframe(
        df[display_cols].rename(columns={
            "carbon_intensity_tCO2e":"Carbon Intensity","physical_risk_score":"Physical Risk",
            "transition_risk_score":"Transition Risk",
            "adj_physical_risk":f"Adj Physical ({scenario})","adj_transition_risk":f"Adj Transition ({scenario})",
            "var_95":"VaR 95%","cvar_95":"CVaR 95%","carbon_cost_exposure_usd_m":"Carbon Cost (USD M)",
            "tcfd_disclosure_score":"TCFD Score","green_revenue_pct":"Green Rev %","credit_rating":"Rating",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="font-size:1.1rem">Full Risk Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Every company across every risk dimension in one view. Darker red = higher risk or worse score. Darker green = better. Ideal for quick portfolio scanning.</div>', unsafe_allow_html=True)
    heat_cols = ["physical_risk_score", "transition_risk_score", "var_95", "cvar_95",
                 "carbon_intensity_tCO2e", "tcfd_disclosure_score", "green_revenue_pct"]
    heat_labels = ["Physical Risk", "Transition Risk", "VaR 95%", "CVaR 95%",
                   "Carbon Intensity", "TCFD Score", "Green Revenue %"]
    heat_df = df.set_index("company")[heat_cols].copy()
    # Normalise each column 0-1 for color (invert TCFD and green_revenue so red = bad)
    heat_norm = heat_df.copy()
    for col in heat_cols:
        col_min, col_max = heat_df[col].min(), heat_df[col].max()
        if col_max > col_min:
            heat_norm[col] = (heat_df[col] - col_min) / (col_max - col_min)
        else:
            heat_norm[col] = 0.5
    # Invert good metrics so red always = bad
    for col in ["tcfd_disclosure_score", "green_revenue_pct"]:
        heat_norm[col] = 1 - heat_norm[col]
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_norm.values,
        x=heat_labels,
        y=heat_norm.index.tolist(),
        colorscale=[[0, "#3fb950"], [0.5, "#e3b341"], [1, "#f85149"]],
        showscale=True,
        text=heat_df.round(1).values,
        texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono"),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        colorbar=dict(title="Risk Level", tickfont=dict(color="#8b949e"), titlefont=dict(color="#8b949e"),
                      tickvals=[0, 0.5, 1], ticktext=["Low", "Medium", "High"]),
    ))
    fig_heat.update_layout(
        **PLOT_LAYOUT, height=520,
        title=dict(text="Portfolio Risk Heatmap — All Companies × All Dimensions", font=dict(family="DM Serif Display", size=15, color="#e6edf3")),
        xaxis=dict(side="top", tickfont=dict(family="DM Sans", color="#e6edf3", size=11)),
        yaxis=dict(tickfont=dict(family="DM Sans", color="#e6edf3", size=11), autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('<div class="icard icard-blue">🔍 <b>How to read this</b> — scan each row to see a company&#39;s full risk profile. Scan down each column to compare all companies on one dimension. Red = high risk or poor score. Green = low risk or strong score.</div>', unsafe_allow_html=True)

    st.download_button("⬇️ Download CSV", df[display_cols].to_csv(index=False).encode("utf-8"),
                       "tcfd_climate_risk.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;border-top:1px solid #30363d;display:flex;justify-content:space-between;align-items:center">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8b949e">
    SmartHaven Digital &nbsp;·&nbsp; TCFD Climate Risk Intelligence Platform P02 &nbsp;·&nbsp; Faith Ndinda, Nairobi
  </div>
  <div style="display:flex;gap:0.8rem">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8b949e">Python</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8b949e">Streamlit</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8b949e">Plotly</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#8b949e">Claude AI</span>
  </div>
</div>
""", unsafe_allow_html=True)
