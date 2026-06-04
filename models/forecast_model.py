forecast_model.py 
"""
Climate Risk Forecasting Model (2025–2030) — SmartHaven Digital
================================================================
Generates forward-looking climate risk projections per company using:
  - Linear trend extrapolation on historical scores
  - Carbon intensity trajectory modelling
  - Scenario-conditioned VaR forecasts
  - Transition pathway scoring (early mover vs laggard)
  - Early warning signals for tipping points
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


FORECAST_YEARS = [2024, 2025, 2026, 2027, 2028, 2029, 2030]

# ── Policy/regulatory tightening assumptions per scenario ─────────────────────
SCENARIO_TRAJECTORIES = {
    "1.5C": {
        "carbon_price_path":      [30, 45, 60, 75, 85, 95, 105],  # USD/tonne
        "physical_risk_multiplier": [1.0, 1.08, 1.18, 1.30, 1.44, 1.58, 1.75],
        "transition_multiplier":    [1.0, 1.15, 1.32, 1.52, 1.72, 1.95, 2.20],
        "policy_stringency":        [1.0, 1.20, 1.45, 1.75, 2.10, 2.50, 3.00],
    },
    "2C": {
        "carbon_price_path":      [20, 28, 36, 44, 50, 56, 62],
        "physical_risk_multiplier": [1.0, 1.10, 1.22, 1.36, 1.52, 1.68, 1.88],
        "transition_multiplier":    [1.0, 1.10, 1.22, 1.35, 1.50, 1.65, 1.82],
        "policy_stringency":        [1.0, 1.12, 1.26, 1.42, 1.60, 1.80, 2.02],
    },
    "3C": {
        "carbon_price_path":      [10, 12, 15, 18, 22, 26, 30],
        "physical_risk_multiplier": [1.0, 1.15, 1.32, 1.55, 1.82, 2.12, 2.48],
        "transition_multiplier":    [1.0, 1.05, 1.10, 1.16, 1.23, 1.30, 1.38],
        "policy_stringency":        [1.0, 1.05, 1.10, 1.16, 1.22, 1.28, 1.35],
    },
}

# ── Sector decarbonisation trajectories ──────────────────────────────────────
# How fast each sector is expected to reduce carbon intensity (% per year)
SECTOR_DECARBONISATION_RATES = {
    "Energy":         {"1.5C": -0.12, "2C": -0.08, "3C": -0.03},
    "Utilities":      {"1.5C": -0.10, "2C": -0.07, "3C": -0.02},
    "Oil & Gas":      {"1.5C": -0.06, "2C": -0.04, "3C": -0.01},
    "Banking":        {"1.5C": -0.08, "2C": -0.06, "3C": -0.02},
    "Telecoms":       {"1.5C": -0.09, "2C": -0.07, "3C": -0.03},
    "Consumer Goods": {"1.5C": -0.07, "2C": -0.05, "3C": -0.02},
    "Materials":      {"1.5C": -0.05, "2C": -0.03, "3C": -0.01},
    "Insurance":      {"1.5C": -0.08, "2C": -0.06, "3C": -0.02},
    "Media":          {"1.5C": -0.09, "2C": -0.07, "3C": -0.03},
}


def forecast_carbon_intensity(company_row, scenario, years=FORECAST_YEARS):
    """
    Project carbon intensity trajectory to 2030.
    Early movers (high green_revenue_pct, high TCFD score) decline faster.
    """
    sector = company_row.get("sector", "Banking")
    base_rate = SECTOR_DECARBONISATION_RATES.get(sector, {}).get(scenario, -0.05)

    # Adjust for company behaviour
    green_rev = company_row.get("green_revenue_pct", 5) / 100
    tcfd_score = company_row.get("tcfd_disclosure_score", 50) / 100
    capex_intensity = company_row.get("energy_transition_capex_usd_millions", 10) / max(
        company_row.get("revenue_usd_millions", 100), 1
    )

    # Better disclosure + higher green revenue + more capex = faster decarbonisation
    company_adjustment = (green_rev * 0.04) + (tcfd_score * 0.02) + (capex_intensity * 0.5)
    adjusted_rate = base_rate - company_adjustment  # more negative = faster reduction

    base_intensity = company_row.get("carbon_intensity_tCO2e", 100)
    base_year = 2023

    projections = []
    for i, yr in enumerate(years):
        years_forward = yr - base_year
        projected = base_intensity * ((1 + adjusted_rate) ** years_forward)
        # Add uncertainty band
        uncertainty = base_intensity * 0.05 * np.sqrt(years_forward)
        projections.append({
            "year": yr,
            "carbon_intensity": round(max(0, projected), 2),
            "upper_bound": round(max(0, projected + uncertainty), 2),
            "lower_bound": round(max(0, projected - uncertainty), 2),
            "annual_reduction_rate": round(adjusted_rate * 100, 2),
        })

    return pd.DataFrame(projections)


def forecast_risk_scores(company_row, scenario, years=FORECAST_YEARS):
    """
    Project physical and transition risk scores forward to 2030
    under chosen scenario trajectory.
    """
    traj = SCENARIO_TRAJECTORIES[scenario]
    base_physical = company_row.get("physical_risk_score", 5)
    base_transition = company_row.get("transition_risk_score", 5)
    stranded = company_row.get("stranded_asset_risk", "medium")

    stranded_multiplier = {"low": 0.9, "medium": 1.0, "high": 1.15}.get(stranded, 1.0)

    projections = []
    for i, yr in enumerate(years):
        phys = min(10, base_physical * traj["physical_risk_multiplier"][i])
        trans = min(10, base_transition * traj["transition_multiplier"][i] * stranded_multiplier)
        carbon_cost = (
            company_row.get("carbon_intensity_tCO2e", 100) *
            traj["carbon_price_path"][i] *
            company_row.get("revenue_usd_millions", 100) / 1_000_000
        )
        composite = (phys * 0.4 + trans * 0.4 + min(10, carbon_cost / 50) * 0.2)

        projections.append({
            "year": yr,
            "physical_risk": round(min(10, phys), 2),
            "transition_risk": round(min(10, trans), 2),
            "carbon_cost_usd_m": round(carbon_cost, 2),
            "composite_risk": round(min(10, composite), 2),
            "carbon_price": traj["carbon_price_path"][i],
        })

    return pd.DataFrame(projections)


def forecast_var(company_row, scenario, years=FORECAST_YEARS):
    """
    Project Climate VaR forward — how does revenue at risk change to 2030?
    """
    np.random.seed(42)
    risk_df = forecast_risk_scores(company_row, scenario, years)
    results = []

    for _, row in risk_df.iterrows():
        n = 5000
        physical_shock = np.random.normal(row["physical_risk"] * 0.012,
                                           row["physical_risk"] * 0.008, n)
        transition_shock = np.random.normal(row["transition_risk"] * 0.015,
                                             row["transition_risk"] * 0.010, n)
        carbon_cost_pct = (row["carbon_cost_usd_m"] /
                           max(company_row.get("revenue_usd_millions", 100), 1))
        carbon_shock = np.random.uniform(0, carbon_cost_pct * 2, n)
        losses = np.clip(physical_shock + transition_shock + carbon_shock, 0, None) * 100

        results.append({
            "year": int(row["year"]),
            "var_95": round(np.percentile(losses, 95), 2),
            "cvar_95": round(losses[losses >= np.percentile(losses, 95)].mean(), 2),
            "median_loss": round(np.median(losses), 2),
        })

    return pd.DataFrame(results)


def detect_tipping_points(risk_df, threshold_physical=7.5, threshold_composite=7.0):
    """
    Identify the year when a company crosses critical risk thresholds.
    These are early warning signals for investors and management.
    """
    warnings = []

    physical_breach = risk_df[risk_df["physical_risk"] >= threshold_physical]
    if not physical_breach.empty:
        warnings.append({
            "type": "🌡️ Physical Risk Tipping Point",
            "year": int(physical_breach.iloc[0]["year"]),
            "value": physical_breach.iloc[0]["physical_risk"],
            "threshold": threshold_physical,
            "implication": "Physical asset damage costs likely to exceed insurance coverage",
        })

    transition_breach = risk_df[risk_df["transition_risk"] >= 8.5]
    if not transition_breach.empty:
        warnings.append({
            "type": "⚡ Transition Risk Tipping Point",
            "year": int(transition_breach.iloc[0]["year"]),
            "value": transition_breach.iloc[0]["transition_risk"],
            "threshold": 8.5,
            "implication": "Carbon costs may render core business model unviable",
        })

    composite_breach = risk_df[risk_df["composite_risk"] >= threshold_composite]
    if not composite_breach.empty:
        warnings.append({
            "type": "⚠️ Composite Risk Tipping Point",
            "year": int(composite_breach.iloc[0]["year"]),
            "value": composite_breach.iloc[0]["composite_risk"],
            "threshold": threshold_composite,
            "implication": "Combined risk likely to trigger credit rating downgrade",
        })

    return warnings


def score_transition_pathway(company_row, carbon_df, scenario):
    """
    Score how well positioned a company is for the transition.
    Returns Early Mover / On Track / Laggard / Stranded classification.
    """
    tcfd = company_row.get("tcfd_disclosure_score", 50)
    green_rev = company_row.get("green_revenue_pct", 5)
    capex = company_row.get("energy_transition_capex_usd_millions", 10)
    revenue = max(company_row.get("revenue_usd_millions", 100), 1)
    capex_intensity = (capex / revenue) * 100

    # Rate of decarbonisation
    if len(carbon_df) >= 2:
        start = carbon_df.iloc[0]["carbon_intensity"]
        end = carbon_df.iloc[-1]["carbon_intensity"]
        reduction_pct = ((start - end) / max(start, 1)) * 100
    else:
        reduction_pct = 0

    score = (
        (tcfd / 100) * 25 +
        (min(green_rev, 50) / 50) * 25 +
        (min(capex_intensity, 5) / 5) * 25 +
        (min(reduction_pct, 40) / 40) * 25
    )

    if score >= 75:
        return "🟢 Early Mover", score
    elif score >= 55:
        return "🔵 On Track", score
    elif score >= 35:
        return "🟡 Laggard", score
    else:
        return "🔴 Stranded Risk", score


def run_all_forecasts(company_row, scenario):
    """Run complete forecast suite for one company."""
    carbon_df = forecast_carbon_intensity(company_row, scenario)
    risk_df = forecast_risk_scores(company_row, scenario)
    var_df = forecast_var(company_row, scenario)
    tipping_points = detect_tipping_points(risk_df)
    pathway, pathway_score = score_transition_pathway(company_row, carbon_df, scenario)

    return {
        "carbon": carbon_df,
        "risk": risk_df,
        "var": var_df,
        "tipping_points": tipping_points,
        "pathway": pathway,
        "pathway_score": round(pathway_score, 1),
    }
