import pandas as pd
import numpy as np


def compute_climate_var(carbon_intensity, physical_risk, transition_risk, confidence=0.95):
    """
    Climate Value-at-Risk (CVaR) — simplified Monte Carlo.
    Returns estimated % revenue at risk under climate scenarios.
    """
    np.random.seed(42)
    n = 10000

    # Physical risk shock: flood, drought, extreme heat disruption
    physical_shock = np.random.normal(
        loc=physical_risk * 0.012,
        scale=physical_risk * 0.008,
        size=n
    )

    # Transition risk shock: carbon pricing, stranded assets, policy
    transition_shock = np.random.normal(
        loc=transition_risk * 0.015,
        scale=transition_risk * 0.010,
        size=n
    )

    # Carbon cost component: higher intensity = higher exposure
    carbon_cost = (carbon_intensity / 500) * np.random.uniform(0.005, 0.025, size=n)

    total_loss = physical_shock + transition_shock + carbon_cost
    total_loss = np.clip(total_loss, 0, None)

    var = np.percentile(total_loss, confidence * 100)
    cvar = total_loss[total_loss >= var].mean()

    return round(var * 100, 2), round(cvar * 100, 2)


def scenario_analysis(df, scenario="1.5C"):
    """
    Apply TCFD-aligned scenario multipliers to risk scores.
    Scenarios: 1.5C (aggressive transition), 2C (moderate), 3C (physical-dominant)
    """
    multipliers = {
        "1.5C": {"physical": 0.8, "transition": 1.6, "carbon_cost_usd_per_tonne": 85},
        "2C":   {"physical": 1.2, "transition": 1.2, "carbon_cost_usd_per_tonne": 50},
        "3C":   {"physical": 1.8, "transition": 0.7, "carbon_cost_usd_per_tonne": 25},
    }
    m = multipliers[scenario]
    result = df.copy()
    result["adj_physical_risk"] = (result["physical_risk_score"] * m["physical"]).clip(0, 10).round(2)
    result["adj_transition_risk"] = (result["transition_risk_score"] * m["transition"]).clip(0, 10).round(2)
    result["carbon_cost_exposure_usd_m"] = (
        result["carbon_intensity_tCO2e"] * result["revenue_usd_millions"] * m["carbon_cost_usd_per_tonne"] / 1_000_000
    ).round(2)
    return result


def classify_risk(row):
    avg = (row["physical_risk_score"] + row["transition_risk_score"]) / 2
    if avg >= 7.5:
        return "🔴 High"
    elif avg >= 5.0:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def load_and_enrich(path="data/climate_risk_data.csv"):
    df = pd.read_csv(path)
    df = df[df["year"] == df["year"].max()].copy()
    df["risk_category"] = df.apply(classify_risk, axis=1)
    df["var_95"], df["cvar_95"] = zip(*df.apply(
        lambda r: compute_climate_var(r["carbon_intensity_tCO2e"], r["physical_risk_score"], r["transition_risk_score"]),
        axis=1
    ))
    return df
