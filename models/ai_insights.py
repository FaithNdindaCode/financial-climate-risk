"""
AI Insights Module — SmartHaven Digital
=========================================
Generates narrative climate risk analysis using Claude API.
Reads live dashboard data and produces structured insights:
  - Executive summary
  - Top risk flags
  - Sector outlook
  - Geospatial/satellite data recommendations
  - Investor action points
"""

import json


def build_analysis_prompt(df, scenario, top_risk_companies, sector_stats, forecast_data=None):
    """Build a structured prompt for Claude to analyse the dashboard data."""

    company_summary = []
    for _, row in top_risk_companies.iterrows():
        company_summary.append(
            f"- {row['company']} ({row['sector']}): "
            f"Physical risk {row.get('physical_risk_score', 'N/A')}/10, "
            f"Transition risk {row.get('transition_risk_score', 'N/A')}/10, "
            f"Carbon intensity {row.get('carbon_intensity_tCO2e', 'N/A')} tCO2e, "
            f"Climate VaR {row.get('var_95', 'N/A')}% revenue, "
            f"TCFD score {row.get('tcfd_disclosure_score', 'N/A')}/100, "
            f"Rating: {row.get('credit_rating', 'N/A')}"
        )

    sector_summary = []
    for sector, stats in sector_stats.items():
        sector_summary.append(
            f"- {sector}: avg carbon intensity {stats['avg_carbon']:.0f} tCO2e, "
            f"avg VaR {stats['avg_var']:.1f}%, "
            f"companies: {stats['count']}"
        )

    forecast_text = ""
    if forecast_data:
        forecast_text = f"""
FORECAST DATA (2025-2030 under {scenario} scenario):
{forecast_data}
"""

    return f"""You are a senior climate risk analyst at a leading African development finance institution.
Analyse the following TCFD climate risk data for NSE-listed Kenyan companies under the {scenario} warming scenario.

SCENARIO CONTEXT:
- {scenario} scenario: {"Aggressive energy transition, high carbon prices ($85/tonne by 2030), policy stringency" if scenario == "1.5C" else "Moderate transition balanced with physical risk growth" if scenario == "2C" else "Physical risk dominant, weak transition policy, severe weather events by 2030"}

TOP RISK COMPANIES:
{chr(10).join(company_summary)}

SECTOR BREAKDOWN:
{chr(10).join(sector_summary)}
{forecast_text}

KENYA MARKET CONTEXT:
- NSE is a thin, illiquid market — standard greenium models don't apply directly
- 70% of climate exposure is county-level physical risk, not firm-specific
- Mobile money penetration (M-Pesa) acts as a household climate shock buffer
- DFI concessional finance is more accessible than commercial green bonds for most issuers
- Voluntary carbon markets (REDD+) are the primary monetisation pathway, not compliance carbon

Respond in this EXACT JSON structure (no markdown, no preamble, pure JSON):
{{
  "executive_summary": "3-4 sentence plain-English summary of the overall risk picture. Name specific companies and numbers. Tell a story.",
  "top_3_risks": [
    {{"risk": "risk title", "company_or_sector": "who is affected", "detail": "specific financial implication with numbers", "urgency": "Immediate/2026/2028/2030"}},
    {{"risk": "risk title", "company_or_sector": "who is affected", "detail": "specific financial implication with numbers", "urgency": "Immediate/2026/2028/2030"}},
    {{"risk": "risk title", "company_or_sector": "who is affected", "detail": "specific financial implication with numbers", "urgency": "Immediate/2026/2028/2030"}}
  ],
  "sector_outlook": [
    {{"sector": "sector name", "outlook": "Positive/Cautious/Negative", "reasoning": "one sentence"}}
  ],
  "satellite_data_gaps": "2-3 sentences on what Planetek/KSA satellite data (drought indices, flood mapping, land change) would change about this analysis — be specific about which risk scores would shift and why",
  "investor_actions": [
    "specific actionable recommendation 1",
    "specific actionable recommendation 2",
    "specific actionable recommendation 3"
  ],
  "kenya_specific_insight": "One paragraph on what makes this risk picture uniquely Kenyan — things a Western climate model would miss"
}}"""


def compute_sector_stats(df):
    """Compute sector-level stats for the prompt."""
    stats = {}
    for sector in df["sector"].unique():
        sector_df = df[df["sector"] == sector]
        stats[sector] = {
            "avg_carbon": sector_df["carbon_intensity_tCO2e"].mean(),
            "avg_var": sector_df["var_95"].mean(),
            "count": len(sector_df),
        }
    return stats


async def generate_insights_stream(prompt):
    """
    Call Claude API and stream the response.
    Returns the full JSON response text.
    """
    import urllib.request
    import json

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["content"][0]["text"]


def generate_insights_sync(prompt):
    """Synchronous version using requests."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            text = result["content"][0]["text"]
            # Strip any markdown fences
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
    except Exception as e:
        return {"error": str(e)}


def generate_executive_summary(df, scenario):
    """
    Generate a quick rule-based executive summary without API call.
    Used as fallback or instant preview before AI analysis loads.
    """
    high_risk = df[df["risk_category"] == "🔴 High"]
    highest_var = df.nlargest(1, "var_95").iloc[0]
    best_disclosure = df.nlargest(1, "tcfd_disclosure_score").iloc[0]
    worst_carbon = df.nlargest(1, "carbon_intensity_tCO2e").iloc[0]

    scenario_context = {
        "1.5C": "aggressive energy transition with carbon prices reaching $85/tonne",
        "2C": "moderate transition balanced with growing physical risk",
        "3C": "physical risk-dominant pathway with severe weather intensification",
    }

    summary = (
        f"Under the **{scenario}** scenario ({scenario_context.get(scenario, '')}), "
        f"**{len(high_risk)} of {len(df)} companies** are in the high-risk category. "
        f"**{worst_carbon['company']}** ({worst_carbon['sector']}) carries the highest carbon intensity "
        f"at {worst_carbon['carbon_intensity_tCO2e']:.0f} tCO₂e — "
        f"making it most vulnerable to carbon pricing. "
        f"**{highest_var['company']}** faces the highest revenue at risk at "
        f"**{highest_var['var_95']:.1f}% Climate VaR**. "
        f"**{best_disclosure['company']}** leads on TCFD disclosure "
        f"({best_disclosure['tcfd_disclosure_score']}/100) — "
        f"positioning it best for green finance access."
    )
    return summary
