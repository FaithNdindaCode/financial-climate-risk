# 🌍 TCFD Climate Risk Dashboard — P02

**SmartHaven Digital | Financial Engineering Portfolio**

TCFD-aligned physical and transition risk analysis for NSE-listed Kenyan companies, with Monte Carlo Climate VaR simulation and multi-scenario modelling.

## Live Demo
🔗 [View on Streamlit Cloud](https://your-app-url.streamlit.app) ← update after deploy

## What this model does
- Scores companies on physical risk (floods, drought, heat) and transition risk (carbon pricing, stranded assets, policy)
- Runs 10,000-iteration Monte Carlo to compute Climate VaR and CVaR at 95% confidence
- Applies TCFD-aligned scenarios: 1.5°C, 2°C, and 3°C warming pathways (NGFS framework)
- Calculates carbon cost exposure per company under each scenario
- Tracks TCFD disclosure scores vs actual risk exposure

## Tech stack
- **Python** — pandas, numpy, scikit-learn, statsmodels
- **Visualisation** — Plotly
- **App** — Streamlit
- **Deployment** — Streamlit Cloud (auto-deploy from GitHub)

## Folder structure
```
financial-climate-risk/
├── app.py                  ← Streamlit entry point
├── requirements.txt
├── data/
│   └── climate_risk_data.csv
└── models/
    └── tcfd_model.py       ← Monte Carlo CVaR + scenario logic
```

## Run locally
```bash
git clone https://github.com/FaithNdindaCode/financial-climate-risk.git
cd financial-climate-risk
pip install -r requirements.txt
streamlit run app.py
```

## TCFD pillars covered
| Pillar | Coverage |
|---|---|
| Governance | Disclosure score tracking |
| Strategy | Scenario analysis (1.5C / 2C / 3C) |
| Risk Management | Physical + transition risk scoring |
| Metrics & Targets | Carbon intensity, VaR, CVaR, carbon cost exposure |

## Author
**Faith Ndinda** — SmartHaven Digital, Nairobi  
BCom Accounting & Finance, CUEA  
[GitHub](https://github.com/FaithNdindaCode) · [Portfolio](https://smarthaven-agent.netlify.app)
