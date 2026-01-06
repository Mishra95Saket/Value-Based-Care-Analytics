# Value-Based Care Analytics: Preventable Readmissions & Cost Leakage (Synthetic Claims)

A **senior-level Healthcare Data Analyst** portfolio project that mirrors real payer + hospital workflows:
- **Synthetic claims + admissions data generator** (no PHI)
- End-to-end **readmissions analytics pipeline**
- BI-ready outputs + **executive dashboard** (offline HTML + optional Streamlit)

---

## Business Questions Answered

### 1) Which diagnoses have the highest **preventable 30-day readmission** rates?
**Output:** `data/processed/diagnosis_summary.csv`  
Includes admissions, 30D readmissions, preventable readmission events (proxy), and preventable share.

### 2) Which patient groups are **highest risk**?
**Output:** `data/processed/patient_risk_scores.csv`  
Includes transparent 0–100 risk score + risk tier + drivers (age, SDI, chronic burden, utilization).

### 3) What is the **cost impact** of preventable readmissions?
**Outputs:**
- `data/processed/kpi_summary.csv` (leadership KPIs)
- `data/processed/readmissions_events.csv` (event-level spend)

### 4) Which interventions would save the most money?
**Output:** `data/processed/intervention_roi.csv`  
Simulated ROI for follow-up calls, medication reconciliation, and care coordination.

---

## Repo Structure

```text
healthcare-readmissions-analytics/
├── data/raw/                # generated synthetic data
├── data/processed/          # BI-ready tables
├── src/                     # pipeline scripts
├── sql/                     # example SQL
└── dashboard/               # HTML dashboard + Streamlit app
```

---

## How to Run

### Install
```bash
pip install -r requirements.txt
```

### 1) Generate synthetic data
```bash
python src/generate_synthetic_claims.py --n_members 5000 --seed 42
```

### 2) Build analytics tables + KPIs
```bash
python src/build_analytics_tables.py
```

### 3) Generate the dashboard (downloadable HTML)
```bash
python src/make_html_dashboard.py
```

Open:
- `dashboard/readmissions_dashboard.html`

### Optional: Run interactive app
```bash
streamlit run dashboard/streamlit_app.py
```

---

## Dashboard Explanation (What each section means)

- **KPI 1: 30-Day Readmission Rate** — overall rate of next admission within 30 days of discharge
- **KPI 2: Preventable Readmission Spend** — total paid amount on readmission admissions flagged preventable (proxy)
- **Top Diagnoses (Preventable Events)** — diagnoses driving avoidable readmissions (actionable targets)
- **Risk Tier Distribution** — how many members are Low/Medium/High risk (care management targeting)
- **Intervention Net Savings + ROI Table** — which programs deliver best expected net savings

---

## Notes / Assumptions

- “Preventable” is a **proxy label** generated using condition mix, SDI, chronic burden, and follow-up behavior.
- Readmission is defined as **the next admission within 1–30 days** for a member.
- ROI is a simplified simulation for portfolio purposes.

---

## Strong Next Steps (to level this up further)

1. Risk-adjusted benchmarking by hospital (indirect standardization)
2. Add pharmacy fills and compute PDC adherence
3. Train an ML model (logistic regression/XGBoost) + calibration + SHAP
4. Deploy Streamlit publicly and link it in README
