import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Preventable Readmissions Dashboard", layout="wide")
processed = st.sidebar.text_input("Processed data folder", "data/processed")

paths = {
    "kpi": os.path.join(processed, "kpi_summary.csv"),
    "dx": os.path.join(processed, "diagnosis_summary.csv"),
    "risk": os.path.join(processed, "patient_risk_scores.csv"),
    "roi": os.path.join(processed, "intervention_roi.csv"),
}

if not all(os.path.exists(p) for p in paths.values()):
    st.error("Processed files not found. Run: python src/build_analytics_tables.py")
    st.stop()

kpi = pd.read_csv(paths["kpi"]).iloc[0]
dx = pd.read_csv(paths["dx"])
risk = pd.read_csv(paths["risk"])
roi = pd.read_csv(paths["roi"])

st.title("Value-Based Care — Preventable Readmissions & Cost Leakage (Synthetic Data)")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Admissions", f"{int(kpi.total_admissions):,}")
c2.metric("30D Readmissions", f"{int(kpi.readmissions_30d):,}")
c3.metric("30D Readmission Rate", f"{kpi.readmission_rate_30d*100:.2f}%")
c4.metric("Preventable Readmission Spend", f"${kpi.preventable_readmission_paid:,.0f}")

st.subheader("Top Diagnoses (Preventable Readmission Events)")
top_dx = dx.sort_values("preventable_readmission_events", ascending=False).head(10)
st.plotly_chart(px.bar(top_dx, x="primary_condition_group", y="preventable_readmission_events"), use_container_width=True)

st.subheader("Risk Tier Distribution")
tier = risk["risk_tier"].value_counts().reindex(["High","Medium","Low"]).fillna(0).reset_index()
tier.columns = ["risk_tier","members"]
st.plotly_chart(px.pie(tier, names="risk_tier", values="members", hole=0.45), use_container_width=True)

st.subheader("Intervention ROI Simulation")
st.plotly_chart(px.bar(roi.sort_values("estimated_net_savings", ascending=False), x="intervention", y="estimated_net_savings"), use_container_width=True)
st.dataframe(roi.sort_values("estimated_net_savings", ascending=False), use_container_width=True)
