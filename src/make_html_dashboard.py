from __future__ import annotations
"""
Generate an executive-style offline HTML dashboard using Plotly.
Output: dashboard/readmissions_dashboard.html
"""
import argparse, os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--out_html", type=str, default="dashboard/readmissions_dashboard.html")
    args = ap.parse_args()

    kpi = pd.read_csv(os.path.join(args.processed_dir, "kpi_summary.csv")).iloc[0].to_dict()
    dx = pd.read_csv(os.path.join(args.processed_dir, "diagnosis_summary.csv"))
    risk = pd.read_csv(os.path.join(args.processed_dir, "patient_risk_scores.csv"))
    roi = pd.read_csv(os.path.join(args.processed_dir, "intervention_roi.csv"))

    top_dx = dx.sort_values("preventable_readmission_events", ascending=False).head(8)
    tier = risk["risk_tier"].value_counts().reindex(["High","Medium","Low"]).fillna(0)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type":"indicator"},{"type":"indicator"}],
               [{"type":"bar"},{"type":"pie"}],
               [{"type":"bar"},{"type":"table"}]],
        subplot_titles=("30-Day Readmission Rate",
                        "Preventable Readmission Spend",
                        "Top Diagnoses: Preventable Readmission Events",
                        "Risk Tier Distribution",
                        "Intervention: Net Savings (Simulation)",
                        "Top ROI Interventions"),
        vertical_spacing=0.10,
        horizontal_spacing=0.10
    )

    fig.add_trace(go.Indicator(
        mode="number",
        value=float(kpi["readmission_rate_30d"])*100,
        number={"suffix":"%"},
        title={"text": f"Readmission Rate (As of {kpi['as_of_date']})"}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="number",
        value=float(kpi["preventable_readmission_paid"]),
        number={"prefix":"$"},
        title={"text":"Preventable Readmission Spend"}
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=top_dx["primary_condition_group"],
        y=top_dx["preventable_readmission_events"],
        name="Preventable Events"
    ), row=2, col=1)

    fig.add_trace(go.Pie(
        labels=tier.index.tolist(),
        values=tier.values.tolist(),
        hole=0.45
    ), row=2, col=2)

    roi2 = roi.sort_values("estimated_net_savings", ascending=False)
    fig.add_trace(go.Bar(
        x=roi2["intervention"],
        y=roi2["estimated_net_savings"],
        name="Net Savings"
    ), row=3, col=1)

    table = roi2[["intervention","expected_readmission_reduction_pct","estimated_savings","estimated_program_cost","estimated_net_savings","roi"]].copy()
    table["expected_readmission_reduction_pct"] = (table["expected_readmission_reduction_pct"]*100).round(1).astype(str) + "%"
    fig.add_trace(go.Table(
        header={"values": list(table.columns)},
        cells={"values": [table[c].tolist() for c in table.columns]}
    ), row=3, col=2)

    fig.update_layout(
        title="Value-Based Care Analytics — Preventable Readmissions & Cost Leakage (Synthetic Data)",
        height=1050,
        margin=dict(l=30,r=30,t=80,b=30),
        showlegend=False
    )

    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)
    fig.write_html(args.out_html, include_plotlyjs="cdn")
    print("Wrote:", args.out_html)

if __name__ == "__main__":
    main()
