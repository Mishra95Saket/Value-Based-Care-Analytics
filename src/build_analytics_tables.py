from __future__ import annotations
"""
Build processed tables and KPIs for the readmissions analytics project.

Inputs (data/raw):
- members.csv
- admissions.csv
- claims.csv

Outputs (data/processed):
- admissions_enriched.csv
- readmissions_events.csv
- diagnosis_summary.csv
- hospital_summary.csv
- kpi_summary.csv
- patient_risk_scores.csv
- intervention_roi.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

def compute_readmission_flags(adm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adm = adm.copy()
    adm["admit_date"] = pd.to_datetime(adm["admit_date"])
    adm["discharge_date"] = pd.to_datetime(adm["discharge_date"])
    adm = adm.sort_values(["member_id","admit_date"]).reset_index(drop=True)

    g = adm.groupby("member_id", sort=False)
    adm["next_admit_date"] = g["admit_date"].shift(-1)
    adm["next_admission_id"] = g["admission_id"].shift(-1)
    adm["days_to_next_admit"] = (adm["next_admit_date"] - adm["discharge_date"]).dt.days
    adm["is_30d_readmission"] = ((adm["days_to_next_admit"] >= 1) & (adm["days_to_next_admit"] <= 30)).astype(int)

    events = adm.loc[adm["is_30d_readmission"]==1, [
        "member_id","admission_id","discharge_date","next_admission_id","next_admit_date","days_to_next_admit",
        "primary_condition_group","hospital_id","inpatient_paid_amount","preventable_proxy","followup_within_7d"
    ]].rename(columns={
        "admission_id":"index_admission_id",
        "discharge_date":"index_discharge_date",
        "primary_condition_group":"index_condition_group",
        "hospital_id":"index_hospital_id",
        "inpatient_paid_amount":"index_inpatient_paid_amount",
        "preventable_proxy":"index_preventable_proxy",
        "followup_within_7d":"index_followup_within_7d",
    })

    readm = adm[["admission_id","admit_date","primary_condition_group","preventable_proxy","inpatient_paid_amount"]].rename(columns={
        "admission_id":"next_admission_id",
        "admit_date":"readmit_admit_date",
        "primary_condition_group":"readmit_condition_group",
        "preventable_proxy":"readmit_preventable_proxy",
        "inpatient_paid_amount":"readmit_inpatient_paid_amount",
    })
    events = events.merge(readm, on="next_admission_id", how="left")
    events["readmission_event_total_paid"] = events["index_inpatient_paid_amount"] + events["readmit_inpatient_paid_amount"]
    return adm, events

def build_util_features(members: pd.DataFrame, admissions: pd.DataFrame, claims: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    admissions = admissions.copy()
    claims = claims.copy()
    admissions["admit_date"] = pd.to_datetime(admissions["admit_date"])
    claims["claim_date"] = pd.to_datetime(claims["claim_date"])
    start = as_of - pd.Timedelta(days=365)

    adm_12m = admissions[(admissions["admit_date"] >= start) & (admissions["admit_date"] <= as_of)]
    clm_12m = claims[(claims["claim_date"] >= start) & (claims["claim_date"] <= as_of)]

    prior_adm = adm_12m.groupby("member_id").size().rename("prior_admissions_12m")
    ed_visits = clm_12m[clm_12m["cpt"].isin(["A0427","99214"])].groupby("member_id").size().rename("ed_visits_12m")
    outpatient = clm_12m[clm_12m["claim_type"]=="OUTPATIENT"].groupby("member_id").size().rename("outpatient_visits_12m")
    no_follow = adm_12m.groupby("member_id")["followup_within_7d"].apply(lambda s: float((1-s).mean()) if len(s) else 0).rename("no_followup_rate")

    feats = pd.concat([prior_adm, ed_visits, outpatient, no_follow], axis=1).reset_index().fillna(0)
    return feats

def score_risk(members: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = members.merge(feats, on="member_id", how="left").fillna(0)
    age = df["age"].clip(18, 90)
    chronic = df["chronic_count"].clip(0, 6)
    sdi = df["sdi"].clip(0, 1)
    prior_adm = df["prior_admissions_12m"].clip(0, 10)
    prior_ed = df["ed_visits_12m"].clip(0, 20)
    outpt = df["outpatient_visits_12m"].clip(0, 60)
    no_follow = df["no_followup_rate"].clip(0, 1)

    raw = (
        0.22*(age-18)/72
        + 0.22*(chronic/6)
        + 0.20*sdi
        + 0.16*(prior_adm/10)
        + 0.10*(prior_ed/20)
        + 0.05*(outpt/60)
        + 0.05*no_follow
    )
    score = (raw / raw.max()) * 100
    df["readmission_risk_score"] = score.round(1)
    df["risk_tier"] = pd.cut(df["readmission_risk_score"], bins=[-1, 33, 66, 101], labels=["Low","Medium","High"])
    return df[[
        "member_id","age","sex","state","plan_type","sdi","chronic_count",
        "prior_admissions_12m","ed_visits_12m","outpatient_visits_12m","no_followup_rate",
        "readmission_risk_score","risk_tier"
    ]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--as_of_date", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    members = pd.read_csv(os.path.join(args.raw_dir, "members.csv"))
    admissions = pd.read_csv(os.path.join(args.raw_dir, "admissions.csv"))
    claims = pd.read_csv(os.path.join(args.raw_dir, "claims.csv"))

    admissions_enriched, readm_events = compute_readmission_flags(admissions)

    admissions_enriched["admit_date"] = pd.to_datetime(admissions_enriched["admit_date"])
    as_of = pd.to_datetime(args.as_of_date) if args.as_of_date else admissions_enriched["admit_date"].max()

    feats = build_util_features(members, admissions_enriched, claims, as_of)
    risk = score_risk(members, feats)

    # Dx summary
    dx = admissions_enriched.groupby("primary_condition_group").agg(
        admissions=("admission_id","count"),
        readmissions_30d=("is_30d_readmission","sum"),
        avg_inpatient_paid=("inpatient_paid_amount","mean"),
    ).reset_index()
    dx["readmission_rate_30d"] = (dx["readmissions_30d"] / dx["admissions"]).replace([np.inf,np.nan],0)

    preventable_events = readm_events.copy()
    preventable_events["is_preventable_readmission_event"] = ((preventable_events["index_preventable_proxy"]==1) & (preventable_events["days_to_next_admit"].between(1,30))).astype(int)
    prev_by_dx = preventable_events.groupby("index_condition_group").agg(
        preventable_readmission_events=("is_preventable_readmission_event","sum"),
        total_readmission_events=("index_admission_id","count"),
        avoidable_paid=("readmit_inpatient_paid_amount", "sum"),
    ).reset_index().rename(columns={"index_condition_group":"primary_condition_group"})
    dx = dx.merge(prev_by_dx, on="primary_condition_group", how="left").fillna(0)
    dx["preventable_share_of_readmissions"] = (dx["preventable_readmission_events"] / dx["total_readmission_events"]).replace([np.inf,np.nan],0)
    dx = dx.sort_values(["preventable_readmission_events","readmissions_30d"], ascending=False)

    # Hospital summary
    hosp = admissions_enriched.groupby("hospital_id").agg(
        admissions=("admission_id","count"),
        readmissions_30d=("is_30d_readmission","sum"),
        avg_paid=("inpatient_paid_amount","mean"),
    ).reset_index()
    hosp["readmission_rate_30d"] = (hosp["readmissions_30d"]/hosp["admissions"]).replace([np.inf,np.nan],0)
    hosp = hosp.sort_values("readmission_rate_30d", ascending=False)

    # KPIs
    total_adm = int(len(admissions_enriched))
    total_readm = int(admissions_enriched["is_30d_readmission"].sum())
    readm_rate = total_readm/total_adm if total_adm else 0
    total_inpatient_paid = float(admissions_enriched["inpatient_paid_amount"].sum())
    preventable_readm_paid = float(preventable_events.loc[preventable_events["is_preventable_readmission_event"]==1, "readmit_inpatient_paid_amount"].sum())
    avg_readm_paid = float(preventable_events["readmit_inpatient_paid_amount"].mean()) if len(preventable_events) else 0
    high_risk_members = int((risk["risk_tier"]=="High").sum())

    kpi = pd.DataFrame([{
        "as_of_date": as_of.date().isoformat(),
        "total_admissions": total_adm,
        "readmissions_30d": total_readm,
        "readmission_rate_30d": round(readm_rate,4),
        "total_inpatient_paid": round(total_inpatient_paid,2),
        "preventable_readmission_paid": round(preventable_readm_paid,2),
        "avg_readmission_paid": round(avg_readm_paid,2),
        "high_risk_members": high_risk_members,
    }])

    # Intervention ROI
    avoidable = preventable_readm_paid
    interventions = [
        ("Post-discharge follow-up (7d)", 0.07, 18.0),
        ("Medication reconciliation", 0.05, 28.0),
        ("Care coordination program", 0.10, 65.0),
    ]
    touches = max(int((risk["risk_tier"]=="High").sum()), 1)
    roi_rows=[]
    for name, red, cost_per in interventions:
        savings = avoidable * red
        program_cost = touches * cost_per
        roi = (savings - program_cost)/program_cost if program_cost else 0
        roi_rows.append({
            "intervention": name,
            "expected_readmission_reduction_pct": red,
            "avoidable_paid_baseline": round(avoidable,2),
            "estimated_savings": round(savings,2),
            "estimated_program_cost": round(program_cost,2),
            "estimated_net_savings": round(savings-program_cost,2),
            "roi": round(roi,3),
        })
    roi_df = pd.DataFrame(roi_rows).sort_values("estimated_net_savings", ascending=False)

    # Save
    admissions_enriched.to_csv(os.path.join(args.out_dir, "admissions_enriched.csv"), index=False)
    readm_events.to_csv(os.path.join(args.out_dir, "readmissions_events.csv"), index=False)
    dx.to_csv(os.path.join(args.out_dir, "diagnosis_summary.csv"), index=False)
    hosp.to_csv(os.path.join(args.out_dir, "hospital_summary.csv"), index=False)
    risk.to_csv(os.path.join(args.out_dir, "patient_risk_scores.csv"), index=False)
    roi_df.to_csv(os.path.join(args.out_dir, "intervention_roi.csv"), index=False)
    kpi.to_csv(os.path.join(args.out_dir, "kpi_summary.csv"), index=False)

    print("Wrote processed tables to:", args.out_dir)
    print(kpi.to_string(index=False))

if __name__ == "__main__":
    main()
