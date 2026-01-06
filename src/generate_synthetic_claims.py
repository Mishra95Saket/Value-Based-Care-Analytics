
"""
Generate synthetic healthcare claims + admissions dataset tailored for readmissions analytics.

Outputs (data/raw):
- members.csv
- admissions.csv
- claims.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

CONDITIONS = [
    ("CHF", ["I50.9","I50.1","I11.0"], 291),
    ("COPD", ["J44.9","J44.1"], 190),
    ("DIABETES", ["E11.9","E11.65"], 640),
    ("PNEUMONIA", ["J18.9","J13"], 193),
    ("SEPSIS", ["A41.9","R65.20"], 871),
    ("CKD", ["N18.3","N18.4","N18.5"], 694),
    ("HTN", ["I10"], 301),
]
CPT_OUTPATIENT = ["99213","99214","93000","36415","83036","80053","71045","A0427","G0439"]
PROVIDERS = [f"P{str(i).zfill(5)}" for i in range(1, 801)]
HOSPITALS = [f"H{str(i).zfill(4)}" for i in range(1, 121)]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

def make_members(rng: np.random.Generator, n: int) -> pd.DataFrame:
    member_id = [f"M{str(i).zfill(7)}" for i in range(1, n+1)]
    age = rng.integers(18, 91, size=n)
    sex = rng.choice(["F","M"], size=n, p=[0.52, 0.48])
    state = rng.choice(["TX","CA","FL","NY","GA","NC","IL","AZ","WA","NJ"], size=n)
    sdi = np.clip(rng.normal(0.45, 0.22, size=n), 0, 1)
    plan_type = rng.choice(["HMO","PPO","Medicare Advantage"], size=n, p=[0.35,0.45,0.20])
    chronic_lambda = 0.8 + 0.03*np.clip(age-45, 0, None) + 0.9*sdi
    chronic_count = np.clip(rng.poisson(chronic_lambda), 0, 6)
    return pd.DataFrame({
        "member_id": member_id,
        "age": age,
        "sex": sex,
        "state": state,
        "sdi": np.round(sdi,3),
        "plan_type": plan_type,
        "chronic_count": chronic_count,
    })

def random_date(rng: np.random.Generator, start: datetime, end: datetime, size: int) -> np.ndarray:
    delta = (end - start).days
    days = rng.integers(0, delta+1, size=size)
    return np.array([start + timedelta(days=int(d)) for d in days], dtype="datetime64[ns]")

def make_admissions(rng: np.random.Generator, members: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    n = len(members)
    base = 0.12 + 0.01*(members["age"]>65).astype(float) + 0.03*members["chronic_count"].clip(0,6)
    adm_count = rng.poisson(base.to_numpy()*3)
    adm_count = np.clip(adm_count, 0, 6)

    rows = []
    adm_id = 1
    for i, m in members.iterrows():
        k = int(adm_count[i])
        if k == 0:
            continue
        admit_dates = np.sort(random_date(rng, start, end - timedelta(days=5), k))
        for ad in admit_dates:
            weights = np.array([1.2,1.1,1.1,0.9,0.7,0.8,0.6], dtype=float)
            weights = weights * (1 + 0.06*m["chronic_count"] + 0.008*max(m["age"]-50,0))
            weights = weights / weights.sum()
            cond_idx = rng.choice(len(CONDITIONS), p=weights)
            cond, icd_list, drg = CONDITIONS[cond_idx]
            icd10 = rng.choice(icd_list)
            los = int(np.clip(rng.normal(4.2, 2.0), 1, 18))
            discharge = (pd.Timestamp(ad) + pd.Timedelta(days=los)).to_datetime64()
            hospital_id = rng.choice(HOSPITALS)
            attending_provider_id = rng.choice(PROVIDERS)

            preventable_base = {"CHF":0.55,"COPD":0.50,"DIABETES":0.35,"PNEUMONIA":0.40,"SEPSIS":0.20,"CKD":0.25,"HTN":0.18}[cond]
            preventable = rng.random() < np.clip(preventable_base + 0.25*m["sdi"] + 0.06*m["chronic_count"], 0, 0.95)

            cond_mult = {"CHF":1.10,"COPD":1.00,"DIABETES":0.85,"PNEUMONIA":0.95,"SEPSIS":1.55,"CKD":1.25,"HTN":0.80}[cond]
            base_cost = rng.lognormal(mean=8.7, sigma=0.35)
            paid = base_cost * cond_mult * (1 + 0.10*(los-4))
            paid = float(np.clip(paid, 1800, 90000))

            followup_7d = rng.random() < np.clip(0.62 - 0.20*m["sdi"] - 0.06*m["chronic_count"], 0.05, 0.90)

            rows.append({
                "admission_id": f"A{adm_id:09d}",
                "member_id": m["member_id"],
                "hospital_id": hospital_id,
                "attending_provider_id": attending_provider_id,
                "admit_date": pd.Timestamp(ad).date().isoformat(),
                "discharge_date": pd.Timestamp(discharge).date().isoformat(),
                "length_of_stay": los,
                "primary_condition_group": cond,
                "primary_icd10": icd10,
                "drg": drg,
                "preventable_proxy": int(preventable),
                "followup_within_7d": int(followup_7d),
                "inpatient_paid_amount": round(paid,2),
            })
            adm_id += 1

    admissions = pd.DataFrame(rows)
    if admissions.empty:
        return admissions

    # simulate readmissions
    m_map = members.set_index("member_id")[["age","sdi","chronic_count"]]
    ad2 = admissions.merge(m_map, left_on="member_id", right_index=True, how="left")
    x = (
        -2.2
        + 0.018*(ad2["age"]-50)
        + 1.2*ad2["sdi"]
        + 0.28*ad2["chronic_count"]
        + 0.55*ad2["preventable_proxy"]
        + 0.70*(1-ad2["followup_within_7d"])
        + 0.35*ad2["primary_condition_group"].isin(["CHF","COPD","PNEUMONIA"]).astype(int)
    )
    p = sigmoid(x.to_numpy())
    will_readmit = rng.random(len(ad2)) < np.clip(p, 0.01, 0.55)
    discharge_dt = pd.to_datetime(ad2["discharge_date"])
    eligible = discharge_dt <= (pd.Timestamp(end) - pd.Timedelta(days=2))
    will_readmit = will_readmit & eligible.to_numpy()

    readmit_rows = []
    for idx, flag in enumerate(will_readmit):
        if not flag:
            continue
        row = admissions.iloc[idx]
        dis = pd.Timestamp(row["discharge_date"])
        gap = int(np.clip(rng.normal(12, 7), 2, 30))
        readmit_date = dis + pd.Timedelta(days=gap)
        los2 = int(np.clip(rng.normal(3.8, 1.8), 1, 15))
        discharge2 = readmit_date + pd.Timedelta(days=los2)

        if rng.random() < 0.72:
            cond = row["primary_condition_group"]
            icd_list = [c[1] for c in CONDITIONS if c[0]==cond][0]
            drg = [c[2] for c in CONDITIONS if c[0]==cond][0]
        else:
            cond, icd_list, drg = CONDITIONS[rng.integers(0,len(CONDITIONS))]
        icd10 = rng.choice(icd_list)
        preventable = int(rng.random() < 0.65)
        followup_7d = 0

        base_cost = rng.lognormal(mean=8.65, sigma=0.35)
        cond_mult = {"CHF":1.05,"COPD":1.00,"DIABETES":0.85,"PNEUMONIA":0.95,"SEPSIS":1.60,"CKD":1.20,"HTN":0.80}[cond]
        paid = float(np.clip(base_cost*cond_mult*(1+0.10*(los2-4)), 1700, 95000))

        readmit_rows.append({
            "admission_id": f"A{len(admissions)+len(readmit_rows)+1:09d}",
            "member_id": row["member_id"],
            "hospital_id": row["hospital_id"],
            "attending_provider_id": row["attending_provider_id"],
            "admit_date": readmit_date.date().isoformat(),
            "discharge_date": discharge2.date().isoformat(),
            "length_of_stay": los2,
            "primary_condition_group": cond,
            "primary_icd10": icd10,
            "drg": drg,
            "preventable_proxy": preventable,
            "followup_within_7d": followup_7d,
            "inpatient_paid_amount": round(paid,2),
        })

    if readmit_rows:
        admissions = pd.concat([admissions, pd.DataFrame(readmit_rows)], ignore_index=True)

    admissions["admit_date"] = pd.to_datetime(admissions["admit_date"])
    admissions["discharge_date"] = pd.to_datetime(admissions["discharge_date"])
    admissions = admissions.sort_values(["member_id","admit_date"]).reset_index(drop=True)
    admissions["admit_date"] = admissions["admit_date"].dt.date.astype(str)
    admissions["discharge_date"] = admissions["discharge_date"].dt.date.astype(str)
    return admissions

def make_claims(rng: np.random.Generator, members: pd.DataFrame, admissions: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    rows = []
    clm_id = 1
    for _, m in members.iterrows():
        lam = 8 + 0.20*m["age"] + 2.0*m["chronic_count"] + 6.5*m["sdi"]
        n_claims = int(np.clip(rng.poisson(lam/10), 2, 40))
        dates = random_date(rng, start, end, n_claims)
        for d in dates:
            cpt = rng.choice(CPT_OUTPATIENT)
            provider_id = rng.choice(PROVIDERS)
            paid = float(np.clip(rng.lognormal(4.2, 0.55), 10, 1200))
            cond, icds, _ = CONDITIONS[rng.integers(0, len(CONDITIONS))]
            icd10 = rng.choice(icds)
            rows.append({
                "claim_id": f"C{clm_id:011d}",
                "member_id": m["member_id"],
                "claim_date": pd.Timestamp(d).date().isoformat(),
                "claim_type": "OUTPATIENT",
                "provider_id": provider_id,
                "cpt": cpt,
                "icd10": icd10,
                "paid_amount": round(paid,2),
            })
            clm_id += 1

    if not admissions.empty:
        for _, a in admissions.iterrows():
            rows.append({
                "claim_id": f"C{clm_id:011d}",
                "member_id": a["member_id"],
                "claim_date": a["admit_date"],
                "claim_type": "INPATIENT",
                "provider_id": a["hospital_id"],
                "cpt": None,
                "icd10": a["primary_icd10"],
                "paid_amount": float(a["inpatient_paid_amount"]),
            })
            clm_id += 1

    claims = pd.DataFrame(rows)
    claims["claim_date"] = pd.to_datetime(claims["claim_date"]).dt.date.astype(str)
    return claims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_members", type=int, default=5000)
    ap.add_argument("--start_date", type=str, default="2024-01-01")
    ap.add_argument("--end_date", type=str, default="2025-12-31")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="data/raw")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start = datetime.fromisoformat(args.start_date)
    end = datetime.fromisoformat(args.end_date)

    members = make_members(rng, args.n_members)
    admissions = make_admissions(rng, members, start, end)
    claims = make_claims(rng, members, admissions, start, end)

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    members.to_csv(os.path.join(args.output_dir, "members.csv"), index=False)
    admissions.to_csv(os.path.join(args.output_dir, "admissions.csv"), index=False)
    claims.to_csv(os.path.join(args.output_dir, "claims.csv"), index=False)

    print(f"Wrote: {args.output_dir}/members.csv ({len(members):,} rows)")
    print(f"Wrote: {args.output_dir}/admissions.csv ({len(admissions):,} rows)")
    print(f"Wrote: {args.output_dir}/claims.csv ({len(claims):,} rows)")

if __name__ == "__main__":
    main()
