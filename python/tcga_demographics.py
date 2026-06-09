"""
Demographics (sex, race, age) of TCGA patients per cohort,
downloaded via GDC API. Covers all patients in the ML dataset.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import json
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
GDC_URL     = "https://api.gdc.cancer.gov/cases"
FIELDS      = "submitter_id,demographic.gender,demographic.race,demographic.age_at_index"

PROJECT_MAP = {
    "TCGA_BRCA": "TCGA-BRCA", "TCGA_KIRC": "TCGA-KIRC",
    "TCGA_LUAD": "TCGA-LUAD", "TCGA_LUSC": "TCGA-LUSC",
    "TCGA_OV":   "TCGA-OV",   "TCGA_PRAD": "TCGA-PRAD",
    "TCGA_STAD": "TCGA-STAD", "TCGA_THCA": "TCGA-THCA",
}

os.makedirs(FIGURES_DIR, exist_ok=True)


def fetch_demo(project_id):
    payload = {
        "filters": {"op": "in", "content": {
            "field": "project.project_id", "value": [project_id]}},
        "fields": FIELDS,
        "format": "JSON",
        "size": 2000,
    }
    r = requests.post(GDC_URL, json=payload, timeout=60)
    r.raise_for_status()
    rows = []
    for h in r.json()["data"]["hits"]:
        demo = h.get("demographic", {})
        rows.append({
            "submitter_id": h["submitter_id"],
            "gender": demo.get("gender", "unknown"),
            "race":   demo.get("race", "unknown"),
            "age":    demo.get("age_at_index"),
        })
    return pd.DataFrame(rows)


all_rows = []

for cohort, proj in PROJECT_MAP.items():
    print(f"Fetching {proj}...")
    try:
        df = fetch_demo(proj)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    df["cohort"] = cohort
    all_rows.append(df)

    n = len(df)
    age_mean = df["age"].dropna().astype(float).mean()
    age_std  = df["age"].dropna().astype(float).std()
    print(f"  N={n}  age={age_mean:.1f}+/-{age_std:.1f}")
    print(f"  Sex:  {df['gender'].value_counts().to_dict()}")
    print(f"  Race: {df['race'].value_counts().to_dict()}")

full = pd.concat(all_rows, ignore_index=True)
full["age"] = pd.to_numeric(full["age"], errors="coerce")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n\n=== SUMMARY TABLE ===")
for cohort in PROJECT_MAP:
    sub = full[full["cohort"] == cohort]
    age = sub["age"].dropna()
    print(f"\n{cohort}  (N={len(sub)})")
    print(f"  Age : mean={age.mean():.1f}  median={age.median():.1f}  "
          f"std={age.std():.1f}  range=[{age.min():.0f}-{age.max():.0f}]")
    print(f"  Sex : {sub['gender'].value_counts().to_dict()}")
    top_races = sub["race"].value_counts().head(4).to_dict()
    print(f"  Race: {top_races}")

# ── Save CSV ─────────────────────────────────────────────────────────────────
out_csv = os.path.join(OUTPUT_DIR, "TCGA_Demographics.csv")
full.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── Figures ──────────────────────────────────────────────────────────────────
cohorts = list(PROJECT_MAP.keys())

# 1. Age distribution per cohort
fig, ax = plt.subplots(figsize=(12, 5))
data_for_box = [full[full["cohort"] == c]["age"].dropna().values for c in cohorts]
ax.boxplot(data_for_box, labels=[c.replace("TCGA_", "") for c in cohorts], patch_artist=True)
ax.set_title("Age distribution per TCGA cohort")
ax.set_ylabel("Age at diagnosis (years)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "demo_age_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. Sex per cohort
sex_pivot = (full.groupby(["cohort", "gender"])
               .size().unstack(fill_value=0)
               .reindex(cohorts))
sex_pivot.index = [c.replace("TCGA_", "") for c in sex_pivot.index]
sex_pivot.plot(kind="bar", stacked=True, figsize=(10, 5),
               color={"male": "#1976D2", "female": "#D32F2F",
                      "unknown": "#BDBDBD"})
plt.title("Sex distribution per TCGA cohort")
plt.xlabel(""); plt.ylabel("N patients"); plt.legend(title="Sex")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "demo_sex_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3. Race per cohort (top-4 categories)
RACE_MAP = {
    "white": "White",
    "black or african american": "Black/AA",
    "asian": "Asian",
    "american indian or alaska native": "Native Am.",
    "not reported": "Not reported",
    "unknown": "Unknown",
}
full["race_clean"] = full["race"].str.lower().map(RACE_MAP).fillna("Other")
race_pivot = (full.groupby(["cohort", "race_clean"])
                .size().unstack(fill_value=0)
                .reindex(cohorts))
race_pivot.index = [c.replace("TCGA_", "") for c in race_pivot.index]
race_pivot.plot(kind="bar", stacked=True, figsize=(12, 5))
plt.title("Race/Ethnicity distribution per TCGA cohort")
plt.xlabel(""); plt.ylabel("N patients"); plt.legend(title="Race", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "demo_race_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Figures saved: demo_age_boxplot.png, demo_sex_bar.png, demo_race_bar.png")

