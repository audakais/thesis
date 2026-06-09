"""
Per-cohort analysis of sex, age and ethnicity distribution
across tumour samples (target=1). Produces one figure per cohort
with 3 subplots and a summary CSV table.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

DATASET_DIR = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load demographics ────────────────────────────────────────────────────────
demo = pd.read_csv(os.path.join(OUTPUT_DIR, "TCGA_Demographics.csv"))
demo["submitter_id_12"] = demo["submitter_id"].str[:12]
demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
demo["age_group"] = pd.cut(demo["age"],
    bins=[0, 40, 50, 60, 70, 120],
    labels=["<40", "40–49", "50–59", "60–69", "70+"])
RACE_MAP = {
    "white": "White",
    "black or african american": "Black/AA",
    "asian": "Asian",
    "american indian or alaska native": "Native Am.",
    "native hawaiian or other pacific islander": "Pacific Isl.",
}
demo["race_clean"] = demo["race"].str.lower().map(RACE_MAP).fillna("Other/NR")
demo["gender_clean"] = demo["gender"].str.capitalize().replace("Unknown", "Other/NR")

# ── Load targets ─────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if "_vs_gtex" not in f]

all_rows = []
for fp in files:
    cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    df = pd.read_csv(fp, usecols=["submitter_id", "target"])
    df["submitter_id_12"] = df["submitter_id"].str[:12]
    df["cohort"] = cohort
    all_rows.append(df)

targets = pd.concat(all_rows, ignore_index=True)
data = targets.merge(
    demo[["submitter_id_12", "gender_clean", "race_clean", "age_group", "age"]],
    on="submitter_id_12", how="left"
)

# Keep only tumor samples
tumors = data[data["target"] == 1].copy()

summary_rows = []

for cohort in sorted(tumors["cohort"].unique()):
    sub = tumors[tumors["cohort"] == cohort]
    n   = len(sub)
    label = cohort.replace("TCGA_", "")
    print(f"\n{'='*50}")
    print(f"  {cohort}  (N tumor = {n})")
    print(f"{'='*50}")

    if sub["gender_clean"].isna().all() or len(sub) == 0:
        print("  No demographics available, skipping.")
        continue

    # ── Sex ──────────────────────────────────────────────────────────────────
    sex = sub["gender_clean"].value_counts()
    sex_pct = (sex / n * 100).round(1)
    print("Sex:")
    for k, v in sex.items():
        print(f"  {k:<15} {v:>5}  ({sex_pct[k]:.1f}%)")

    # ── Age ──────────────────────────────────────────────────────────────────
    age_valid = sub["age"].dropna()
    print(f"Age: mean={age_valid.mean():.1f}  median={age_valid.median():.1f}  "
          f"std={age_valid.std():.1f}  range=[{age_valid.min():.0f}–{age_valid.max():.0f}]")
    age_grp = sub["age_group"].value_counts().sort_index()
    for k, v in age_grp.items():
        print(f"  {str(k):<8} {v:>5}  ({v/n*100:.1f}%)")

    # ── Race ─────────────────────────────────────────────────────────────────
    race = sub["race_clean"].value_counts()
    race_pct = (race / n * 100).round(1)
    print("Race/Ethnicity:")
    for k, v in race.items():
        print(f"  {k:<20} {v:>5}  ({race_pct[k]:.1f}%)")

    # ── Chi-square sex vs race (if enough categories) ─────────────────────────
    ct = pd.crosstab(sub["gender_clean"], sub["race_clean"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2 and ct.values.sum() > 0:
        try:
            chi2, p_chi, _, _ = chi2_contingency(ct)
            print(f"Chi2 sex×race: χ²={chi2:.2f}  p={p_chi:.4f}"
                  f"  ({'significant' if p_chi<0.05 else 'not significant'})")
        except Exception:
            p_chi = None
    else:
        p_chi = None

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{cohort} — Tumor incidence by demographics  (N={n})", fontsize=13)

    # Sex
    sex.plot(kind="bar", ax=axes[0],
             color=["#D32F2F" if "Female" in str(k) else "#1976D2" if "Male" in str(k)
                    else "#9E9E9E" for k in sex.index])
    axes[0].set_title("Sex"); axes[0].set_xlabel("")
    axes[0].set_ylabel("N tumor patients"); axes[0].tick_params(axis="x", rotation=0)
    for bar, (_, v) in zip(axes[0].patches, sex.items()):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"{v/n*100:.1f}%", ha="center", fontsize=9)

    # Age group
    age_grp.plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_title("Age group"); axes[1].set_xlabel("Age")
    axes[1].set_ylabel("N tumor patients"); axes[1].tick_params(axis="x", rotation=0)
    for bar, v in zip(axes[1].patches, age_grp.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v/n*100:.1f}%", ha="center", fontsize=8)

    # Race
    race_plot = race[race > 2]  # hide categories with < 3 patients
    colors = {"White": "#78909C", "Black/AA": "#5D4037", "Asian": "#F57C00",
              "Other/NR": "#BDBDBD", "Native Am.": "#388E3C", "Pacific Isl.": "#7B1FA2"}
    race_plot.plot(kind="bar", ax=axes[2],
                   color=[colors.get(k, "#BDBDBD") for k in race_plot.index])
    axes[2].set_title("Race / Ethnicity"); axes[2].set_xlabel("")
    axes[2].set_ylabel("N tumor patients"); axes[2].tick_params(axis="x", rotation=30)
    for bar, v in zip(axes[2].patches, race_plot.values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v/n*100:.1f}%", ha="center", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"incidence_{cohort}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")

    # ── Summary row ───────────────────────────────────────────────────────────
    top_race = race.index[0]
    def _n(grp): return int(race.get(grp, 0))
    def _p(grp): return round(race_pct.get(grp, 0.0), 1)
    summary_rows.append({
        "Cohort":           cohort,
        "N_Tumor":          n,
        "Age_Mean":         round(age_valid.mean(), 1),
        "Age_Median":       round(age_valid.median(), 1),
        "Age_Std":          round(age_valid.std(), 1),
        "Female_N":         int(sex.get("Female", 0)),
        "Female_Pct":       round(sex.get("Female", 0) / n * 100, 1),
        "Male_N":           int(sex.get("Male", 0)),
        "Male_Pct":         round(sex.get("Male", 0) / n * 100, 1),
        "White_N":          _n("White"),
        "White_Pct":        _p("White"),
        "Black_AA_N":       _n("Black/AA"),
        "Black_AA_Pct":     _p("Black/AA"),
        "Asian_N":          _n("Asian"),
        "Asian_Pct":        _p("Asian"),
        "NativeAm_N":       _n("Native Am."),
        "NativeAm_Pct":     _p("Native Am."),
        "OtherNR_N":        _n("Other/NR"),
        "OtherNR_Pct":      _p("Other/NR"),
        "Chi2_sex_race_p":  round(p_chi, 4) if p_chi is not None else None,
    })

summary = pd.DataFrame(summary_rows)
out_csv = os.path.join(OUTPUT_DIR, "Tumor_Incidence_Demographics.csv")
summary.to_csv(out_csv, index=False)
print(f"\n\nSummary saved: {out_csv}")
print(summary.to_string(index=False))

