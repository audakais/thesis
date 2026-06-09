"""
Kaplan-Meier survival analysis for top RF biomarker per TCGA cohort.
Uses GDC /analysis/survival endpoint (Overall Survival).
Median-splits tumor patients by top-gene expression; log-rank test.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import json
import time
import glob
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

DATASET_DIR = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SURV_URL    = "https://api.gdc.cancer.gov/analysis/survival"

PROJECT_MAP = {
    "TCGA_BRCA": "TCGA-BRCA", "TCGA_KIRC": "TCGA-KIRC",
    "TCGA_LUAD": "TCGA-LUAD", "TCGA_LUSC": "TCGA-LUSC",
    "TCGA_PRAD": "TCGA-PRAD", "TCGA_STAD": "TCGA-STAD",
    "TCGA_THCA": "TCGA-THCA",
}

os.makedirs(FIGURES_DIR, exist_ok=True)

# Build ENSG -> symbol lookup from the named biomarkers file
_named_path = os.path.join(OUTPUT_DIR, "Biomarkers_Final_Named.csv")
_named = pd.read_csv(_named_path)[['Gene', 'symbol']].dropna()
SYMBOL_MAP = dict(zip(_named['Gene'], _named['symbol']))
# mygene-resolved fallbacks for top genes not in Biomarkers_Final_Named.csv
SYMBOL_MAP.update({
    "ENSG00000169550": "MUC15",
    "ENSG00000204305": "AGER",
    "ENSG00000241644": "INMT",
    "ENSG00000242110": "AMACR",
})


def fetch_survival(project_id):
    filt = json.dumps({"op": "in", "content": {
        "field": "cases.project.project_id", "value": [project_id]}})
    r = requests.get(SURV_URL, params={"filters": filt, "size": 10000}, timeout=60)
    r.raise_for_status()
    donors = r.json()["results"][0]["donors"]
    rows = [{"submitter_id": d["submitter_id"][:12],
             "time":  float(d["time"]),
             "event": 0 if d["censored"] else 1}
            for d in donors if d["time"] > 0]
    return pd.DataFrame(rows)


all_stats = []

files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if "_vs_gtex" not in f]

for fp in files:
    cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    proj   = PROJECT_MAP.get(cohort)
    if not proj:
        continue
    rf_path = os.path.join(OUTPUT_DIR, f"Biomarkers_RF_{cohort}.csv")
    if not os.path.exists(rf_path):
        continue

    top_ensg = pd.read_csv(rf_path).iloc[0]["Gene"]
    top_gene = SYMBOL_MAP.get(top_ensg, top_ensg)

    df = pd.read_csv(fp)
    if top_ensg not in df.columns:
        continue

    df["submitter_id_12"] = df["submitter_id"].astype(str).str[:12]
    expr = np.log2(pd.to_numeric(df[top_ensg], errors="coerce").fillna(0) + 1.0)
    df["expr"] = expr.values

    tumor_df = df[df["target"] == 1][["submitter_id_12", "expr"]].drop_duplicates("submitter_id_12")

    print(f"\n{cohort}: fetching survival data...")
    try:
        surv = fetch_survival(proj)
        time.sleep(0.3)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    surv = surv.rename(columns={"submitter_id": "submitter_id_12"})
    merged = tumor_df.merge(surv, on="submitter_id_12", how="inner")
    merged = merged[merged["time"] > 0].dropna(subset=["time", "event", "expr"])
    if len(merged) < 20:
        print(f"  too few matched ({len(merged)}), skip.")
        continue

    median_expr = merged["expr"].median()
    hi = merged[merged["expr"] >= median_expr]
    lo = merged[merged["expr"] <  median_expr]

    lr = logrank_test(hi["time"], lo["time"], hi["event"], lo["event"])
    p  = lr.p_value

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for grp, sub, color in [("High", hi, "#D32F2F"), ("Low", lo, "#1976D2")]:
        kmf.fit(sub["time"], sub["event"], label=f"{grp} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_title(f"{cohort} — OS by {top_gene} expression  (log-rank p={p:.4f})", fontsize=11)
    ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"survival_{cohort}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    sig = "YES" if p < 0.05 else "NO"
    n_events = int(merged["event"].sum())
    print(f"  gene={top_gene} ({top_ensg}) | n={len(merged)} | events={n_events} | p={p:.4f} | sig={sig}")
    all_stats.append({"Cohort": cohort, "Gene": top_gene, "ENSG": top_ensg, "N": len(merged),
                      "N_events": n_events, "LogRank_P": round(p, 6), "Significant": sig})

if all_stats:
    out_csv = os.path.join(OUTPUT_DIR, "Survival_Analysis.csv")
    pd.DataFrame(all_stats).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(pd.DataFrame(all_stats).to_string(index=False))

