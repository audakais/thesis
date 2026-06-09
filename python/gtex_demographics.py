"""
Demographics (sex, age) of GTEx samples predicted as high-risk (P(tumor)>=0.5).
Downloads GTEx v8 SubjectPhenotypes (public). Race/ethnicity is not available
in the public GTEx v8 release (requires dbGaP access).
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
PHENO_URL   = ("https://storage.googleapis.com/adult-gtex/annotations/v8/"
               "metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt")
PHENO_LOCAL = os.path.join(os.path.dirname(base_dir), "data", "gtex_raw", "GTEx_v8_SubjectPhenotypes.txt")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Download subject phenotypes if not present
if not os.path.exists(PHENO_LOCAL):
    print("Downloading GTEx v8 SubjectPhenotypes...")
    r = requests.get(PHENO_URL, timeout=60)
    r.raise_for_status()
    with open(PHENO_LOCAL, "wb") as f:
        f.write(r.content)
    print("  Done.")

pheno = pd.read_csv(PHENO_LOCAL, sep="\t")
# Columns: SUBJID, SEX (1=Male, 2=Female), AGE (bracket), DTHHRDY
pheno["SEX_label"] = pheno["SEX"].map({1: "Male", 2: "Female"})

preds = pd.read_csv(os.path.join(OUTPUT_DIR, "GTEx_Predictions.csv"))
high_risk = preds[preds["Predicted_Class"] == 1].copy()

print(f"\nTotal GTEx samples evaluated : {len(preds)}")
print(f"High-risk (P>=0.5)           : {len(high_risk)}")

if high_risk.empty:
    print("No high-risk samples — demographics table not applicable.")
else:
    # SUBJID = first two dash-separated fields: GTEX-XXXXX
    high_risk["SUBJID"] = high_risk["SAMPID"].str.extract(r"^(GTEX-[^-]+)")

    merged = high_risk.merge(pheno[["SUBJID", "SEX_label", "AGE"]], on="SUBJID", how="left")

    print("\n--- High-risk GTEx samples ---")
    print(merged[["SAMPID", "Cohort", "Tumor_Prob", "SEX_label", "AGE"]].to_string(index=False))

    # Summary stats
    print("\n--- Sex distribution ---")
    print(merged["SEX_label"].value_counts().to_string())

    print("\n--- Age distribution ---")
    print(merged["AGE"].value_counts().sort_index().to_string())

    print("\n--- By cohort ---")
    print(merged.groupby("Cohort")[["Tumor_Prob"]].agg(["count", "mean"]).to_string())

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "GTEx_HighRisk_Demographics.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Plot only if demographics resolved
    sex_counts = merged["SEX_label"].value_counts()
    age_counts = merged["AGE"].value_counts().sort_index()
    if not sex_counts.empty or not age_counts.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if not sex_counts.empty:
            sex_counts.plot.bar(ax=axes[0], color=["#1976D2", "#D32F2F"][:len(sex_counts)])
        axes[0].set_title("Sex — GTEx high-risk samples"); axes[0].set_xlabel("")
        if not age_counts.empty:
            age_counts.plot.bar(ax=axes[1], color="steelblue")
        axes[1].set_title("Age bracket — GTEx high-risk samples"); axes[1].set_xlabel("Age")
        plt.tight_layout()
        out_fig = os.path.join(FIGURES_DIR, "gtex_highrisk_demographics.png")
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved: {out_fig}")
    else:
        print("  No demographic data resolved — figure skipped.")

print("\nNote: Race/ethnicity is not available in the public GTEx v8 release.")
print("dbGaP accession phs000424 is required for protected phenotype data.")

