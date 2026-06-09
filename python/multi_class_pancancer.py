"""
Multi-class pan-cancer classification on TCGA tumor samples.

Predicts the TCGA project (BRCA vs LUAD vs KIRC vs ...) using only tumor
samples (target=1) from all cohorts, on a common feature space (intersection
of the top-1500 highest-variance genes per cohort).

Output:
  - MultiClass_PanCancer_Report.csv         (precision/recall/f1 per class)
  - MultiClass_PanCancer_Biomarkers.csv     (mean RF importance across all genes)
  - MultiClass_PanCancer_ConfMatrix.csv     (confusion matrix)
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

INPUT_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")
N_FOLDS    = 5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "dataset_TCGA_*.csv")))
    if not files:
        print(f"No dataset_TCGA_*.csv files found in {INPUT_DIR}.")
        return

    # 1. Load tumor samples (target=1) for each cohort and compute gene intersection.
    print("=== LOADING DATA AND COMPUTING FEATURE INTERSECTION ===")
    cohort_dfs = []
    gene_sets  = []
    for fp in files:
        cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
        df = pd.read_csv(fp)
        if 'target' not in df.columns or 'cancer_project' not in df.columns:
            print(f"  {cohort}: unexpected schema, skipping.")
            continue
        df_t = df[df['target'] == 1].copy()
        if df_t.empty:
            print(f"  {cohort}: 0 tumor samples, skipping.")
            continue
        cohort_dfs.append(df_t)
        gene_cols = [c for c in df_t.columns if c.startswith("ENSG")]
        gene_sets.append(set(gene_cols))
        print(f"  {cohort}: {len(df_t)} tumors, {len(gene_cols)} genes")

    if len(cohort_dfs) < 2:
        print("\nAt least 2 tumor cohorts required for multi-class classification. Stopping.")
        return

    common_genes = sorted(set.intersection(*gene_sets))
    print(f"\nGenes in INTERSECTION across {len(cohort_dfs)} cohorts: {len(common_genes)}")
    if len(common_genes) < 50:
        print(f"WARNING: very small intersection ({len(common_genes)} genes). "
              "Multi-class signal may be weak.")

    # 2. Concatenate tumor samples on the common feature space.
    keep_cols = ['submitter_id', 'cancer_project'] + common_genes
    big_df = pd.concat([d[keep_cols] for d in cohort_dfs], ignore_index=True)
    print(f"Concatenated dataset: {big_df.shape}")

    y = big_df['cancer_project'].astype(str)
    groups = big_df['submitter_id'].astype(str).str[:12]
    X = big_df[common_genes].apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    X = np.log2(X + 1.0)
    X = X.fillna(X.median(numeric_only=True))

    print(f"\nClass distribution:")
    print(y.value_counts().to_string())
    n_patients = groups.nunique()
    print(f"Unique patients: {n_patients}")
    if n_patients < N_FOLDS:
        print(f"ERROR: patients < {N_FOLDS} folds. Stopping.")
        return

    # 3. Multi-class Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )
    gkf = GroupKFold(n_splits=N_FOLDS)

    print("\n=== CROSS-VALIDATION (5-fold GroupKFold by patient) ===")
    cv_res = cross_validate(
        model, X, y, groups=groups, cv=gkf,
        scoring=['accuracy', 'f1_macro'], n_jobs=1
    )
    print(f"Accuracy:  {cv_res['test_accuracy'].mean():.4f} +/- {cv_res['test_accuracy'].std():.4f}")
    print(f"F1 Macro:  {cv_res['test_f1_macro'].mean():.4f} +/- {cv_res['test_f1_macro'].std():.4f}")

    # 4. Out-of-fold predictions for report and confusion matrix
    print("\n=== OOF PREDICTIONS ===")
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=gkf, n_jobs=1)

    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose().round(4)
    out_rep = os.path.join(OUTPUT_DIR, "MultiClass_PanCancer_Report.csv")
    rep_df.to_csv(out_rep)
    print(rep_df.to_string())
    print(f"-> {out_rep}")

    classes = sorted(y.unique().tolist())
    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                             columns=[f"pred_{c}" for c in classes])
    out_cm = os.path.join(OUTPUT_DIR, "MultiClass_PanCancer_ConfMatrix.csv")
    cm_df.to_csv(out_cm)
    print(f"\nConfusion matrix -> {out_cm}")

    # 5. Full fit for pan-cancer biomarker importances
    print("\n=== FINAL FIT FOR FEATURE IMPORTANCES ===")
    model.fit(X, y)
    bio = pd.DataFrame({'Gene': X.columns, 'Importance': model.feature_importances_})
    bio = bio.sort_values('Importance', ascending=False).reset_index(drop=True)
    out_bio = os.path.join(OUTPUT_DIR, "MultiClass_PanCancer_Biomarkers.csv")
    bio.to_csv(out_bio, index=False)
    print(f"\nTop 20 pan-cancer biomarkers:")
    print(bio.head(20).to_string(index=False))
    print(f"\n-> {out_bio}")


if __name__ == "__main__":
    main()

