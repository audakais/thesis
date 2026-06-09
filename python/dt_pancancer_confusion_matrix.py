"""
Pan-cancer multi-class confusion matrix using Decision Tree.

Same data pipeline as multi_class_pancancer.py (tumor-only, common gene
intersection, log2-TPM, GroupKFold OOF predictions) but uses a
DecisionTreeClassifier instead of RF.

Output:
  - MultiClass_PanCancer_DT_Report.csv
  - MultiClass_PanCancer_DT_ConfMatrix.csv
  - figures/confusion_matrix_DT_pancancer.png
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

INPUT_DIR   = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
N_FOLDS     = 5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "dataset_TCGA_*.csv")))
    files = [f for f in files if '_vs_gtex' not in f]
    if not files:
        print(f"No dataset_TCGA_*.csv files found in {INPUT_DIR}.")
        return

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
        print("At least 2 cohorts required. Stopping.")
        return

    common_genes = sorted(set.intersection(*gene_sets))
    print(f"\nCommon genes across {len(cohort_dfs)} cohorts: {len(common_genes)}")

    keep_cols = ['submitter_id', 'cancer_project'] + common_genes
    big_df    = pd.concat([d[keep_cols] for d in cohort_dfs], ignore_index=True)
    print(f"Concatenated dataset: {big_df.shape}")

    y      = big_df['cancer_project'].astype(str)
    groups = big_df['submitter_id'].astype(str).str[:12]
    X      = big_df[common_genes].apply(pd.to_numeric, errors='coerce')
    X      = X.dropna(axis=1, how='all')
    X      = np.log2(X + 1.0)
    X      = X.fillna(X.median(numeric_only=True))

    print(f"\nClass distribution:\n{y.value_counts().to_string()}")

    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
    )
    gkf = GroupKFold(n_splits=N_FOLDS)

    print("\n=== OOF PREDICTIONS (5-fold GroupKFold) ===")
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=gkf)

    rep    = classification_report(y, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose().round(4)
    out_rep = os.path.join(OUTPUT_DIR, "MultiClass_PanCancer_DT_Report.csv")
    rep_df.to_csv(out_rep)
    print(rep_df.to_string())
    print(f"-> {out_rep}")

    classes = sorted(y.unique().tolist())
    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm,
                         index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])
    out_cm = os.path.join(OUTPUT_DIR, "MultiClass_PanCancer_DT_ConfMatrix.csv")
    cm_df.to_csv(out_cm)
    print(f"\nConfusion matrix -> {out_cm}")

    # Plot: row-normalised colour, raw counts as text
    labels_short = [c.replace("TCGA-", "") for c in classes]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_short)))
    ax.set_xticklabels(labels_short, rotation=35, ha='right', fontsize=12)
    ax.set_yticks(range(len(labels_short)))
    ax.set_yticklabels(labels_short, fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=13)
    ax.set_ylabel('True label', fontsize=13)
    ax.set_title('Pan-Cancer Multi-Class DT — OOF Confusion Matrix\n'
                 'Colour = row-normalised rate  |  Number = raw OOF count',
                 fontsize=13, pad=14)

    for i in range(len(labels_short)):
        for j in range(len(labels_short)):
            val   = cm[i, j]
            color = 'white' if cm_norm[i, j] > 0.55 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=11,
                    color=color, fontweight='bold' if i == j else 'normal')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Recall (row-normalised)', fontsize=11)

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "confusion_matrix_DT_pancancer.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out_png}")


if __name__ == "__main__":
    main()
