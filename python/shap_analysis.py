"""
SHAP (SHapley Additive exPlanations) analysis for each TCGA cohort.
Trains RF on 70% of data, computes TreeExplainer SHAP values on test set,
saves beeswarm summary plot for top-20 features per cohort.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATASET_DIR = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
META_COLS   = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
               'sample_type', 'target']
TOP_N       = 20
RANDOM_STATE = 42

os.makedirs(FIGURES_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if '_vs_gtex' not in f]

for fp in files:
    cohort  = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    rf_path = os.path.join(OUTPUT_DIR, f"Biomarkers_RF_{cohort}.csv")
    if not os.path.exists(rf_path):
        continue

    df = pd.read_csv(fp)
    if df['target'].nunique() < 2:
        continue

    top_genes = pd.read_csv(rf_path).head(50)['Gene'].tolist()
    top_genes = [g for g in top_genes if g in df.columns]
    if not top_genes:
        continue

    X = np.log2(df[top_genes].apply(pd.to_numeric, errors='coerce').fillna(0) + 1.0)
    y = df['target'].astype(int).values

    X_train, X_test, y_train, _ = train_test_split(
        X.values, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=250, max_depth=7,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    # shap_vals[1] = contribution to class 1 (tumor)
    shap_tumor = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[:, :, 1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_tumor, X_test,
        feature_names=top_genes,
        max_display=TOP_N,
        show=False, plot_size=None
    )
    plt.title(f'{cohort} — SHAP values (top-{TOP_N} features, tumor class)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"shap_{cohort}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

