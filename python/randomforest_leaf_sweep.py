"""
Esperimento min_samples_leaf: confronto RF con leaf=10, 50, 100.
Stessa pipeline di randomforest.py (GroupKFold, SMOTE, variance-top-500).
Produce Summary_RF_LeafSweep.csv e figura di confronto F1/AUC.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

INPUT_DIR   = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

META_COLS = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
             'sample_type', 'target']

N_FOLDS       = 5
FOLD_FEATURES = 500
LEAF_VALUES   = [10, 50, 100]


def run_cohort_leaf(file_path, leaf):
    cohort = os.path.basename(file_path).replace("dataset_", "").replace(".csv", "")

    df = pd.read_csv(file_path)
    if 'target' not in df.columns or 'submitter_id' not in df.columns:
        return None
    if df['target'].nunique() < 2:
        return None

    groups = df['submitter_id'].astype(str).str[:12]
    if groups.nunique() < N_FOLDS:
        return None

    y = df['target'].astype(int)
    X = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    X = np.log2(X + 1.0)
    X = X.fillna(X.median(numeric_only=True))

    n_fold_feat = min(FOLD_FEATURES, X.shape[1])
    gkf   = GroupKFold(n_splits=N_FOLDS)
    smote = SMOTE(random_state=42)
    X_arr = X.values
    y_arr = y.values
    g_arr = groups.values

    fold_f1, fold_auc, train_f1 = [], [], []

    for train_idx, test_idx in gkf.split(X_arr, y_arr, groups=g_arr):
        X_tr_orig, y_tr_orig = X_arr[train_idx], y_arr[train_idx]
        X_te_full, y_te      = X_arr[test_idx],  y_arr[test_idx]

        top_idx = np.argsort(np.var(X_tr_orig, axis=0))[::-1][:n_fold_feat]
        X_tr_f  = X_tr_orig[:, top_idx]
        X_te_f  = X_te_full[:, top_idx]

        X_tr_s, y_tr_s = smote.fit_resample(X_tr_f, y_tr_orig)

        model = RandomForestClassifier(n_estimators=250, max_depth=7,
                                       min_samples_leaf=leaf,
                                       random_state=42, n_jobs=-1)
        model.fit(X_tr_s, y_tr_s)

        y_pred_tr = model.predict(X_tr_f)
        y_pred    = model.predict(X_te_f)
        y_prob    = model.predict_proba(X_te_f)[:, 1]

        train_f1.append(f1_score(y_tr_orig, y_pred_tr, average='macro'))
        fold_f1.append(f1_score(y_te, y_pred, average='macro'))
        fold_auc.append(roc_auc_score(y_te, y_prob))

    del df, X, y
    gc.collect()

    return {
        'Cohort':   cohort,
        'Leaf':     leaf,
        'F1_Macro': round(float(np.mean(fold_f1)),  4),
        'F1_Std':   round(float(np.std(fold_f1)),   4),
        'ROC_AUC':  round(float(np.mean(fold_auc)), 4),
        'Train_F1': round(float(np.mean(train_f1)), 4),
        'Gap_F1':   round(float(np.mean(train_f1)) - float(np.mean(fold_f1)), 4),
    }


def plot_sweep(df_all):
    cohorts = sorted(df_all['Cohort'].unique())
    metrics = ['F1_Macro', 'ROC_AUC', 'Gap_F1']
    titles  = ['Val F1 Macro', 'ROC-AUC', 'Overfitting Gap (Train-Val F1)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric, title in zip(axes, metrics, titles):
        for cohort in cohorts:
            sub = df_all[df_all['Cohort'] == cohort].sort_values('Leaf')
            ax.plot(sub['Leaf'], sub[metric], marker='o', label=cohort, linewidth=1.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('min_samples_leaf')
        ax.set_ylabel(metric)
        ax.set_xticks(LEAF_VALUES)
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)

    plt.suptitle('RF — min_samples_leaf Sweep (10 / 50 / 100)', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "rf_leaf_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura: {out}")


def main():
    print("=== RF LEAF SWEEP (10 / 50 / 100) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "dataset_TCGA_*.csv")))
    files = [f for f in files if '_vs_gtex' not in f]

    rows = []
    for leaf in LEAF_VALUES:
        print(f"\n--- min_samples_leaf = {leaf} ---")
        for fp in files:
            cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
            print(f"  {cohort}...", end=" ", flush=True)
            try:
                r = run_cohort_leaf(fp, leaf)
                if r:
                    rows.append(r)
                    print(f"F1={r['F1_Macro']} AUC={r['ROC_AUC']} Gap={r['Gap_F1']}")
                else:
                    print("skip")
            except Exception as e:
                print(f"ERROR: {e}")

    if rows:
        df_all = pd.DataFrame(rows)
        out_csv = os.path.join(OUTPUT_DIR, "Summary_RF_LeafSweep.csv")
        df_all.to_csv(out_csv, index=False)
        print(f"\nCSV: {out_csv}")

        pivot = df_all.pivot_table(index='Cohort', columns='Leaf',
                                   values=['F1_Macro', 'ROC_AUC', 'Gap_F1'])
        print("\n=== CONFRONTO ===")
        print(pivot.to_string())

        plot_sweep(df_all)


if __name__ == "__main__":
    main()
