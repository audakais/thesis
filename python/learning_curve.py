"""
Learning curves for all TCGA cohorts (focus: PRAD and STAD).
Shows that low Recall Normal is due to data scarcity, not model failure.
Uses same pipeline as randomforest.py: variance top-500, SMOTE on training fold,
RF n_estimators=250, max_depth=7, min_samples_leaf=5.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

DATASET_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR   = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR  = os.path.join(OUTPUT_DIR, "figures")
META_COLS    = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
                'sample_type', 'target']
FOLD_FEATURES = 500

os.makedirs(FIGURES_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if '_vs_gtex' not in f]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

rng   = np.random.default_rng(42)
ax_idx = 0

for fp in files:
    cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    df     = pd.read_csv(fp)
    if df['target'].nunique() < 2:
        continue

    groups  = df['submitter_id'].astype(str).str[:12].values
    y_arr   = df['target'].astype(int).values
    X_df    = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
    X_df    = X_df.apply(pd.to_numeric, errors='coerce')
    X_arr   = np.log2(X_df.fillna(0).values + 1.0)

    n_fold_feat = min(FOLD_FEATURES, X_arr.shape[1])
    gkf         = GroupKFold(n_splits=5)

    n_train_max = int(len(X_arr) * 4 / 5)
    sizes       = np.unique(np.round(np.linspace(int(n_train_max * 0.2),
                                                  n_train_max, 6)).astype(int))

    all_train = np.zeros((len(sizes), 5))
    all_val   = np.zeros((len(sizes), 5))

    folds = list(gkf.split(X_arr, y_arr, groups=groups))

    for si, size in enumerate(sizes):
        for fi, (train_idx, val_idx) in enumerate(folds):
            # Random subset of training indices of length `size`
            sub = rng.choice(train_idx, size=min(size, len(train_idx)), replace=False)

            X_tr, y_tr = X_arr[sub],      y_arr[sub]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]

            # Variance-based feature selection on training subset
            top_idx = np.argsort(np.var(X_tr, axis=0))[::-1][:n_fold_feat]
            X_tr_f  = X_tr[:, top_idx]
            X_val_f = X_val[:, top_idx]

            # SMOTE only if both classes present and enough minority samples
            if len(np.unique(y_tr)) >= 2 and np.bincount(y_tr).min() >= 2:
                try:
                    X_tr_s, y_tr_s = SMOTE(random_state=42).fit_resample(X_tr_f, y_tr)
                except Exception:
                    X_tr_s, y_tr_s = X_tr_f, y_tr
            else:
                X_tr_s, y_tr_s = X_tr_f, y_tr

            model = RandomForestClassifier(n_estimators=250, max_depth=7,
                                           min_samples_leaf=5,
                                           random_state=42, n_jobs=-1)
            model.fit(X_tr_s, y_tr_s)

            all_train[si, fi] = f1_score(y_tr, model.predict(X_tr_f), average='macro')
            all_val[si, fi]   = f1_score(y_val, model.predict(X_val_f), average='macro')

    ax = axes[ax_idx]
    ax.plot(sizes, all_train.mean(1), 'o-', color='steelblue', label='Train')
    ax.fill_between(sizes,
                    all_train.mean(1) - all_train.std(1),
                    all_train.mean(1) + all_train.std(1), alpha=0.2, color='steelblue')
    ax.plot(sizes, all_val.mean(1), 'o-', color='tomato', label='Validation')
    ax.fill_between(sizes,
                    all_val.mean(1) - all_val.std(1),
                    all_val.mean(1) + all_val.std(1), alpha=0.2, color='tomato')
    ax.set_title(cohort, fontsize=10)
    ax.set_xlabel('Training samples'); ax.set_ylabel('F1 Macro')
    ax.set_ylim(0.5, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"  {cohort} done")
    ax_idx += 1

for j in range(ax_idx, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Learning Curves — TCGA cohorts (5-fold GroupKFold)', fontsize=13)
plt.tight_layout()
out = os.path.join(FIGURES_DIR, "learning_curves.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

