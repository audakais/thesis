import os
base_dir = os.path.dirname(os.path.abspath(__file__))
"""Confusion matrices (OOF) for selected cohorts — standalone, no pipeline impact."""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

INPUT_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES    = os.path.join(OUTPUT_DIR, "figures")
META_COLS  = ['sample_id','sample_uuid','submitter_id','cancer_project','sample_type','target']
COHORTS    = ["TCGA_PRAD", "TCGA_THCA"]
N_FOLDS, FOLD_FEAT = 5, 500

for cohort in COHORTS:
    path = os.path.join(INPUT_DIR, f"dataset_{cohort}.csv")
    df   = pd.read_csv(path)
    y    = df['target'].astype(int)
    g    = df['submitter_id'].astype(str).str[:12]
    X    = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
    X    = np.log2(X.apply(pd.to_numeric, errors='coerce').fillna(0).values + 1.0)

    gkf   = GroupKFold(n_splits=N_FOLDS)
    smote = SMOTE(random_state=42)
    y_arr, g_arr = y.values, g.values
    all_true, all_pred = [], []

    for train_idx, test_idx in gkf.split(X, y_arr, groups=g_arr):
        X_tr, y_tr = X[train_idx], y_arr[train_idx]
        X_te, y_te = X[test_idx],  y_arr[test_idx]
        top_idx    = np.argsort(np.var(X_tr, axis=0))[::-1][:FOLD_FEAT]
        X_tr_f, X_te_f = X_tr[:, top_idx], X_te[:, top_idx]
        X_tr_s, y_tr_s = smote.fit_resample(X_tr_f, y_tr)
        model = RandomForestClassifier(n_estimators=250, max_depth=7,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_tr_s, y_tr_s)
        all_true.extend(y_te)
        all_pred.extend(model.predict(X_te_f))

    cm     = confusion_matrix(all_true, all_pred)
    tn, fp, fn, tp = cm.ravel()
    labels = ['Normal (0)', 'Tumor (1)']

    print(f"\n=== {cohort} — OOF Confusion Matrix ===")
    print(f"  {'':15} Pred Normal  Pred Tumor")
    print(f"  {'True Normal':15} {tn:>10}   {fp:>9}")
    print(f"  {'True Tumor':15} {fn:>10}   {tp:>9}")
    print(f"  Specificity (Recall Normal): {tn/(tn+fp):.4f}")
    print(f"  Sensitivity (Recall Tumor) : {tp/(tp+fn):.4f}")
    print(f"  False Positive Rate        : {fp/(tn+fp):.4f}")
    print(f"  False Negative Rate        : {fn/(tp+fn):.4f}")
    print(classification_report(all_true, all_pred,
                                target_names=['Normal','Tumor'], digits=4))

    # Figure
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel('Predicted label', fontsize=13)
    ax.set_ylabel('True label', fontsize=13)
    ax.set_title(f'{cohort}\nOOF Confusion Matrix (5-fold GroupKFold)', fontsize=14, pad=12)
    n_tot = cm.sum()
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / n_tot * 100
            ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                    color='white' if val > cm.max()/2 else 'black', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(FIGURES, f"confusion_matrix_{cohort}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
