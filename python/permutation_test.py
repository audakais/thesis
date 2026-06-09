"""Permutation test: shuffle labels N times and compare F1/AUC to real labels."""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

INPUT_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
N_FOLDS    = 5
FOLD_FEAT  = 500
N_PERM     = 100
META_COLS  = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
              'sample_type', 'target']
COHORTS    = ["TCGA_LUAD", "TCGA_PRAD"]  # most suspicious: AUC=1.0 and lowest F1


def cv_f1_auc(X_arr, y_arr, g_arr, seed=42):
    gkf   = GroupKFold(n_splits=N_FOLDS)
    smote = SMOTE(random_state=seed)
    f1s, aucs = [], []
    for train_idx, test_idx in gkf.split(X_arr, y_arr, groups=g_arr):
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_te, y_te = X_arr[test_idx],  y_arr[test_idx]
        if len(np.unique(y_te)) < 2:
            continue
        top_idx = np.argsort(np.var(X_tr, axis=0))[::-1][:FOLD_FEAT]
        X_tr_f, X_te_f = X_tr[:, top_idx], X_te[:, top_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        X_tr_s, y_tr_s = smote.fit_resample(X_tr_f, y_tr)
        model = RandomForestClassifier(n_estimators=250, max_depth=7,
                                       min_samples_leaf=5,
                                       random_state=seed, n_jobs=-1)
        model.fit(X_tr_s, y_tr_s)
        y_prob = model.predict_proba(X_te_f)[:, 1]
        y_pred = model.predict(X_te_f)
        f1s.append(f1_score(y_te, y_pred, average='macro'))
        aucs.append(roc_auc_score(y_te, y_prob))
    return np.mean(f1s), np.mean(aucs)


for cohort in COHORTS:
    path = f"{INPUT_DIR}/dataset_{cohort}.csv"
    df   = pd.read_csv(path)
    y    = df['target'].astype(int).values
    g    = df['submitter_id'].astype(str).str[:12].values
    X    = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
    X    = np.log2(X.apply(pd.to_numeric, errors='coerce').fillna(0).values + 1.0)

    real_f1, real_auc = cv_f1_auc(X, y, g)

    perm_f1s, perm_aucs = [], []
    rng = np.random.default_rng(0)
    for i in range(N_PERM):
        y_perm = rng.permutation(y)
        f1, auc = cv_f1_auc(X, y_perm, g, seed=i)
        perm_f1s.append(f1)
        perm_aucs.append(auc)

    perm_f1s  = np.array(perm_f1s)
    perm_aucs = np.array(perm_aucs)
    p_f1  = (perm_f1s  >= real_f1).mean()
    p_auc = (perm_aucs >= real_auc).mean()

    print(f"\n=== {cohort} ===")
    print(f"  Real    F1={real_f1:.4f}  AUC={real_auc:.4f}")
    print(f"  Perm    F1={perm_f1s.mean():.4f}+-{perm_f1s.std():.4f}  "
          f"AUC={perm_aucs.mean():.4f}+-{perm_aucs.std():.4f}")
    print(f"  p-value F1={p_f1:.3f}  AUC={p_auc:.3f}  "
          f"({'SIGNIFICANT' if p_f1 < 0.05 else 'not significant'})")
