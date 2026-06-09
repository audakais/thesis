import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

INPUT_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")

META_COLS = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
             'sample_type', 'target']

N_FOLDS       = 5
FOLD_FEATURES = 500  # per-fold variance selection on training data only


def run_cohort(file_path):
    cohort = os.path.basename(file_path).replace("dataset_", "").replace(".csv", "")
    print(f"\n=== {cohort} ===")

    df = pd.read_csv(file_path)
    print(f"  Shape: {df.shape}")

    if 'target' not in df.columns or 'submitter_id' not in df.columns:
        print("  SKIP: required columns missing.")
        return None
    if df['target'].nunique() < 2:
        print(f"  SKIP: single-class dataset.")
        return None

    groups = df['submitter_id'].astype(str).str[:12]
    if groups.nunique() < N_FOLDS:
        print(f"  SKIP: only {groups.nunique()} unique patients < {N_FOLDS} folds.")
        return None

    y = df['target'].astype(int)
    X = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    X = np.log2(X + 1.0)
    X = X.fillna(X.median(numeric_only=True))

    n_normal = int((y == 0).sum())
    n_tumor  = int((y == 1).sum())
    n_fold_feat = min(FOLD_FEATURES, X.shape[1])
    print(f"  Patients={groups.nunique()} | Normal={n_normal} | Tumor={n_tumor} "
          f"| Pool={X.shape[1]} | FoldFeatures={n_fold_feat}")

    gkf   = GroupKFold(n_splits=N_FOLDS)
    smote = SMOTE(random_state=42)
    X_arr = X.values
    y_arr = y.values
    g_arr = groups.values
    col_names = np.array(X.columns)

    fold_acc, fold_f1, fold_auc, fold_r0, fold_r1, train_f1 = [], [], [], [], [], []
    # Accumulate importance in full 1500-gene space (zeros for non-selected genes)
    fold_importances = np.zeros((N_FOLDS, X_arr.shape[1]))

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_arr, y_arr, groups=g_arr)):
        X_tr_orig, y_tr_orig = X_arr[train_idx], y_arr[train_idx]
        X_te_full, y_te      = X_arr[test_idx],  y_arr[test_idx]

        # Feature selection: variance on training data only
        top_idx = np.argsort(np.var(X_tr_orig, axis=0))[::-1][:n_fold_feat]
        X_tr_f  = X_tr_orig[:, top_idx]
        X_te_f  = X_te_full[:, top_idx]

        X_tr_s, y_tr_s = smote.fit_resample(X_tr_f, y_tr_orig)

        model = RandomForestClassifier(n_estimators=250, max_depth=7,
                                       min_samples_leaf=5,
                                       random_state=42, n_jobs=-1)
        model.fit(X_tr_s, y_tr_s)

        y_pred_tr = model.predict(X_tr_f)
        y_pred    = model.predict(X_te_f)
        y_prob    = model.predict_proba(X_te_f)[:, 1]

        train_f1.append(f1_score(y_tr_orig, y_pred_tr, average='macro'))
        fold_acc.append(accuracy_score(y_te, y_pred))
        fold_f1.append(f1_score(y_te, y_pred, average='macro'))
        fold_auc.append(roc_auc_score(y_te, y_prob))
        fold_r0.append(recall_score(y_te, y_pred, pos_label=0))
        fold_r1.append(recall_score(y_te, y_pred, pos_label=1))

        fold_importances[fold_i, top_idx] = model.feature_importances_

    summary = {
        'Cohort':        cohort,
        'N_Patients':    groups.nunique(),
        'N_Normal':      n_normal,
        'N_Tumor':       n_tumor,
        'N_Features':    n_fold_feat,
        'Accuracy':      round(float(np.mean(fold_acc)), 4),
        'Accuracy_Std':  round(float(np.std(fold_acc)),  4),
        'F1_Macro':      round(float(np.mean(fold_f1)),  4),
        'ROC_AUC':       round(float(np.mean(fold_auc)), 4),
        'Recall_Normal': round(float(np.mean(fold_r0)),  4),
        'Recall_Tumor':  round(float(np.mean(fold_r1)),  4),
    }
    gap = round(float(np.mean(train_f1)) - float(np.mean(fold_f1)), 4)
    summary['Train_F1'] = round(float(np.mean(train_f1)), 4)
    summary['Gap_F1']   = gap
    print(f"  Acc={summary['Accuracy']}+/-{summary['Accuracy_Std']} "
          f"F1_train={summary['Train_F1']} F1_val={summary['F1_Macro']} gap={gap} "
          f"AUC={summary['ROC_AUC']} Rec0={summary['Recall_Normal']} Rec1={summary['Recall_Tumor']}")

    importances = np.mean(fold_importances, axis=0)
    biomarkers  = pd.DataFrame({'Gene': col_names, 'Importance': importances})
    biomarkers  = biomarkers.sort_values('Importance', ascending=False).reset_index(drop=True)
    out_path    = os.path.join(OUTPUT_DIR, f"Biomarkers_RF_{cohort}.csv")
    biomarkers.to_csv(out_path, index=False)
    print(f"  -> {out_path}")

    del df, X, y
    gc.collect()
    return summary


def main():
    print("=== RANDOM FOREST MULTI-COHORT ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "dataset_TCGA_*.csv")))
    files = [f for f in files if '_vs_gtex' not in f]
    if not files:
        print(f"No dataset_TCGA_*.csv files found in {INPUT_DIR}.")
        return

    summaries = []
    for fp in files:
        try:
            s = run_cohort(fp)
            if s is not None:
                summaries.append(s)
        except Exception as e:
            print(f"  ERROR on {os.path.basename(fp)}: {e}")
            import traceback; traceback.print_exc()

    if summaries:
        df_sum = pd.DataFrame(summaries).sort_values('F1_Macro', ascending=False)
        out    = os.path.join(OUTPUT_DIR, "Summary_RF_AllCohorts.csv")
        df_sum.to_csv(out, index=False)
        print(f"\n=== SUMMARY ===")
        print(df_sum.to_string(index=False))
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
