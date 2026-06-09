"""
Interpretable decision tree on top-N RF biomarkers per cohort.

For each dataset_TCGA_*.csv:
  1. Reads the top-N genes from Biomarkers_RF_<cohort>.csv produced by randomforest.py.
  2. Trains a DecisionTree (depth=3) on those genes only (log2-TPM, balanced weights).
  3. Validates with 5-fold GroupKFold by patient.
  4. Saves the rules as plaintext (export_text).

Output: outputs/interpretable_rules/rules_<cohort>.txt + summary CSV.
Must be run AFTER randomforest.py.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupKFold, cross_val_score

warnings.filterwarnings('ignore')

INPUT_DIR     = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR    = os.path.join(os.path.dirname(base_dir), "outputs")
RULES_DIR     = os.path.join(OUTPUT_DIR, "interpretable_rules")
META_COLS     = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
                 'sample_type', 'target']

TOP_N_GENES   = 50   # number of RF genes to include in the tree
TREE_DEPTH    = None  # no depth limit
N_FOLDS       = 5


def run_cohort(cohort, dataset_path, rf_path):
    df = pd.read_csv(dataset_path)
    if df['target'].nunique() < 2:
        print(f"  SKIP: single-class dataset.")
        return None

    rf = pd.read_csv(rf_path)
    if not {'Gene', 'Importance'}.issubset(rf.columns):
        print(f"  SKIP: unexpected RF schema.")
        return None

    top_genes = rf.head(TOP_N_GENES)['Gene'].tolist()
    top_genes = [g for g in top_genes if g in df.columns]
    if not top_genes:
        print(f"  SKIP: no RF genes found in dataset.")
        return None

    groups = df['submitter_id'].astype(str).str[:12]
    if groups.nunique() < N_FOLDS:
        print(f"  SKIP: only {groups.nunique()} patients < {N_FOLDS} folds.")
        return None

    y = df['target'].astype(int)
    X = df[top_genes].apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    X = np.log2(X + 1.0)
    X = X.fillna(X.median(numeric_only=True))

    clf = DecisionTreeClassifier(
        max_depth=TREE_DEPTH,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
    )
    gkf = GroupKFold(n_splits=N_FOLDS)
    f1 = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring='f1_macro')

    # Fit on full data to export human-readable rules
    clf.fit(X, y)
    rules = export_text(
        clf,
        feature_names=list(X.columns),
        class_names=['Normal', 'Tumor'],
    )

    out_path = os.path.join(RULES_DIR, f"rules_{cohort}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Cohort: {cohort}\n")
        f.write(f"Top {len(X.columns)} RF features (depth={TREE_DEPTH}, log2-TPM, balanced)\n")
        f.write(f"5-fold GroupKFold F1 macro: {f1.mean():.4f} (+/- {f1.std():.4f})\n")
        f.write("=" * 60 + "\n")
        f.write(rules)
    print(f"  F1={f1.mean():.4f}+/-{f1.std():.4f} -> {out_path}")
    return {
        'Cohort':   cohort,
        'F1_Macro': round(float(f1.mean()), 4),
        'F1_Std':   round(float(f1.std()), 4),
        'TopN':     len(X.columns),
        'Depth':    TREE_DEPTH,
        'Rules':    out_path,
    }


def main():
    os.makedirs(RULES_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "dataset_TCGA_*.csv")))
    if not files:
        print(f"No dataset_TCGA_*.csv files found in {INPUT_DIR}.")
        return

    summaries = []
    for fp in files:
        cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
        rf_path = os.path.join(OUTPUT_DIR, f"Biomarkers_RF_{cohort}.csv")
        if not os.path.exists(rf_path):
            print(f"\n{cohort}: {os.path.basename(rf_path)} not found "
                  "(run randomforest.py first). Skipping.")
            continue
        print(f"\n=== {cohort} ===")
        try:
            s = run_cohort(cohort, fp, rf_path)
            if s is not None:
                summaries.append(s)
        except Exception as e:
            print(f"  ERROR: {e}")

    if summaries:
        out = os.path.join(OUTPUT_DIR, "Summary_InterpretableTrees.csv")
        pd.DataFrame(summaries).to_csv(out, index=False)
        print(f"\n=== SUMMARY -> {out} ===")
        print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

