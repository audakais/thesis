import os
base_dir = os.path.dirname(os.path.abspath(__file__))
"""
Decision Tree biomarker selector -- no max_depth constraint.
Identical to decision_tree_analysis.py except:
  - max_depth removed (unrestricted tree)
  - outputs Biomarkers_DT_TCGA_*.csv (Gene, Importance) per cohort
    matching the format of Biomarkers_RF_TCGA_*.csv
Used exclusively for FFNN feature selection.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")

META_COLS = ['sample_id', 'sample_uuid', 'submitter_id',
             'cancer_project', 'sample_type', 'target']

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if '_vs_gtex' not in f]

for fp in files:
    cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    df = pd.read_csv(fp)

    if df['target'].nunique() < 2:
        print("%s: single class, skip." % cohort)
        continue

    y = df['target']
    X = df.drop(columns=[c for c in META_COLS if c in df.columns]).fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    dt = DecisionTreeClassifier(
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
        # max_depth intentionally omitted
    )
    dt.fit(X_tr, y_tr)
    y_pred = dt.predict(X_te)

    acc   = accuracy_score(y_te, y_pred)
    mf1   = f1_score(y_te, y_pred, average='macro', zero_division=0)
    npos  = int(np.sum(dt.feature_importances_ > 0))
    depth = dt.get_depth()

    print("%s: depth=%d  N_genes=%d  Acc=%.4f  MacroF1=%.4f" %
          (cohort, depth, npos, acc, mf1))

    # Save in same format as Biomarkers_RF_*.csv (Gene, Importance)
    out_df = pd.DataFrame({
        'Gene':       X.columns,
        'Importance': dt.feature_importances_
    }).sort_values('Importance', ascending=False)

    out_path = os.path.join(OUTPUT_DIR, "Biomarkers_DT_%s.csv" % cohort)
    out_df.to_csv(out_path, index=False)
    print("  Saved: %s" % out_path)
