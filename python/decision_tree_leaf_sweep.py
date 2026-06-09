import os
base_dir = os.path.dirname(os.path.abspath(__file__))
"""
Decision Tree min_samples_leaf sweep: 10, 20, 50, 100
Evaluates each cohort on 30% stratified hold-out.
Outputs: Summary_DT_LeafSweep.csv + figures/dt_leaf_sweep.png
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

BASE_DIR    = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LEAF_VALUES = [10, 20, 50, 100]

META_COLS = ['sample_id', 'sample_uuid', 'submitter_id',
             'cancer_project', 'sample_type', 'target']

os.makedirs(FIGURES_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if '_vs_gtex' not in f]

rows = []

for fp in files:
    cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    df = pd.read_csv(fp)
    if df['target'].nunique() < 2:
        continue

    y = df['target']
    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    X = X.fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    print("--- %s (N=%d) ---" % (cohort, len(df)))

    for leaf in LEAF_VALUES:
        dt = DecisionTreeClassifier(
            max_depth=7,
            min_samples_split=20,
            min_samples_leaf=leaf,
            random_state=42,
            class_weight='balanced'
        )
        dt.fit(X_tr, y_tr)
        y_pred = dt.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        mf1  = f1_score(y_te, y_pred, average='macro',    zero_division=0)
        wf1  = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        npos = int(np.sum(dt.feature_importances_ > 0))

        print("  leaf=%3d  Acc=%.4f  MacroF1=%.4f  N_genes=%d" %
              (leaf, acc, mf1, npos))

        rows.append({
            'Cohort':           cohort,
            'min_samples_leaf': leaf,
            'Accuracy':         round(acc, 4),
            'MacroF1':          round(mf1, 4),
            'WeightedF1':       round(wf1, 4),
            'N_genes_nonzero':  npos,
        })

df_res = pd.DataFrame(rows)
df_res.to_csv(os.path.join(OUTPUT_DIR, "Summary_DT_LeafSweep.csv"), index=False)
print("\nSaved: Summary_DT_LeafSweep.csv")
print(df_res.to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────────────────
cohorts = df_res['Cohort'].unique()
n       = len(cohorts)
colors  = ['#1565C0', '#0288D1', '#4CAF50', '#F57C00']
markers = ['o', 's', '^', 'D']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, ylim in zip(axes,
                             ['Accuracy', 'MacroF1'],
                             [(0.85, 1.01), (0.80, 1.01)]):
    for i, leaf in enumerate(LEAF_VALUES):
        vals = [df_res[(df_res['Cohort']==c) &
                       (df_res['min_samples_leaf']==leaf)][metric].values[0]
                for c in cohorts]
        ax.plot(range(n), vals, color=colors[i], marker=markers[i],
                linewidth=1.5, markersize=6,
                label='min_samples_leaf=%d' % leaf)
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.replace('TCGA_', 'TCGA-') for c in cohorts],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title('DT %s by min_samples_leaf' % metric, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Decision Tree -- min_samples_leaf Sweep', fontsize=12)
plt.tight_layout()
out = os.path.join(FIGURES_DIR, "dt_leaf_sweep.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print("Figure: %s" % out)
