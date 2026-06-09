"""DT pipeline: GroupKFold (patient-level) + SMOTE + leaf=100. Same CV as RF."""
import pandas as pd, numpy as np, csv, os, warnings
base_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

BASE  = os.path.join(base_dir, "ml_dataset_project_batches")
OUT   = os.path.join(os.path.dirname(base_dir), "outputs")
FIGS  = OUT + '/figures'
META  = ['sample_id','sample_uuid','submitter_id','cancer_project','sample_type','target']
LEAF  = 100
COHORTS = ['TCGA_BRCA','TCGA_KIRC','TCGA_LUAD','TCGA_LUSC','TCGA_PRAD','TCGA_STAD','TCGA_THCA']
imp = SimpleImputer(strategy='median')

summary = []
for cohort in COHORTS:
    df = pd.read_csv(f'{BASE}/dataset_{cohort}.csv')
    feats = [c for c in df.columns if c not in META]
    X = imp.fit_transform(df[feats].values)
    y = df['target'].values
    groups = df['submitter_id'].str[:12].values

    gkf = GroupKFold(n_splits=5)
    f1s, imps_list = [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        k = max(1, min(4, int((ytr==0).sum())-1))
        Xtr_s, ytr_s = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr, ytr)
        dt = DecisionTreeClassifier(max_depth=None, min_samples_leaf=LEAF,
                                     class_weight='balanced', random_state=42)
        dt.fit(Xtr_s, ytr_s)
        f1s.append(f1_score(yte, dt.predict(Xte), average='macro'))
        imps_list.append(dt.feature_importances_)

    f1_cv = float(np.mean(f1s)); f1_std = float(np.std(f1s))

    mean_imp = np.mean(imps_list, axis=0)
    ranked = sorted(zip(feats, mean_imp), key=lambda x: -x[1])
    with open(f'{OUT}/Biomarkers_DT_{cohort}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['Gene','Importance'])
        for gene, iv in ranked[:50]: w.writerow([gene, iv])

    Xtr_h, Xte_h, ytr_h, yte_h = train_test_split(X, y, test_size=0.3,
                                                     stratify=y, random_state=42)
    k = max(1, min(4, int((ytr_h==0).sum())-1))
    Xtr_hs, ytr_hs = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr_h, ytr_h)
    dt_h = DecisionTreeClassifier(max_depth=None, min_samples_leaf=LEAF,
                                   class_weight='balanced', random_state=42)
    dt_h.fit(Xtr_hs, ytr_hs)
    pred_h = dt_h.predict(Xte_h)
    tn, fp, fn, tp = confusion_matrix(yte_h, pred_h).ravel()
    acc_ho = float(accuracy_score(yte_h, pred_h))
    f1_ho  = float(f1_score(yte_h, pred_h, average='macro'))
    spec = round(tn/(tn+fp), 3) if (tn+fp) > 0 else 0
    sens = round(tp/(tp+fn), 3) if (tp+fn) > 0 else 0

    cm_arr = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal (0)','Tumor (1)'],
                yticklabels=['Normal (0)','Tumor (1)'])
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
    ax.set_title('%s\nDT (GroupKFold, SMOTE, min_samples_leaf=100) — 30%% hold-out\n'
                 'Acc=%.4f  MacroF1=%.4f' % (cohort, acc_ho, f1_ho))
    plt.tight_layout()
    plt.savefig(f'{FIGS}/confusion_matrix_DT_{cohort}.png', dpi=150, bbox_inches='tight')
    plt.close()

    summary.append({'Cohort':cohort,'F1_Macro':round(f1_cv,4),'F1_Std':round(f1_std,4),
                    'Acc_HO':round(acc_ho,4),'F1_HO':round(f1_ho,4),
                    'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Spec':spec,'Sens':sens})
    print(f'{cohort:15s} CV_F1={f1_cv:.4f}+-{f1_std:.4f}  '
          f'HO Acc={acc_ho:.4f} F1={f1_ho:.4f}  TN={tn} FP={fp} FN={fn} TP={tp}')

with open(OUT+'/Summary_DT_GroupKFold_SMOTE_leaf100.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    w.writeheader(); w.writerows(summary)
print('Done.')
