"""
Unbiased cancer risk prediction for healthy GTEx individuals.

Method: 5-fold CV where both GTEx healthy AND TCGA tumor samples are split
across folds. Each fold's test set contains both classes, enabling full
classification metrics (precision, recall, F1, specificity). TCGA tumor
samples held out per fold are replaced by the remaining 80% during training.

Run AFTER gtex_integration.py and randomforest.py.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

DATASET_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR   = os.path.join(os.path.dirname(base_dir), "outputs")
TOP_N        = 50
N_FOLDS      = 5
RANDOM_STATE = 42

TISSUE_MAP = {
    "TCGA_BRCA": "Breast - Mammary Tissue",
    "TCGA_KIRC": "Kidney - Cortex",
    "TCGA_LUAD": "Lung",
    "TCGA_LUSC": "Lung",
    "TCGA_OV":   "Ovary",
    "TCGA_PRAD": "Prostate",
    "TCGA_STAD": "Stomach",
    "TCGA_THCA": "Thyroid",
}


def run_cohort(cohort, vs_path, rf_path):
    df = pd.read_csv(vs_path)
    if df['target'].nunique() < 2:
        print(f"  SKIP: single-class dataset.")
        return None

    top_genes = pd.read_csv(rf_path).head(TOP_N)['Gene'].tolist()
    top_genes = [g for g in top_genes if g in df.columns]
    if not top_genes:
        print(f"  SKIP: no top genes found.")
        return None

    X = np.log2(df[top_genes].apply(pd.to_numeric, errors='coerce').fillna(0) + 1.0).values
    y = df['target'].astype(int).values

    # Stratified K-fold on the full dataset (both TCGA tumor and GTEx healthy)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_probs      = np.zeros(len(df))
    all_preds      = np.zeros(len(df), dtype=int)
    fold_reports   = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = RandomForestClassifier(
            n_estimators=250, max_depth=7,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X[train_idx], y[train_idx])

        probs = model.predict_proba(X[test_idx])[:, 1]
        preds = model.predict(X[test_idx])
        all_probs[test_idx] = probs
        all_preds[test_idx] = preds

        rep = classification_report(
            y[test_idx], preds,
            target_names=['GTEx Healthy', 'TCGA Tumor'],
            output_dict=True, zero_division=0
        )
        fold_reports.append(rep)

    # Aggregate metrics across folds
    def mean_metric(key, sub):
        return round(float(np.mean([r[key][sub] for r in fold_reports])), 4)

    print(f"  --- 5-fold CV metrics (mean across folds) ---")
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'GTEx Healthy':<14} {mean_metric('GTEx Healthy','precision'):>10.4f} "
          f"{mean_metric('GTEx Healthy','recall'):>8.4f} "
          f"{mean_metric('GTEx Healthy','f1-score'):>8.4f}")
    print(f"  {'TCGA Tumor':<14} {mean_metric('TCGA Tumor','precision'):>10.4f} "
          f"{mean_metric('TCGA Tumor','recall'):>8.4f} "
          f"{mean_metric('TCGA Tumor','f1-score'):>8.4f}")
    print(f"  {'F1 Macro':<14} {mean_metric('macro avg','f1-score'):>28.4f}")
    print(f"  {'Accuracy':<14} {mean_metric('accuracy',''):>28.4f}" if False else
          f"  Accuracy (mean): {round(float(np.mean([r['accuracy'] for r in fold_reports])),4):.4f}")

    # GTEx-specific output
    gtex_mask  = y == 0
    gtex_probs = all_probs[gtex_mask]
    n_high     = int((gtex_probs >= 0.5).sum())
    print(f"  GTEx samples: {gtex_mask.sum()} | High-risk (>=0.5): {n_high} ({100*n_high/gtex_mask.sum():.1f}%)")
    print(f"  Mean tumor prob (GTEx): {gtex_probs.mean():.4f} +/- {gtex_probs.std():.4f}")

    gtex_df = df[gtex_mask].copy()
    result  = pd.DataFrame({
        'SAMPID':          gtex_df['sample_id'].values,
        'GTEx_Tissue':     TISSUE_MAP.get(cohort, cohort),
        'Cohort':          cohort,
        'Tumor_Prob':      gtex_probs.round(4),
        'Predicted_Class': all_preds[gtex_mask],
    })
    result.attrs.update({
        'precision_healthy': mean_metric('GTEx Healthy', 'precision'),
        'recall_healthy':    mean_metric('GTEx Healthy', 'recall'),
        'f1_healthy':        mean_metric('GTEx Healthy', 'f1-score'),
        'precision_tumor':   mean_metric('TCGA Tumor',   'precision'),
        'recall_tumor':      mean_metric('TCGA Tumor',   'recall'),
        'f1_tumor':          mean_metric('TCGA Tumor',   'f1-score'),
        'f1_macro':          mean_metric('macro avg',    'f1-score'),
        'accuracy':          round(float(np.mean([r['accuracy'] for r in fold_reports])), 4),
    })
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vs_files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*_vs_gtex.csv")))
    vs_files = [f for f in vs_files if 'combat' not in f]

    all_results  = []
    summary_rows = []

    for fp in vs_files:
        cohort  = os.path.basename(fp).replace("dataset_", "").replace("_vs_gtex.csv", "")
        rf_path = os.path.join(OUTPUT_DIR, f"Biomarkers_RF_{cohort}.csv")
        if not os.path.exists(rf_path):
            print(f"\n{cohort}: RF biomarkers not found, skipping.")
            continue
        print(f"\n=== {cohort} ===")
        try:
            result = run_cohort(cohort, fp, rf_path)
            if result is not None:
                all_results.append(result)
                probs = result['Tumor_Prob'].values
                a     = result.attrs
                summary_rows.append({
                    'Cohort':            cohort,
                    'GTEx_Tissue':       TISSUE_MAP.get(cohort, cohort),
                    'N_GTEx_Samples':    len(result),
                    'Mean_Tumor_Prob':   round(float(probs.mean()), 4),
                    'Std_Tumor_Prob':    round(float(probs.std()),  4),
                    'High_Risk_N':       int((probs >= 0.5).sum()),
                    'High_Risk_Pct':     round(100 * (probs >= 0.5).mean(), 2),
                    'Prec_Healthy':      a['precision_healthy'],
                    'Recall_Healthy':    a['recall_healthy'],
                    'F1_Healthy':        a['f1_healthy'],
                    'Prec_Tumor':        a['precision_tumor'],
                    'Recall_Tumor':      a['recall_tumor'],
                    'F1_Tumor':          a['f1_tumor'],
                    'F1_Macro':          a['f1_macro'],
                    'Accuracy':          a['accuracy'],
                })
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        pd.concat(all_results).to_csv(
            os.path.join(OUTPUT_DIR, "GTEx_Predictions.csv"), index=False)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out = os.path.join(OUTPUT_DIR, "GTEx_Predictions_Summary.csv")
        summary.to_csv(out, index=False)
        print(f"\n=== SUMMARY ===")
        print(summary.to_string(index=False))
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()

