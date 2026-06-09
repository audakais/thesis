"""
Integrates GTEx healthy samples with TCGA tumor samples to build
balanced datasets for cancer prediction (healthy vs. tumor).

Input:
  - GTEx TPM matrix (.gct.gz) and sample attributes (.txt)
  - Existing TCGA per-cohort CSVs (ml_dataset_project_batches/)

Output:
  - ml_dataset_project_batches/dataset_<cohort>_vs_gtex.csv
    (target=0: GTEx healthy, target=1: TCGA tumor)

Run AFTER create_ml_dataset.py.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import gzip
import numpy as np
import pandas as pd

# --- Paths ---
GTEX_TPM_FILE   = os.path.join(os.path.dirname(base_dir), "data", "gtex_raw", "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz")
GTEX_ATTR_FILE  = os.path.join(os.path.dirname(base_dir), "data", "gtex_raw", "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt.txt")
TCGA_DIR        = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR      = TCGA_DIR

# GTEx tissue name -> TCGA cohort mapping
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

MIN_SAMPLES  = 100
RANDOM_STATE = 42


def load_gtex(tpm_path, attr_path, target_tissues):
    print("Loading GTEx attributes...")
    attr = pd.read_csv(attr_path, sep='\t', usecols=['SAMPID', 'SMTSD'])
    attr = attr[attr['SAMPID'].str.startswith('GTEX-')]
    relevant = attr[attr['SMTSD'].isin(target_tissues)].copy()
    print(f"  Relevant samples: {len(relevant)} across {len(target_tissues)} tissues")

    print("Reading GTEx TPM header to identify columns...")
    with gzip.open(tpm_path, 'rt') as f:
        f.readline()
        f.readline()
        header = f.readline().rstrip('\n').split('\t')

    gct_id_set = set(header[2:])
    relevant['GCT_ID'] = relevant['SAMPID'].where(relevant['SAMPID'].isin(gct_id_set))
    relevant = relevant.dropna(subset=['GCT_ID'])
    keep_gct_ids = relevant['GCT_ID'].tolist()
    print(f"  Matched {len(keep_gct_ids)} samples in GCT file")

    usecols = [header[0], header[1]] + keep_gct_ids
    print(f"  Loading {len(keep_gct_ids)} columns...")

    with gzip.open(tpm_path, 'rt') as f:
        f.readline()
        f.readline()
        tpm = pd.read_csv(f, sep='\t', usecols=usecols)

    tpm['Name'] = tpm['Name'].str.split('.').str[0]
    tpm = tpm.drop(columns=['Description'], errors='ignore')
    tpm = tpm.drop_duplicates(subset='Name')
    tpm = tpm.set_index('Name').T
    tpm.index.name = 'SAMPID'
    tpm = tpm.reset_index()

    df = tpm.merge(relevant[['GCT_ID', 'SMTSD']].rename(columns={'GCT_ID': 'SAMPID'}), on='SAMPID')
    print(f"GTEx loaded: {df.shape[0]} samples")
    return df


def build_dataset(cohort, tcga_path, gtex_df, tissue_name):
    print(f"\n=== {cohort} ===")

    # Load TCGA tumor samples only
    tcga = pd.read_csv(tcga_path)
    tcga_tumor = tcga[tcga['target'] == 1].copy()
    if len(tcga_tumor) < MIN_SAMPLES:
        print(f"  SKIP: only {len(tcga_tumor)} tumor samples.")
        return None

    # Filter GTEx for matching tissue
    gtex_tissue = gtex_df[gtex_df['SMTSD'] == tissue_name].copy()
    if len(gtex_tissue) < MIN_SAMPLES:
        print(f"  SKIP: only {len(gtex_tissue)} GTEx samples for '{tissue_name}'.")
        return None

    # Find common ENSG genes between TCGA and GTEx
    tcga_genes = [c for c in tcga_tumor.columns if c.startswith('ENSG')]
    gtex_genes = [c for c in gtex_tissue.columns if c.startswith('ENSG')]
    common_genes = sorted(set(tcga_genes) & set(gtex_genes))
    if len(common_genes) < 100:
        print(f"  SKIP: only {len(common_genes)} common genes.")
        return None
    print(f"  Common genes: {len(common_genes)}")

    # Balance: use min(n_tumor, n_gtex) samples, capped at 500 per class
    n = min(len(tcga_tumor), len(gtex_tissue), 500)
    tcga_sample  = tcga_tumor.sample(n=n, random_state=RANDOM_STATE)
    gtex_sample  = gtex_tissue.sample(n=n, random_state=RANDOM_STATE)

    # Build TCGA block
    tcga_block = tcga_sample[common_genes].copy()
    tcga_block['sample_id']      = tcga_sample['sample_id'].values
    tcga_block['submitter_id']   = tcga_sample['submitter_id'].values
    tcga_block['cancer_project'] = tcga_sample['cancer_project'].values
    tcga_block['sample_type']    = 'Primary Tumor'
    tcga_block['target']         = 1

    # Build GTEx block
    gtex_block = gtex_sample[common_genes].copy()
    gtex_block['sample_id']      = gtex_sample['SAMPID'].values
    gtex_block['submitter_id']   = gtex_sample['SAMPID'].values
    gtex_block['cancer_project'] = cohort
    gtex_block['sample_type']    = 'GTEx Healthy'
    gtex_block['target']         = 0

    meta_cols = ['sample_id', 'submitter_id', 'cancer_project', 'sample_type', 'target']
    combined = pd.concat([
        tcga_block[meta_cols + common_genes],
        gtex_block[meta_cols + common_genes],
    ], ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, f"dataset_{cohort}_vs_gtex.csv")
    combined.to_csv(out_path, index=False)
    print(f"  {n} tumor + {n} healthy -> {out_path} ({combined.shape})")
    return {
        'Cohort':       cohort,
        'GTEx_Tissue':  tissue_name,
        'N_Tumor':      n,
        'N_Healthy':    n,
        'N_Genes':      len(common_genes),
    }


def main():
    target_tissues = set(TISSUE_MAP.values())
    gtex_df = load_gtex(GTEX_TPM_FILE, GTEX_ATTR_FILE, target_tissues)

    summaries = []
    for cohort, tissue in TISSUE_MAP.items():
        tcga_path = os.path.join(TCGA_DIR, f"dataset_{cohort}.csv")
        if not os.path.exists(tcga_path):
            print(f"\n{cohort}: TCGA CSV not found, skipping.")
            continue
        result = build_dataset(cohort, tcga_path, gtex_df, tissue)
        if result:
            summaries.append(result)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        print("\n=== SUMMARY ===")
        print(summary_df.to_string(index=False))
        print("\nDatasets ready. Now run randomforest.py pointing to *_vs_gtex.csv files.")


if __name__ == "__main__":
    main()

