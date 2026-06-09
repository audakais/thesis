"""
Converts GTEx TPM .gz to a compact Parquet file.

Memory-efficient: reads the gz in chunks of 500 genes, keeps only the
tissue-relevant sample columns and the genes needed by any cohort model.
Peak RAM is ~50 MB regardless of the full matrix size.

Run once before predict_gtex.py.
"""
import gzip
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import pandas as pd

GZ       = os.path.join(os.path.dirname(base_dir), "data", "gtex_raw", "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz")
ATTR     = os.path.join(os.path.dirname(base_dir), "data", "gtex_raw", "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt.txt")
OUT      = os.path.join(os.path.dirname(base_dir), "data", "gtex_raw", "gtex_tpm.parquet")
RF_DIR   = os.path.join(os.path.dirname(base_dir), "outputs")
TOP_N    = 50

TISSUES = {
    "Breast - Mammary Tissue", "Kidney - Cortex", "Lung",
    "Ovary", "Prostate", "Stomach", "Thyroid",
}


def collect_needed_genes():
    genes = set()
    for fp in glob.glob(os.path.join(RF_DIR, "Biomarkers_RF_*.csv")):
        genes.update(pd.read_csv(fp).head(TOP_N)['Gene'].tolist())
    print(f"  Needed genes across all cohorts: {len(genes)}")
    return genes


def main():
    # 1. Get relevant sample IDs
    print("Reading GTEx attributes...")
    attr = pd.read_csv(ATTR, sep='\t', usecols=['SAMPID', 'SMTSD'])
    attr = attr[
        attr['SAMPID'].str.startswith('GTEX-') &
        attr['SMTSD'].isin(TISSUES)
    ].drop_duplicates('SAMPID')

    # 2. Read gz header to match sample IDs to columns
    with gzip.open(GZ, 'rt') as f:
        f.readline(); f.readline()
        header = f.readline().rstrip('\n').split('\t')

    sample_set = set(attr['SAMPID'])
    keep_samples = [c for c in header[2:] if c in sample_set]
    print(f"  Tissue-relevant samples in GCT: {len(keep_samples)}")

    # 3. Collect needed genes
    needed_genes = collect_needed_genes()

    # 4. Chunk-read: only keep needed gene rows and relevant sample columns
    usecols = ['Name'] + keep_samples
    chunks  = []
    found   = 0

    print("Chunk-reading gz (500 genes/chunk, only needed genes kept)...")
    with gzip.open(GZ, 'rt') as f:
        f.readline(); f.readline()
        reader = pd.read_csv(
            f, sep='\t', usecols=usecols, chunksize=500,
        )
        for chunk in reader:
            chunk['Name'] = chunk['Name'].str.split('.').str[0]
            chunk = chunk[chunk['Name'].isin(needed_genes)]
            chunk = chunk.drop_duplicates(subset='Name').set_index('Name')
            if not chunk.empty:
                chunks.append(chunk)
                found += len(chunk)

    if not chunks:
        print("ERROR: no needed genes found in GTEx.")
        return

    df = pd.concat(chunks)
    df = df[~df.index.duplicated()]
    print(f"  Kept {len(df)} genes x {len(df.columns)} samples")

    # Transpose to samples x genes for column-efficient parquet reads
    df.T.to_parquet(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

