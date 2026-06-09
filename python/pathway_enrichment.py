"""
Pathway enrichment analysis for top-50 RF biomarkers per cohort.
Uses the Enrichr API (KEGG_2021_Human and GO_Biological_Process_2023).
Outputs top-10 enriched pathways per cohort as CSV and bar plot.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import time
import glob
import numpy as np
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
LIBRARIES   = ["KEGG_2021_Human", "GO_Biological_Process_2023"]
TOP_N       = 50

os.makedirs(FIGURES_DIR, exist_ok=True)

# Load gene name mapping if available
name_map = {}
name_file = os.path.join(OUTPUT_DIR, "BIOMARCATORI_FINALI_NOMINATIVI.csv")
if os.path.exists(name_file):
    nm = pd.read_csv(name_file)
    sym_col = next((c for c in nm.columns if c.lower() == 'symbol'), None)
    if 'Gene' in nm.columns and sym_col:
        name_map = dict(zip(nm['Gene'], nm[sym_col]))


def ensg_to_symbol(genes):
    return [name_map.get(g, g) for g in genes]


def enrichr_query(gene_symbols, library):
    payload = {'list': '\n'.join(gene_symbols), 'description': library}
    r = requests.post(f"{ENRICHR_URL}/addList", files=payload, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    user_id = r.json().get('userListId')
    time.sleep(0.5)
    r2 = requests.get(f"{ENRICHR_URL}/enrich",
                      params={'userListId': user_id, 'backgroundType': library},
                      timeout=30)
    if r2.status_code != 200:
        return pd.DataFrame()
    data = r2.json().get(library, [])
    rows = [{'Rank': d[0], 'Term': d[1], 'P_value': d[2],
             'Adj_P': d[6], 'Overlap': d[8]} for d in data]
    return pd.DataFrame(rows)


files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "Biomarkers_RF_TCGA_*.csv")))
all_results = []

for fp in files:
    cohort = os.path.basename(fp).replace("Biomarkers_RF_", "").replace(".csv", "")
    genes  = pd.read_csv(fp).head(TOP_N)['Gene'].tolist()
    symbols = ensg_to_symbol(genes)
    # Filter out ENSG IDs that weren't mapped
    symbols = [str(s) for s in symbols if isinstance(s, str) and not s.startswith('ENSG')]
    if len(symbols) < 5:
        print(f"{cohort}: too few gene symbols mapped, skipping.")
        continue

    print(f"\n=== {cohort} ({len(symbols)} symbols) ===")
    cohort_rows = []
    for lib in LIBRARIES:
        try:
            df = enrichr_query(symbols, lib)
            if df.empty:
                continue
            df['Cohort']  = cohort
            df['Library'] = lib
            top = df.head(10)
            cohort_rows.append(top)
            print(f"  {lib}: top term = {top.iloc[0]['Term']} (adj.p={top.iloc[0]['Adj_P']:.3e})")
        except Exception as e:
            print(f"  {lib}: ERROR {e}")

    if not cohort_rows:
        continue

    cohort_df = pd.concat(cohort_rows, ignore_index=True)
    all_results.append(cohort_df)

    # Bar plot for KEGG
    kegg = cohort_df[cohort_df['Library'] == 'KEGG_2021_Human'].head(10)
    if not kegg.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(kegg['Term'][::-1], -kegg['Adj_P'].apply(lambda x: np.log10(x + 1e-300))[::-1],
                color='steelblue')
        ax.set_xlabel('-log10(adj. p-value)')
        ax.set_title(f'{cohort} — KEGG pathway enrichment (top-10)')
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"pathway_{cohort}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {out}")

if all_results:
    out_csv = os.path.join(OUTPUT_DIR, "Pathway_Enrichment.csv")
    pd.concat(all_results).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

