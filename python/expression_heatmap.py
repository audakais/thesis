"""
Heatmap of top-20 RF biomarker expression (log2 TPM+1) across tumor and
normal samples for each TCGA cohort.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TOP_N       = 20

os.makedirs(FIGURES_DIR, exist_ok=True)

# Build Ensembl → gene symbol mapping
import mygene
_named = os.path.join(OUTPUT_DIR, "Biomarkers_Final_Named.csv")
symbol_map = {}
if os.path.exists(_named):
    _nm = pd.read_csv(_named)
    if 'Gene' in _nm.columns and 'symbol' in _nm.columns:
        symbol_map = dict(zip(_nm['Gene'], _nm['symbol'].fillna('')))

def resolve_symbols(ensg_list):
    missing = [g for g in ensg_list if not symbol_map.get(g)]
    if missing:
        mg = mygene.MyGeneInfo()
        res = mg.querymany(missing, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
        for r in res:
            if 'symbol' in r:
                symbol_map[r['query']] = r['symbol']
    return [symbol_map.get(g, '') or g for g in ensg_list]

files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if '_vs_gtex' not in f]

for fp in files:
    cohort  = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    rf_path = os.path.join(OUTPUT_DIR, f"Biomarkers_RF_{cohort}.csv")
    if not os.path.exists(rf_path):
        continue

    df       = pd.read_csv(fp)
    top_genes = pd.read_csv(rf_path).head(TOP_N)['Gene'].tolist()
    top_genes = [g for g in top_genes if g in df.columns]
    if not top_genes:
        continue

    X   = np.log2(df[top_genes].apply(pd.to_numeric, errors='coerce').fillna(0) + 1.0)
    y   = df['target'].astype(int)

    # Sample max 100 per class for readability
    norm_idx  = y[y == 0].index[:100]
    tumor_idx = y[y == 1].index[:100]
    sel_idx   = norm_idx.tolist() + tumor_idx.tolist()

    plot_df = X.loc[sel_idx].T
    plot_df.index = resolve_symbols(list(plot_df.index))
    labels  = ['Normal'] * len(norm_idx) + ['Tumor'] * len(tumor_idx)

    col_colors = ['#2196F3' if l == 'Normal' else '#F44336' for l in labels]

    g = sns.clustermap(
        plot_df, col_colors=col_colors,
        cmap='RdBu_r', center=0, z_score=0,
        figsize=(12, 6), yticklabels=True, xticklabels=False,
        dendrogram_ratio=0.1, cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    g.ax_heatmap.set_title(f'{cohort} — top-{TOP_N} RF biomarkers', pad=12)
    g.ax_heatmap.set_ylabel('Gene')

    from matplotlib.patches import Patch
    g.ax_col_dendrogram.legend(
        handles=[Patch(color='#2196F3', label='Normal'),
                 Patch(color='#F44336', label='Tumor')],
        loc='center', ncol=2, fontsize=9
    )

    out = os.path.join(FIGURES_DIR, f"heatmap_{cohort}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

