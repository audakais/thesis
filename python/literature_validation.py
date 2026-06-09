"""
Literature validation: compares the top-N RF biomarkers against curated lists
of known oncogenic drivers for each TCGA cohort.

ENSG -> Gene Symbol mapping strategy:
  1. Use BIOMARCATORI_FINALI_NOMINATIVI.csv (mapping already in repo)
  2. If missing, try mygene (requires `pip install mygene` and internet access)
  3. Otherwise the gene remains unmapped and cannot be matched

Curated lists sourced from:
  - COSMIC Cancer Gene Census v98
  - TCGA marker papers (Cancer Genome Atlas, Nature 2012-2017)
  - Bailey et al. 2018, "Comprehensive Characterization of Cancer Driver Genes"
  - Hoadley et al. 2018, Cell, "Cell-of-origin patterns" (Pan-Cancer Atlas)

Must be run AFTER randomforest.py.
"""
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
import glob

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(base_dir), "outputs")
RF_FILES_GLOB   = os.path.join(OUTPUT_DIR, "Biomarkers_RF_*.csv")
SYMBOL_MAP_FILE = os.path.join(OUTPUT_DIR, "BIOMARCATORI_FINALI_NOMINATIVI.csv")

TOP_N = 50

# Known drivers per cohort (HGNC gene symbols). Non-exhaustive but covers the
# most frequently cited drivers/biomarkers from the literature.
KNOWN_DRIVERS = {
    "TCGA_BRCA": {
        "BRCA1", "BRCA2", "TP53", "ESR1", "ERBB2", "PIK3CA", "MYC", "GATA3",
        "PTEN", "CDH1", "MUC1", "KRT5", "KRT14", "COL10A1", "MMP11", "UBE2C",
        "CAV1", "FOXA1", "MAP3K1", "MAP2K4", "AKT1", "RB1", "CDKN1B", "NCOR1",
    },
    "TCGA_KIRC": {
        "VHL", "PBRM1", "BAP1", "SETD2", "MTOR", "KDM5C", "PIK3CA", "TP53",
        "PTEN", "TSC1", "TSC2", "SFRP1", "CA9", "NDUFA4L2", "ANGPTL4", "EGLN3",
    },
    "TCGA_LUAD": {
        "KRAS", "EGFR", "TP53", "KEAP1", "STK11", "NF1", "BRAF", "MET", "ALK",
        "RET", "ROS1", "ERBB2", "RBM10", "SETD2", "ARID1A", "SMARCA4", "SFTPC",
    },
    "TCGA_LUSC": {
        "TP53", "CDKN2A", "FGFR1", "PIK3CA", "KEAP1", "NFE2L2", "NOTCH1", "SOX2",
        "PTEN", "RB1", "HRAS", "CCND1", "NF1", "FAM107A",
    },
    "TCGA_PRAD": {
        "AR", "TMPRSS2", "ERG", "PTEN", "TP53", "FOXA1", "SPOP", "CDK12",
        "KMT2C", "KMT2D", "ATM", "CHD1", "FANCD2", "KLK3", "NKX3-1",
    },
    "TCGA_STAD": {
        "TP53", "CDH1", "ARID1A", "RHOA", "PIK3CA", "KRAS", "MUC6", "ERBB2",
        "ERBB3", "SMAD4", "APC", "CTNNB1", "MLH1", "CTHRC1",
    },
    "TCGA_THCA": {
        "BRAF", "HRAS", "NRAS", "KRAS", "RET", "PTEN", "TG", "TPO", "TERT",
        "TSHR", "NTRK1", "NTRK3", "TMT1B", "DUOX1", "DUOX2",
    },
    "TCGA_OV": {
        "TP53", "BRCA1", "BRCA2", "RB1", "NF1", "CDK12", "RAD51C", "PALB2",
        "BRIP1", "CCNE1", "MYC", "KRAS", "PIK3CA", "PTEN",
    },
}


def load_symbol_map_from_csv():
    if not os.path.exists(SYMBOL_MAP_FILE):
        return {}
    df = pd.read_csv(SYMBOL_MAP_FILE)
    if not {'Gene', 'symbol'}.issubset(df.columns):
        return {}
    return {str(g): str(s) for g, s in zip(df['Gene'], df['symbol'])
            if pd.notna(g) and pd.notna(s)}


def map_via_mygene(ensg_ids):
    """Best-effort online mapping. Returns dict ENSG -> symbol; empty on failure."""
    try:
        import mygene
    except ImportError:
        print("  mygene not installed. Install with: pip install mygene")
        return {}

    print(f"  Querying mygene for {len(ensg_ids)} ENSG IDs...")
    try:
        mg = mygene.MyGeneInfo()
        res = mg.querymany(
            list(ensg_ids),
            scopes='ensembl.gene',
            fields='symbol',
            species='human',
            verbose=False,
        )
    except Exception as e:
        print(f"  mygene query failed: {e}")
        return {}

    out = {}
    for r in res:
        if isinstance(r, dict) and r.get('symbol') and r.get('query'):
            out[r['query']] = r['symbol']
    return out


def main():
    files = sorted(glob.glob(RF_FILES_GLOB))
    if not files:
        print(f"No Biomarkers_RF_*.csv files in {OUTPUT_DIR}. "
              "Run randomforest.py first.")
        return

    sym_map = load_symbol_map_from_csv()
    print(f"Symbol map loaded from CSV: {len(sym_map)} entries.")

    # Load top-N genes for each cohort and collect ENSG IDs missing from the map
    rf_data = {}
    all_top_ensg = set()
    for fp in files:
        cohort = os.path.basename(fp).replace("Biomarkers_RF_", "").replace(".csv", "")
        rf = pd.read_csv(fp).head(TOP_N)
        rf_data[cohort] = rf
        all_top_ensg.update(rf['Gene'].astype(str).tolist())

    missing = sorted([g for g in all_top_ensg if g not in sym_map])
    if missing:
        print(f"\nMapping missing for {len(missing)} ENSG IDs. Trying mygene...")
        extra = map_via_mygene(missing)
        sym_map.update(extra)
        print(f"  Mapped: {len(extra)} additional. Total: {len(sym_map)}.")

    rows = []
    for cohort, rf in rf_data.items():
        rf = rf.copy()
        rf['Symbol'] = rf['Gene'].astype(str).map(sym_map).fillna(rf['Gene'].astype(str))

        drivers = KNOWN_DRIVERS.get(cohort)
        if drivers is None:
            print(f"\n{cohort}: no literature list for this cohort (skip).")
            continue

        hits = rf[rf['Symbol'].isin(drivers)]
        rate = round(100.0 * len(hits) / len(rf), 2) if len(rf) else 0.0
        rows.append({
            'Cohort':                cohort,
            'TopN':                  len(rf),
            'Known_Drivers_in_List': len(drivers),
            'Hits_in_TopN':          len(hits),
            'Hit_Rate_Pct':          rate,
            'Hit_Symbols':           ', '.join(hits['Symbol'].tolist()),
        })
        print(f"\n{cohort}: {len(hits)}/{len(rf)} top biomarkers are known drivers "
              f"({rate}%)")
        if len(hits):
            print(f"  Hits: {hits['Symbol'].tolist()}")

    if rows:
        out = os.path.join(OUTPUT_DIR, "Literature_Validation.csv")
        df_out = pd.DataFrame(rows).sort_values('Hit_Rate_Pct', ascending=False)
        df_out.to_csv(out, index=False)
        print(f"\n=== REPORT -> {out} ===")
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()

