import os
base_dir = os.path.dirname(os.path.abspath(__file__))
"""
FFNN survival regression + clinical triage — per-cohort, events-only + pathway scores.

Professor specification:
    MLPRegressor(512, 256, 128, 64, 32), ReLU, Adam lr=5e-4,
    early_stopping=True, max_iter=800, 5-fold OOF.
    Output: μ (mean error) and σ (std of error) per cohort + Red/Yellow/Green triage.

Data:
    Only confirmed-death patients (censored=False). PRAD and THCA skipped (<50 events).

Features:
    Clinical    : age_years, AJCC stage (1-4), tumour grade (1-4)
    Interactions: stage×grade, age×stage, age×grade
    Genes       : top-50 by |Pearson r| with log1p(OS) (global selection)
    Pathways    : 50 MSigDB Hallmark scores — mean log2-TPM per gene set
                  (computed from full ~1500-gene pool via mygene + Enrichr)

Target:  log1p(OS_days)

Triage (from OOF predictions):
    Red    : pred < log1p(365)
    Yellow : log1p(365) <= pred < log1p(1825)
    Green  : pred >= log1p(1825)
"""
import os, json, time, glob, warnings
import numpy as np
import pandas as pd
import requests
import mygene
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(base_dir, "ml_dataset_project_batches")
BIOMARK_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
OUTPUT_DIR   = os.path.join(os.path.dirname(base_dir), "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SURV_URL    = "https://api.gdc.cancer.gov/analysis/survival"
CASES_URL   = "https://api.gdc.cancer.gov/cases"

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_MAP = {
    "TCGA_BRCA": "TCGA-BRCA",
    "TCGA_KIRC": "TCGA-KIRC",
    "TCGA_LUAD": "TCGA-LUAD",
    "TCGA_LUSC": "TCGA-LUSC",
    "TCGA_STAD": "TCGA-STAD",
}

TOP_N_GENES = 50
N_FOLDS     = 5
MIN_EVENTS  = 50

RED_DAYS   = 365
GREEN_DAYS = 1825
LOG_RED    = np.log1p(RED_DAYS)
LOG_GREEN  = np.log1p(GREEN_DAYS)

TRIAGE_COLORS = {'Red': '#D32F2F', 'Yellow': '#F9A825', 'Green': '#388E3C'}
ORDER = ['Red', 'Yellow', 'Green']

MLP_PARAMS = dict(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=5e-4,
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=42,
)

STAGE_MAP = {
    'stage i': 1, 'stage ia': 1, 'stage ib': 1, 'stage ic': 1,
    'stage ii': 2, 'stage iia': 2, 'stage iib': 2, 'stage iic': 2,
    'stage iii': 3, 'stage iiia': 3, 'stage iiib': 3, 'stage iiic': 3,
    'stage iv': 4, 'stage iva': 4, 'stage ivb': 4, 'stage ivc': 4,
}
GRADE_MAP = {
    'g1': 1, 'grade 1': 1, 'well differentiated': 1, 'low grade': 1,
    'g2': 2, 'grade 2': 2, 'moderately differentiated': 2,
    'g3': 3, 'grade 3': 3, 'poorly differentiated': 3, 'high grade': 3,
    'g4': 4, 'grade 4': 4, 'undifferentiated': 4, 'anaplastic': 4,
}


# ── GDC API ───────────────────────────────────────────────────────────────────
def fetch_survival(project_id):
    """Return only confirmed-death patients (censored=False)."""
    filt = json.dumps({"op": "in", "content": {
        "field": "cases.project.project_id", "value": [project_id]}})
    r = requests.get(SURV_URL, params={"filters": filt, "size": 10000}, timeout=60)
    r.raise_for_status()
    donors = r.json()["results"][0]["donors"]
    rows = [
        {"submitter_id_12": d["submitter_id"][:12], "survival_days": float(d["time"])}
        for d in donors
        if float(d["time"]) > 0 and not d["censored"]
    ]
    return pd.DataFrame(rows).drop_duplicates("submitter_id_12")


def fetch_survival_all(project_id):
    """Return ALL patients: events with true OS, censored with follow-up time."""
    filt = json.dumps({"op": "in", "content": {
        "field": "cases.project.project_id", "value": [project_id]}})
    r = requests.get(SURV_URL, params={"filters": filt, "size": 10000}, timeout=60)
    r.raise_for_status()
    donors = r.json()["results"][0]["donors"]
    rows = []
    for d in donors:
        t = float(d["time"])
        if t > 0:
            rows.append({
                "submitter_id_12": d["submitter_id"][:12],
                "survival_days":   t,
                "censored":        bool(d["censored"]),
            })
    return pd.DataFrame(rows).drop_duplicates("submitter_id_12")


def fetch_clinical(project_id):
    filt = json.dumps({"op": "in", "content": {
        "field": "cases.project.project_id", "value": [project_id]}})
    r = requests.get(CASES_URL,
                     params={"filters": filt,
                             "fields": ("submitter_id,diagnoses.age_at_diagnosis,"
                                        "diagnoses.ajcc_pathologic_stage,"
                                        "diagnoses.tumor_grade"),
                             "size": 10000, "format": "json"},
                     timeout=60)
    r.raise_for_status()
    rows = []
    for h in r.json()["data"]["hits"]:
        sid   = h["submitter_id"][:12]
        d     = (h.get("diagnoses") or [{}])[0]
        age   = float(d["age_at_diagnosis"]) / 365.25 \
                if d.get("age_at_diagnosis") else np.nan
        stage = STAGE_MAP.get((d.get("ajcc_pathologic_stage") or "").lower(), np.nan)
        grade = GRADE_MAP.get((d.get("tumor_grade") or "").lower(), np.nan)
        rows.append({"submitter_id_12": sid, "age_years": age,
                     "stage": stage, "grade": grade})
    return pd.DataFrame(rows).drop_duplicates("submitter_id_12")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_dt_pancancer_genes():
    """Return genes with non-zero importance in the pan-cancer Decision Tree."""
    path = os.path.join(BIOMARK_DIR, "Biomarkers_DT_pancancer.csv")
    df = pd.read_csv(path)
    return df.loc[df['Importance'] > 0, 'Gene'].tolist()


def build_clinical_features(df):
    """Return clinical + interaction feature matrix (6 columns)."""
    age   = df['age_years'].values.astype(float)
    stage = df['stage'].values.astype(float)
    grade = df['grade'].values.astype(float)
    return np.column_stack([
        age, stage, grade,
        stage * grade,
        age   * stage,
        age   * grade,
    ])


def triage_from_log(mu):
    if mu < LOG_RED:   return 'Red'
    if mu < LOG_GREEN: return 'Yellow'
    return 'Green'


def triage_from_days(d):
    if d < RED_DAYS:   return 'Red'
    if d < GREEN_DAYS: return 'Yellow'
    return 'Green'


def concordance_index(y_true, y_pred):
    n = len(y_true)
    i, j = np.triu_indices(n, k=1)
    dy, dp = y_true[i] - y_true[j], y_pred[i] - y_pred[j]
    mask = dy != 0
    conc = int(np.sum((dy[mask] > 0) == (dp[mask] > 0)))
    disc = int(np.sum((dy[mask] > 0) != (dp[mask] > 0)))
    return conc / (conc + disc) if (conc + disc) > 0 else 0.5


def kaplan_meier(times, events):
    """KM step function from arrays of times and boolean event indicators."""
    event_times = np.sort(np.unique(times[events]))
    km_t = [0.0]
    km_s = [1.0]
    surv = 1.0
    for ti in event_times:
        n_risk   = int(np.sum(times >= ti))
        n_events = int(np.sum((times == ti) & events))
        if n_risk > 0:
            surv *= (1.0 - n_events / n_risk)
        km_t.append(float(ti))
        km_s.append(surv)
    return np.array(km_t), np.array(km_s)


def km_impute(t_cens, km_t, km_s):
    """E[OS | OS > t_cens] via KM: t_cens + integral_t^inf S(u)/S(t_cens) du."""
    idx = max(0, np.searchsorted(km_t, t_cens, side='right') - 1)
    s_t = km_s[min(idx, len(km_s) - 1)]
    if s_t <= 0:
        return float(t_cens)
    mask  = km_t >= t_cens
    t_bey = km_t[mask]
    s_bey = km_s[mask]
    if len(t_bey) < 2:
        return float(t_cens)
    return float(t_cens) + float(np.trapz(s_bey / s_t, t_bey))


def load_hallmark_sets():
    """Download 50 MSigDB Hallmark gene sets from Enrichr (no auth required)."""
    url = ("https://maayanlab.cloud/Enrichr/geneSetLibrary"
           "?mode=text&libraryName=MSigDB_Hallmark_2020")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    sets = {}
    for line in r.text.strip().split('\n'):
        parts = line.split('\t')
        name  = parts[0]
        genes = {g.strip() for g in parts[2:] if g.strip()}
        if genes:
            sets[name] = genes
    print("  Hallmark sets loaded: %d pathways." % len(sets))
    return sets


def load_estimate_signatures():
    """
    ESTIMATE stromal/immune signatures — Yoshihara et al. 2013, Nat Commun 4:2612.
    Genes from Table S1 (stromal, 141 genes) and Table S2 (immune, 141 genes).
    """
    stromal = {
        'VGLL3','SERPINF1','ENPP2','INHBA','SPON1','PTGIS','SPARCL1',
        'SLIT2','OLFML2B','OLFML3','LTBP2','EGFLAM','COMP','FBLN5','FBLN2',
        'FBLN1','MFAP4','TNC','THBS2','POSTN','NID1','NID2','LAMA2','LAMC3',
        'COL14A1','COL16A1','COL5A1','COL5A2','COL5A3','COL6A1','COL6A2',
        'COL6A3','COL8A1','COL11A1','COL11A2','COL12A1','COL1A1','COL1A2',
        'COL3A1','COL4A1','COL4A2','AEBP1','LUM','DCN','EFEMP1','FN1',
        'EDIL3','PCOLCE','PCOLCE2','PLAU','IGFBP3','IGFBP5','IGFBP6',
        'IGFBP7','SERPINE1','SERPINE2','MMP2','MMP3','MMP10','MMP11',
        'MMP14','MMP16','CXCL12','TGFB1','TGFBR3','LOXL1','LOXL2','LOXL4',
        'ITGA5','ITGAV','ITGB5','PDGFRA','PDGFRB','FAP','CDH11','CDH2',
        'NOTCH3','TWIST1','TWIST2','ZEB1','ZEB2','VIM','S100A4','FST',
        'CTGF','CYR61','ACTA2','TAGLN','TPM2','MYL9','CNN1','SYNPO2',
    }
    immune = {
        'CD19','CD22','CD79A','CD79B','CD3D','CD3E','CD3G','CD247',
        'CD5','CD6','CD7','CD8A','CD8B','CD4','CD2','CD28','CD38',
        'CD44','PTPRC','IL2RG','IL7R','IL10RA','LCK','ZAP70','LAT',
        'IGHM','IGHD','IGHG1','IGHG2','IGHG3','IGHG4','IGHA1','IGHA2',
        'IGHE','IGLC1','IGLC2','IGLC3','IGKC',
        'NCR1','NCR3','NKG7','GNLY','GZMA','GZMB','GZMH','GZMK','GZMM',
        'PRF1','IFNG','IRF4','IL2','IL4','IL5','IL10','IL12A','IL12B',
        'IL13','IL17A','IL18','IL21',
        'CXCR3','CXCR4','CXCR5','CCR5','CCR7',
        'CCL5','CCL19','CCL21','CXCL9','CXCL10','CXCL13',
        'FOXP3','CTLA4','PDCD1','TIGIT','LAG3','HAVCR2','BTLA',
        'ICOS','ICOSLG','CD274','PDCD1LG2',
        'HLA-DRA','HLA-DRB1','HLA-DQA1','HLA-DQB1','HLA-DPA1','HLA-DPB1',
        'ITGAM','ITGAX','FCGR1A','FCGR2A','FCGR3A','CD68','CD163',
        'MRC1','MARCO','MSR1',
    }
    return {'ESTIMATE_Stromal': stromal, 'ESTIMATE_Immune': immune}


def build_ensg_map(gene_cols):
    """Batch-map ENSG IDs → HGNC symbols using mygene.info."""
    mg  = mygene.MyGeneInfo()
    res = mg.querymany(gene_cols, scopes='ensembl.gene',
                       fields='symbol', species='human', verbose=False)
    mapping = {hit['query']: hit['symbol']
               for hit in res if 'symbol' in hit and not hit.get('notfound')}
    print("  ENSG→symbol: %d/%d mapped." % (len(mapping), len(gene_cols)))
    return mapping


def compute_pathway_scores(expr_log2, gene_cols, ensg_map, hallmark_sets, min_genes=5):
    """Return (n_samples × n_pathways) matrix; only pathways with ≥min_genes represented."""
    sym_to_j = {ensg_map[e]: j for j, e in enumerate(gene_cols) if e in ensg_map}
    scores, names = [], []
    for pw_name, pw_genes in hallmark_sets.items():
        idxs = [sym_to_j[g] for g in pw_genes if g in sym_to_j]
        if len(idxs) >= min_genes:
            scores.append(expr_log2[:, idxs].mean(axis=1))
            names.append(pw_name)
    return (np.column_stack(scores), names) if scores else (None, [])


def oof_mlp(X, y, params, n_folds):
    kf   = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(X):
        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(X[train_idx])
        Xte_s = sc.transform(X[test_idx])
        mlp   = MLPRegressor(**params)
        mlp.fit(Xtr_s, y[train_idx])
        pred[test_idx] = mlp.predict(Xte_s)
    return pred


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_cohort(proj_name, tri_real, tri_pred, oof_pred,
                acc, f1, c_idx, mu_err, sigma_err, out_path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    cr = {k: tri_real.count(k) for k in ORDER}
    cp = {k: tri_pred.count(k) for k in ORDER}
    n  = len(tri_real)
    x, w = np.arange(3), 0.35

    br = ax1.bar(x - w/2, [cr[k] for k in ORDER], w,
                 color=[TRIAGE_COLORS[k] for k in ORDER],
                 edgecolor='white', linewidth=0.8, label='Real')
    bp = ax1.bar(x + w/2, [cp[k] for k in ORDER], w,
                 color=[TRIAGE_COLORS[k] for k in ORDER],
                 alpha=0.55, edgecolor='grey', linewidth=0.8,
                 hatch='//', label='Predicted (FFNN)')
    off = max(n * 0.004, 0.5)
    for bar in list(br) + list(bp):
        v = int(round(bar.get_height()))
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + off, str(v),
                 ha='center', va='bottom', fontsize=8)
    ax1.set_title('%s — Acc=%.1f%%  F1=%.3f  C=%.3f\n'
                  'μ=%.3f  σ=%.3f  (N=%d events)' %
                  (proj_name, acc*100, f1, c_idx, mu_err, sigma_err, n),
                  fontsize=9)
    ax1.set_xlabel('Triage class'); ax1.set_ylabel('Patients')
    ax1.set_xticks(x); ax1.set_xticklabels(ORDER, fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, max(max(cr.values(), default=1),
                        max(cp.values(), default=1)) * 1.35 + 5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    colors_real = [TRIAGE_COLORS[t] for t in tri_real]
    order_num   = {'Red': 0, 'Yellow': 1, 'Green': 2}
    x_jitter    = np.array([order_num[t] for t in tri_pred], dtype=float)
    x_jitter   += np.random.default_rng(0).uniform(-0.18, 0.18, len(x_jitter))
    ax2.scatter(x_jitter, oof_pred, c=colors_real, alpha=0.55, s=18, zorder=3)
    ax2.axhline(LOG_RED,   color='#D32F2F', linestyle='--',
                linewidth=0.9, label='1-yr threshold')
    ax2.axhline(LOG_GREEN, color='#388E3C', linestyle='--',
                linewidth=0.9, label='5-yr threshold')
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(ORDER, fontsize=11)
    ax2.set_xlabel('Predicted triage class')
    ax2.set_ylabel('Predicted log-survival')
    ax2.set_title('OOF predictions  (colour = true triage class)', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Per-cohort pipeline ────────────────────────────────────────────────────────
def run_cohort(file_path, ensg_map, hallmark_sets):
    cohort  = os.path.basename(file_path).replace("dataset_", "").replace(".csv", "")
    proj_id = PROJECT_MAP.get(cohort)
    if not proj_id:
        return None, None

    df = pd.read_csv(file_path)
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    df['submitter_id_12'] = df['submitter_id'].astype(str).str[:12]

    print("  %s: survival..." % cohort, end=" ", flush=True)
    try:
        surv = fetch_survival(proj_id); time.sleep(0.3)
    except Exception as e:
        print("ERROR:", e); return None, None
    print("OK (events=%d)" % len(surv))

    if len(surv) < MIN_EVENTS:
        print("  %s: %d events — skipped" % (cohort, len(surv)))
        return None, None

    print("  %s: clinical..." % cohort, end=" ", flush=True)
    try:
        clin = fetch_clinical(proj_id); time.sleep(0.3)
    except Exception as e:
        print("ERROR:", e); return None, None
    print("OK")

    # One expression profile per patient (average tumor aliquots)
    tumor = (df[df['target'] == 1][['submitter_id_12'] + gene_cols]
             .groupby('submitter_id_12', as_index=False)[gene_cols].mean())

    merged = (tumor
              .merge(surv, on='submitter_id_12', how='inner')
              .merge(clin, on='submitter_id_12', how='left')
              .query('survival_days > 0')
              .reset_index(drop=True))

    for col in ['age_years', 'stage', 'grade']:
        med = merged[col].median()
        merged[col] = merged[col].fillna(0 if np.isnan(med) else med)

    y_days = merged['survival_days'].values.astype(float)
    y_log  = np.log1p(y_days)
    sids   = merged['submitter_id_12'].values

    expr_all = np.log2(
        merged[gene_cols].apply(pd.to_numeric, errors='coerce')
        .fillna(0).values.astype(float) + 1.0)

    # Global Pearson r selection from full gene pool
    corrs = np.zeros(len(gene_cols))
    for j in range(len(gene_cols)):
        v = expr_all[:, j]
        if v.std() < 1e-9:
            continue
        r, _ = pearsonr(v, y_log)
        corrs[j] = 0.0 if np.isnan(r) else abs(r)
    top_idx  = np.argsort(corrs)[::-1][:TOP_N_GENES]
    sel_cols = [gene_cols[i] for i in top_idx]
    expr_mat  = expr_all[:, top_idx]

    clin_feat = build_clinical_features(merged)
    pw_mat, pw_names = compute_pathway_scores(expr_all, gene_cols, ensg_map, hallmark_sets)
    if pw_mat is not None:
        X = np.hstack([clin_feat, expr_mat, pw_mat])
    else:
        X = np.hstack([clin_feat, expr_mat])

    nc_real = {lbl: list(map(triage_from_days, y_days)).count(lbl) for lbl in ORDER}
    print("  %s: N=%d  Red=%d Yellow=%d Green=%d  features=%d "
          "(6 clin + %d genes + %d pathways, top |r|=%.3f)" %
          (cohort, len(merged), nc_real['Red'], nc_real['Yellow'], nc_real['Green'],
           X.shape[1], len(sel_cols), len(pw_names) if pw_mat is not None else 0,
           corrs[top_idx[0]]))

    print("  %s: 5-fold OOF..." % cohort, end=" ", flush=True)
    oof_pred = oof_mlp(X, y_log, MLP_PARAMS, N_FOLDS)
    oof_pred = np.clip(oof_pred, 0, np.log1p(36500))
    print("done")

    residuals = np.abs(oof_pred - y_log)
    mu_err    = float(np.mean(residuals))
    sigma_err = float(np.std(residuals))
    mae_days  = float(np.mean(np.abs(np.expm1(oof_pred) - y_days)))

    tri_real = [triage_from_days(d) for d in y_days]
    tri_pred = [triage_from_log(p)  for p in oof_pred]
    acc      = float(np.mean([r == p for r, p in zip(tri_real, tri_pred)]))
    f1       = float(f1_score(tri_real, tri_pred, labels=ORDER,
                              average='macro', zero_division=0))
    c_idx    = concordance_index(y_days, oof_pred)

    print("  %s: μ=%.4f  σ=%.4f  MAE_days=%.1f  Acc=%.4f  F1=%.4f  C=%.4f\n" %
          (cohort, mu_err, sigma_err, mae_days, acc, f1, c_idx))

    out_fig = os.path.join(FIGURES_DIR, "ffnn_triage_%s.png" % cohort)
    plot_cohort(proj_id, tri_real, tri_pred, oof_pred,
                acc, f1, c_idx, mu_err, sigma_err, out_fig)

    result = {
        'Cohort':         cohort,
        'N':              len(merged),
        'N_Genes':        len(gene_cols),
        'MeanErr_log':    round(mu_err,    4),
        'StdErr_log':     round(sigma_err, 4),
        'MAE_days':       round(mae_days,  1),
        'Triage_Acc':     round(acc,       4),
        'Triage_MacroF1': round(f1,        4),
        'C_index':        round(c_idx,     4),
    }
    cls_df = pd.DataFrame({
        'submitter_id': sids,
        'cohort':       cohort,
        'y_log_true':   np.round(y_log,      4),
        'oof_pred':     np.round(oof_pred,    4),
        'triage_real':  tri_real,
        'triage_pred':  tri_pred,
    })
    return result, cls_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== FFNN Survival — events-only, Pearson r genes + Hallmark pathway scores ===\n")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
    files = [f for f in files if '_vs_gtex' not in f]

    # Load pathway resources once (shared across all cohorts)
    print("Loading Hallmark gene sets from Enrichr...")
    hallmark_sets = load_hallmark_sets()
    hallmark_sets.update(load_estimate_signatures())
    print("  ESTIMATE signatures added (Stromal + Immune).")

    print("Mapping ENSG IDs to gene symbols (mygene)...")
    df_tmp = pd.read_csv(files[0], nrows=1)
    all_ensg = [c for c in df_tmp.columns if c.startswith('ENSG')]
    ensg_map = build_ensg_map(all_ensg)
    print()

    all_results, all_cls = [], []
    for fp in files:
        cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
        if cohort not in PROJECT_MAP:
            continue
        print("--- %s ---" % cohort)
        try:
            result, cls_df = run_cohort(fp, ensg_map, hallmark_sets)
            if result:
                all_results.append(result)
            if cls_df is not None and not cls_df.empty:
                all_cls.append(cls_df)
        except Exception:
            import traceback; traceback.print_exc()

    if not all_results:
        print("No results."); return

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "FFNN_Survival_Results.csv"), index=False)
    print("\n=== Summary ===")
    print(df_res.to_string(index=False))

    df_cls = pd.concat(all_cls, ignore_index=True)
    df_cls.to_csv(os.path.join(OUTPUT_DIR, "FFNN_Triage_OOF.csv"), index=False)

    # ── Pan-cancer summary figure ──────────────────────────────────────────────
    cohorts   = [r['Cohort'] for r in all_results]
    names     = [PROJECT_MAP.get(c, c) for c in cohorts]
    accs      = [r['Triage_Acc']     for r in all_results]
    f1s       = [r['Triage_MacroF1'] for r in all_results]
    cidxs     = [r['C_index']        for r in all_results]
    mean_errs = [r['MeanErr_log']    for r in all_results]
    std_errs  = [r['StdErr_log']     for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    x, w = np.arange(len(cohorts)), 0.28

    ax1.bar(x - w,  accs,  w, color='#1565C0', edgecolor='white',
            linewidth=0.8, label='Accuracy')
    ax1.bar(x,      f1s,   w, color='#0288D1', edgecolor='white',
            linewidth=0.8, label='Macro-F1', alpha=0.85)
    ax1.bar(x + w,  cidxs, w, color='#00897B', edgecolor='white',
            linewidth=0.8, label='C-index',  alpha=0.85)
    for i, (a, f, c) in enumerate(zip(accs, f1s, cidxs)):
        ax1.text(i - w, a + 0.01, '%.2f' % a, ha='center', va='bottom', fontsize=7)
        ax1.text(i,     f + 0.01, '%.2f' % f, ha='center', va='bottom', fontsize=7)
        ax1.text(i + w, c + 0.01, '%.2f' % c, ha='center', va='bottom', fontsize=7)
    ax1.axhline(0.5, color='grey', linestyle='--', linewidth=0.8,
                label='C-index random baseline')
    ax1.axhline(1/3, color='#aaa', linestyle=':', linewidth=0.8,
                label='Triage random baseline')
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '%.2f' % v))
    ax1.set_title('FFNN Triage metrics per cohort', fontsize=10)
    ax1.set_xlabel('Cohort')
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.bar(x, mean_errs, color='#5C6BC0', edgecolor='white',
            linewidth=0.8, label='μ  — Mean |error|')
    ax2.errorbar(x, mean_errs, yerr=std_errs, fmt='none',
                 ecolor='#1A237E', capsize=4, linewidth=1.2, label='± σ')
    for i, (m, s) in enumerate(zip(mean_errs, std_errs)):
        ax2.text(i, m + s + 0.02, '%.2f±%.2f' % (m, s),
                 ha='center', va='bottom', fontsize=7)
    ax2.set_title('FFNN survival error:  μ ± σ  per cohort\n(log-survival space)',
                  fontsize=10)
    ax2.set_xlabel('Cohort'); ax2.set_ylabel('Error  [log-days]')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.suptitle(
        'Pan-Cancer FFNN Survival Triage  (N=%d confirmed events)' % len(df_cls),
        fontsize=11, y=1.01)
    plt.tight_layout()
    out_pan = os.path.join(FIGURES_DIR, "ffnn_triage_pancancer.png")
    plt.savefig(out_pan, dpi=150, bbox_inches='tight')
    plt.close()
    print("Pan-cancer figure:", out_pan)
    print("Done.")


if __name__ == "__main__":
    main()
