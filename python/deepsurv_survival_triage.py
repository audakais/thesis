import os
base_dir = os.path.dirname(os.path.abspath(__file__))
"""
DeepHit survival regression + clinical triage — per-cohort.

Architecture: DeepHit neural network (Lee et al. 2018, NeurIPS)
    Network : MLPVanilla(512, 256, 128, 64, 32), ReLU, BatchNorm, Dropout=0.1
    Loss    : NLL-PMF + alpha * DeepHit ranking loss  (directly optimises concordance)
    Training: ALL patients (events + censored), 5-fold OOF
              Adam lr=5e-4, weight_decay=1e-4, early_stopping patience=25, max_epochs=800
    Ensemble: N_SEEDS=5 models per fold, averaged survival functions (reduces variance)

vs CoxPH (previous version):
    - DeepHit loss directly optimises ranking/concordance, not log-likelihood
    - Seed ensemble reduces initialisation variance (critical for small cohorts like BRCA)
    - L2 regularisation (weight_decay=1e-4) reduces overfitting

vs FFNN (MLPRegressor + MSE):
    - DeepHit handles censoring natively — no patients discarded
    - Directly optimises the ranking behind the C-index

Output (same format as FFNN, compatible with professor specification):
    mu  = mean |log1p(pred_days) - log1p(true_days)|  on confirmed-death patients
    sig = std  of the same residuals
    Triage: Red pred<365d, Yellow 365-1825d, Green >=1825d
"""
import os, json, time, glob, warnings
import numpy as np
import pandas as pd
import requests
import mygene
import torch
import torch.nn as nn
import torchtuples as tt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(base_dir, "ml_dataset_project_batches")
OUTPUT_DIR  = os.path.join(os.path.dirname(base_dir), "outputs")
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
TOP_N_GENES   = 50
N_FOLDS       = 5
N_SEEDS       = 5      # seed ensemble per fold
NUM_DURATIONS = 100    # DeepHit time discretisation bins
MIN_EVENTS    = 50
RED_DAYS    = 365
GREEN_DAYS  = 1825
LOG_RED     = np.log1p(RED_DAYS)
LOG_GREEN   = np.log1p(GREEN_DAYS)
TRIAGE_COLORS = {'Red': '#D32F2F', 'Yellow': '#F9A825', 'Green': '#388E3C'}
ORDER = ['Red', 'Yellow', 'Green']

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
def fetch_survival_events(project_id):
    filt = json.dumps({"op": "in", "content": {
        "field": "cases.project.project_id", "value": [project_id]}})
    r = requests.get(SURV_URL, params={"filters": filt, "size": 10000}, timeout=60)
    r.raise_for_status()
    donors = r.json()["results"][0]["donors"]
    rows = [{"submitter_id_12": d["submitter_id"][:12],
             "survival_days": float(d["time"])}
            for d in donors if float(d["time"]) > 0 and not d["censored"]]
    return pd.DataFrame(rows).drop_duplicates("submitter_id_12")


def fetch_survival_all(project_id):
    filt = json.dumps({"op": "in", "content": {
        "field": "cases.project.project_id", "value": [project_id]}})
    r = requests.get(SURV_URL, params={"filters": filt, "size": 10000}, timeout=60)
    r.raise_for_status()
    donors = r.json()["results"][0]["donors"]
    rows = [{"submitter_id_12": d["submitter_id"][:12],
             "survival_days": float(d["time"]),
             "censored": bool(d["censored"])}
            for d in donors if float(d["time"]) > 0]
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
        age   = float(d["age_at_diagnosis"]) / 365.25 if d.get("age_at_diagnosis") else np.nan
        stage = STAGE_MAP.get((d.get("ajcc_pathologic_stage") or "").lower(), np.nan)
        grade = GRADE_MAP.get((d.get("tumor_grade") or "").lower(), np.nan)
        rows.append({"submitter_id_12": sid, "age_years": age,
                     "stage": stage, "grade": grade})
    return pd.DataFrame(rows).drop_duplicates("submitter_id_12")


# ── Pathway resources ─────────────────────────────────────────────────────────
def load_gene_sets():
    url = ("https://maayanlab.cloud/Enrichr/geneSetLibrary"
           "?mode=text&libraryName=MSigDB_Hallmark_2020")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    sets = {}
    for line in r.text.strip().split('\n'):
        parts = line.split('\t')
        genes = {g.strip() for g in parts[2:] if g.strip()}
        if genes:
            sets[parts[0]] = genes
    # ESTIMATE signatures (Yoshihara et al. 2013 Nat Commun)
    sets['ESTIMATE_Stromal'] = {
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
    sets['ESTIMATE_Immune'] = {
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
    print("  Gene sets loaded: %d (50 Hallmark + 2 ESTIMATE)." % len(sets))
    return sets


def build_ensg_map(gene_cols):
    mg  = mygene.MyGeneInfo()
    res = mg.querymany(gene_cols, scopes='ensembl.gene',
                       fields='symbol', species='human', verbose=False)
    mapping = {hit['query']: hit['symbol']
               for hit in res if 'symbol' in hit and not hit.get('notfound')}
    print("  ENSG->symbol: %d/%d mapped." % (len(mapping), len(gene_cols)))
    return mapping


def compute_pathway_scores(expr_log2, gene_cols, ensg_map, gene_sets, min_genes=5):
    sym_to_j = {ensg_map[e]: j for j, e in enumerate(gene_cols) if e in ensg_map}
    scores, names = [], []
    for name, genes in gene_sets.items():
        idxs = [sym_to_j[g] for g in genes if g in sym_to_j]
        if len(idxs) >= min_genes:
            scores.append(expr_log2[:, idxs].mean(axis=1))
            names.append(name)
    return (np.column_stack(scores), names) if scores else (None, [])


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_clinical_features(df):
    age   = df['age_years'].values.astype(float)
    stage = df['stage'].values.astype(float)
    grade = df['grade'].values.astype(float)
    return np.column_stack([age, stage, grade, stage*grade, age*stage, age*grade])


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


# ── DeepSurv OOF ─────────────────────────────────────────────────────────────
def median_survival_days(surv_df):
    times  = surv_df.index.values
    result = np.full(len(surv_df.columns), float(times[-1]))
    for i, col in enumerate(surv_df.columns):
        s = surv_df[col].values
        below = np.where(s < 0.5)[0]
        if len(below) > 0:
            result[i] = float(times[below[0]])
    return result


def oof_deephit(X, y_days, events, n_folds, n_seeds=N_SEEDS, num_durations=NUM_DURATIONS):
    """
    5-fold OOF DeepHit with seed ensemble.
    - Loss: NLL-PMF + alpha*ranking → directly optimises concordance.
    - Seed ensemble: n_seeds models per fold, survival curves averaged before scoring.
    - L2 via weight_decay=1e-4 in Adam.
    Returns:
        pred_days   : predicted median OS days (for triage/mu/sigma)
        surv_oof_df : OOF survival functions on common time grid (for concordance-td)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    pred_days    = np.full(len(y_days), np.nan)
    common_times = np.sort(np.unique(y_days[events])).astype(np.float32)
    surv_matrix  = np.ones((len(y_days), len(common_times)), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        n_val   = max(20, int(0.1 * len(train_idx)))
        val_idx = train_idx[-n_val:]
        tr_idx  = train_idx[:-n_val]

        sc    = StandardScaler()
        Xtr   = sc.fit_transform(X[tr_idx]).astype(np.float32)
        Xval  = sc.transform(X[val_idx]).astype(np.float32)
        Xte   = sc.transform(X[test_idx]).astype(np.float32)

        # Discretise time into num_durations bins for DeepHit PMF output
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_tr     = labtrans.fit_transform(
            y_days[tr_idx].astype(float), events[tr_idx].astype(float))
        y_val_dh = labtrans.transform(
            y_days[val_idx].astype(float), events[val_idx].astype(float))

        # Seed ensemble: n_seeds initialisations, average survival functions
        surv_sum    = None
        surv_df_ref = None
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            net = tt.practical.MLPVanilla(
                in_features  = X.shape[1],
                num_nodes    = [512, 256, 128, 64, 32],
                out_features = labtrans.out_features,
                batch_norm   = True,
                dropout      = 0.1,
                activation   = nn.ReLU,
            )
            model = DeepHitSingle(
                net,
                tt.optim.Adam(lr=5e-4, weight_decay=1e-4),
                alpha          = 0.2,
                sigma          = 0.1,
                duration_index = labtrans.cuts,
            )
            model.fit(
                Xtr, y_tr,
                batch_size = min(256, len(tr_idx)),
                epochs     = 800,
                callbacks  = [tt.callbacks.EarlyStopping(patience=25)],
                val_data   = (Xval, y_val_dh),
                verbose    = False,
            )
            surv_df = model.predict_surv_df(Xte)
            if surv_sum is None:
                surv_sum    = surv_df.values.copy()
                surv_df_ref = surv_df
            else:
                surv_sum += surv_df.values

        surv_avg = pd.DataFrame(
            surv_sum / n_seeds,
            index   = surv_df_ref.index,
            columns = surv_df_ref.columns,
        )

        pred_days[test_idx] = median_survival_days(surv_avg)

        raw_times = surv_avg.index.values.astype(np.float32)
        for local_i, global_i in enumerate(test_idx):
            s_raw = surv_avg.iloc[:, local_i].values.astype(np.float32)
            surv_matrix[global_i] = np.interp(
                common_times, raw_times, s_raw,
                left=1.0, right=float(s_raw[-1]))

    surv_oof_df = pd.DataFrame(surv_matrix.T, index=common_times.astype(float))
    return pred_days, surv_oof_df


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_cohort(proj_name, tri_real, tri_pred, pred_days_ev,
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
                 hatch='//', label='Predicted (DeepSurv)')
    off = max(n * 0.004, 0.5)
    for bar in list(br) + list(bp):
        v = int(round(bar.get_height()))
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + off, str(v),
                 ha='center', va='bottom', fontsize=8)
    ax1.set_title('%s  Acc=%.1f%%  F1=%.3f  C=%.3f\n'
                  'mu=%.3f  sigma=%.3f  (N=%d events)' %
                  (proj_name, acc*100, f1, c_idx, mu_err, sigma_err, n), fontsize=9)
    ax1.set_xlabel('Triage class'); ax1.set_ylabel('Patients')
    ax1.set_xticks(x); ax1.set_xticklabels(ORDER, fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, max(max(cr.values(), default=1),
                        max(cp.values(), default=1)) * 1.35 + 5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    log_pred  = np.log1p(pred_days_ev)
    order_num = {'Red': 0, 'Yellow': 1, 'Green': 2}
    x_pos = np.array([order_num[t] for t in tri_pred], dtype=float)
    x_pos += np.random.default_rng(0).uniform(-0.18, 0.18, len(x_pos))
    colors_real = [TRIAGE_COLORS[t] for t in tri_real]
    ax2.scatter(x_pos, log_pred, c=colors_real, alpha=0.55, s=18, zorder=3)
    ax2.axhline(LOG_RED,   color='#D32F2F', linestyle='--', linewidth=0.9, label='1-yr')
    ax2.axhline(LOG_GREEN, color='#388E3C', linestyle='--', linewidth=0.9, label='5-yr')
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(ORDER, fontsize=11)
    ax2.set_xlabel('Predicted triage class')
    ax2.set_ylabel('Predicted log-survival')
    ax2.set_title('OOF predictions  (colour = true triage class)', fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Per-cohort pipeline ───────────────────────────────────────────────────────
def run_cohort(file_path, ensg_map, gene_sets):
    cohort  = os.path.basename(file_path).replace("dataset_", "").replace(".csv", "")
    proj_id = PROJECT_MAP.get(cohort)
    if not proj_id:
        return None, None

    df = pd.read_csv(file_path)
    gene_cols = [c for c in df.columns if c.startswith("ENSG")]
    df['submitter_id_12'] = df['submitter_id'].astype(str).str[:12]

    print("  %s: survival..." % cohort, end=" ", flush=True)
    try:
        surv_ev = fetch_survival_events(proj_id); time.sleep(0.3)
    except Exception as e:
        print("ERROR:", e); return None, None
    print("OK (%d events)" % len(surv_ev))
    if len(surv_ev) < MIN_EVENTS:
        print("  %s: skipped" % cohort); return None, None

    try:
        surv_all = fetch_survival_all(proj_id); time.sleep(0.3)
    except Exception as e:
        print("ERROR fetch_all:", e); return None, None

    print("  %s: clinical..." % cohort, end=" ", flush=True)
    try:
        clin = fetch_clinical(proj_id); time.sleep(0.3)
    except Exception as e:
        print("ERROR:", e); return None, None
    print("OK")

    tumor = (df[df['target'] == 1][['submitter_id_12'] + gene_cols]
             .groupby('submitter_id_12', as_index=False)[gene_cols].mean())
    merged = (tumor
              .merge(surv_all, on='submitter_id_12', how='inner')
              .merge(clin,     on='submitter_id_12', how='left')
              .query('survival_days > 0')
              .reset_index(drop=True))

    for col in ['age_years', 'stage', 'grade']:
        med = merged[col].median()
        merged[col] = merged[col].fillna(0 if np.isnan(med) else med)

    t_all    = merged['survival_days'].values.astype(float)
    is_event = ~merged['censored'].values.astype(bool)
    sids_all = merged['submitter_id_12'].values

    expr_log2 = np.log2(
        merged[gene_cols].apply(pd.to_numeric, errors='coerce')
        .fillna(0).values.astype(float) + 1.0)

    # Gene selection on events only (true OS, no noise)
    y_log_ev = np.log1p(t_all[is_event])
    corrs = np.zeros(len(gene_cols))
    for j in range(len(gene_cols)):
        v = expr_log2[is_event, j]
        if v.std() < 1e-9:
            continue
        r, _ = pearsonr(v, y_log_ev)
        corrs[j] = 0.0 if np.isnan(r) else abs(r)
    top_idx  = np.argsort(corrs)[::-1][:TOP_N_GENES]
    expr_mat  = expr_log2[:, top_idx]

    clin_feat = build_clinical_features(merged)
    pw_mat, pw_names = compute_pathway_scores(expr_log2, gene_cols, ensg_map, gene_sets)
    X = np.hstack([clin_feat, expr_mat, pw_mat]) if pw_mat is not None \
        else np.hstack([clin_feat, expr_mat])

    n_ev   = int(is_event.sum())
    n_cens = len(merged) - n_ev
    nc_ev  = {lbl: list(map(triage_from_days, t_all[is_event])).count(lbl) for lbl in ORDER}
    print("  %s: N_total=%d (events=%d + censored=%d)  "
          "Red=%d Yellow=%d Green=%d  features=%d (%d pathways)" %
          (cohort, len(merged), n_ev, n_cens,
           nc_ev['Red'], nc_ev['Yellow'], nc_ev['Green'],
           X.shape[1], len(pw_names)))

    print("  %s: 5-fold OOF DeepHit (x%d seeds)..." % (cohort, N_SEEDS),
          end=" ", flush=True)
    pred_days_all, surv_oof_df = oof_deephit(X, t_all, is_event, N_FOLDS)
    pred_days_all = np.clip(pred_days_all, 1.0, 36500.0)
    print("done")

    pred_days_ev = pred_days_all[is_event]
    y_days_ev    = t_all[is_event]

    residuals = np.abs(np.log1p(pred_days_ev) - np.log1p(y_days_ev))
    mu_err    = float(np.mean(residuals))
    sigma_err = float(np.std(residuals))
    mae_days  = float(np.mean(np.abs(pred_days_ev - y_days_ev)))

    tri_real = [triage_from_days(d) for d in y_days_ev]
    tri_pred = [triage_from_days(d) for d in pred_days_ev]
    acc      = float(np.mean([r == p for r, p in zip(tri_real, tri_pred)]))
    f1       = float(f1_score(tri_real, tri_pred, labels=ORDER,
                              average='macro', zero_division=0))

    # concordance-td: uses ALL patients + KM censoring correction (proper Cox metric)
    ev_eval = EvalSurv(surv_oof_df, t_all, is_event.astype(float), censor_surv='km')
    c_idx   = float(ev_eval.concordance_td())

    print("  %s: mu=%.4f  sigma=%.4f  MAE_days=%.1f  Acc=%.4f  F1=%.4f  C=%.4f\n" %
          (cohort, mu_err, sigma_err, mae_days, acc, f1, c_idx))

    out_fig = os.path.join(FIGURES_DIR, "deephit_triage_%s.png" % cohort)
    plot_cohort(proj_id, tri_real, tri_pred, pred_days_ev,
                acc, f1, c_idx, mu_err, sigma_err, out_fig)

    result = {
        'Cohort':         cohort,
        'N_events':       n_ev,
        'N_total':        len(merged),
        'N_Genes':        len(gene_cols),
        'MeanErr_log':    round(mu_err,    4),
        'StdErr_log':     round(sigma_err, 4),
        'MAE_days':       round(mae_days,  1),
        'Triage_Acc':     round(acc,       4),
        'Triage_MacroF1': round(f1,        4),
        'C_index':        round(c_idx,     4),
    }
    cls_df = pd.DataFrame({
        'submitter_id': sids_all[is_event],
        'cohort':       cohort,
        'true_days':    np.round(y_days_ev,    1),
        'pred_days':    np.round(pred_days_ev, 1),
        'triage_real':  tri_real,
        'triage_pred':  tri_pred,
    })
    return result, cls_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== DeepHit Survival — ranking loss NN + seed ensemble + Hallmark + ESTIMATE ===\n")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
    files = [f for f in files if '_vs_gtex' not in f]

    print("Loading gene sets (Hallmark + ESTIMATE)...")
    gene_sets = load_gene_sets()

    print("Mapping ENSG IDs to gene symbols...")
    df_tmp   = pd.read_csv(files[0], nrows=1)
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
            result, cls_df = run_cohort(fp, ensg_map, gene_sets)
            if result:
                all_results.append(result)
            if cls_df is not None and not cls_df.empty:
                all_cls.append(cls_df)
        except Exception:
            import traceback; traceback.print_exc()

    if not all_results:
        print("No results."); return

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "DeepHit_Survival_Results.csv"), index=False)
    print("\n=== Summary ===")
    print(df_res.to_string(index=False))

    df_cls = pd.concat(all_cls, ignore_index=True)
    df_cls.to_csv(os.path.join(OUTPUT_DIR, "DeepHit_Triage_OOF.csv"), index=False)

    # Pan-cancer figure
    cohorts = [r['Cohort']         for r in all_results]
    names   = [PROJECT_MAP.get(c,c) for c in cohorts]
    accs    = [r['Triage_Acc']     for r in all_results]
    f1s     = [r['Triage_MacroF1'] for r in all_results]
    cidxs   = [r['C_index']        for r in all_results]
    mu_errs = [r['MeanErr_log']    for r in all_results]
    sg_errs = [r['StdErr_log']     for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    x, w = np.arange(len(cohorts)), 0.28
    ax1.bar(x-w, accs,  w, color='#1565C0', edgecolor='white', linewidth=0.8, label='Accuracy')
    ax1.bar(x,   f1s,   w, color='#0288D1', edgecolor='white', linewidth=0.8, label='Macro-F1', alpha=0.85)
    ax1.bar(x+w, cidxs, w, color='#00897B', edgecolor='white', linewidth=0.8, label='C-index',  alpha=0.85)
    for i, (a, f, c) in enumerate(zip(accs, f1s, cidxs)):
        ax1.text(i-w, a+0.01, '%.2f'%a, ha='center', va='bottom', fontsize=7)
        ax1.text(i,   f+0.01, '%.2f'%f, ha='center', va='bottom', fontsize=7)
        ax1.text(i+w, c+0.01, '%.2f'%c, ha='center', va='bottom', fontsize=7)
    ax1.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='C random')
    ax1.axhline(1/3, color='#aaa', linestyle=':', linewidth=0.8,  label='Triage random')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('DeepHit Triage metrics per cohort', fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax1.legend(fontsize=7, ncol=2); ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2.bar(x, mu_errs, color='#5C6BC0', edgecolor='white', linewidth=0.8, label='mu error')
    ax2.errorbar(x, mu_errs, yerr=sg_errs, fmt='none', color='#1A237E',
                 capsize=4, linewidth=1.2)
    for i, (m, s) in enumerate(zip(mu_errs, sg_errs)):
        ax2.text(i, m+s+0.02, '%.3f+/-%.3f'%(m,s), ha='center', va='bottom', fontsize=7)
    ax2.set_title('Mean error mu +/- sigma (log scale)', fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('|log1p(pred) - log1p(true)|')
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out_pan = os.path.join(FIGURES_DIR, "deephit_triage_pancancer.png")
    plt.savefig(out_pan, dpi=150, bbox_inches='tight')
    plt.close()
    print("Pan-cancer figure:", out_pan)
    print("Done.")


if __name__ == "__main__":
    main()
