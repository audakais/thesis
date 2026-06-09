# Identification of Biomarkers from Gene Expression Data

Machine-learning pipeline for the identification of cancer biomarkers from bulk
RNA-seq gene-expression profiles. Tumour samples from The Cancer Genome Atlas (TCGA)
are classified against matched normal tissue, candidate biomarker genes are ranked
and validated, and a pan-cancer model distinguishes the tissue of origin across
multiple tumour types. Healthy reference samples from GTEx (v8) are used as an
external control set.

## Cohorts

Eight TCGA projects are analysed: **BRCA, KIRC, LUAD, LUSC, OV, PRAD, STAD, THCA**.
OV is excluded from tumour-vs-normal classification because no normal samples are
available for that project.

## Repository structure

```
Biomarker_Identification/
├── README.md
├── requirements.txt        # Python dependencies (unpinned)
├── data/                   # raw inputs (not shipped) — see data/README.md
├── python/                 # analysis scripts (one stage of the pipeline each)
│   ├── run_gtex.sh         # GTEx external-validation run order
│   └── run_downstream.sh   # downstream-analysis run order
├── sql/
│   └── 04_create_db.sql    # PostgreSQL schema (expression database)
└── outputs/
    ├── figures/            # all figures produced by the pipeline (figures named as referenced in the thesis)
    ├── csv/                # result tables (biomarker lists, summaries, predictions)
    └── interpretable_rules/# plain-text decision-tree rules per cohort
```

## Data sources

- **TCGA** RNA-seq (STAR – Counts / TPM) and clinical metadata, downloaded from the
  NCI Genomic Data Commons (https://portal.gdc.cancer.gov).
- **GTEx v8** normal-tissue expression, from the GTEx Portal
  (https://gtexportal.org).

## Database

Downloaded data are stored in a **PostgreSQL** database whose schema is defined in
`sql/04_create_db.sql` (tables: `Individuals`, `Patients`, `SampleTypes`, `Samples`,
`Biomarkers`, `Measurements`). `gdc_download.py` retrieves the files from the GDC API
and populates the database; `create_ml_dataset.py` exports one analysis-ready matrix
per cohort.

## Requirements

- Python 3.10+
- Core: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`
- Imbalanced learning (SMOTE): `imbalanced-learn`
- Model interpretation: `shap`
- Gene-ID mapping and enrichment: `mygene`, `requests` (Enrichr API)
- Survival models: `lifelines`, `torch`, `torchtuples`, `pycox`
- Database: `psycopg2` (pip package `psycopg2-binary`)
- File I/O: `openpyxl` (Excel), `pyarrow` (Parquet), `python-docx`

Install the dependencies (the same `pip install` commands are listed in `requirements.txt`), for example:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn imbalanced-learn \
            shap mygene requests lifelines torch torchtuples pycox psycopg2-binary \
            openpyxl pyarrow python-docx
```

## Pipeline

The scripts are organised by stage. Each script reads the per-cohort datasets
exported from the database and writes its results into `outputs/`.

**1. Data acquisition and dataset construction**
- `gdc_download.py` — download TCGA files from the GDC and load them into PostgreSQL.
- `create_ml_dataset.py` — export one expression matrix per cohort (tumour + normal).
- `mapgenes.py` — map Ensembl gene IDs to HGNC symbols.
- `convert_gtex_parquet.py`, `gtex_integration.py` — prepare and align the GTEx
  reference matrices to the TCGA gene space.

**2. Biomarker selection (Random Forest)**
- `randomforest.py` — per-cohort tumour-vs-normal Random Forest; exports ranked
  biomarker tables and feature-importance figures.
- `randomforest_leaf_sweep.py` — sensitivity analysis over `min_samples_leaf`.
- `permutation_test.py` — permutation test of classification performance.
- `shap_analysis.py` — SHAP values for the selected features.

**3. Interpretable models (Decision Tree)**
- `decision_tree_analysis.py`, `interpretable_tree.py`,
  `decision_tree_biomarker_selector.py`, `decision_tree_leaf_sweep.py` — compact,
  human-readable trees on the top Random-Forest genes; rules saved under
  `outputs/interpretable_rules/`.
- `dt_groupkfold_smote_leaf100.py` — patient-level GroupKFold with SMOTE.
- `dt_pancancer_confusion_matrix.py`, `pancancer_confusion_matrix.py`,
  `multi_class_pancancer.py` — pan-cancer (tissue-of-origin) classification.
- `confusion_matrix_report.py` — confusion-matrix figures and reports.

**4. External validation on GTEx**
- `predict_gtex.py` — apply the trained classifiers to GTEx normal samples.
- `dt_gtex_analysis.py` — decision-tree analysis of the TCGA-vs-GTEx setting.

**5. Survival and clinical triage**
- `survival_analysis.py` — Kaplan–Meier / log-rank survival analysis.
- `ffnn_survival_triage.py` — feed-forward neural-network survival regression with
  Red / Yellow / Green triage.
- `deepsurv_survival_triage.py` — DeepHit survival model (concordance-optimised,
  seed ensemble).

**6. Biological interpretation and reporting**
- `pathway_enrichment.py` — KEGG/Hallmark pathway enrichment (Enrichr API).
- `literature_validation.py` — cross-check selected genes against the literature.
- `biomarker_overlap.py` — overlap of biomarker sets across cohorts.
- `expression_heatmap.py` — expression heatmaps / clustermaps of top genes.
- `learning_curve.py` — learning curves.
- `tcga_demographics.py`, `gtex_demographics.py`, `tumor_incidence_analysis.py` —
  demographic and incidence summaries.

`run_gtex.sh` and `run_downstream.sh` document a convenient execution order for the
GTEx validation and the downstream-analysis stages.

## Outputs

- `outputs/csv/` — ranked biomarker tables (`Biomarkers_RF_TCGA_*.csv`), per-cohort
  and pan-cancer summaries (`Summary_RF_AllCohorts.csv`,
  `MultiClass_PanCancer_*.csv`), GTEx predictions, pathway enrichment, literature
  validation, survival analysis, and demographic tables.
- `outputs/figures/` — feature-importance plots, confusion matrices, heatmaps,
  pathway plots, survival/triage plots and demographic charts
  (named as referenced in the thesis).
- `outputs/interpretable_rules/` — plain-text decision-tree rules per cohort.

## Notes

- All paths are resolved relative to each script's own location, so the project can
  be run from anywhere. The per-cohort expression matrices are expected in
  `python/ml_dataset_project_batches/`, results are written to `outputs/`, and the
  external raw inputs that are not shipped with this repository (the GTEx bulk files
  and the GDC export) are expected under a `data/` folder at the project root.
- Random Forest and Decision Tree use patient-level partitioning (GroupKFold) so that
  samples from the same patient never appear in both the training and the test fold.
