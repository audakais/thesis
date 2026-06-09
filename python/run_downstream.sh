#!/bin/bash
cd "$(dirname "$0")"

run() {
    echo "=== $1 ==="
    python3 "$1" 2>&1 | tail -4
    echo ""
}

run shap_analysis.py
run biomarker_overlap.py
run learning_curve.py
run multi_class_pancancer.py
run expression_heatmap.py
run decision_tree_analysis.py
run survival_analysis.py
run tumor_incidence_analysis.py
run pathway_enrichment.py
run literature_validation.py
run tcga_demographics.py
run randomforest_leaf_sweep.py
run ffnn_survival_triage.py

echo "=== ALL DONE ==="
