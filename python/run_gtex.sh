#!/bin/bash
cd "$(dirname "$0")"

echo "=== gtex_integration.py ==="
python3 gtex_integration.py 2>&1 | grep -E '(Saved|ERROR|Skipping|vs_gtex|cohort|TCGA)'

echo ""
echo "=== predict_gtex.py ==="
python3 predict_gtex.py 2>&1 | grep -E '(Saved|ERROR|F1|Macro|GTEx|high.risk|cohort|TCGA)'

echo ""
echo "=== gtex_demographics.py ==="
python3 gtex_demographics.py 2>&1 | tail -5

echo ""
echo "=== ALL DONE ==="
