#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONOPTIMIZE
PYTHON=${PYTHON:-python3}
CXX=${CXX:-g++}
export CXX
"$PYTHON" -c 'import numpy; assert __debug__, "Python assertions must be enabled"'
mkdir -p .build/results
# A failed rerun must not leave an old aggregate success report.
rm -f .build/results/summary.json .build/results/manifest.json
"$PYTHON" prepare.py > .build/results/prepare.log 2>&1
"$PYTHON" run_gqa2_layout.py > .build/results/gqa2_layout.log 2>&1
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror test_gqa2_policy.cpp -o .build/test_gqa2_policy
./.build/test_gqa2_policy > .build/results/gqa2_policy_results.json
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror test_mq2_masks.cpp -o .build/test_mq2_masks
./.build/test_mq2_masks > .build/results/mq2_masks_results.json
"$PYTHON" check_negative_controls.py > .build/results/negative_controls_masks.log 2>&1
echo 'Structural tests passed. Running full CPU numerical model (several minutes)...'
"$PYTHON" mq2_rowlane_prototype.py > .build/results/mq2.log 2>&1
"$PYTHON" audit_error_budget.py > .build/results/error_budget.log 2>&1
"$PYTHON" summarize.py
