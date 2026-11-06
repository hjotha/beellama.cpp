#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p .build results
cxx=${CXX:-c++}
for mode in default 0 1; do
 flag=""
 if [ "$mode" != default ]; then flag="-DRKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96=$mode"; fi
 "$cxx" -std=c++17 -O2 -Wall -Wextra -Werror $flag tests/policy.cpp -o ".build/policy-$mode"
 ".build/policy-$mode" > "results/policy-$mode.txt"
done
if "$cxx" -std=c++17 -DRKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96=2 tests/policy.cpp -o .build/invalid >results/invalid-switch.txt 2>&1; then
 echo 'FAIL: invalid macro accepted'; exit 1
fi
.build/policy-1 --geometry > results/geometry.csv
python3 tests/check_windows.py > results/windows.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 tests/numeric.py > results/numeric-summary.json
cat results/policy-*.txt results/windows.txt results/numeric-summary.json
