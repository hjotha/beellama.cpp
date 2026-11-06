#!/bin/sh
# CPU-only: requires a POSIX shell, Python 3 + NumPy, g++ and cpp.
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
export CXX="${CXX:-g++}" CPP="${CPP:-cpp}"
export QROT_BUILD_DIR="$PWD/.build"
mkdir -p "$QROT_BUILD_DIR"
rm -f "$QROT_BUILD_DIR/results.json" "$QROT_BUILD_DIR/structural-results.json" \
    "$QROT_BUILD_DIR/transform-results.json"
python3 verify_sources.py
"$CXX" -std=c++17 -O2 -fno-fast-math -ffp-contract=off -shared -fPIC \
    cpu_reference.cpp -o "$QROT_BUILD_DIR/cpu_reference.so"
python3 test_structure.py
python3 test_qrot.py
python3 verify_sources.py
printf '\nPASS: CPU-only suite completed. Results: .build/results.json\n'
