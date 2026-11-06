# Portable package validation, 2026-11-06 UTC

Fresh command: `sh run_all.sh`, run from the package directory. Exit status 0.
The script commands and compiler flags are in `run_all.sh`.

- 40 numerical scenarios / 104 runs; 36 local-two-head vs local-six-head bitwise comparisons.
- Six GQA2 preprocessor/layout tests.
- 5,598,720 policy combinations plus four flag combinations.
- 895,320 selectors; 329,855,172 row classifications; 179,064 phantom checks.
- Numerical phase: 274.38158066498 seconds, not a GPU benchmark.
- Maximum absolute OUT error: 0.00018272082498071995.
- Maximum absolute LSE error: 1.4576248370090639e-05.
- Mutation controls were detected and official error limits were unchanged.
- Aggregate numerical, geometry, negative-control, policy and layout results exactly match historical results after excluding CPU elapsed time.

Extra smoke validation: copied only package files into a nested temporary
location, excluding generated outputs, and called `python3 prepare.py` and
`python3 run_gqa2_layout.py` there from a different working directory. Both
exited 0. The comparison function's AST source matched its original excerpt
SHA-256, and all distributed Python files parsed successfully.

`manifest.json` identifies the tested package sources and runtime versions.
`summary.json` is the fresh compact result; `comparison_to_historical.json`
records the historical comparison. Per-run logs, binaries and reconstructed
outputs were removed after preserving these compact results. Regenerate them
with `sh run_all.sh`; they stay under ignored `.build/`.

No HLSL/DXIL compilation, generated-header provenance check, current-main
integration, Windows/UWP build, PSO, GPU/Xbox run, real-model quality check,
optional long numeric case or optional contracted-PV run was performed.
