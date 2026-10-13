# F5/F7 auxiliary preparation handoff — 2026-10-05

Purpose: avoid reimplementing completed auxiliary preparation. Read this at the F5/F7 checkpoints. Review compatibility, then reuse the existing tools/runbooks where appropriate. This document does not authorize blind patch application, promotion, deployment, or service changes.

## Provenance and boundaries
- Root: /home/hjotha/occupancy-aux-20261005
- Base: dirty snapshot of /home/hjotha/beellama.cpp, branch feat/occupancy-position-split, HEAD d9db0b88155ec4350942306696378c57ac0485c8.
- Captured: 2026-10-05T01:16:33.317873+00:00; 6421 source files. This is NOT a clean HEAD baseline.
- Original source manifest: /home/hjotha/occupancy-aux-20261005/snapshot-manifest.json.
- All 6421 pre-existing snapshot files were independently verified unchanged at completion. Auxiliaries added only the listed files, without GPU tests, heavy builds, service actions, or canonical source edits.
- Full per-file hashes: deliverables/f5-review-summary.json and deliverables/f7-review-summary.json beneath the root above.
- Patch/archive hashes were independently rechecked at 03:22 UTC with no drift.
- Main session retains F4, build, GPU, and authorized production lifecycle ownership. Both preparation tasks are complete and intentionally idle.

## F5: CPU benchmark/telemetry preparation
Source directory: /home/hjotha/occupancy-aux-20261005/f5-benchmark

Artifacts:
- deliverables/f5-preparation.patch
  SHA256 062b8f840a643a17b2e4b6d05444a502c11e992901d01aec183167190084a49b
- deliverables/f5-preparation.tar.gz
  SHA256 a84331aab99e43422993f9cea72b07916bd2e2dcebf7a6d2140645123f0f9864
- deliverables/f5-review-summary.json

Added files:
- docs/occupancy-f5-aux-report.md
- tools/occupancy-f5/README.md
- tools/occupancy-f5/f5_harness.py
- tools/occupancy-f5/fixtures/complete-synthetic.json
- tools/occupancy-f5/fixtures/expected-context.json
- tools/occupancy-f5/fixtures/insufficient-repetitions.json
- tools/occupancy-f5/fixtures/make_fixtures.py
- tools/occupancy-f5/fixtures/malformed.json
- tools/occupancy-f5/fixtures/missing-evidence.json
- tools/occupancy-f5/fixtures/nonfinite.json
- tools/occupancy-f5/fixtures/wrong-provenance.json
- tools/occupancy-f5/tests/test_f5_harness.py

Independent validation: python3 -B -m unittest discover -s tools/occupancy-f5/tests — 55 tests, exit 0.
Independent bypass probes were repaired: empty expected provenance rejects; one-token reuse plus full-prefix re-prefill fails all restore gates even with consistently rebound evidence.
Includes both-arm warmup, strict context/provenance, quality/capacity/restore/cycle evidence bindings, absence of Vulkan attention copies below P, and swap-in/out checks.
This is CPU fixture/schema/gate arithmetic preparation ONLY. Genuine F5 acceptance is false until the actual benchmark/quality/memory campaign is run and reviewed on the final build.

## F7: rollback/provenance preparation
Source directory: /home/hjotha/occupancy-aux-20261005/f7-rollback

Artifacts:
- deliverables/f7-preparation.patch
  SHA256 dd8c0c88281b0f9b8a13920a240e23fdce424ebbbbb7cf427b5b4f1d025df7ee
- deliverables/f7-preparation.tar.gz
  SHA256 e6e97d1fe754fd1a800677b6b858c8169e5f822e10fdb8d94c95da47e0b37551
- deliverables/f7-review-summary.json

Added files:
- docs/occupancy-f7-aux-report.md
- tools/occupancy-f7/README.md
- tools/occupancy-f7/aux-snapshot-verification.json
- tools/occupancy-f7/elfdeps.py
- tools/occupancy-f7/f7-inventory-observed.json
- tools/occupancy-f7/f7-readiness-report.json
- tools/occupancy-f7/f7_inventory.py
- tools/occupancy-f7/test_f7_inventory.py
- tools/occupancy-f7/test_f7_regressions.py

Independent validation:
- python3 -B tools/occupancy-f7/test_f7_inventory.py — 93 tests, exit 0
- python3 -B tools/occupancy-f7/test_f7_regressions.py — 66 tests, exit 0
- Total 159. Fixture ResourceWarnings about unclosed files remain warnings, not failures.
Repaired arbitrary verified-string acceptance and missing mandatory gates (independent probes exit 3), partial-snapshot false integrity, unsafe paths/symlink escapes, canonical build-directory containment, and missing baseline/config/closure evidence.
Actual F7 remains NOT READY: baseline reconstruction, config hashes, runtime closure, XXL/XXXL rollback and canary are unrun. DT_NEEDED closure is not proof of runtime dlopen, loader resolution, or startup. Reconstructing a baseline does not recover an overwritten original binary.

## Reuse checklist
1. At F5/F7, read the corresponding report, README, and manifest before designing replacement tooling.
2. Revalidate artifact hashes and per-file provenance.
3. Review patch applicability and compatibility against the CURRENT dirty main tree and final interfaces.
4. Reuse selectively only within authorization; never copy the whole snapshot or overwrite core/F4/checklist work.
5. Rerun preparation tests after integration, then collect real acceptance evidence separately.
6. Keep preparation completeness separate from F5 acceptance and F7 promotion readiness. No automatic promotion/deployment.
