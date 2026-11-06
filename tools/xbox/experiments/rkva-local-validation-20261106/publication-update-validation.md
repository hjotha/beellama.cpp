# Publication update validation, 2026-11-06

Parent checked: 2616d3833ba83aff67fd05d513a855c6e537f782.
All unchanged package files matched the remote Git blob hashes before integration.

Rerun in the isolated publication package:
- `python3 -m unittest discover -s qualification -v`: 10 passed.
- `python3 lds-wave64/test_lds.py`: 6 passed.
- ABBA synthetic example: `not_qualified`, as required.
- `git apply --check ../candidate.patch` in `lds-wave64/baseline`: passed.
- Input ZIP SHA256 values matched the supplied delivery hashes.
- Package manifest regenerated over all published package files, excluding itself
  and caches/build output; verified after generation.

The earlier Q_rot/WHT/MQ2 results remain historical; their suites were not rerun
for this documentation and isolated LDS/ABBA extension. No DXC/DXIL, MSVC,
GPU/console, performance or deployment test ran for the LDS candidate.
No production shader, generated header, host gate or build script was changed.
