#!/usr/bin/env python3
"""Check new MQ split arithmetic against the independent F64 attention oracle.

This reuses per-query F32 emulation to validate split/reducer numerics; it does
not emulate inter-query LDS sharing or replace a compiled-shader console probe.
Run with --long for the 6k/16k/35k cache ladder. Default cases cover split edges.
"""
import argparse
import os

# Keep the CPU-only oracle within one BLAS worker.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import rkva_range_kvarn4_opt_model as m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(20261020)
    counts = [1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 513]
    if args.long:
        counts += [6144, 16384, 16385, 34816]
    worst_out = worst_lse = 0.0
    for active in counts:
        lay = m.Layout(96256, active + 64, 128)
        kb, vb = m.build_plane(rng, lay, active), m.build_plane(rng, lay, active)
        kp, vp = m.Plane(kb, lay, active), m.Plane(vb, lay, active)
        ks, exact = m.oracle_sources(lay, kb, active, value=False)
        vs, _ = m.oracle_sources(lay, vb, active, value=True)
        q = rng.normal(0, 1, (m.N_HEAD, m.D)).astype(np.float32)
        pos = lay.origin + active - 1
        reference, lse_reference = m.oracle_attention(q, ks, vs, exact, active, 1.0 / 16.0)
        for mode, tile in ((1, 4), (2, 2)):
            split, _, output, lse = m.run_optimized(
                kp, vp, q, active, pos, 1.0 / 16.0, mode, 64, ks, vs,
                prefix=True, mq_tile=tile,
            )
            assert split == m.kvarn4_mq_split(active, mode, tile)[0]
            scale = max(1.0, float(np.max(np.abs(reference))))
            err_out = float(np.max(np.abs(output - reference))) / scale
            err_lse = float(np.max(np.abs(lse - lse_reference)))
            # Same order-of-magnitude gate as the existing optimized-kernel oracle.
            assert err_out <= 2e-5 and err_lse <= 2e-5, (active, mode, err_out, err_lse)
            worst_out, worst_lse = max(worst_out, err_out), max(worst_lse, err_lse)
    print(f"MQ split oracle: passed counts={counts} max_scaled_O={worst_out:.3e} max_LSE={worst_lse:.3e}")


if __name__ == "__main__":
    main()
