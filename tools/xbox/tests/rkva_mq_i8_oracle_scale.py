#!/usr/bin/env python3
"""Offline scale/Q8 audit of the serial-prefix probe; no network or GPU.

Replays its RNG, boundaries and rollback schedule (excluding capacity edge).
Hardware JSON contains maxima only, so componentwise relative HW error and
logit error cannot be recovered. The Q8 column is a separate algebraic model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rkva_kvarn_serial_prefix_model as sp
import rkva_kvarn_serial_prefix_probe as probe
import rkva_mq_i8_quality as Q


class NoNetwork:
    def call(self, *args, **kwargs):
        pass


class Audit(probe.Probe):
    def configure(self):
        self.layer = probe.Layer(np, sp, self.rng, self.ptt)
        # Same RNG use as the probe, with no full-capacity allocation needed.
        self.layer.canon = sp.Canon(self.rng, 512)

    def append(self, rows, reset=False):
        target = rows if reset else self.layer.live + rows
        self.layer.canon.append(target - self.layer.canon.live)
        self.layer.live = target

    def rewind(self, back):
        self.layer.live -= back
        self.layer.canon.rewind(self.layer.live)

    def expect_failure(self, label, payload_positions=None, n=3):
        self.rng.normal(0, 1, (len(payload_positions) if payload_positions else n, 24, 256))
        self.batch(1, label + ' then nq1')

    def batch(self, n, label):
        live = self.layer.live
        q = self.rng.normal(0, 1, (n, 24, 256)).astype(np.float32)
        positions = [probe.ORIGIN + live - n + i for i in range(n)]
        want, lse = self.layer.expected(q, positions)
        qr = Q.butterfly_wht(q)
        approx = want.copy()
        approx_lse = lse.copy()
        for j in range(n):
            count = live - n + j + 1
            # MQ2 only; nq1 follows exact decode. Last odd query is paired
            # with an inactive phantom, so its own prefix defines fast_hi.
            pair_count = live - n + (j // 2) * 2 + 1
            fast = max(0, min(pair_count & ~127, pair_count - 128)) & ~7
            if n == 1 or not fast:
                continue
            ks, ex = sp.serial_sources(self.layer.canon, self.layer.lay, count, False)
            vs, _ = sp.serial_sources(self.layer.canon, self.layer.lay, count, True)
            for head in range(24):
                kv = head // 6
                k = ks[:, kv].astype(np.float64)
                v = vs[:, kv].astype(np.float64)
                v[~ex] = v[~ex] @ Q.H256
                s = k @ (q[j, head].astype(np.float64) @ Q.H256) / 16
                s[ex] = k[ex] @ q[j, head].astype(np.float64) / 16
                groups = (fast + 127) // 128
                class Store:
                    pass
                store = Store()
                store.groups = groups
                store.tail_begin = groups * 128
                store.kplanes = [[Q.record_planes(b) for b in
                                  self.layer.canon.record(g, False)[2*kv:2*kv+2]]
                                 for g in range(groups)]
                s[:fast] = Q.i8_scores(qr[j, head], store, True)[:fast]
                mx = s.max()
                w = np.exp(s - mx)
                den = w.sum()
                approx[j, head] = w @ v / den
                approx_lse[j, head] = mx + np.log(den)
        delta = approx - want
        rms = float(np.sqrt(np.mean(want ** 2)))
        self.results.append(dict(label=label, live=live, nq=n,
            oracle_rms=rms, oracle_median_abs=float(np.median(abs(want))),
            oracle_p95_abs=float(np.quantile(abs(want), .95)),
            oracle_max_abs=float(abs(want).max()),
            q8_out_max_abs=float(abs(delta).max()),
            q8_out_rms=float(np.sqrt(np.mean(delta ** 2))),
            q8_relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(want), 1e-30)),
            q8_lse_max_abs=float(abs(approx_lse[np.isfinite(lse)] - lse[np.isfinite(lse)]).max())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hardware', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    audit = Audit(np, NoNetwork(), np.random.default_rng(20261266), 128, [2], False)
    audit.run()
    if args.hardware:
        hw = json.loads(args.hardware.read_text())['runs'][0]['results']
        # Repeated decode labels are matched in sequence, never by label alone.
        cursor = 0
        for row in audit.results:
            while cursor < len(hw):
                h = hw[cursor]
                cursor += 1
                if all(h.get(k) == row[k] for k in ('label', 'live', 'nq')):
                    row['hardware_maxima'] = h['vs_oracle']
                    if 'out_max_abs' in h['vs_oracle']:
                        row['hardware_max_over_oracle_rms'] = h['vs_oracle']['out_max_abs'] / row['oracle_rms']
                    break
            else:
                raise AssertionError('hardware case missing: ' + str(row))
    args.output.write_text(json.dumps(dict(seed=20261266, ptt=128,
        capacity_edge=False, note='synthetic probe distribution, not model activations',
        results=audit.results), indent=2) + '\n')
    print(f'{len(audit.results)} oracle cases -> {args.output}')


if __name__ == '__main__':
    main()
