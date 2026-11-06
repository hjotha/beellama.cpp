"""Exact CPU comparison gate excerpt; no remote probe or socket functionality.

Original source and excerpt SHA-256 are recorded in provenance.json.
"""

def compare(np, got_out, got_lse, want_out, want_lse):
    finite = np.isfinite(want_lse)
    if not np.array_equal(np.isfinite(got_lse), finite):
        return dict(ok=False, reason='LSE finiteness mismatch')
    if not np.all(np.isfinite(got_out)):
        return dict(ok=False, reason='non-finite output')
    ok = bool(np.all(np.abs(got_out - want_out) <= 3e-4 + 3e-4 * np.abs(want_out)) and
              np.all(np.abs(got_lse[finite] - want_lse[finite]) <=
                     3e-4 + 3e-5 * np.abs(want_lse[finite])))
    return dict(ok=ok, out_max_abs=float(np.max(np.abs(got_out - want_out))),
                lse_max_abs=float(np.max(np.abs(got_lse[finite] - want_lse[finite])))
                if finite.any() else 0.0)
