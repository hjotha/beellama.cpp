import collections
import re
import statistics

log = open('/home/hjotha/beellama.cpp/docs/benchmarks/mtp-dflash2-shadow-early-20260929/shadow-every4-early-on.server.log', errors='replace').read()

pattern = re.compile(
    r'SHADOWv2 job=(\d+) epoch=(\d+) request=(\d+) pos0=(\d+) anchor=(\d+) c=(\d+) d=(\d+) '
    r'matched=(\d+) prefix_ok=(\d+) position_ok=(\d+) ready=(\d+) remaining=(\d+) usable=(\d+) '
    r'usable_tokens=(\d+) done_us=(\d+) decision_us=(\d+) work_us=(\d+) dec_us=(\d+) sel_us=(\d+) '
    r'confirmed=\[([^\]]*)\] proposed=\[([^\]]*)\]')

jobs = []
for m in pattern.finditer(log):
    g = m.groups()
    jobs.append({
        'id': int(g[0]), 'epoch': int(g[1]), 'request': int(g[2]),
        'pos0': int(g[3]), 'anchor': int(g[4]), 'c': int(g[5]), 'd': int(g[6]),
        'matched': int(g[7]), 'prefix_ok': int(g[8]), 'position_ok': int(g[9]),
        'ready': int(g[10]), 'remaining': int(g[11]), 'usable': int(g[12]),
        'usable_tokens': int(g[13]),
        'confirmed': [int(x) for x in g[19].split(',')] if g[19] else [],
        'proposed': [int(x) for x in g[20].split(',')] if g[20] else [],
    })

print("jobs:", len(jobs))

# position -> token map from every job's confirmed round
pos_map = {}
for j in jobs:
    for i, tok in enumerate(j['confirmed']):
        pos_map[j['pos0'] + 1 + i] = tok

usable = [j for j in jobs if j['usable'] == 1]
prefix = [j for j in jobs if j['prefix_ok'] == 1]

rows = []
missing_hits = 0
for j in usable:
    c, d = j['c'], j['d']
    ut = j['usable_tokens']
    suffix = j['proposed'][c:c + ut]
    L = 0
    missing = False
    for k in range(len(suffix)):
        tok = pos_map.get(j['pos0'] + c + 1 + k)
        if tok is None:
            missing = True
            break
        if tok != suffix[k]:
            break
        L += 1
    gain = (L + 1) - c
    missing_hits += missing
    rows.append((j['pos0'], c, len(suffix), L, gain, missing))

Ls = [r[3] for r in rows]
gains = [r[4] for r in rows]
print("usable jobs:", len(usable), "| prefix_ok jobs:", len(prefix))
print("suffix first-token match (L>=1):", sum(1 for L in Ls if L >= 1), f"({100*sum(1 for L in Ls if L>=1)/len(Ls):.1f}%)" if Ls else 0)
print("L distribution:", dict(sorted(collections.Counter(Ls).items())))
print("mean L:", round(statistics.mean(Ls), 2) if Ls else None, "| median L:", statistics.median(Ls) if Ls else None)
print("gain distribution:", dict(sorted(collections.Counter(gains).items())))
print("gain > 0:", sum(1 for g in gains if g > 0), f"({100*sum(1 for g in gains if g>0)/len(gains):.1f}%)" if gains else 0)
print("mean gain:", round(statistics.mean(gains), 2) if gains else None)
print("total MTP tokens in these rounds (sum c):", sum(j['c'] for j in usable))
print("total hybrid tokens (sum L+1):", sum(L + 1 for L in Ls))
print("comparisons truncated by missing continuation:", missing_hits)

# c distribution among usable
print("c dist usable:", dict(sorted(collections.Counter(j['c'] for j in usable).items())))