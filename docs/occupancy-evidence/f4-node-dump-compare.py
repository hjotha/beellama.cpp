import collections, sys

def parse(path):
    D = collections.defaultdict(list)
    for line in open(path):
        if not line.startswith("NODE "): continue
        r = dict(t.split("=",1) for t in line.split()[1:] if "=" in t)
        D[r["tag"]].append(r)
    return D

D = parse(sys.argv[1])
def key(r): return (int(r["eval"]), r["name"], r["op"], r["ne"], r["nb"], r["bytes"])
M = {}
for tag, rows in D.items():
    c = collections.Counter(key(r) for r in rows)
    dups = {k:v for k,v in c.items() if v>1}
    print("%-9s rows=%d unique=%d ambiguous_keys=%d" % (tag, len(rows), len(c), len(dups)))
    for k,v in list(dups.items())[:5]: print("    AMBIG", k[1], k[2], k[3], k[4], "x%d"%v)
    M[tag] = {key(r): r for r in rows}

a, b = M.get("split_a1",{}), M.get("split_a2",{})
print("\nA/A2 split_a1 vs split_a2: only1=%d only2=%d differing=%d" % (
    len(set(a)-set(b)), len(set(b)-set(a)),
    sum(1 for k in set(a)&set(b) if a[k]["hash"]!=b[k]["hash"])))

ORDER = ["l_out-2","attn_norm-3","Qcur_full-3","Kproj-3","Vproj-3","Qcur_reshaped-3",
         "Qcur_normed-3","Kcur_normed-3","gate_reshaped-3","Qcur-3","Kcur-3","Vcur-3",
         "kq_mask","inp_pos","fa_kv_k-3","fa_kv_v-3","fa_out-3","attn_pregate-3",
         "gate_sigmoid-3","attn_gated-3","attn_output-3","attn_residual-3",
         "attn_post_norm-3","ffn_out-3","post_ffn-3","l_out-3"]
def rank(n):
    for i,r in enumerate(ORDER):
        if n==r: return i
    return 999

p = M.get("pure",{})
print("\nsplit_a1 vs pure:")
print("  only_split=%d only_pure=%d" % (len(set(a)-set(p)), len(set(p)-set(a))))
onlyp = collections.Counter(k[1] for k in set(p)-set(a))
if onlyp: print("  only_pure names:", dict(onlyp))
diffs = collections.defaultdict(lambda:[0,None,None])
for k in set(a)&set(p):
    x,y = a[k], p[k]
    if x["hash"]!=y["hash"] or x["backend"]!=y["backend"] or x["dtype"]!=y["dtype"]:
        n=k[1]; diffs[n][0]+=1
        if diffs[n][1] is None: diffs[n][1]=(x,y,k)
if not diffs: print("  NO DIFFERENCES"); sys.exit()
print("  first differences in canonical operator order:")
for n in sorted(diffs, key=rank):
    x,y,k = diffs[n][1]
    print("   %-22s evals=%-4d op=%s ne=%s | hash %s vs %s | dtype %s/%s | backend %s/%s | bytes %s/%s"
          % (n, diffs[n][0], x["op"], x["ne"], x["hash"], y["hash"], x["dtype"], y["dtype"],
             x["backend"], y["backend"], x["bytes"], y["bytes"]))
