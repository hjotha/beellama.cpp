"""Small nonnegative latency model; no third-party dependencies."""
import math
import statistics as st

def features(n, context):
    l = context / 32768
    return [1., l] if n == 0 else [1., n / 8, l, n / 8 * l]

def nnls(x, y):
    if len(x) < len(x[0]) + 2:
        raise ValueError("insufficient independent calibration points")
    # Reject rank-deficient designs; repeated/collinear probes cannot identify terms.
    matrix = [list(map(float,row)) for row in x]
    rank = 0
    for col in range(len(x[0])):
        pivot = max(range(rank,len(matrix)), key=lambda i:abs(matrix[i][col]))
        if abs(matrix[pivot][col]) < 1e-10:
            continue
        matrix[rank],matrix[pivot] = matrix[pivot],matrix[rank]
        scale=matrix[rank][col]
        matrix[rank]=[v/scale for v in matrix[rank]]
        for i in range(rank+1,len(matrix)):
            scale=matrix[i][col]
            matrix[i]=[a-scale*b for a,b in zip(matrix[i],matrix[rank])]
        rank+=1
    if rank < len(x[0]):
        raise ValueError("rank-deficient probe design")
    # Coordinate descent, nonnegative coefficients, normalized features.
    b = [0.] * len(x[0])
    for _ in range(4000):
        old = b[:]
        for j in range(len(b)):
            den = sum(row[j] ** 2 for row in x)
            if den == 0:
                raise ValueError("unidentified feature")
            b[j] = max(0., sum(row[j] * (v - sum(row[k]*b[k] for k in range(len(b)) if k != j))
                               for row, v in zip(x, y)) / den)
        if max(abs(a-c) for a,c in zip(old,b)) < 1e-7:
            break
    return b


def effective(n, p):
    """Labeled fallback only; p is fitted survival, NOT draft confidence p_min."""
    return sum(p ** k for k in range(n + 1))

def fit(rows, *, synthetic=False):
    from identity import check_rows
    identity=check_rows(rows,synthetic)
    result={"mode":"local_empirical","identity":identity,"synthetic":synthetic}
    for kind in ("n0", "mtp"):
        subset = [r for r in rows if (r["n"] == 0) == (kind == "n0")]
        groups = {}
        for r in subset:
            groups.setdefault((r["n"], r["context"]), []).append(r)
        points = [{**v[0], **{k: st.median(r[k] for r in v) for k in
                   ("cycle_us", "effective")}, "measured_context_mean":
                   st.mean(r.get("measured_context_mean",r["context"]) for r in v)} for v in groups.values()]
        if not points: raise ValueError("both N0 and MTP probes required")
        x = [features(r["n"], r["measured_context_mean"]) for r in points]
        b = nnls(x, [r["cycle_us"] for r in points])
        p = 0. if kind == "n0" else min((i/1000 for i in range(1001)),
            key=lambda p: sum((effective(r["n"],p)-r["effective"])**2 for r in points))
        result[kind] = {"coeff_us": b, "fallback_survival_p": p,
            "acceptance_points":[{"n":r["n"],"context":r["measured_context_mean"],
                                 "effective":r["effective"]} for r in points],
            "context_range": [min(r["measured_context_mean"] for r in points), max(r["measured_context_mean"] for r in points)],
            "n_values": sorted({r["n"] for r in points})}
    return result

def predict_detail(model,n,context,*,allow_extrapolation=False):
    m=model["n0" if n==0 else "mtp"]
    outside=not (m["context_range"][0]<=context<=m["context_range"][1] and
                 min(m["n_values"])<=n<=max(m["n_values"]))
    if outside and not allow_extrapolation: raise ValueError("outside local empirical domain")
    us=sum(a*b for a,b in zip(features(n,context),m["coeff_us"]))
    if us<=0: raise ValueError("nonpositive prediction")
    points=sorted((r["context"],r["effective"]) for r in m["acceptance_points"] if r["n"]==n)
    if n==0: e=1.; source="N0 exact one-token cycle"
    elif points:
        lo=max((p for p in points if p[0]<=context),default=points[0])
        hi=min((p for p in points if p[0]>=context),default=points[-1])
        e=lo[1] if hi[0]==lo[0] else lo[1]+(hi[1]-lo[1])*(context-lo[0])/(hi[0]-lo[0])
        source="empirical committed/cycle; linear context interpolation"
    else:
        e=effective(n,m["fallback_survival_p"]);source="geometric fallback for unobserved N"
    return {"tps":e*1e6/us,"cycle_us":us,"effective_tokens":e,"extrapolated":outside,
            "acceptance_source":source,"mode":"local_empirical",
            "domain":{"context":m["context_range"],"n_values":m["n_values"]},
            "identity":model["identity"],"uncertainty":"not estimated; inspect held-out errors and run dispersion"}

def predict(model,n,context,*,allow_extrapolation=False):
    return predict_detail(model,n,context,allow_extrapolation=allow_extrapolation)["tps"]

def validate(rows,axis,*,synthetic=False):
    from identity import check_rows
    check_rows(rows,synthetic)
    errors=[];folds=[]
    for value in sorted({r[axis] for r in rows}):
        if axis=="n" and value==0: continue
        train=[r for r in rows if r[axis]!=value];test=[r for r in rows if r[axis]==value]
        try: m=fit(train,synthetic=synthetic)
        except ValueError as e:
            folds.append({"holdout":value,"blocked":str(e)});continue
        # Edge holdouts deliberately test extrapolation; never hide this in output.
        preds=[predict_detail(m,r["n"],r.get("measured_context_mean",r["context"]),
                              allow_extrapolation=True) for r in test]
        err=[100*abs(p["tps"]/r["tps"]-1) for p,r in zip(preds,test)]
        errors.extend(err)
        folds.append({"holdout":value,"median_ape_percent":st.median(err),
                      "extrapolated_predictions":sum(p["extrapolated"] for p in preds),
                      "acceptance_sources":sorted({p["acceptance_source"] for p in preds})})
    return {"axis":axis,"median_ape_percent":st.median(errors) if errors else None,"folds":folds}

DEFAULT_CONTEXTS=(4096,8192,16384,24576,32768)
DEFAULT_NS=(0,1,2,4,6,8)
def validate_plan(contexts=DEFAULT_CONTEXTS,ns=DEFAULT_NS):
    contexts=sorted(set(contexts));ns=sorted(set(ns))
    if not contexts or min(contexts)<=0 or not ns or min(ns)!=0: raise ValueError("positive contexts and N0 required")
    points=[(n,c) for c in contexts for n in ns]
    designs=[("full",points)]
    designs += [("context="+str(c),[(n,l) for n,l in points if l!=c]) for c in contexts]
    designs += [("N="+str(k),[(n,l) for n,l in points if n!=k]) for k in ns if k]
    for name,pts in designs:
        for is_n0 in (True,False):
            x=[features(n,l) for n,l in pts if (n==0)==is_n0]
            if not x: raise ValueError("empty design: "+name)
            try: nnls(x,[1.]*len(x))
            except ValueError as e: raise ValueError(name+": "+str(e)) from e
    return {"identifiable":True,"context_holdouts":len(contexts),"N_holdouts":len(ns)-1,
            "contexts":contexts,"ns":ns,"execution":"one explicitly qualified point per invocation",
            "qualification_order":"4k N0/N2/N6 first; review VRAM/graphs before each expansion",
            "memory_qualified":False}

def roofline_us(weight_bytes, kv_bytes, flops, bandwidth_gbs, compute_tflops):
    if min(bandwidth_gbs,compute_tflops)<=0: raise ValueError("positive rates required")
    return max((weight_bytes+kv_bytes)/(bandwidth_gbs*1e9),flops/(compute_tflops*1e12))*1e6

def interval(values):
    m=st.median(values)
    return {"median":m,"min":min(values),"max":max(values),"mad":st.median(abs(x-m) for x in values)}
