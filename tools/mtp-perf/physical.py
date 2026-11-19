"""Cross-architecture physical path. All hardware/shape parameters supplied with provenance.
No constants or calibrated profiles for RTX4070/T4 are bundled.
"""
import argparse, json, math
from acceptance import expected
from model import roofline_us

def bound(param,side):
    if not isinstance(param,dict) or not param.get("source"): raise ValueError("parameter provenance required")
    lo,mid,hi=(param[k] for k in ("low","value","high"))
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (lo,mid,hi)):
        raise ValueError("finite bounds required")
    if not 0<=lo<=mid<=hi: raise ValueError("invalid parameter bounds")
    return param[side]

def shape(term,n,l):
    # Explicit physical work model, units named in caller. No implicit nominal GPU scaling.
    if not term.get("source"): raise ValueError("physical work provenance required")
    value=sum(term.get(k,0)*v for k,v in (("constant",1),("per_n",n),("per_context",l),("per_n_context",n*l)))
    if value<0 or not math.isfinite(value): raise ValueError("invalid physical work")
    return value

def predict_physical(profile,workload,acceptance,n,context,allow_extrapolation=False):
    if profile.get("mode") not in ("prior_transfer","target_calibrated"): raise ValueError("unknown mode")
    if not profile.get("hardware_id") or not workload.get("signature"): raise ValueError("identity required")
    domain=workload["domain"]
    outside=not (domain["context"][0]<=context<=domain["context"][1] and n in domain["n_values"])
    if outside and not allow_extrapolation: raise ValueError("outside physical workload domain")
    if profile["mode"]=="target_calibrated":
        evidence=profile.get("calibration",{})
        if evidence.get("hardware_id")!=profile["hardware_id"] or evidence.get("workload_signature")!=workload["signature"] or not evidence.get("probe_ids"):
            raise ValueError("target calibration evidence missing/mismatched")
        cd=evidence.get("domain",{})
        if "context" not in cd or "n_values" not in cd: raise ValueError("calibration domain missing")
        outside=outside or not (cd["context"][0]<=context<=cd["context"][1] and n in cd["n_values"])
        if outside and not allow_extrapolation: raise ValueError("outside target calibration domain")
    elif not profile.get("transfer_assumptions"):
        raise ValueError("prior transfer assumptions required")
    if acceptance.get("workload_signature")!=workload["signature"]: raise ValueError("acceptance workload mismatch")
    alo,ahi=acceptance["context_domain"]
    if not alo<=context<=ahi and not allow_extrapolation: raise ValueError("acceptance context extrapolation")
    tokens=expected(acceptance,n)
    # Explicit sensitivity bounds on effective emission (not statistical confidence).
    tl,th=acceptance["effective_bounds"]
    if not 0<tl<=tokens<=th: raise ValueError("invalid effective-token bounds")
    profiles=[]
    for side in ("low","value","high"):
        # cost low uses high rates; cost high uses low rates.
        rate_side={"low":"high","value":"value","high":"low"}[side]
        parts={"draft_us":0.,"verify_us":0.,"sample_us":0.,"catchup_us":0.,
               "host_us":bound(profile["host_us"],side),"launch_us":0.,"communication_us":0.}
        kernel_parts=[]
        for kernel in workload["kernels"]:
            if n==0 and kernel.get("mtp_only"): continue
            family=profile["families"][kernel["family"]]
            if family.get("kernel_signature")!=kernel["kernel_signature"]:
                raise ValueError("kernel family signature mismatch")
            # Peak/spec rates * dimensionless family efficiencies; target-calibrated
            # values must originate from serving-matched probes recorded above.
            bw=bound(profile["bandwidth_gbs"],rate_side)*bound(family["bandwidth_efficiency"],rate_side)
            compute=bound(profile["compute_tflops"],rate_side)*bound(family["compute_efficiency"],rate_side)
            if min(bw,compute)<=0: raise ValueError("positive effective hardware rates required")
            if max(bound(family["bandwidth_efficiency"],"high"),bound(family["compute_efficiency"],"high"))>1:
                raise ValueError("efficiency exceeds one")
            weights=shape(kernel["weight_bytes"],n,context)
            kv=shape(kernel["kv_bytes"],n,context)
            flops=shape(kernel["flops"],n,context)
            launches=shape(kernel["launches"],n,context)
            memory_us=(weights+kv)/(bw*1e9)*1e6
            compute_us=flops/(compute*1e12)*1e6
            cost=roofline_us(weights,kv,flops,bw,compute)
            phase=kernel["phase"]+"_us"
            if phase not in ("draft_us","verify_us","sample_us","catchup_us"): raise ValueError("unknown phase")
            parts[phase]+=cost
            parts["launch_us"]+=launches*bound(profile["launch_us"],side)
            kernel_parts.append({"family":kernel["family"],"phase":phase,"us":cost,
                                 "memory_bound_us":memory_us,"compute_bound_us":compute_us})
        comm=workload["communication"]
        count=shape(comm["messages"],n,context)
        amount=shape(comm["bytes"],n,context)
        if count or amount:
            if profile["mode"]=="target_calibrated" and not profile["calibration"].get("communication_probe_ids"):
                raise ValueError("target communication calibration missing")
            bw=bound(profile["communication_gbs"],rate_side)
            if bw<=0: raise ValueError("communication bandwidth missing")
            parts["communication_us"]=count*bound(profile["communication_latency_us"],side)+amount/(bw*1e9)*1e6
        total=sum(parts.values())
        if total<=0: raise ValueError("nonpositive cycle cost")
        profiles.append({"cycle_us":total,"breakdown":parts,"kernels":kernel_parts})
    low,central,high=profiles
    return {"mode":profile["mode"],"data_kind":profile.get("data_kind","prior"),"hardware_id":profile["hardware_id"],"workload_signature":workload["signature"],
        "tps":tokens*1e6/central["cycle_us"],"effective_tokens":tokens,**central,
        "sensitivity_tps":[tl*1e6/high["cycle_us"],th*1e6/low["cycle_us"]],
        "interval_kind":"parameter sensitivity envelope, not a confidence interval",
        "extrapolated":outside or not alo<=context<=ahi,
        "domain":domain,"acceptance_source":acceptance["source"],
        "parameter_provenance":{"profile":profile,"workload":workload,"acceptance":acceptance},
        "limitations":["serialized critical-path assumption; overlap must be calibrated",
                       "shape/wave/SM effects are family/domain-specific efficiencies",
                       "no accuracy claim without serving-matched target holdout"]}

def main():
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("--out",required=True)
    a=p.parse_args();d=json.load(open(a.input))
    result=predict_physical(d["profile"],d["workload"],d["acceptance"],d["n"],d["context"])
    with open(a.out,"w") as f:json.dump(result,f,indent=2)

def calibrate_target(profile,workload,probes,*,synthetic=False):
    """Requires 3 device-only memory-bound AND compute-bound probes per family."""
    import copy, statistics
    result=copy.deepcopy(profile)
    if not probes: raise ValueError("no target probes")
    for p in probes:
        if p.get("data_kind") != ("synthetic" if synthetic else "measured"):
            raise ValueError("probe data kind mismatch; explicit synthetic mode required")
        if p.get("hardware_id")!=profile["hardware_id"] or p.get("workload_signature")!=workload["signature"] or not p.get("probe_id"):
            raise ValueError("probe identity mismatch")
        if p["device_us"]<=0: raise ValueError("invalid device timing")
    def measured(values,source):
        if len(values)<3: raise ValueError("three probes per identifiable parameter required")
        return {"low":min(values),"value":statistics.median(values),"high":max(values),"source":source}
    for name,family in result["families"].items():
        for kind,field,rate,amount,scale in (
            ("memory","bandwidth_efficiency","bandwidth_gbs","bytes",1e3),
            ("compute","compute_efficiency","compute_tflops","flops",1e6)):
            selected=[p for p in probes if p["family"]==name and p["bound"]==kind]
            if any(p["kernel_signature"]!=family["kernel_signature"] for p in selected):
                raise ValueError("probe kernel mismatch")
            vals=[p[amount]/(p["device_us"]*bound(profile[rate],"value")*scale) for p in selected]
            if any(not 0<v<=1 for v in vals): raise ValueError("invalid measured efficiency")
            family[field]=measured(vals,"target device probes: "+",".join(p["probe_id"] for p in selected))
    for field in ("host_us","launch_us"):
        result[field]=measured([p[field] for p in probes],"target host timeline probes")
    result["mode"]="target_calibrated"
    result["data_kind"]="synthetic" if synthetic else "measured"
    result["calibration"]={"hardware_id":profile["hardware_id"],"workload_signature":workload["signature"],
        "probe_ids":[p["probe_id"] for p in probes],
        "domain":{"context":[min(p["context"] for p in probes),max(p["context"] for p in probes)],
                  "n_values":sorted({p["n"] for p in probes})}}
    if any(v for term in workload["communication"].values() for key,v in term.items() if key!="source"):
        comm=profile.get("communication_calibration",{})
        if comm.get("hardware_id")!=profile["hardware_id"] or not comm.get("probe_ids"):
            raise ValueError("separate target communication probes required")
        result["calibration"]["communication_probe_ids"]=comm["probe_ids"]
    return result

if __name__=="__main__":main()
