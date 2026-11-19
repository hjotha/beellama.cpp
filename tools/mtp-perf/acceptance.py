"""Empirical survival is unconditional per completed cycle, never confidence p_min."""
import re
def from_cycles(cycles,n):
    if not cycles: raise ValueError("no cycles")
    if n<0: raise ValueError("negative cap")
    committed=sum(c["committed"] for c in cycles)
    # Replays/multiple passes may count an accepted token twice. Preserve actual
    # emission but do not pretend these counters identify per-position survival.
    eligible=[c for c in cycles if c["passes"]==1 and not c["stopped"]
              and 0<=c["accepted_drafts"]<=c["proposed"]<=n
              and 0<=c["committed"]-c["accepted_drafts"]<=1]
    result={"source":"MTPCYCLE committed","effective":committed/len(cycles),
            "cycles":len(cycles),"position_cycles":len(eligible),"n_max":n,
            "all_cycles_included_in_effective":True}
    if not eligible:
        result["position_status"]="unidentified: replay, stop or noncanonical bonus; use empirical emission"
        return result
    k=len(eligible)
    result.update(source="MTPCYCLE position survival",position_status="identified on eligible cycles only; inspect excluded count",
        excluded_position_cycles=len(cycles)-len(eligible),
        survival=[sum(c["accepted_drafts"]>=i for c in eligible)/k for i in range(1,n+1)],
        proposal_survival=[sum(c["proposed"]>=i for c in eligible)/k for i in range(1,n+1)],
        bonus_mean=sum(c["committed"]-c["accepted_drafts"] for c in eligible)/k)
    return result

def from_counters(steps,generated,accepted,bonus_mean,source):
    if steps<=0 or not source: raise ValueError("counter denominator/provenance required")
    n=max(len(generated),len(accepted))
    generated=list(generated)+[0]*(n-len(generated))
    accepted=list(accepted)+[0]*(n-len(accepted))
    result={"source":source,"n_max":n,"survival":[x/steps for x in accepted],
            "proposal_survival":[x/steps for x in generated],"bonus_mean":bonus_mean,
            "steps":steps}
    expected(result,n)  # validate; acc/gen is conditional and is NOT used here.
    return result

def parse_stats(text):
    """Cumulative impl-level diagnostics; includes warmup. Never treated as request deltas."""
    rows=[]
    for line in text.splitlines():
        if "MTPSTATS " not in line: continue
        fields=dict(re.findall(r"(\w+)=([^ ]+)",line))
        rows.append({"type":fields["type"],"steps":int(fields["steps"]),
            "generated":[int(x) for x in fields["gen_per_pos"].strip("[]").split(",") if x],
            "accepted":[int(x) for x in fields["acc_per_pos"].strip("[]").split(",") if x],
            "scope":"cumulative implementation; includes warmup; bonus not supplied"})
    return rows

def expected(profile,n):
    if not profile.get("source"): raise ValueError("acceptance provenance required")
    if n<0 or n>profile["n_max"]: raise ValueError("acceptance proposal cap out of domain")
    if "survival" not in profile:
        if n!=profile["n_max"]: raise ValueError("empirical effective only valid at observed cap")
        return profile["effective"]
    s=profile["survival"]; g=profile["proposal_survival"]
    if len(s)<profile["n_max"] or len(g)<profile["n_max"]: raise ValueError("incomplete position counts")
    if any(not 0<=a<=b<=1 for a,b in zip(s,g)): raise ValueError("invalid survival")
    if any(a<b for seq in (s,g) for a,b in zip(seq,seq[1:])): raise ValueError("survival must be nonincreasing")
    bonus=profile["bonus_mean"]
    if not 0<=bonus<=1: raise ValueError("invalid bonus")
    return bonus+sum(s[:n])
