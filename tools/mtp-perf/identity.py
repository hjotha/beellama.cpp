"""Strict experiment identity, shared by JSON/CSV and local fit."""
import hashlib, json
FIELDS = ("hardware_id","gpu_uuid","driver","model_sha256","binary_sha256","build_id",
          "backend","kernel_profile","kv_target","kv_draft","graphs","sampling",
          "batch","ubatch","p_min","corpus_sha256","generation","parallel")
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"))
def check_identity(identity):
    for key in FIELDS:
        if key not in identity or identity[key] is None or identity[key] == "":
            raise ValueError("missing identity: "+key)
    for key in ("hardware_id","gpu_uuid","driver","build_id","backend","kernel_profile","graphs"):
        if isinstance(identity[key],str) and identity[key].lower() in ("unknown","none","unverified","n/a"):
            raise ValueError("unqualified identity: "+key)
    if not isinstance(identity["sampling"],dict) or not identity["sampling"]:
        raise ValueError("explicit sampling settings required")
    for key in ("model_sha256","binary_sha256","corpus_sha256"):
        value=identity[key]
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid digest: "+key)
    if identity["parallel"]!=1: raise ValueError("single-request telemetry only")
    return {k:identity[k] for k in FIELDS}
def identity_hash(identity):
    return hashlib.sha256(canonical(check_identity(identity)).encode()).hexdigest()
def check_rows(rows, synthetic=False):
    if not rows: raise ValueError("empty dataset")
    if synthetic:
        if any(r.get("data_kind")!="synthetic" for r in rows):
            raise ValueError("synthetic bypass requires every row explicitly marked synthetic")
        return {"data_kind":"synthetic"}
    first=None; prompts={}
    for row in rows:
        if row.get("data_kind")!="measured": raise ValueError("real rows must be marked measured")
        identity=check_identity(row.get("identity",{}))
        if row.get("regime_id")!=identity_hash(identity): raise ValueError("identity digest mismatch")
        if first is None: first=identity
        if canonical(first)!=canonical(identity): raise ValueError("incompatible calibration regimes")
        for key in ("n","context","measured_context_mean","context_min","context_max",
                    "cycle_us","effective","tps","coverage"):
            if key not in row: raise ValueError("missing real row field: "+key)
        if row["coverage"].get("status")!="complete_except_bootstrap":
            raise ValueError("incomplete cycle coverage")
        for key in ("batch","ubatch","p_min"):
            if key in row and row[key]!=identity[key]: raise ValueError("row/identity mismatch: "+key)
        h=row.get("prompt_sha256")
        if not isinstance(h,str) or len(h)!=64: raise ValueError("missing prompt identity")
        # Corpus is invariant globally; token slice is invariant only at same prompt length.
        if row["context"] in prompts and prompts[row["context"]]!=h:
            raise ValueError("different token slices at the same context")
        prompts[row["context"]]=h
    return first
