#!/usr/bin/env python3
"""Isolated, fail-closed MTP calibration runner. Python standard library only."""
import argparse, csv, hashlib, json, os, pathlib, re, socket, statistics as st
import subprocess as sp, threading, time, urllib.request
from model import fit, predict, validate, interval, validate_plan
from identity import check_identity, identity_hash, check_rows, canonical
from acceptance import from_cycles, parse_stats

def command(args):
    return sp.check_output(args,text=True,timeout=10).strip()

def gpu():
    fields="memory.free,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw"
    vals=command(["nvidia-smi","--id=0","--query-gpu="+fields,"--format=csv,noheader,nounits"]).split(",")
    return dict(zip(fields.split(","),map(float,vals)))

def guard(required):
    processes=command(["nvidia-smi","--id=0","--query-compute-apps=pid","--format=csv,noheader,nounits"])
    free=gpu()["memory.free"]
    if processes.strip() or free < required:
        raise RuntimeError(f"GPU busy/insufficient budget: compute PIDs={processes!r}; free={free} MiB, required={required}")

def api(port,path,body=None):
    req=urllib.request.Request(f"http://127.0.0.1:{port}"+path,
        None if body is None else json.dumps(body).encode(),{"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r:
        return json.load(r)

def parse_cycles(text):
    rows=[]; phases=None
    for line in text.splitlines():
        if "MTPCYCLE_BEGIN " in line:
            if phases is not None: raise ValueError("overlapping/incomplete cycle")
            begin=dict(re.findall(r"(\w+)=([\w.\-]+)",line))
            phases={"draft_us":0.,"target_us":0.,"catchup_us":0.,"accept_us":0.,"sample_us":0.}
        elif phases is not None and ("MTPPHASE " in line or "MTPTGT " in line):
            d=dict(re.findall(r"(\w+)=([\w.\-]+)",line))
            phase=d.get("phase")
            key={"draft":"draft_us","sync":"catchup_us","verify":"accept_us","verify_rollback":"accept_us","n0_sample":"accept_us"}.get(phase)
            if "MTPTGT " in line: key="target_us"
            if key: phases[key]+=float(d["us"])
            phases["sample_us"]+=float(d.get("sample_accept_us",0))
        elif "MTPCYCLE " in line:
            if phases is None: raise ValueError("binary lacks MTPCYCLE_BEGIN instrumentation")
            d=dict(re.findall(r"(\w+)=([\w.\-]+)",line))
            d={k:(v if k=="tier" else int(v)) for k,v in d.items()}
            if any(int(begin[k])!=d[k] for k in ("request","slot","cycle")):
                raise ValueError("cycle marker mismatch")
            total=sum(phases[k] for k in ("draft_us","target_us","catchup_us","accept_us"))
            d.update(phases)
            d["residual_us"]=d["wall_us"]-total
            if d["wall_us"] <= 0 or d["committed"] < 1 or d["residual_us"] < -100:
                raise ValueError("invalid/overlapping cycle accounting")
            rows.append(d); phases=None
    if phases is not None: raise ValueError("truncated cycle trace")
    return rows

def server_args(a,n,c):
    args=[a.binary,"--model",a.model,"--host","127.0.0.1","--port",str(a.port),
          "--parallel","1","--ctx-size",str(c+a.generate+32),"--batch-size",str(a.batch),
          "--ubatch-size",str(a.ubatch),"--n-gpu-layers","99","--device","CUDA0",
          "--flash-attn","on","--fit","off","--cache-ram","0","--no-context-shift",
          "--cache-type-k","kvarn4","--cache-type-v","kvarn4"]
    if n: args+=["--spec-type","draft-mtp","--spec-draft-n-max",str(n),
                  "--spec-draft-p-min","0.65","--spec-draft-type-k","kvarn4","--spec-draft-type-v","kvarn4"]
    return args

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def summarize_cycles(cycles,prompt_context,predicted_n,elapsed_s,n):
    wall=sum(r["wall_us"] for r in cycles)
    tokens=sum(r["committed"] for r in cycles)
    anchors=[r["anchor"] for r in cycles]
    ids=[(r["request"],r["slot"],r["cycle"]) for r in cycles]
    if not cycles or len(set(ids))!=len(ids) or len({(r["request"],r["slot"]) for r in cycles})!=1:
        raise ValueError("missing/duplicate/mixed cycle coverage")
    if predicted_n-tokens!=1: raise ValueError("expected exactly one bootstrap token outside cycles")
    if wall<=0 or wall>elapsed_s*1e6*1.01:
        raise ValueError("cycle wall exceeds request wall")
    if anchors!=sorted(anchors) or min(anchors)<prompt_context or max(anchors)>prompt_context+predicted_n:
        raise ValueError("invalid context anchors")
    return {"measured_context_mean":st.mean(anchors),"context_min":min(anchors),"context_max":max(anchors),
        "cycle_us":wall/len(cycles),"effective":tokens/len(cycles),"tps":tokens*1e6/wall,
        "end_to_end_tps":predicted_n/elapsed_s,
        "coverage":{"status":"complete_except_bootstrap","cycle_tokens":tokens,"bootstrap_tokens":1,
                    "response_tokens":predicted_n,"cycle_wall_us":wall,"request_wall_us":elapsed_s*1e6,
                    "cycle_tps_denominator":"sum MTPCYCLE wall_us",
                    "e2e_tps_denominator":"HTTP request start through final SSE"},
        "acceptance":from_cycles(cycles,n)}

def run_identity(a):
    supplied=json.loads(pathlib.Path(a.identity).read_text())
    check_identity(supplied)
    # Executed ONLY in an authorized window, after guard. No 10GB read in plan/tests.
    actual={"model_sha256":sha(a.model),"binary_sha256":sha(a.binary),
        "corpus_sha256":sha(a.prompt),"gpu_uuid":command(["nvidia-smi","--id=0","--query-gpu=uuid","--format=csv,noheader"]),
        "driver":command(["nvidia-smi","--id=0","--query-gpu=driver_version","--format=csv,noheader"]),
        "hardware_id":socket.gethostname()+":"+command(["uname","-m"]),
        "batch":a.batch,"ubatch":a.ubatch,"p_min":.65,"generation":a.generate,"parallel":1,
        "graphs":"disabled:GGML_CUDA_DISABLE_GRAPHS=1","backend":"CUDA","kv_target":"kvarn4/kvarn4","kv_draft":"kvarn4/kvarn4",
        "sampling":{"temperature":1.0,"top_k":20,"top_p":.95,"min_p":.05,"seed":a.seed,"ignore_eos":True}}
    manifest=json.loads(pathlib.Path(a.build_manifest).read_text())
    artifacts=manifest.get("artifacts",{})
    if str(pathlib.Path(a.binary).resolve()) not in artifacts:
        raise ValueError("binary absent from build artifact manifest")
    for path,digest in artifacts.items():
        if sha(path)!=digest: raise ValueError("build artifact digest mismatch: "+path)
    actual["build_id"]=sha(a.build_manifest)
    for k,v in actual.items():
        if canonical(supplied[k])!=canonical(v): raise ValueError("runtime identity mismatch: "+k)
    return supplied

def run_point(a,n,c,out):
    guard(a.required_free)
    # Refuse a listening endpoint, including unrelated local services.
    with socket.socket() as s: s.bind(("127.0.0.1",a.port))
    logpath=out/f"n{n}-c{c}.log"
    cmd=server_args(a,n,c)
    (out/f"n{n}-c{c}.command.json").write_text(json.dumps(cmd))
    samples=[]; stop=threading.Event(); low=threading.Event()
    with logpath.open("w") as log:
        process=sp.Popen(cmd,stdout=log,stderr=log,env={**os.environ,"GGML_MTP_PROF":"1","GGML_CUDA_DISABLE_GRAPHS":"1"})
        def monitor():
            while not stop.is_set():
                try:
                    sample={"time":time.time(),**gpu()}; samples.append(sample)
                    if sample["memory.free"] < a.reserve:
                        low.set(); process.terminate(); return
                except Exception:
                    low.set(); process.terminate(); return
                stop.wait(.25)
        t=threading.Thread(target=monitor); t.start()
        try:
            for _ in range(180):
                if process.poll() is not None: raise RuntimeError("server exited; inspect log")
                try:
                    if api(a.port,"/health").get("status")=="ok": break
                except Exception: time.sleep(1)
            else: raise RuntimeError("health timeout")
            source=pathlib.Path(a.prompt).read_text()
            ids=api(a.port,"/tokenize",{"content":source,"add_special":True})["tokens"]
            if len(ids)<c: raise ValueError("prompt corpus must contain enough tokens; no silent repetition")
            ids=ids[:c]
            prompt_hash=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            results=[]
            for rep in range(a.repeats+1):
                if low.is_set(): raise RuntimeError("VRAM reserve crossed")
                offset=logpath.stat().st_size
                payload={"prompt":ids,"n_predict":a.generate,"seed":a.seed,"temperature":1.0,
                         "top_k":20,"top_p":.95,"min_p":.05,"cache_prompt":False,
                         "ignore_eos":True,"stream":True}
                start=time.perf_counter(); first=None; final=None
                req=urllib.request.Request(f"http://127.0.0.1:{a.port}/completion",
                      json.dumps(payload).encode(),{"Content-Type":"application/json"})
                with urllib.request.urlopen(req,timeout=600) as response:
                    for line in response:
                        if not line.startswith(b"data: "): continue
                        raw=line[6:].strip()
                        if raw==b"[DONE]": continue
                        data=json.loads(raw)
                        if data.get("content") and first is None: first=time.perf_counter()
                        if data.get("stop"): final=data
                elapsed=time.perf_counter()-start
                if final is None: raise RuntimeError("missing final response")
                timings=final["timings"]
                if timings["prompt_n"] != c: raise ValueError("context mismatch or cached prefill")
                if timings["predicted_n"] != a.generate: raise ValueError("incomplete generation")
                with logpath.open() as f: f.seek(offset); cycles=parse_cycles(f.read())
                if not cycles: raise RuntimeError("no cycle telemetry: build instrumented checkout")
                (out/f"n{n}-c{c}-r{rep}.json").write_text(json.dumps({"response":final,"cycles":cycles},indent=2))
                if rep:
                    wall=sum(r["wall_us"] for r in cycles); tokens=sum(r["committed"] for r in cycles)
                    results.append({**summarize_cycles(cycles,c,timings["predicted_n"],elapsed,n),
                        "data_kind":"measured","identity":a.regime,"regime_id":identity_hash(a.regime),"n":n,"context":c,"rep":rep,"p_min":.65,"batch":a.batch,"ubatch":a.ubatch,
                        "prompt_sha256":prompt_hash,"cycle_us":wall/len(cycles),"effective":tokens/len(cycles),
                        "tps":tokens*1e6/wall,"server_tps":timings["predicted_per_second"],
                        "prefill_ms":timings["prompt_ms"],"ttft_ms":None if first is None else (first-start)*1000,
                        "elapsed_s":elapsed,"decode_tokens":tokens,"cycles":len(cycles),
                        "vram_free_min_mib":min(s["memory.free"] for s in samples),
                        "vram_used_peak_mib":max(s["memory.used"] for s in samples)})
            return results
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except sp.TimeoutExpired: process.kill(); process.wait()
            stop.set(); t.join(timeout=12)
            (out/f"n{n}-c{c}.gpu.json").write_text(json.dumps(samples,indent=2))
            (out/f"n{n}-c{c}.impl-stats.json").write_text(json.dumps(parse_stats(logpath.read_text()),indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["plan","run","analyze"])
    p.add_argument("--out",required=True); p.add_argument("--binary"); p.add_argument("--model"); p.add_argument("--prompt")
    p.add_argument("--contexts",default="4096,8192,16384,24576,32768"); p.add_argument("--ns",default="0,1,2,4,6,8")
    p.add_argument("--batch",type=int,default=2048); p.add_argument("--ubatch",type=int,default=256)
    p.add_argument("--generate",type=int,default=256); p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--seed",type=int,default=1234); p.add_argument("--port",type=int,default=18097)
    p.add_argument("--required-free",type=int,default=11500); p.add_argument("--reserve",type=int,default=768)
    p.add_argument("--inputs",nargs="+",help="analyze only: merge independently qualified rows.json files")
    p.add_argument("--identity"); p.add_argument("--build-manifest")
    p.add_argument("--qualified-point",help="N,context: run ONLY this previously reviewed point")
    p.add_argument("--window-authorized",action="store_true",help="operator has authorized the current GPU window")
    a=p.parse_args(); out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if a.repeats<3 or not 0<a.ubatch<=a.batch or a.reserve<512: p.error("repeats>=3, 0<UB<=B, reserve>=512 required")
    if a.mode in ("plan","run"):
        design=validate_plan(list(map(int,a.contexts.split(","))),list(map(int,a.ns.split(","))))
    if a.mode=="plan":
        (out/"plan.json").write_text(json.dumps({**vars(a),"design":design},indent=2)); print("Identifiable plan only; GPU execution disabled"); return
    if a.mode=="run":
        if not a.window_authorized or not a.qualified_point:
            p.error("authorized window and one explicitly qualified point required; no automatic matrix")
        if not all((a.binary,a.model,a.prompt,a.identity,a.build_manifest)):
            p.error("binary/model/prompt/identity/build-manifest required")
        n,c=map(int,a.qualified_point.split(","))
        if n not in design["ns"] or c not in design["contexts"]: p.error("point outside validated plan")
        if (out/"rows.json").exists(): p.error("refusing to overwrite existing experiment")
        guard(a.required_free)  # before model hash / process launch
        a.regime=run_identity(a)
        metadata={**vars(a),
                  "commit":command(["git","rev-parse","HEAD"]),"diff":command(["git","diff"]),
                  "gpu":gpu(),"host":socket.gethostname()}
        (out/"metadata.json").write_text(json.dumps(metadata,indent=2)); rows=[]
        rows=run_point(a,n,c,out)
        (out/"rows.json").write_text(json.dumps(rows,indent=2))
    if a.inputs:
        if a.mode!="analyze": p.error("--inputs is analyze-only")
        rows=[r for source in a.inputs for r in json.loads(pathlib.Path(source).read_text())]
        check_rows(rows)
        (out/"rows.json").write_text(json.dumps(rows,indent=2))
    else:
        rows=json.loads((out/"rows.json").read_text())
    check_rows(rows)
    with (out/"rows.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    curves=[]
    for n,c in sorted({(r["n"],r["context"]) for r in rows}):
        group=[r for r in rows if (r["n"],r["context"])==(n,c)]
        curves.append({"n":n,"context":c,"tps":interval([r["tps"] for r in group])})
    (out/"curves.json").write_text(json.dumps(curves,indent=2))
    report={"curves":curves,"holdout":[validate(rows,"n"),validate(rows,"context")]}
    try: report["model"]=fit(rows)
    except ValueError as e: report["calibration_blocked"]=str(e)
    (out/"analysis.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
