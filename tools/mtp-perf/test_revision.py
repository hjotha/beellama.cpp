"""CPU-only fixtures. Numbers below are fictional, not hardware predictions."""
import copy, json, unittest
from unittest.mock import patch
from identity import *
from acceptance import *
from physical import predict_physical,calibrate_target
from model import validate_plan,fit,predict_detail,features,effective
from bench import summarize_cycles,parse_cycles

def identity():
    d={k:"fixture" for k in FIELDS}
    d.update(model_sha256="a"*64,binary_sha256="b"*64,corpus_sha256="c"*64,
             parallel=1,batch=2048,ubatch=256,p_min=.65,generation=256,
             sampling={"seed":1234,"temperature":1.},graphs="disabled-explicit")
    return d
def rows():
    ident=identity()
    result=[]
    for c in (4096,8192,16384,24576,32768):
        for n in (0,1,2,4,6,8):
            us=sum(a*b for a,b in zip(features(n,c+100),[9000,2000] if not n else [9000,4000,2000,3000]))
            e=effective(n,.8)
            result.append(dict(data_kind="measured",identity=copy.deepcopy(ident),regime_id=identity_hash(ident),
                n=n,context=c,measured_context_mean=c+100,context_min=c,context_max=c+200,
                cycle_us=us,effective=e,tps=e*1e6/us,prompt_sha256=("%064x"%c),
                coverage={"status":"complete_except_bootstrap"}))
    return result
def parameter(x,spread=.2):
    return {"low":x*(1-spread),"value":x,"high":x*(1+spread),"source":"fictional CPU-test fixture"}
def term(x=0,**kw):return {"constant":x,**kw,"source":"fictional shape fixture"}
def inputs():
    profile={"mode":"prior_transfer","hardware_id":"fictional-A","transfer_assumptions":["fictional efficiency transfer"],
        "bandwidth_gbs":parameter(100),"compute_tflops":parameter(10),
        "host_us":parameter(10),"launch_us":parameter(2),
        "communication_gbs":parameter(1),"communication_latency_us":parameter(5),
        "families":{"weights":{"kernel_signature":"fixture-matvec","bandwidth_efficiency":parameter(.5),
                              "compute_efficiency":parameter(.5)}}}
    work={"signature":"fictional-work","domain":{"context":[100,1000],"n_values":[0,1,2]},
        "kernels":[{"family":"weights","kernel_signature":"fixture-matvec","phase":"verify",
            "weight_bytes":term(1e8),"kv_bytes":term(per_context=1024),"flops":term(1e6),
            "launches":term(3)},
            {"family":"weights","kernel_signature":"fixture-matvec","phase":"draft","mtp_only":True,
            "weight_bytes":term(per_n=1e6),"kv_bytes":term(per_n_context=128),
            "flops":term(per_n=1e5),"launches":term(per_n=1)}],
        "communication":{"messages":term(),"bytes":term()}}
    acc={"source":"fictional measured survival","workload_signature":work["signature"],"context_domain":[100,1000],
        "n_max":2,"survival":[.8,.4],"proposal_survival":[1,.7],"bonus_mean":1,"effective_bounds":[2.,2.4]}
    return profile,work,acc

class RevisionTests(unittest.TestCase):
    def test_plan_default_holdouts(self):
        d=validate_plan()
        self.assertEqual(d["context_holdouts"],5);self.assertFalse(d["memory_qualified"])
        with self.assertRaises(ValueError):validate_plan([4096,16384,32768],[0,2,6])
    def test_identity_required_and_mixed(self):
        for key in FIELDS:
            r=rows();del r[0]["identity"][key]
            with self.assertRaises(ValueError,msg=key):fit(r)
        r=rows();r[0]["identity"]["gpu_uuid"]="other";r[0]["regime_id"]=identity_hash(r[0]["identity"])
        with self.assertRaises(ValueError):fit(r)
        r=rows();r[0]["prompt_sha256"]="f"*64
        with self.assertRaises(ValueError):fit(r)
    def test_synthetic_bypass_explicit(self):
        with self.assertRaises(ValueError):fit(rows(),synthetic=True)
    def test_identity_roundtrip_json_csv_cell(self):
        r=rows()[0];cell=canonical(r["identity"])
        self.assertEqual(identity_hash(json.loads(cell)),r["regime_id"])
        check_rows(json.loads(json.dumps(rows())))
    def test_measured_context_domain(self):
        m=fit(rows())
        self.assertEqual(m["n0"]["context_range"][0],4196)
        with self.assertRaises(ValueError):predict_detail(m,2,100000)
        self.assertTrue(predict_detail(m,2,100000,allow_extrapolation=True)["extrapolated"])
        self.assertIn("empirical",predict_detail(m,2,8000)["acceptance_source"])
        self.assertIn("fallback",predict_detail(m,3,8000)["acceptance_source"])
    def test_position_survival_and_cap(self):
        p=from_counters(10,[10,6,3],[8,4,1],1,"synthetic counters")
        self.assertAlmostEqual(expected(p,3),2.3)
        self.assertAlmostEqual(expected(p,2),2.2)
        with self.assertRaises(ValueError):expected(p,4)
        with self.assertRaises(ValueError):from_counters(10,[10,2],[8,3],1,"bad")
    def test_cycle_coverage_and_bonus(self):
        cyc=[dict(request=1,slot=0,cycle=i,anchor=100+i,wall_us=10,
                  proposed=2,accepted_drafts=1,committed=2,passes=1,stopped=0) for i in (1,2)]
        r=summarize_cycles(cyc,100,5,.1,2)
        self.assertEqual(r["coverage"]["bootstrap_tokens"],1)
        self.assertEqual(r["end_to_end_tps"],50)
        self.assertEqual(r["acceptance"]["bonus_mean"],1)
        self.assertEqual(r["measured_context_mean"],101.5)
        with self.assertRaises(ValueError):summarize_cycles(cyc,100,6,.1,2)
        cyc[0]["passes"]=2
        self.assertEqual(from_cycles(cyc,2)["effective"],2)
    def test_counter_parse(self):
        x=parse_stats("MTPSTATS type=draft-mtp steps=10 gen_drafts=10 acc_drafts=8 gen_tokens=16 acc_tokens=12 gen_per_pos=[10,6] acc_per_pos=[8,4]")[0]
        self.assertEqual(x["accepted"],[8,4])
    def test_hardware_bandwidth_host_compute(self):
        p,w,a=inputs();base=predict_physical(p,w,a,2,500)
        p["hardware_id"]="fictional-B";p["bandwidth_gbs"]=parameter(200)
        faster=predict_physical(p,w,a,2,500)
        self.assertGreater(faster["tps"],base["tps"])
        p["host_us"]=parameter(1e6)
        self.assertLess(predict_physical(p,w,a,2,500)["tps"],base["tps"])
        p,w,a=inputs();w["kernels"][0]["flops"]=term(1e12)
        slow=predict_physical(p,w,a,2,500)
        p["compute_tflops"]=parameter(20)
        self.assertGreater(predict_physical(p,w,a,2,500)["tps"],slow["tps"])
    def test_ranges_domain_prior_and_provenance(self):
        p,w,a=inputs();r=predict_physical(p,w,a,2,500)
        self.assertLess(r["sensitivity_tps"][0],r["tps"]);self.assertGreater(r["sensitivity_tps"][1],r["tps"])
        self.assertEqual(r["mode"],"prior_transfer");self.assertIn("not a confidence",r["interval_kind"])
        self.assertAlmostEqual(sum(r["breakdown"].values()),r["cycle_us"])
        with self.assertRaises(ValueError):predict_physical(p,w,a,2,2000)
        del p["host_us"]["source"]
        with self.assertRaises(ValueError):predict_physical(p,w,a,2,500)
    def test_calibration_requires_actual_evidence(self):
        p,w,a=inputs();p["mode"]="target_calibrated"
        with self.assertRaises(ValueError):predict_physical(p,w,a,2,500)
        p,w,a=inputs();probes=[]
        for kind in ("memory","compute"):
            for i in range(3):
                probes.append(dict(data_kind="synthetic",hardware_id=p["hardware_id"],workload_signature=w["signature"],
                    probe_id=kind+str(i),family="weights",kernel_signature="fixture-matvec",bound=kind,
                    device_us=10,bytes=500000,flops=50000000,host_us=10,launch_us=2,n=2,context=100+i*450))
        calibrated=calibrate_target(p,w,probes,synthetic=True)
        self.assertEqual(calibrated["data_kind"],"synthetic")
        with self.assertRaises(ValueError): calibrate_target(p,w,probes)
        r=predict_physical(calibrated,w,a,2,500)
        self.assertEqual(r["mode"],"target_calibrated")
        calibrated["hardware_id"]="different"
        with self.assertRaises(ValueError):predict_physical(calibrated,w,a,2,500)
    def test_communication_separate(self):
        p,w,a=inputs();r=predict_physical(p,w,a,2,500)
        w["communication"]={"messages":term(2),"bytes":term(1e6)}
        r2=predict_physical(p,w,a,2,500)
        self.assertGreater(r2["breakdown"]["communication_us"],0)
        self.assertLess(r2["tps"],r["tps"])
    def test_truncated_cycle_rejected(self):
        with self.assertRaises(ValueError):parse_cycles("MTPCYCLE_BEGIN request=1 slot=0 cycle=1")
    def test_plan_and_run_gate_use_no_gpu(self):
        import tempfile,sys
        from bench import main
        with tempfile.TemporaryDirectory() as d, patch("bench.command",side_effect=AssertionError("external command forbidden")), patch("bench.gpu",side_effect=AssertionError("GPU forbidden")):
            with patch.object(sys,"argv",["bench.py","plan","--out",d]): main()
            with patch.object(sys,"argv",["bench.py","run","--out",d]):
                with self.assertRaises(SystemExit):main()
    def test_n0_skips_draft(self):
        p,w,a=inputs();a["effective_bounds"]=[1,1]
        r=predict_physical(p,w,a,0,500)
        self.assertEqual(r["breakdown"]["draft_us"],0)
        self.assertEqual(r["effective_tokens"],1)
if __name__=="__main__":unittest.main()
