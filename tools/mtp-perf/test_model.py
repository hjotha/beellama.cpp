import unittest
from model import *
from bench import parse_cycles,server_args,guard
from unittest.mock import patch
class Tests(unittest.TestCase):
    def test_cycle_partition(self):
        text="""MTPTGT n_tokens=4096 us=99999
MTPCYCLE_BEGIN request=1 slot=0 cycle=1
MTPPHASE phase=draft us=10 drafted=2
MTPTGT n_tokens=3 us=20
MTPPHASE phase=sync us=5
MTPPHASE phase=verify us=7 sample_accept_us=3 n_draft=2
MTPCYCLE request=1 slot=0 cycle=1 anchor=4096 tier=local proposed=2 accepted_drafts=1 target_nq=3 passes=1 committed=2 stopped=0 wall_us=50"""
        r=parse_cycles(text)[0]
        self.assertEqual(r["target_us"],20)
        self.assertEqual(r["residual_us"],8)
        self.assertEqual(r["sample_us"],3)
    def test_missing_instrumentation(self):
        with self.assertRaises(ValueError): parse_cycles("MTPCYCLE wall_us=3 committed=1")
    def test_gpu_guard(self):
        with patch("bench.command",return_value="119180"),patch("bench.gpu",return_value={"memory.free":235}):
            with self.assertRaises(RuntimeError): guard(11500)
    def test_sparse(self):
        with self.assertRaises(ValueError): fit([{"n":0,"context":4096,"cycle_us":1,"effective":1}])
    def test_synthetic_fit_holdout(self):
        rows=[]
        for c in (4096,8192,16384,24576,32768):
            for n in (0,1,2,4,6,8):
                coef=[9000,2000] if not n else [9000,4000,2000,3000]
                us=sum(a*b for a,b in zip(features(n,c),coef))
                eff=effective(n,.8)
                rows.append(dict(data_kind="synthetic",n=n,context=c,cycle_us=us,effective=eff,tps=eff*1e6/us))
        m=fit(rows,synthetic=True)
        self.assertLess(abs(predict(m,4,12000)/(effective(4,.8)*1e6/(9000+2000+2000*12000/32768+1500*12000/32768))-1),.005)
        self.assertLess(validate(rows,"n",synthetic=True)["median_ape_percent"],.1)
        self.assertLess(validate(rows,"context",synthetic=True)["median_ape_percent"],.1)
    def test_rank_deficient(self):
        with self.assertRaises(ValueError): nnls([[1,1]]*6,[3]*6)
    def test_mixed_regime(self):
        with self.assertRaises(ValueError): fit([{"batch":512},{"batch":2048}])
    def test_roofline(self):
        self.assertEqual(roofline_us(1e9,0,0,100,1),10000)
    def test_n0_effective(self):
        self.assertEqual(effective(0,.2),1)
if __name__=="__main__": unittest.main()
