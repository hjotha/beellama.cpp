"""Reproduce the largest-error boundary and expose pointwise official gate usage."""
import json
from pathlib import Path
from output_paths import RESULTS
import numpy as np
import mq2_rowlane_prototype as m
rng=np.random.default_rng(20261106)
target=(72704,0,255,9)
for origin in (0,72704):
 for tail in (0,128):
  for live,nq in ((127,3),(128,2),(129,5),(255,9),(256,3),(257,9),(449,3),(513,5),(1025,3)):
   c=m.sp.Canon(rng,max(live+128,2048));c.append(live)
   q=rng.normal(size=(nq,24,256)).astype(m.F)
   if (origin,tail,live,nq)!=target:continue
   r=m.sp.Resident(c,m.km.Layout(origin,max(live+128,2048),tail,ring=256));r.reset(live-nq);r.append(live)
   pos=np.arange(origin+live-nq,origin+live,dtype=np.int64)
   want=m.oracle(c,r,q,pos);got=m.run(r,q,pos,(1,448),6)
   out_err=np.abs(got[0]-want[0]);lse_err=np.abs(got[1]-want[1]);finite=np.isfinite(want[1])
   out_budget=3e-4+3e-4*np.abs(want[0]);lse_budget=3e-4+3e-5*np.abs(want[1])
   r={'case':dict(origin=origin,tail=tail,live=live,nq=nq,split_k=1,split_rows=448),'official_gate_unchanged':m.probe.compare(np,*got,*want),'out_max_fraction_of_allowed_error':float(np.max(out_err/out_budget)),'lse_max_fraction_of_allowed_error':float(np.max(lse_err[finite]/lse_budget[finite])),'out_relative_l2':float(np.linalg.norm(got[0]-want[0])/np.linalg.norm(want[0])),'lse_relative_l2':float(np.linalg.norm(got[1]-want[1])/np.linalg.norm(want[1])),'out_atol':3e-4,'out_rtol':3e-4,'lse_atol':3e-4,'lse_rtol':3e-5,'gate_statement':'Every component must satisfy abs(error) <= atol + rtol*abs(reference); finite LSE mask must match, and all OUT components must be finite.'}
   assert r['official_gate_unchanged']['ok']
   (RESULTS/'error_budget_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0)
raise AssertionError('target not reached')
