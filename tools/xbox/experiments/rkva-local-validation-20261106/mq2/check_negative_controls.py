"""Falsification tests for serial-prefix representation and sealed-group safety."""
from pathlib import Path
from output_paths import RESULTS
import json,numpy as np
import mq2_rowlane_prototype as m
mutation=[]
for tail,row,w,active in ((0,128,255,256),(128,127,255,256),(128,128,256,257)):
    correct=m.kind(row,w,72704,tail);bad=m.kind(row,active,72704,tail)
    assert correct!=bad
    mutation.append(dict(tail=tail,row=row,query_window=w,active_count=active,correct=correct,broken=bad,detected=True))
sp=m.sp;km=m.km
rng=np.random.default_rng(1);c=sp.Canon(rng,512);c.append(256)
r=sp.Resident(c,km.Layout(72704,512,0,ring=256));r.reset(255);r.append(256)
P,_=sp.planes(r);P.window=255
try:P.region('rec',1,0)
except AssertionError as e:seal={'detected':True,'reason':str(e)}
else:raise AssertionError('incomplete group read allowed')
r={'success':True,'active_count_instead_of_prefix':mutation,'unsealed_group_read':seal}
(RESULTS/'negative_controls_masks.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
