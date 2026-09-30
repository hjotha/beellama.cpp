"""Audit same-cadence early/late runs and the sync-only/cancel regressions."""
import json
from pathlib import Path
import re
import sys

root=Path(__file__).resolve().parent/(sys.argv[1] if len(sys.argv)>1 else 'review-v2')
results=json.loads((root/'results.json').read_text())
by_mode={}
for r in results:by_mode.setdefault(r['mode'],{})[r['scenario']]=r
for left,right in [('early-off','early-on'),('empty-primary-baseline','empty-primary-early-on')]:
    assert by_mode[left].keys()==by_mode[right].keys()
    for key in by_mode[left]:assert by_mode[left][key]['content_sha256']==by_mode[right][key]['content_sha256'],(left,right,key)
extract=lambda line:{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
all_metrics={};paired_records={}
for path in sorted(root.glob('*.server-output.txt')):
    text=path.read_text()
    totals=re.findall(r'shadow auxiliary: (.*)',text)
    if not totals:continue
    counts=extract(totals[-1])
    accounting=extract(re.findall(r'shadow accounting: (.*)',text)[-1])
    order=list(dict.fromkeys(map(int,re.findall(r'task\s+(\d+)\s*\|\s*processing task',text))))
    launches={};observed={};cancelled={};records={}
    for line in text.splitlines():
        offset=line.find('SHADOWv2 ')
        if offset<0:continue
        line=line[offset:] # logger prefix and raw stderr can share a line
        if line.startswith('SHADOWv2 launch '):
            f=extract(line);assert f['job'] not in launches;launches[f['job']]=f
        elif line.startswith('SHADOWv2 cancel '):
            f=extract(line);assert f['job'] not in cancelled
            f['reason']=re.search(r'reason=(\w+)',line)[1];cancelled[f['job']]=f
        elif line.startswith('SHADOWv2 job='):
            f=extract(line);assert f['job'] not in observed
            def ids(key):
                value=re.search(key+r'=\[([^]]*)\]',line)[1]
                return list(map(int,value.split(','))) if value else []
            f['confirmed']=ids('confirmed');f['proposed']=ids('proposed')
            matched=0
            for x,y in zip(f['confirmed'],f['proposed']):
                if x!=y:break
                matched+=1
            assert f['matched']==matched
            assert f['prefix_ok']==int(bool(f['confirmed']) and matched==len(f['confirmed']))
            assert f['ready']==int(f['done_us']<=f['decision_us'])
            assert not f['usable'] or (f['ready'] and f['prefix_ok'] and f['position_ok'] and f['remaining']>0)
            assert f['usable_tokens']<=min(f['remaining'],4)
            observed[f['job']]=f
            key=(order.index(f['request']),f['pos0'],f['anchor'])
            assert key not in records
            records[key]=f
    assert set(observed).isdisjoint(cancelled)
    assert set(launches)==set(observed)|set(cancelled),(path,'unaccounted jobs')
    assert counts['launched']==len(launches)==accounting['observed']+accounting['cancelled']
    assert accounting['observed']==len(observed) and accounting['cancelled']==len(cancelled)
    assert accounting['pending']==0 and counts['errors']==0
    assert counts['ready']==sum(f['ready'] for f in observed.values())
    assert counts['late']==len(observed)-counts['ready']
    assert counts['prefix_mismatch']==len(observed)-counts['prefix_match']
    assert counts['usable_tokens']==sum(f['usable_tokens'] for f in observed.values())
    assert counts['prefix_match']==sum(f['prefix_ok'] for f in observed.values())
    assert counts['usable_blocks']==sum(f['usable'] for f in observed.values())
    name=path.name.removesuffix('.server-output.txt')
    paired_records[name]=records
    without_warmup=[f for key,f in records.items() if key[0]!=0]
    all_metrics[name]={'counts':counts,'accounting':accounting,'cancelled_primary':sum(f['reason']=='primary_aborted' for f in cancelled.values()),'excluding_warmup':{'observed':len(without_warmup),'ready':sum(f['ready'] for f in without_warmup),'usable':sum(f['usable'] for f in without_warmup)}}
assert all_metrics['sync-only-early-on']['counts']['launched']==0,'EVERY=0 launched work'
assert all_metrics['empty-primary-early-on']['cancelled_primary']>0,'Empty-primary cancellation was not exercised'
a=paired_records['early-off'];b=paired_records['early-on'];common=sorted(set(a)&set(b))
assert common
for key in common:assert a[key]['confirmed']==b[key]['confirmed'],('Different target path',key)
comparison={'paired_anchors':len(common),'only_off':len(set(a)-set(b)),'only_on':len(set(b)-set(a))}
for label,rows in [('off',[a[k] for k in common]),('on',[b[k] for k in common])]:
    comparison[label]={key:sum(r[key] for r in rows) for key in ['ready','prefix_ok','usable','usable_tokens']}
output={'modes':all_metrics,'paired_comparison':comparison}
(root/'review-summary.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
print('Hashes, every=0, primary cancellation, observer accounting and anchor pairing validated')
