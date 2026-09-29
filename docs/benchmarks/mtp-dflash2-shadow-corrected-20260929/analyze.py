"""Validate corrected observer event accounting and summarize preserved runs."""
import json
from pathlib import Path
import re
import statistics
import sys

root=Path(__file__).resolve().parent/(sys.argv[1] if len(sys.argv)>1 else 'paired-v1')
results=json.loads((root/'results.json').read_text())
reference={r['scenario']:r['content_sha256'] for r in results if r['mode']=='mtp-n4'}
if reference:
    for r in results:
        assert r['content_sha256']==reference[r['scenario']],(r['mode'],r['scenario'],'output differs')
metrics={}
for path in sorted(root.glob('*.server.log')):
    text=path.read_text()
    summary=re.findall(r'shadow auxiliary: (.*)',text)
    if not summary:continue
    fields=lambda s:{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',s)}
    last=fields(summary[-1])
    account=fields(re.findall(r'shadow accounting: (.*)',text)[-1])
    stages=fields(re.findall(r'shadow stages: (.*)',text)[-1])
    rows=[]
    for line in text.splitlines():
        if not line.startswith('SHADOWv2 job='):continue
        r=fields(line)
        tokens=lambda key:list(map(int,re.search(key+r'=\[([^]]*)\]',line)[1].split(','))) if re.search(key+r'=\[([^]]*)\]',line)[1] else []
        confirmed=tokens('confirmed'); proposed=tokens('proposed')
        matched=0
        for x,y in zip(confirmed,proposed):
            if x!=y:break
            matched+=1
        assert r['matched']==matched,line
        assert r['prefix_ok']==int(bool(confirmed) and matched==len(confirmed)),line
        assert r['ready']==int(r['done_us']<=r['decision_us']),line
        assert r['remaining']==max(0,len(proposed)-len(confirmed)),line
        assert not r['usable'] or (r['ready'] and r['prefix_ok'] and r['position_ok'] and r['remaining']>0),line
        assert not r['usable_tokens'] or r['usable'],line
        assert r['usable_tokens']<=min(r['remaining'],4),line
        rows.append(r)
    assert len({r['job'] for r in rows})==len(rows),'duplicated observation'
    assert last['launched']==account['observed']+account['cancelled']+account['pending'],(path,'unreconciled launches')
    assert len(rows)==account['observed'],(path,'missing observation records')
    for field,actual in [('ready',sum(r['ready'] for r in rows)),('late',sum(not r['ready'] for r in rows)),('prefix_match',sum(r['prefix_ok'] for r in rows)),('usable_blocks',sum(r['usable'] for r in rows)),('usable_tokens',sum(r['usable_tokens'] for r in rows))]:
        assert last[field]==actual,(path,field,last[field],actual)
    assert account['pending']==0,(path,'unfinished worker accounting')
    assert last['errors']==0,(path,'auxiliary failed')
    stage_means={k:round(stages.get(k,0)/last['launched']/1000,3) if last['launched'] else None for k in ['worker_us','decode_us','selector_us']}
    work_us=stages.get('worker_us',0)
    other_us=work_us-stages.get('decode_us',0)-stages.get('selector_us',0)
    stage_means['other_worker_us']=round(other_us/last['launched']/1000,3) if last['launched'] else None
    metrics[path.stem]={'counts':last,'accounting':account,'mean_worker_stage_ms':stage_means,'main_thread_totals_ms':{k:round(stages.get(k,0)/1000,3) for k in ['inject_submit_us','stage_us']},'matched_percent':round(100*last['prefix_match']/max(account['observed'],1),2),'ready_percent':round(100*last['ready']/max(account['observed'],1),2),'usable_percent':round(100*last['usable_blocks']/max(account['observed'],1),2),'by_request':{str(req):{'observed':len(rs),'ready':sum(x['ready'] for x in rs),'prefix_match':sum(x['prefix_ok'] for x in rs),'usable':sum(x['usable'] for x in rs),'usable_tokens':sum(x['usable_tokens'] for x in rs)} for req in sorted({r['request'] for r in rows}) for rs in [[r for r in rows if r['request']==req]]}}
(root/'observer-summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps({k:{x:v[x] for x in ['counts','accounting','mean_worker_stage_ms','main_thread_totals_ms','matched_percent','ready_percent','usable_percent']} for k,v in metrics.items()},indent=2))
print('Validated response hashes and all observer event invariants')
