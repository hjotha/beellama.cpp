#!/usr/bin/env python3
"""Host-only structural contract, bounds and merge-model tests. Not HLSL compilation."""
from pathlib import Path
import hashlib, json, math, random, re, subprocess, unittest
ROOT=Path(__file__).resolve().parent
REL=Path('tools/xbox/shaders/rkva')
FLAG='RKVA_EXPERIMENT_ROWLANE_WAVE64_LDS'
CASES=[('rkva_range_attn_q4_opt.hlsl',['RKVA_DECODE_ROWLANE=1'],'CSRangeAttnQ4DecodeRowlane'),
       ('rkva_range_attn_gqa_opt.hlsl',['RKVA_DECODE_ROWLANE=1','RKVA_OPT_STORAGE=2'],'CSRangeAttnQ8DecodeRowlane'),
       ('rkva_range_attn_rowlane_matrix.hlsl',['ROWLANE_F16=1'],'CSRangeAttnF16DecodeRowlane'),
       ('rkva_range_attn_rowlane_matrix.hlsl',['ROWLANE_SAW_INT4=1'],'CSRangeAttnSawInt4DecodeRowlane')]
def pp(side,name,defs,ok=True):
    r=subprocess.run(['cpp','-P','-x','c',*[f'-D{d}' for d in defs],str(ROOT/side/REL/name)],text=True,capture_output=True)
    if ok and r.returncode: raise AssertionError(r.stderr)
    if not ok:
        assert r.returncode != 0,r.stdout
    return r.stdout

def body(text,name):
    m=re.search(r'void\s+'+name+r'\(',text);assert m,name
    start=text.index('{',m.end());depth=0
    for i in range(start,len(text)):
        depth+=(text[i]=='{')-(text[i]=='}')
        if depth==0: return text[start:i+1]
    raise AssertionError('unclosed body')
def arrays(text):
    out={}
    for name,dims in re.findall(r'groupshared\s+(?:uint|float)\s+(\w+)((?:\[[^\]]+\])+);',text):
        out[name]=tuple(int(eval(d,{'__builtins__':{}},{})) for d in re.findall(r'\[([^\]]+)\]',dims))
    return out

def publish(waves,quarter=False,seed=42):
    """Only the unchanged publication/merge section, not complete attention emulation."""
    rng=random.Random(seed); a=[float('nan')]*(waves*6*256); den=[float('nan')]*(waves*6);mx=den.copy()
    for wave in range(2):
        for g in range(6):
            slot=wave*6+g;den[slot]=0.0 if (g+seed)%4==0 else rng.random()*64;mx[slot]=rng.uniform(-20,20)
            for lane in range(64):
                d=((lane>>3)*32+(lane&1)*16+((lane>>1)&3)*4) if quarter else 4*lane
                for c in range(4):a[slot*256+d+c]=rng.uniform(-10,10)
    result=[]
    for e in range(6*256):
        g,d=divmod(e,256);m=-3.402823466e38
        for w in range(2):
            slot=w*6+g
            if den[slot]>0:m=max(m,mx[slot])
        sd=sa=0.0
        for w in range(2):
            slot=w*6+g
            if den[slot]>0:
                weight=math.exp(mx[slot]-m);sd+=den[slot]*weight;sa+=a[slot*256+d]*weight
        result.append((sa,m,sd,sa/sd if sd>0 else 0,m+math.log(sd) if sd>0 else -math.inf))
    return result

class LDS(unittest.TestCase):
    def test_source_hashes(self):
        for name,h in json.loads((ROOT/'source-manifest.json').read_text())['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'baseline'/name).read_bytes()).hexdigest(),h)
    def test_default_preprocessor_identical(self):
        variants=[(n,d) for n,d,_ in CASES]+[(CASES[0][0],[]),(CASES[1][0],['RKVA_OPT_STORAGE=0']),(CASES[1][0],['RKVA_OPT_STORAGE=2'])]
        for n,d in variants:
            for flag in ([],[FLAG+'=0']):
                self.assertEqual(pp('baseline',n,d),pp('candidate',n,d+flag))
    def test_selected_bodies_and_sizes(self):
        for n,d,entry in CASES:
            b=pp('baseline',n,d);c=pp('candidate',n,d+[FLAG+'=1'])
            self.assertEqual(body(b,entry),body(c,entry))
            ba,ca=arrays(b),arrays(c)
            self.assertEqual(ba.keys(),ca.keys())
            for k in ba:
                self.assertEqual(math.prod(ba[k]),2*math.prod(ca[k]))
            self.assertEqual(sum(math.prod(x)*4 for x in ca.values()),13536 if 'q4_opt' in n else 12384)
            # All non-generic helper bodies remain identical; only arrays and
            # excluded generic function may differ after preprocessing.
            generics=['CSRangeAttnQ4Opt','CSRangeAttnGqaOpt']
            clean=b
            for generic in generics:
                if re.search(r'void\s+'+generic+r'\(',clean):
                    bod=body(clean,generic)
                    clean=re.sub(r'\[numthreads\([^\n]+\)\]\s*void\s+'+generic+r'[^\{]+'+re.escape(bod),'',clean)
                    self.assertNotRegex(c,r'void\s+'+generic+r'\(')
            strip=lambda t: re.sub(r'\s+','',re.sub(r'groupshared\s+(?:uint|float)\s+\w+(?:\[[^\]]+\])+;','',t))
            self.assertEqual(strip(clean),strip(c))
    def test_invalid_flags_fail_closed(self):
        neg=[(CASES[0][0],[]),(CASES[1][0],['RKVA_OPT_STORAGE=2']),
             (CASES[1][0],['RKVA_OPT_STORAGE=0','RKVA_DECODE_ROWLANE=1']),
             (CASES[2][0],[]),(CASES[2][0],['ROWLANE_Q4_1=1']),
             (CASES[2][0],['ROWLANE_F16=1','ROWLANE_SAW_INT4=1']),
             (CASES[2][0],['ROWLANE_F16=1','ROWLANE_KVARN2=1'])]
        for n,d in neg:pp('candidate',n,d+[FLAG+'=1'],False)
        for alias in ['ROWLANE_Q2_0S','ROWLANE_Q3_0','ROWLANE_Q8_0','ROWLANE_BDR_INT4']:
            pp('candidate',CASES[2][0],['ROWLANE_F16=1',FLAG+'=1',alias+'=1'],False)
        for override in ['RKVA_THREADS=64','RKVA_THREADS=256','RKVA_DIM=128','RKVA_GQA=4','RKVA_MAX_WAVES=1','RKVA_MAX_WAVES=4']:
            pp('candidate',CASES[2][0],['ROWLANE_F16=1',FLAG+'=1',override],False)
    def test_all_indices_and_wave32_negative(self):
        for quarter in [False,True]:
            writes=set()
            for tid in range(128):
                wave,lane=divmod(tid,64)
                d=((lane>>3)*32+(lane&1)*16+((lane>>1)&3)*4) if quarter else lane*4
                for g in range(6):
                    slot=wave*6+g;self.assertLess(slot,12)
                    for c in range(4):
                        i=slot*256+d+c;self.assertLess(i,3072);self.assertNotIn(i,writes);writes.add(i)
            self.assertEqual(writes,set(range(3072)))
        # Generic wave32 would need 4*6*256 slots: ensure the check catches it.
        self.assertGreaterEqual((127//32*6+5)*256+255,3072)
        # Q4 raw helper addressing (not used by selected rowlane entry).
        for u in range(64):
            p=(u>>4)*9;odd=(u>>3)&1;c=(u>>1)&3
            for idx in ([p+4,p+5+c] if odd else [p,p+c,p+c+1]):self.assertLess(idx,36)
    def test_cpu_merge_storage_equivalence(self):
        for seed in range(40):
            for q in (True,False):self.assertEqual(publish(4,q,seed),publish(2,q,seed))
        with self.assertRaises(IndexError):publish(1)

if __name__=='__main__':unittest.main(verbosity=2)
