#!/usr/bin/env python3
"""Local CPU validation, NOT a HLSL compiler, GPU emulator or Xbox benchmark."""
import ctypes, json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BUILD=Path(os.environ.get('QROT_BUILD_DIR', str(HERE/'.build'))).resolve()
BUILD.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE/'models'))
import rkva_decode_rowlane_model as rm
sp,km=rm.sp,rm.km
F=np.float32
rng=np.random.default_rng(20261106)
lib=ctypes.CDLL(str(BUILD/'cpu_reference.so'))
lib.reference_wht.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),ctypes.c_uint]

def cpu(x):
    x=np.ascontiguousarray(x,F).reshape(-1,256); y=np.empty_like(x)
    lib.reference_wht(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),len(x))
    return y

def decode_shader(x):
    # Ping-pong LDS writes for tid=0..127; array index is the physical dimension.
    x=np.asarray(x,F).reshape(-1,256).copy()
    ix=np.arange(256)
    for stride in (1,2,4,8,16,32,64):
        other=x[:,ix^stride]
        x=np.where((ix&stride)!=0,other-x,x+other).astype(F)
    lo=x[:,:128]*F(0.08838834764831845)
    hi=x[:,128:]*F(0.08838834764831845)
    return np.concatenate(((lo+hi)*F(0.7071067811865475),(lo-hi)*F(0.7071067811865475)),axis=1)

def bulk_shader_rotation(x):
    # Two heads / 512 LDS floats per group; odd final head padded with zeros.
    x=np.asarray(x,F).reshape(-1,256); h=len(x)
    gs=np.zeros(((h+1)//2,512),F); gs.reshape(-1,256)[:h]=x
    dims=np.arange(512)  # lane*2 and lane*2+1 own both d0,d1.
    for stride in (1,2,4,8,16,32,64):
        oth=gs[:,dims^stride]
        gs=np.where((dims&stride)!=0,oth-gs,gs+oth).astype(F)
    xx=gs*F(0.08838834764831845); yy=gs[:,dims^128]*F(0.08838834764831845)
    rot=np.where(dims&128,yy-xx,xx+yy)*F(0.7071067811865475)
    return rot.reshape(-1,256)[:h].copy()

def bits_equal(a,b):
    return np.array_equal(np.asarray(a,F).view(np.uint32),np.asarray(b,F).view(np.uint32))

def e4_half(c):
    # Transliteration of main rkva_e4m3_codec.hlsli e4m3fn_to_half.
    c=int(c); sign=(c&128)<<8; mag=c&127; exponent=mag>>3; m=mag&7
    if exponent==0:
        if m==0: return sign
        p=2 if m>=4 else (1 if m>=2 else 0)
        return sign|((p+6)<<10)|((m<<(10-p))&1023)
    return sign|((exponent+8)<<10)|(m<<7)

def e4_independent(c):
    m=c&7; e=(c&127)>>3
    return F((-1 if c&128 else 1)*(m*2.0**-9 if e==0 else (1+m/8)*2.0**(e-7)))

# Exhaustive codec conversion before mixed-vector WHT tests.
finite16=np.arange(65536,dtype=np.uint16)
finite16=finite16[(finite16&0x7c00)!=0x7c00]
finite5=np.array([i for i in range(256) if (i&0x7c)!=0x7c],np.uint16)
finite4=np.array([i for i in range(256) if (i&127)!=127],np.uint16)
e4vals=np.array([e4_half(i) for i in finite4],np.uint16).view(np.float16).astype(F)
assert np.array_equal(e4vals,np.array([e4_independent(int(i)) for i in finite4],F))
assert np.array_equal(np.signbit(e4vals),[bool(i&128) for i in finite4])

# Serializing the true Q_raw/Q_rot/positions layout with exact byte offsets.
def gpu_unpack(q,nquery,codec,positions=None):
    q=np.asarray(q,F).reshape(-1,256); nh=len(q)//nquery; ne=q.size; qp=ne//2
    if codec=='F16':
        src=q.astype('<f2').tobytes(); width=2
    elif codec=='E5M2':
        codes=(q.astype('<f2').view(np.uint16)>>8).astype(np.uint8)
        src=codes.tobytes(); width=1
    elif codec=='E4M3FN':
        # Quantization is only fixture generation, not part of tested optimization.
        flat=q.reshape(-1)
        codes=finite4[np.abs(flat[:,None]-e4vals).argmin(axis=1)].astype(np.uint8)
        src=codes.tobytes(); width=1
    else: raise ValueError(codec)
    if positions is None: positions=np.arange(nquery,dtype='<i4')+102400
    src+=np.asarray(positions,dtype='<i4').tobytes()
    dst=bytearray(b'\xcd'*(ne*8+nquery*4+32))
    # Match idx -> packed-byte selection and destination writes in unpack_query.
    raw=np.empty(ne,F)
    for group in range((max(qp,nquery)+255)//256):
        for lane in range(256):
            idx=group*256+lane
            if idx<qp:
                if width==2:
                    word=int.from_bytes(src[idx*4:idx*4+4],'little'); hh=(word&65535,word>>16)
                else:
                    off=(idx//2)*4; word=int.from_bytes(src[off:off+4],'little')>>((idx&1)*16)
                    codes=(word&255,(word>>8)&255)
                    hh=tuple(e4_half(c) if codec=='E4M3FN' else c<<8 for c in codes)
                pair=np.asarray(hh,np.uint16).view(np.float16).astype(F)
                raw[idx*2:idx*2+2]=pair
                dst[idx*8:idx*8+8]=pair.astype('<f4').tobytes()
            if idx<nquery:
                src_off=ne*width+idx*4; dst_off=(qp*4+idx)*4
                dst[dst_off:dst_off+4]=src[src_off:src_off+4]
    rot=bulk_shader_rotation(raw)
    dst[ne*4:ne*8]=rot.astype('<f4').tobytes()
    assert dst[ne*8:ne*8+nquery*4]==np.asarray(positions,dtype='<i4').tobytes()
    assert dst[-32:]==b'\xcd'*32
    assert bits_equal(raw.reshape(-1,256),np.frombuffer(dst,dtype='<f4',count=ne).reshape(-1,256))
    return raw.reshape(-1,256),rot,bytes(dst)

def consume(buf,nquery,nhead,qi,kv):
    # Exactly the patch's q_stride, head0, u, q_rot_base. Both waves read same Q.
    qs=nquery*nhead*256*4; head0=qi*nhead+kv*6
    result=np.empty((2,6,256),F)
    for wave in range(2):
        for lane in range(64):
            for g in range(6):
                off=qs+((head0+g)*256+lane*4)*4
                assert off>=qs and off+16<=2*qs
                result[wave,g,lane*4:lane*4+4]=np.frombuffer(buf,dtype='<f4',count=4,offset=off)
    assert bits_equal(result[0],result[1])
    return result[0]

result={'schema':1,'seed':20261106,'environment':{'python':platform.python_version(),'numpy':np.__version__,'compiler':subprocess.check_output([os.environ.get('CXX','g++'),'--version'],text=True).splitlines()[0],'flags':'-std=c++17 -O2 -fno-fast-math -ffp-contract=off -shared -fPIC'},'qualification':'CPU-only source/layout/numerical validation; no HLSL compilation, DXIL, GPU/console execution, timing or performance claim','transforms':[],'layouts':[],'attention':[],'negative_controls':[]}
# Every finite encoding occurs, padded only to form complete vectors.
for label,vals in [('F16-all-finite',finite16.view(np.float16).astype(F)),('E5M2-all-finite',(finite5<<8).astype(np.uint16).view(np.float16).astype(F)),('E4M3FN-all-finite',e4vals)]:
    perm=rng.permutation(len(vals)); vals=vals[perm]
    xx=np.pad(vals,(0,(-len(vals))%256)).reshape(-1,256)
    want=cpu(xx)
    assert bits_equal(want,decode_shader(xx)) and bits_equal(want,bulk_shader_rotation(xx))
    assert bits_equal(want,km.butterfly_wht(xx))
    result['transforms'].append({'case':label,'encodings':len(vals),'heads':len(xx),'bitwise_equal':True})
for label,xx in [('random-F32',rng.normal(size=(2049,256)).astype(F)),('basis',np.eye(256,dtype=F)),('cancellation-and-signed-zero',np.tile(np.array([0.,-0.,1.,-1.,1e-30,-1e-30,1e20,-1e20],F),(17,32)))]:
    want=cpu(xx)
    assert bits_equal(want,decode_shader(xx)) and bits_equal(want,bulk_shader_rotation(xx))
    result['transforms'].append({'case':label,'heads':len(xx),'bitwise_equal':True})
# Exceptional F32 diagnostic: same classes, not NaN-payload or production-admission claim.
xx=rng.normal(size=(6,256)).astype(F)
xx[0,0]=np.inf;xx[1,2]=-np.inf;xx[2,1]=np.nan;xx[3,::2]=np.finfo(F).max;xx[4,1::2]=-np.finfo(F).max
with np.errstate(all='ignore'):
    aa,bb,cc=cpu(xx),decode_shader(xx),bulk_shader_rotation(xx)
for got in (bb,cc):
    assert np.array_equal(np.isnan(aa),np.isnan(got)) and np.array_equal(np.isposinf(aa),np.isposinf(got)) and np.array_equal(np.isneginf(aa),np.isneginf(got))
result['transforms'].append({'case':'nonfinite-and-overflow-diagnostic','heads':6,'classification_equal':True,'nan_payload_not_compared':True})
for nq,nh in [(1,1),(1,3),(1,5),(3,1),(3,3),(1,6),(1,24),(2,24),(3,24),(9,24)]:
    xx=rng.normal(size=(nq*nh,256)).astype(F)
    for codec in ('F16','E5M2','E4M3FN'):
        raw,rot,buf=gpu_unpack(xx,nq,codec)
        assert bits_equal(rot,cpu(raw))
        if nh%6==0:
            for qi in range(nq):
                for kv in range(nh//6):
                    got=consume(buf,nq,nh,qi,kv)
                    assert bits_equal(got,rot[qi*nh+kv*6:qi*nh+kv*6+6])
        result['layouts'].append({'nquery':nq,'nhead':nh,'codec':codec,'odd_padded_head':bool(nq*nh%2),'passed':True,'decode_runtime_scope':'nquery=1,nhead=6*k only; other shapes exercise producer/address contract only'})
# Deliberately broken transformations/layouts must be distinguishable.
xx=rng.normal(size=(24,256)).astype(F); want=cpu(xx)
x=xx.copy()
for stride in (1,2,4,8,16,32,64,128):
    ix=np.arange(256); oth=x[:,ix^stride];x=np.where(ix&stride,oth-x,x+oth).astype(F)
badnorm=x*F(1/16)
assert not bits_equal(want,badnorm)
result['negative_controls'].append({'case':'collapse_two_normalizations_to_1_over_16','detected':True,'changed_f32_elements':int(np.count_nonzero(want.view(np.uint32)!=badnorm.view(np.uint32)))})
assert not bits_equal(want,want.reshape(24,64,4)[:,::-1,:].reshape(24,256))
result['negative_controls'].append({'case':'reverse_lane_mapping','detected':True})
assert not bits_equal(want,xx)
result['negative_controls'].append({'case':'read_original_plane_as_rotated','detected':True})
(BUILD/'transform-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'stage':'transform-and-layout','status':'PASS','transform_cases':len(result['transforms']),'layout_cases':len(result['layouts']),'negative_controls':len(result['negative_controls'])}),flush=True)

# Existing audited rowlane model; candidate differs ONLY in supplied query rotation.
def attention_case(canon,res,label,codec,forced=None,contract=False,masked=False,negative=False):
    rm.CONTRACT=contract
    q=rng.normal(size=(24,256)).astype(F)
    if codec=='CPU-F32': raw=q; rot=cpu(q)
    else: raw,rot,_=gpu_unpack(q,1,codec)
    pos=res.lay.origin-1 if masked else res.lay.origin+res.live-1
    base=rm.run(*sp.planes(res),raw,res.live,pos,mode=2,forced=forced)
    cand=rm.run(*sp.planes(res),raw,res.live,pos,mode=2,forced=forced,qrot=rot)
    assert bits_equal(base[0],cand[0]) and bits_equal(base[1],cand[1]),label
    count=km.causal_limit(res.lay,res.live,pos)
    ks,exact=sp.serial_sources(canon,res.lay,count,False); vs,_=sp.serial_sources(canon,res.lay,count,True)
    want=km.oracle_attention(raw,ks,vs,exact,count,sp.SCALE)
    eo=np.abs(cand[0].astype(np.float64)-want[0]); finite=np.isfinite(want[1]);el=np.abs(cand[1][finite]-want[1][finite])
    assert np.all(eo<=3e-4+3e-4*np.abs(want[0]))
    assert np.all(el<=3e-4+3e-5*np.abs(want[1][finite]))
    assert np.array_equal(np.isneginf(cand[1]),np.isneginf(want[1]))
    entry={'case':label,'live':res.live,'origin':res.lay.origin,'tail':res.lay.ptt,'codec':codec,'contract_pv':contract,'forced_split':forced,'out_and_lse_bitwise_equal':True,'oracle_out_max_abs':float(np.max(eo)),'oracle_lse_max_abs':float(np.max(el)) if len(el) else 0.,'oracle_tolerance_pass':True}
    result['attention'].append(entry);print(json.dumps(entry),flush=True)
    if negative:
        for name,kwargs in [('original_plane_as_qrot',{'qrot':raw}),('rotated_Q_used_for_original_tail',{'qrot':rot,'bad_tail':True})]:
            bad=rm.run(*sp.planes(res),raw,res.live,pos,mode=2,forced=forced,**kwargs)
            assert not(bits_equal(base[0],bad[0]) and bits_equal(base[1],bad[1])),name
            err=float(np.max(np.abs(base[0]-bad[0])))
            result['negative_controls'].append({'case':name,'detected':True,'attention_out_max_difference':err})

codecs=('CPU-F32','F16','E5M2','E4M3FN')
for origin in (0,102400):
    for tail in (0,128):
        canon=sp.Canon(rng,2048);canon.append(1025)
        lay=km.Layout(origin,2048,tail,ring=256)
        for i,live in enumerate((1,127,128,129,255,256,257,513,1025)):
            res=sp.Resident(canon,lay);res.reset(live)
            forced=(1,live) if i%2==0 else ((live+76)//77+1,77)
            attention_case(canon,res,'boundary',codecs[i%4],forced,contract=i%2==1,
                           negative=(origin==102400 and tail==128 and live==257))
        attention_case(canon,res,'causally-empty','CPU-F32',(10,128),masked=True)
        canon.rewind(1018);res.rewind(1018);canon.append(9);res.append(1027)
        attention_case(canon,res,'rewind-reappend-ring-wrap','E4M3FN',(10,128),contract=True)
# Split/address partition arithmetic at target lengths; not numerical long attention.
result['long_layout_only']=[]
for live in (4096,8192,16384,28672):
    sk,rows=km.kvarn4_decode_split(live,2)
    for origin in (0,102400):
        for tail in (0,128):
            seen=[]
            for s in range(sk):
                begin,end=min(live,s*rows),min(live,(s+1)*rows);tb=live-min(live,tail)
                flo=(max(begin,128 if origin==0 else 0)+127)&~127;fhi=min(end,live&~127,tb)&~127
                if fhi<=flo:flo=fhi=end
                segment=list(range(flo,fhi))+list(range(begin,min(flo,tb)))+list(range(fhi,min(end,tb)))+list(range(max(begin,tb),end))
                assert sorted(segment)==list(range(begin,end));seen.extend(segment)
            assert sorted(seen)==list(range(live))
            result['long_layout_only'].append({'live':live,'origin':origin,'tail':tail,'splits':sk,'rows_per_split':rows,'exact_coverage':True})
result['static_load_tradeoff']={'baseline_q_input_floats_per_group': 1536, 'candidate_qrot_input_floats_per_group': 3072, 'baseline_bytes': 6144, 'candidate_bytes': 12288, 'excludes_tail': True, 'not_measured_dram_traffic': True, 'cache_effect_unknown': True, 'net_performance_unknown': True}
result['status']='PASS'
result['summary']={'attention_cases':len(result['attention']),'transform_cases':len(result['transforms']),'layout_cases':len(result['layouts']),'long_layout_only_cases':len(result['long_layout_only']),'negative_controls':len(result['negative_controls']),'out_lse_ab_bit_differences':0,'oracle_out_max_abs':max(x['oracle_out_max_abs'] for x in result['attention']),'oracle_lse_max_abs':max(x['oracle_lse_max_abs'] for x in result['attention'])}
(BUILD/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary']),flush=True)
