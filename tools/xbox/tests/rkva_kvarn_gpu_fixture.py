"""Independent FP64 serial-prefix oracle over CUDA-produced op16 fixtures."""
import json
import struct
from pathlib import Path
import numpy as np
import rkva_range_kvarn4_opt_model as km


class Fixture:
    def __init__(self,directory,tail):
        base=Path(directory)/f'tail-{tail}'
        self.meta=json.loads(Path(str(base)+'-geometry.json').read_text())
        m=self.meta
        assert m['heads_kv']==4 and m['head_dim']==256 and m['iterations']==16
        assert m['append_tokens']==8 and m['tail']==tail
        self.tail=tail
        self.origin=m['origin']; self.capacity=m['capacity']; self.total=m['tokens']
        self.original=[np.fromfile(str(base)+f'-original-{kind}.f32',dtype='<f4').reshape(self.total,4,256)
                       for kind in ('k','v')]
        rotated=[np.zeros((self.total,4,256),np.float32) for _ in range(2)]
        decoded=[np.zeros_like(rotated[0]) for _ in range(2)]
        rng=np.random.default_rng(2290+tail)
        self.segments=[]
        for index in range(m['packets']):
            upload=Path(str(base)+f'-packet-{index}.bin').read_bytes()
            live,rb,rg,flags,ab,ar,r0,r1=struct.unpack_from('<8I',upload)
            previous=index*8
            assert live==previous+8 and not(r0 or r1)
            assert (flags==1 and index==0 and not ab and not ar) or (flags==0 and ab==previous and ar==8)
            record_bytes=rg*8*8960
            offset=32
            for kind in range(2):
                for group in range(rg):
                    for ph in range(8):
                        start=offset+(group*8+ph)*8960
                        block=km.decode_record(upload[start:start+8960],bool(kind))
                        kv,sl=divmod(ph,2)
                        decoded[kind][(rb+group)*128:(rb+group+1)*128,kv,sl*128:(sl+1)*128]=block
                offset+=record_bytes
            for kind in range(2):
                stage_rows=384 if flags else 8
                stage=np.frombuffer(upload,dtype='<f2',count=stage_rows*1024,offset=offset).reshape(stage_rows,4,256)
                if flags:
                    slot=1+(self.origin//128-1)%2
                    stage=stage[slot*128:slot*128+8]
                rotated[kind][previous:live]=stage.astype(np.float32)
                offset+=stage_rows*2048
            for kind in range(2):
                if tail:
                    got=np.frombuffer(upload,dtype='<f2',count=8*1024,offset=offset).reshape(8,4,256)
                    assert np.array_equal(got,self.original[kind][previous:live].astype(np.float16))
                    offset+=8*2048
            assert offset==len(upload)
            self.segments.append(dict(upload=upload,positions=np.arange(self.origin+previous,self.origin+live,dtype='<i4'),
                                      q=rng.normal(0,1,(8,24,256)).astype('<f4')))
        # Actual wire rows are authoritative data for kernel qualification.
        # Transform/decode in FP64 independently of the Xbox shader algorithm.
        self.body=[x.astype(np.float64)@km.H256 for x in decoded]
        self.partial=[x.astype(np.float64)@km.H256 for x in rotated]
        self.precision=[x.astype(np.float16).astype(np.float64) for x in self.original]
        self.stage_vs_fp64=[]
        for kind in range(2):
            canonical=(self.original[kind].astype(np.float64)@km.H256).astype(np.float16).astype(np.float32)
            self.stage_vs_fp64.append(float(np.max(np.abs(canonical-rotated[kind]))))

    def expected(self,index):
        segment=self.segments[index]
        output=np.empty((8,24,256),np.float64); lse=np.empty((8,24),np.float64)
        for qi,position in enumerate(segment['positions']):
            count=int(position)-self.origin+1
            sources=[]
            for kind in range(2):
                source=self.body[kind][:count].copy()
                partial=count//128*128
                if count%128: source[partial:]=self.partial[kind][partial:count]
                if self.tail: source[max(0,count-self.tail):]=self.precision[kind][max(0,count-self.tail):count]
                sources.append(source)
            for head in range(24):
                kv=head//6
                scores=sources[0][:,kv]@segment['q'][qi,head].astype(np.float64)/16
                maximum=scores.max(); weights=np.exp(scores-maximum)
                denominator=weights.sum()
                output[qi,head]=weights@sources[1][:,kv]/denominator
                lse[qi,head]=maximum+np.log(denominator)
        return output,lse


def load_fixture(directory,tail):
    return Fixture(directory,tail)
