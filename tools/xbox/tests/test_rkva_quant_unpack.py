#!/usr/bin/env python3
"""CPU model of HLSL KVarN unpack and Q2_1/Q3_1 addressing.

This does not execute DXIL or qualify GPU numerical/performance behavior.
The payload has no padding: cross-dword loads must remain within the record.
Run directly with Python 3; deterministic fixtures need no dependencies.
"""
import random, struct
rng=random.Random(20261109)
count=0
for bits in (2,3,4,5,6,8):
    q=[rng.randrange(1<<bits) for _ in range(16384)]
    stream=sum(v << (i*bits) for i,v in enumerate(q)).to_bytes(bits*2048,'little')
    words=struct.unpack('<'+'I'*(len(stream)//4),stream)
    mask=(1<<bits)-1
    def load(index):
        bit=index*bits; w=bit//32; sh=bit%32
        return ((words[w]>>sh)|((words[w+1]<<(32-sh)) if sh+bits>32 else 0))&mask
    for i,x in enumerate(q):
        assert load(i)==x
        count+=1
    for pos in range(128):
        for du in range(32):
            kb=du*(64*bits)+(pos*bits//32)*4
            sh=(pos*bits)&31
            for component in range(4):
                w=(kb+component*(16*bits))//4
                got=((words[w]>>sh)|((words[w+1]<<(32-sh)) if sh+bits>32 else 0))&mask
                assert got==q[(du*4+component)*128+pos]
                count+=1
    for i in range(0,16384,4):
        bit=i*bits; sh=bit%32
        if sh+4*bits<=32:
            word=words[bit//32]>>sh
            got=[(word>>(c*bits))&mask for c in range(4)]
        else:
            got=[load(i+c) for c in range(4)]
        assert got==q[i:i+4]
        count+=4
    if bits == 4:
        # Fast K tile: one dword per dimension supplies eight tokens. Check
        # every tile, both end-of-row and last-record-element boundaries.
        for dim in range(128):
            scale, zero, other = (rng.uniform(-2, 2) for _ in range(3))
            query = rng.uniform(-2, 2)
            for pos0 in range(0, 128, 8):
                word = words[(dim * 128 + pos0) // 8]
                fast_sum = generic_sum = 0.0
                for t in range(8):
                    fast = (word >> (4 * t)) & 15
                    generic = load(dim * 128 + pos0 + t)
                    assert fast == generic
                    fast_dequant = (fast * scale + zero) * other
                    generic_dequant = (generic * scale + zero) * other
                    assert fast_dequant == generic_dequant
                    fast_sum += query * fast_dequant
                    generic_sum += query * generic_dequant
                    count += 1
                assert fast_sum == generic_sum
for bits in (2,3):
    for case in range(1000):
        q=[rng.randrange(1<<bits) for _ in range(32)]
        low=bytes(sum((q[j+8*p]&3)<<(2*p) for p in range(4)) for j in range(8))
        high=sum((q[j]>>2)<<j for j in range(32)) if bits==3 else 0
        words=struct.unpack('<II',low)
        for u in range(8):
            w=words[u&1]>>(2*((u>>1)&3))
            got=[((w>>(8*c))&3)+(((high>>(4*u+c))&1)*4 if bits==3 else 0) for c in range(4)]
            assert got==q[4*u:4*u+4]
            count+=4
print('PASS',count,'scalar/vector/rowlane element comparisons; CPU model only, no GPU execution')
