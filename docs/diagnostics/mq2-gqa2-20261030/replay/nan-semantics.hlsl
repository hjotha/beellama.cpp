#include "logical128-3barrier.hlsl"
ByteAddressBuffer input:register(t0);
RWByteAddressBuffer output:register(u0);
[numthreads(128,1,1)] void main(uint tid:SV_GroupIndex){
 float x=asfloat(input.Load(0));float sum=WaveActiveSum(tid==0?x:0.0f);bool invalid=WaveActiveAnyTrue(!isfinite(sum));
 uint b=tid*32;output.Store(b,asuint(sum));output.Store(b+4,isfinite(sum));output.Store(b+8,invalid);output.Store(b+12,sum!=0);output.Store(b+16,sum==0);output.Store(b+20,isnan(sum));output.Store(b+24,asuint(max(-3.402823466e38f,sum)));output.Store(b+28,asuint(sum!=0?sum:0.0f));
}
