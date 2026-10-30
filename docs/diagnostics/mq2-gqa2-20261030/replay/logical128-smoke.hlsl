#include "logical128.hlsl"
RWByteAddressBuffer output:register(u0);
[numthreads(128,1,1)] void main(uint tid:SV_GroupIndex){
float s=WaveActiveSum(float(tid+1));bool a=WaveActiveAnyTrue(tid==127);bool b=WaveActiveAnyTrue(false);
output.Store(tid*16,WaveGetLaneCount());output.Store(tid*16+4,WaveGetLaneIndex());output.Store(tid*16+8,asuint(s));output.Store(tid*16+12,a&&!b?1:0);}
