RWByteAddressBuffer output:register(u0);
[numthreads(128,1,1)] void main(uint tid:SV_GroupIndex){output.Store(tid*16,WaveGetLaneCount());output.Store(tid*16+4,WaveGetLaneIndex());output.Store(tid*16+8,asuint(WaveActiveSum(float(tid+1))));output.Store(tid*16+12,WaveReadLaneAt(tid,0));}
