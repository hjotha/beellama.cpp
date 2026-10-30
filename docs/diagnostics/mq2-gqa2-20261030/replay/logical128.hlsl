// Diagnostic adapter, NOT native wave128 and NOT original Xbox wave64 arithmetic.
// Whole128-thread group is one logical wave. Valid only for audited uniform sites.
groupshared float emu_values[128];
float emu_sum(float value,uint tid){
 emu_values[tid]=value; GroupMemoryBarrierWithGroupSync();
 [unroll] for(uint stride=64;stride>0;stride>>=1){
  if(tid<stride)emu_values[tid]+=emu_values[tid+stride];
  GroupMemoryBarrierWithGroupSync();
 }
 float result=emu_values[0]; GroupMemoryBarrierWithGroupSync(); return result;
}
bool emu_any(bool value,uint tid){return emu_sum(value?1.0f:0.0f,tid)>0.0f;}
#define WaveGetLaneCount() 128u
#define WaveGetLaneIndex() tid
#define WaveActiveSum(v) emu_sum(v,tid)
#define WaveActiveAnyTrue(v) emu_any(v,tid)
