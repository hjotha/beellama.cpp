// Diagnostic logical128 only; same stride64..1 pairwise tree as first adapter.
// One writer performs each node; no read/write race within each level.
groupshared float emu_values[128];
float emu_sum(float value,uint tid){
 emu_values[tid]=value; GroupMemoryBarrierWithGroupSync();
 if(tid==0){
  [loop] for(uint stride=64;stride>0;stride>>=1){
   [loop] for(uint i=0;i<stride;++i)emu_values[i]+=emu_values[i+stride];
  }
 }
 GroupMemoryBarrierWithGroupSync();
 float result=emu_values[0];GroupMemoryBarrierWithGroupSync();return result;
}
bool emu_any(bool value,uint tid){return emu_sum(value?1.0f:0.0f,tid)>0.0f;}
#define WaveGetLaneCount() 128u
#define WaveGetLaneIndex() tid
#define WaveActiveSum(v) emu_sum(v,tid)
#define WaveActiveAnyTrue(v) emu_any(v,tid)
