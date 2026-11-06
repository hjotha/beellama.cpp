#include "fixtures/gqa2_policy_extract.h"
#include <cassert>
#include <iostream>
using namespace policy;
int main() {
  unsigned long long tested=0,selected=0;
  for (bool flag : {false,true}) for(bool i8 : {false,true})
  for(unsigned mode : {0u,1u,2u,3u}) for(unsigned desc : {1u,2u,3u})
  for(unsigned storage : {0u,1u,2u,3u,8u}) for(unsigned tile : {0u,2u,4u})
  for(unsigned nq : {0u,1u,2u,3u,5u,8u,9u,10u})
  for(unsigned kv : {0u,1u,4u}) for(unsigned ratio : {5u,6u,7u})
  for(unsigned dim : {128u,256u}) for(unsigned tail : {0u,127u,128u})
  for(unsigned wmin : {32u,64u,128u}) for(unsigned wmax : {32u,64u,128u})
  for(bool pso : {false,true}) {
    auto p=kvarn4_gqa2_plan(flag,i8,mode,desc,storage,tile,nq,ratio*kv,kv,dim,tail,wmin,wmax,pso);
    bool want=flag&&!i8&&mode==2&&desc==3&&storage==3&&tile==2&&nq>=2&&nq<=9&&kv>0&&ratio==6&&dim==256&&tail==128&&wmin==64&&wmax==64&&pso;
    assert(p.selected==want);
    assert(p.dispatch_x==(want?3*kv:kv));
    for(unsigned split : {1u,56u,64u}) assert(kvarn4_gqa2_groups(p,(nq+1)/2,split)==p.dispatch_x*((nq+1)/2)*split);
    ++tested; selected+=want;
  }
  assert(mq_gqa2_arm(false,false)==MqGqa2Arm::kOff);
  assert(mq_gqa2_arm(false,true)==MqGqa2Arm::kOff);
  assert(mq_gqa2_arm(true,false)==MqGqa2Arm::kOn);
  assert(mq_gqa2_arm(true,true)==MqGqa2Arm::kConflict);
  std::cout << "{\"success\":true,\"combinations\":"<<tested<<",\"selected\":"<<selected<<",\"arm_checks\":4,\"scope\":\"exact standalone branch policy extract, not D3D12 engine\"}\n";
}
