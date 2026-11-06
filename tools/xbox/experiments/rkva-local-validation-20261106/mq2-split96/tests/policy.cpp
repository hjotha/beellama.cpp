#include "../include/baseline.h"
#include "../include/rkva-range-profile.h"
#include <cassert>
#include <iostream>
#include <cstdint>
namespace rp=rkva_rprof;
uint64_t cases=0, changed=0;
void check(uint32_t n) {
 for(uint32_t mode: {0u,1u,2u,3u,4u,UINT32_MAX}) for(uint32_t tile: {0u,1u,2u,3u,4u,8u}) {
  auto a=baseline::kvarn4_mq_split(n,mode,tile); auto b=rp::kvarn4_mq_split(n,mode,tile); ++cases;
  bool diff=a.split_k!=b.split_k || a.split_rows!=b.split_rows;
  changed+=diff;
  if(!RKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96 || mode!=2 || tile!=2) assert(!diff);
  if(mode==2 && tile==2) {
   assert(b.split_k>=1 && b.split_k<=(RKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96?96u:64u));
   assert(b.split_k<=rp::kOptMaxSplits);
   if(!n) {assert(b.split_k==1 && b.split_rows==0);continue;}
   assert(b.split_rows && b.split_rows%8==0);
   assert(uint64_t(b.split_k)*b.split_rows>=n);
   assert(uint64_t(b.split_k-1)*b.split_rows<n);
   if(n<=16384)assert(!diff);
  }
 }
}
int main(int argc,char**) {
 // Machine-readable geometry consumed by numeric suite, from the actual C++ header.
 if(argc>1) {
  for(uint32_t n: {0u,1u,257u,16384u,16385u,21760u,24576u,24577u,49920u}) {
   auto a=baseline::kvarn4_mq_split(n,2,2);auto b=rp::kvarn4_mq_split(n,2,2);
   std::cout<<n<<","<<a.split_k<<","<<a.split_rows<<","<<b.split_k<<","<<b.split_rows<<"\n";
  }return 0;
 }
 for(uint32_t n=0;n<=1048576;++n)check(n);
 for(uint32_t n: {16777215u,16777216u,1073741824u,2147483647u})check(n);
 auto a=baseline::kvarn4_mq_split(UINT32_MAX,2,2);auto b=rp::kvarn4_mq_split(UINT32_MAX,2,2);
 assert(a.split_k==1 && a.split_rows==0 && b.split_k==1 && b.split_rows==0);
 assert(RKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96 ? changed>0 : changed==0);
 std::cout<<"PASS macro="<<RKVA_EXPERIMENTAL_MQ2_FINE_SPLIT96<<" comparisons="<<cases<<" changed="<<changed<<"; inherited UINT32_MAX overflow reproduced (not valid coverage)\n";
}
