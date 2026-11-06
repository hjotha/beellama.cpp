#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
using uint=uint32_t;
using std::min; using std::max;
#define RKVA_MQ_QT 2
#define RKVA_GQA 6
#define RKVA_TILE 8
#define RKVA_MQ_HEAD0_OFFSET + head_base
#define RKVA_KIND_NONE 0u
#define RKVA_KIND_TAIL 1u
#define RKVA_KIND_STAGE 2u
#define RKVA_KIND_RECORD 3u
uint origin;
#include "fixtures/row_kind_extract.inc"
struct PositionBuffer {std::vector<int32_t> v; uint loads=0; uint Load(uint byte_offset) {assert(byte_offset%4==0); ++loads; return uint(v.at(byte_offset/4));}};
int32_t asint(uint value) {return static_cast<int32_t>(value);}
int main() {
 std::mt19937 gen(20261106); uint64_t calls=0, rows_checked=0,phantom=0,zero_splits=0;
 for(uint org : {0u,72704u}) for(uint precision_tail_tokens : {0u,128u})
 for(uint active_count : {0u,1u,127u,128u,129u,255u,256u,257u,447u,448u,449u,511u,512u,513u,895u,896u,897u,1025u,28672u})
 for(uint n_query : {2u,3u,5u,8u,9u}) for(uint split_rows : {128u,211u,448u,512u,896u,28672u})
 for(uint pattern : {0u,1u,2u}) {
  origin=org; PositionBuffer positions;
  for(uint qi=0;qi<n_query;++qi) {
   int64_t count=pattern==0 ? int64_t(active_count)-n_query+qi+1 : pattern==1 ? gen()%(active_count+1) : qi==0 ? 0:active_count;
   positions.v.push_back(int32_t(int64_t(origin)+count-1));
  }
  uint n_head=24;
  for(uint split=0;split<(active_count+split_rows-1)/split_rows+1;++split)
  for(uint q_first=0;q_first<n_query;q_first+=2)
  for(uint kv_head : {0u,3u}) for(uint head_base : {0u,2u,4u}) {
   #include "fixtures/selector_extract.inc"
   ++calls;
   assert(positions.loads>0);
   assert(body_end_max>=split_begin);
   for(uint j=0;j<2;++j) {
    uint qi=q_first+j,qc=min(qi,n_query-1);
    int64_t c=std::max<int64_t>(0,int64_t(positions.v[qc])-origin+1);
    uint want=qi<n_query ? uint(std::min<int64_t>(active_count,c)):0;
    assert(window[j]==want && tail_begin[j]==want-min(want,precision_tail_tokens));
    assert(head0[j]==qc*n_head+kv_head*6+head_base);
    if(qi>=n_query){++phantom;assert(!act[j]&&end_q[j]==0&&body_end_q[j]==0);}
    uint e=min(want,split_end); if(e<=split_begin){++zero_splits;assert(!act[j]);}
    if(!act[j])continue;
    assert(fast_lo==fast_hi || fast_lo%8==0);
    for(uint row=split_begin;row<e;++row) {
     const bool exact=row>=want-min(want,precision_tail_tokens);
     const bool staged=origin+row<128 || ((row/128)==(want-1)/128 && want%128!=0);
     uint expected=exact?RKVA_KIND_TAIL:staged?RKVA_KIND_STAGE:RKVA_KIND_RECORD;
     assert(row_kind(row,end_q[j],window[j],tail_begin[j])==expected);
     if(row>=fast_lo&&row<fast_hi) assert(expected==RKVA_KIND_RECORD);
     ++rows_checked;
    }
    assert(row_kind(e,end_q[j],window[j],tail_begin[j])==RKVA_KIND_NONE);
   }
  }
 }
 std::cout<<"{\"success\":true,\"selector_calls\":"<<calls<<",\"row_classifications\":"<<rows_checked<<",\"phantom_query_checks\":"<<phantom<<",\"inactive_query_split_checks\":"<<zero_splits<<",\"scope\":\"exact HLSL arithmetic extracts executed as C++, only [unroll] attribute removed; not shader compilation\"}\n";
}
