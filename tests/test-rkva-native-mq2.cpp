// Included by test-rkva-native-mq2.py after actual HLSL scalar loader shims.
// Host-only mathematical/scheduling oracle. Does not execute DXIL or barriers.
#include "ggml-turboquant-nc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
using Row = std::array<float,256>;
static void wht(Row& x, int block) {
    for (int base=0;base<256;base+=block) {
        for(int stride=1;stride<block;stride*=2) for(int i=0;i<block;i+=2*stride)
            for(int j=0;j<stride;++j) {
                float a=x[base+i+j], b=x[base+i+j+stride];
                x[base+i+j]=a+b; x[base+i+j+stride]=a-b;
            }
        for(int i=0;i<block;++i)x[base+i]*=block==16?.25f:.0625f;
    }
}
static Row load(int fmt, ByteAddressBuffer buffer) {
    Row out{};
    for(uint lane=0;lane<64;++lane) {
        if(fmt==3) {
            auto i=mq_indices(buffer,0,lane);
            const uint indices[]={i.x,i.y,i.z,i.w};
            for(int j=0;j<4;++j)out[4*lane+j]=mq_centroids[indices[j]];
        } else {
            auto v=fmt==0?load_q4_1(buffer,0,lane):fmt==1?load_iq4_nl(buffer,0,lane):load_saw_int4(buffer,0,lane);
            uint d=fmt==0?quarter_dim_q4_1(lane):fmt==1?quarter_dim_iq4_nl(lane):quarter_dim_saw_int4(lane);
            for(int j=0;j<4;++j)out[d+j]=v[j];
        }
    }
    return out;
}
static void encode_decode(int fmt,const Row& input,bool key,Row& native,Row& original) {
    if(fmt==3) {
        if(key) {
            ggml_tqnc_key k{}; assert(ggml_tqnc_quantize_key(input.data(),&k));
            std::array<uint8_t,132> padded{}; std::memcpy(padded.data(),&k,sizeof(k));
            native=load(fmt,{padded.data(),padded.size()});
            // Emulate lane float4 dots followed by a balanced 64-lane reduction.
            std::array<float,64> sum{};
            for(int l=0;l<64;++l)for(int d=0;d<4;++d)sum[l]+=native[4*l+d]*native[4*l+d];
            for(int s=32;s>0;s/=2)for(int l=0;l<s;++l)sum[l]+=sum[l+s];
            float scale=ggml_fp16_to_fp32(k.norm)/std::sqrt(sum[0]+1e-16f);
            for(auto& x:native)x*=scale;
            assert(ggml_tqnc_dequantize_key(&k,original.data()));
        } else {
            ggml_tqnc_value v{}; assert(ggml_tqnc_quantize_value(input.data(),&v));
            ByteAddressBuffer b{reinterpret_cast<const uint8_t*>(&v),sizeof(v)};
            for(uint l=0;l<64;++l) {
                auto i=mq_indices(b,0,l); const uint indices[]={i.x,i.y,i.z,i.w};
                for(int j=0;j<4;++j)native[4*l+j]=indices[j]*ggml_fp16_to_fp32(v.scale)+ggml_fp16_to_fp32(v.minimum);
            }
            assert(ggml_tqnc_dequantize_value(&v,original.data()));
        }
    } else {
        auto type=fmt==0?GGML_TYPE_Q4_1:fmt==1?GGML_TYPE_IQ4_NL:GGML_TYPE_SAW_INT4;
        std::vector<uint8_t> packed(ggml_row_size(type,256));
        assert(ggml_quantize_chunk(type,input.data(),packed.data(),0,1,256,nullptr)==packed.size());
        native=load(fmt,{packed.data(),packed.size()});
        ggml_get_type_traits(type)->to_float(packed.data(),original.data(),256);
    }
    Row decoded=native;
    if(fmt==2)wht(decoded,16);
    if(fmt==3 && key)wht(decoded,256);
    for(int d=0;d<256;++d)assert(std::abs(decoded[d]-original[d])<=2e-5f*(1+std::abs(original[d])));
}
struct State { Row acc{};float m=-std::numeric_limits<float>::max(),den=0; };
static float dot_wave(const Row& a,const Row& b,int fmt) {
    std::array<float,64> lanes{};
    for(uint l=0;l<64;++l) {
        uint d=fmt==0?quarter_dim_q4_1(l):fmt==1?quarter_dim_iq4_nl(l):4*l;
        for(int j=0;j<4;++j)lanes[l]+=a[d+j]*b[d+j];
    }
    for(int s=32;s>0;s/=2)for(int l=0;l<s;++l)lanes[l]+=lanes[l+s];
    return lanes[0];
}
static void append(State& s,float score,const Row& v) {
    float next=std::max(s.m,score),old=s.den==0?0:std::exp(s.m-next),p=std::exp(score-next);
    for(int d=0;d<256;++d)s.acc[d]=s.acc[d]*old+v[d]*p;
    s.den=s.den*old+p;s.m=next;
}
static size_t cases=0,checked=0;
static double maxerr=0,maxlse=0;
static void run(int fmt,int rows,int nq,int splits,int origin) {
    constexpr int nhkv=2,gqa=6,nh=nhkv*gqa;
    std::mt19937 rng(1203+fmt+rows+nq);std::uniform_real_distribution<float> dist(-2,2);
    std::vector<Row> k(rows*nhkv),v(k.size()),ko(k.size()),vo(k.size()),q(nq*nh),qo(q.size());
    for(size_t i=0;i<k.size();++i) {
        Row x{},y{};for(auto& f:x)f=dist(rng);for(auto& f:y)f=dist(rng);
        if(i==0)x.fill(0);if(i==1)y.fill(-.375f);
        encode_decode(fmt,x,true,k[i],ko[i]);encode_decode(fmt,y,false,v[i],vo[i]);
    }
    for(size_t i=0;i<q.size();++i) {for(auto& f:qo[i])f=dist(rng);q[i]=qo[i];if(fmt==2)wht(q[i],16);if(fmt==3)wht(q[i],256);}
    std::vector<int> positions(nq),limits(nq);
    for(int i=0;i<nq;++i) {
        // Includes fully masked, full context, one row and non-monotone masks.
        positions[i]=i%4==0?origin-1:i%4==1?origin+rows+3:i%4==2?origin:origin+rows/2;
        limits[i]=positions[i]<origin?0:std::min(rows,positions[i]-origin+1);
    }
    int splitrows=(rows+splits-1)/splits;size_t loads=0;
    std::vector<State> states(splits*nq*nh);
    for(int sp=0;sp<splits;++sp) for(int tile=0;tile<(nq+1)/2;++tile) for(int kh=0;kh<nhkv;++kh)
        for(int row=std::min(rows,sp*splitrows);row<std::min(rows,(sp+1)*splitrows);++row) {
            const Row& shared_k=k[row*nhkv+kh];const Row& shared_v=v[row*nhkv+kh];++loads;
            for(int wave=0;wave<2;++wave) {int qi=tile*2+wave;if(qi>=nq||row>=limits[qi])continue;
                for(int g=0;g<gqa;++g){int qh=qi*nh+kh*gqa+g;append(states[sp*nq*nh+qh],dot_wave(q[qh],shared_k,fmt)*.0625f,shared_v);}
            }
        }
    assert(loads==size_t(rows)*nhkv*((nq+1)/2));
    for(int qi=0;qi<nq;++qi) for(int kh=0;kh<nhkv;++kh)for(int g=0;g<gqa;++g) {
        int qh=qi*nh+kh*gqa+g;Row result{};float m=-std::numeric_limits<float>::max(),den=0;
        for(int sp=0;sp<splits;++sp) {auto& s=states[sp*nq*nh+qh];if(fmt==2)wht(s.acc,16);if(s.den>0)m=std::max(m,s.m);}
        for(int sp=0;sp<splits;++sp) {auto& s=states[sp*nq*nh+qh];if(!s.den)continue;float w=std::exp(s.m-m);den+=s.den*w;for(int d=0;d<256;++d)result[d]+=s.acc[d]*w;}
        if(den)for(auto& f:result)f/=den;
        double lse=den?m+std::log(den):-std::numeric_limits<double>::infinity();
        if(!limits[qi]) {assert(den==0 && std::isinf(lse)&&lse<0);for(float f:result)assert(f==0);continue;}
        std::vector<double> scores(limits[qi]);double mx=-1e300,rd=0;std::array<double,256> ref{};
        for(int r=0;r<limits[qi];++r){double score=0;for(int d=0;d<256;++d)score+=double(qo[qh][d])*ko[r*nhkv+kh][d];scores[r]=score*.0625;mx=std::max(mx,scores[r]);}
        for(int r=0;r<limits[qi];++r){double w=std::exp(scores[r]-mx);rd+=w;for(int d=0;d<256;++d)ref[d]+=w*vo[r*nhkv+kh][d];}
        maxlse=std::max(maxlse,std::abs(lse-(mx+std::log(rd))));assert(std::abs(lse-(mx+std::log(rd)))<5e-5);
        for(int d=0;d<256;++d){double e=std::abs(result[d]-ref[d]/rd);maxerr=std::max(maxerr,e);assert(e<5e-5);++checked;}
    }
    ++cases;
}
int main() {
    for(int f=0;f<4;++f)for(int rows:{1,3,65,129})for(int nq:{1,2,3,9,17})for(int splits:{1,2,5})for(int origin:{0,37})run(f,rows,nq,splits,origin);
    // Large dispatch/odd tail cases without multiplying every earlier axis.
    for(int f=0;f<4;++f)for(int rows:{3,65})for(int nq:{128,255,256})for(int splits:{1,5})
        run(f,rows,nq,splits,std::numeric_limits<int32_t>::max()-68);
    std::printf("PASS native MQ2 host oracle: %zu cases, %zu output elements, max abs OUT %.9g, LSE %.9g; no GPU execution\n",cases,checked,maxerr,maxlse);
}
