#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using uint=unsigned;
using Vec=std::array<float,256>;
struct State { std::array<float,512> shared{}; Vec output{}; };
struct Task {
    struct promise_type {
        Task get_return_object(){return Task{std::coroutine_handle<promise_type>::from_promise(*this)};}
        std::suspend_always initial_suspend() noexcept{return {};}
        std::suspend_always final_suspend() noexcept{return {};}
        void return_void() noexcept{}
        void unhandled_exception(){std::terminate();}
    };
    std::coroutine_handle<promise_type> handle;
    explicit Task(std::coroutine_handle<promise_type> h):handle(h){}
    Task(Task&& t) noexcept:handle(t.handle){t.handle=nullptr;}
    ~Task(){if(handle) handle.destroy();}
};
#include "generated_kernels.inc"
using Kernel=Task(*)(State&, const Vec&, uint);
std::mt19937 schedule_rng(20261106);
uint64_t barrier_rounds=0;
Vec execute(Kernel kernel, const Vec& input) {
    State state;
    state.shared.fill(std::numeric_limits<float>::quiet_NaN());
    std::vector<Task> tasks; tasks.reserve(128);
    for(uint i=0;i<128;i++) tasks.push_back(kernel(state,input,i));
    std::array<uint,128> order; std::iota(order.begin(),order.end(),0);
    for(;;) {
        std::shuffle(order.begin(),order.end(),schedule_rng);
        uint done=0;
        for(uint lane:order) {assert(!tasks[lane].handle.done()); tasks[lane].handle.resume();done+=tasks[lane].handle.done();}
        if(done==128) break;
        if(done!=0) throw std::runtime_error("nonuniform group barrier participation");
        barrier_rounds++;
    }
    return state.output;
}
// Independent pair/blocks butterfly oracle, no XOR lane reads or LDS banking.
Vec oracle(Vec v) {
    for(uint half=0;half<2;half++) for(uint stride=1;stride<128;stride*=2)
        for(uint block=0;block<128;block+=2*stride) for(uint j=0;j<stride;j++) {
            uint lo=half*128+block+j,hi=lo+stride;
            float a=v[lo],b=v[hi]; v[lo]=a+b;v[hi]=a-b;
        }
    for(uint j=0;j<128;j++) {
        float a=v[j]*0.08838834764831845f,b=v[j+128]*0.08838834764831845f;
        v[j]=(a+b)*0.7071067811865475f;v[j+128]=(a-b)*0.7071067811865475f;
    }
    return v;
}
bool equal(float a,float b) {return (std::isnan(a)&&std::isnan(b))||std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b);}
bool equal(const Vec&a,const Vec&b){for(uint i=0;i<256;i++)if(!equal(a[i],b[i]))return false;return true;}
bool poison(const Vec&v){return std::all_of(v.begin(),v.end(),[](float x){return std::isnan(x);});}
struct Partial {float maximum,den;Vec num;};
struct Result {Vec out;float lse;};
Result bad_result(){Result r;r.out.fill(std::bit_cast<float>(0x7fc00000u));r.lse=r.out[0];return r;}
// CPU model of the UNMODIFIED reducer around the extracted WHT body.
Result reduce(const std::vector<Partial>& parts, Kernel kernel) {
    if(parts.size()>256)return bad_result();
    float maximum=-std::numeric_limits<float>::max();
    for(auto&p:parts) {
        if(p.den==0)continue;
        if(!std::isfinite(p.den)||p.den<0||!std::isfinite(p.maximum))return bad_result();
        maximum=std::max(maximum,p.maximum);
    }
    std::vector<float> w(parts.size());float sum=0;
    for(size_t s=0;s<parts.size();s++) {
        if(parts[s].den==0)continue;
        w[s]=std::exp(parts[s].maximum-maximum);
        sum+=parts[s].den*w[s];
    }
    float inv=sum==0?0:1/sum;
    float lse=sum==0?-std::numeric_limits<float>::infinity():maximum+std::log(sum);
    bool invalid=!std::isfinite(sum)||sum<0||!std::isfinite(inv)||(sum!=0&&!std::isfinite(lse));
    Vec acc{};
    for(uint d=0;d<256;d++) {
        for(size_t s=0;s<parts.size();s++)if(parts[s].den!=0){invalid|=!std::isfinite(parts[s].num[d]);acc[d]+=parts[s].num[d]*w[s];}
        invalid|=!std::isfinite(acc[d]);acc[d]*=inv;
    }
    Vec out=execute(kernel,acc);
    for(float x:out)invalid|=!std::isfinite(x);
    return invalid?bad_result():Result{out,lse};
}
int main(){
    assert(std::fesetround(FE_TONEAREST)==0);
    static_assert(std::numeric_limits<float>::is_iec559&&sizeof(float)==4);
    std::mt19937 rng(20261106);
    std::normal_distribution<float> normal(0,2);
    std::vector<Vec> corpus;
    for(uint i=0;i<256;i++){Vec v{};v[i]=1;corpus.push_back(v);v[i]=-1;corpus.push_back(v);}
    for(uint k=0;k<1024;k++) {Vec v;for(float&x:v)x=normal(rng);corpus.push_back(v);}
    for(uint k=0;k<512;k++) {Vec v;for(float&x:v)x=std::ldexp(normal(rng),int(rng()%261)-130);corpus.push_back(v);}
    const std::array<uint32_t,15> adversarial={0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0x7fc12345,0x3f800000,0x3f800001,0xbf800000};
    for(uint32_t bits:adversarial){Vec v;v.fill(std::bit_cast<float>(bits));corpus.push_back(v);for(uint index:{0u,1u,63u,64u,127u,128u,129u,191u,192u,255u}){v.fill(0);v[index]=std::bit_cast<float>(bits);corpus.push_back(v);}}
    for(uint k=0;k<256;k++){Vec v;for(float&x:v)x=std::bit_cast<float>(adversarial[rng()%adversarial.size()]);corpus.push_back(v);}
    uint64_t checked=0,nan=0,inf=0,finite=0,unsafe_detected=0,fused_detected=0,stale_detected=0;
    std::ofstream dense_input("dense_input.f32",std::ios::binary),dense_output("dense_output.f32",std::ios::binary);
    uint dense_count=0;
    for(size_t k=0;k<corpus.size();k++) {
        const auto&v=corpus[k];Vec a=execute(baseline_kernel,v),b=execute(candidate_kernel,v),o=oracle(v);
        if(!equal(a,b)||!equal(o,b))throw std::runtime_error("mapping or float32 order mismatch case "+std::to_string(k));
        checked+=256;
        for(float x:b){nan+=std::isnan(x);inf+=std::isinf(x);finite+=std::isfinite(x);}
        if(k<1536){
            unsafe_detected+=!equal(a,execute(unsafe_kernel,v));
            fused_detected+=!equal(a,execute(fused_kernel,v));
            stale_detected+=!equal(a,execute(stale_kernel,v));
        }
        if(k>=512&&k<576){dense_input.write(reinterpret_cast<const char*>(v.data()),1024);dense_output.write(reinterpret_cast<const char*>(b.data()),1024);dense_count++;}
    }
    assert(unsafe_detected>0&&fused_detected>0&&stale_detected>0);
    uint reduce_cases=0,poison_cases=0,empty_cases=0;
    auto check_reduce=[&](const std::vector<Partial>&parts,int expected){
        Result a=reduce(parts,baseline_kernel),b=reduce(parts,candidate_kernel);
        if(!equal(a.out,b.out)||!equal(a.lse,b.lse))throw std::runtime_error("reducer mismatch");
        if(expected==1){assert(poison(b.out)&&std::isnan(b.lse));poison_cases++;}
        else if(expected==2){assert(std::all_of(b.out.begin(),b.out.end(),[](float x){return x==0;})&&std::isinf(b.lse)&&b.lse<0);empty_cases++;}
        else {assert(!poison(b.out)&&std::isfinite(b.lse));}
        reduce_cases++;
    };
    Partial good{0,1,{}};good.num.fill(2);
    Partial empty{-std::numeric_limits<float>::infinity(),0,{}};empty.num.fill(std::numeric_limits<float>::quiet_NaN());
    for(uint splits:{0u,1u,2u,3u,31u,32u,63u,64u,127u,128u,255u,256u,257u}) {
        check_reduce(std::vector<Partial>(splits,empty),splits>256?1:2);
        if(splits==0)continue;
        std::vector<Partial> p(splits,empty);p.back()=good;check_reduce(p,splits>256?1:0);
        for(uint s=0;s<splits;s++){p[s]=good;p[s].maximum=-float(s%20);p[s].den=float(s%7+1);for(float&x:p[s].num)x=normal(rng);}
        check_reduce(p,splits>256?1:0);
    }
    for(float den:{-1.0f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}){Partial bad=good;bad.den=den;check_reduce({good,bad},1);check_reduce({bad,good},1);}
    for(float mx:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}){Partial bad=good;bad.maximum=mx;check_reduce({good,bad},1);}
    for(float x:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()})for(uint index:{0u,63u,64u,127u,128u,191u,255u}){Partial bad=good;bad.maximum=-1000;bad.num[index]=x;check_reduce({good,bad},1);check_reduce({bad,good},1);}
    Partial huge=good;huge.den=std::numeric_limits<float>::max();check_reduce({huge,huge},1);
    huge=good;huge.num.fill(std::numeric_limits<float>::max());check_reduce({huge,huge},1);check_reduce({huge},1);
    Partial tiny=good;tiny.den=std::numeric_limits<float>::denorm_min();check_reduce({tiny},1);
    std::cout<<"{\n  \"status\":\"PASS\",\n  \"wht_vectors\":"<<corpus.size()<<",\n  \"wht_elements_checked\":"<<checked<<",\n  \"finite_bitwise_comparisons\":"<<finite<<",\n  \"infinity_bitwise_comparisons\":"<<inf<<",\n  \"nan_classification_comparisons\":"<<nan<<",\n  \"oracle\":\"independent scalar pair/block float32 butterfly\",\n  \"scheduler\":\"128 source-derived C++20 coroutines, randomized lane order between group barriers\",\n  \"barrier_rounds_executed\":"<<barrier_rounds<<",\n  \"negative_controls\":{\"unsafe_inplace_vectors_detected\":"<<unsafe_detected<<",\"fused_normalization_vectors_detected\":"<<fused_detected<<",\"stale_bank_vectors_detected\":"<<stale_detected<<"},\n  \"reducer_cases\":"<<reduce_cases<<",\n  \"poison_cases\":"<<poison_cases<<",\n  \"empty_cases\":"<<empty_cases<<",\n  \"dense_check_vectors\":"<<dense_count<<"\n}\n";
}
