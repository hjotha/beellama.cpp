// CPU high-position contract: n_ctx=131072, two 512-token chunks starting at
// position 102400 (not a full 102400-token prefill). Serial vs lookahead logits
// must match bitwise; this exercises high cell IDs/positions without claiming
// long-context end-to-end equivalence.
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <vector>
namespace {
const char * kDefaultModel="/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";
int failures=0;
void check(bool ok,const char*n){std::printf("%s: %s\n",n,ok?"OK":"FAIL");if(!ok)++failures;}
bool file_ok(const char*p){struct stat st;return stat(p,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0;}
bool run(llama_model*model,bool flag,std::vector<float>&out,double&ms){
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=131072;cp.n_batch=512;cp.n_ubatch=256;cp.n_threads=1;cp.n_threads_batch=1;cp.n_seq_max=1;
    llama_context*ctx=llama_new_context_with_model(model,cp);if(!ctx)return false;
    const llama_vocab*v=llama_model_get_vocab(model);llama_token bos=llama_vocab_bos(v);if(bos<0)bos=0;
    setenv("LLAMA_N256_LOOKAHEAD",flag?"1":"0",1);
    const char * se = std::getenv("N256_LONG_START");
    const int start = se && *se ? std::atoi(se) : 102400;
    auto t0=std::chrono::steady_clock::now();
    for(int chunk=0;chunk<2;++chunk){
        llama_batch b=llama_batch_init(512,0,1);b.n_tokens=512;
        for(int i=0;i<512;++i){b.token[i]=bos;b.pos[i]=start+chunk*512+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=0;}
        b.logits[511]=1;
        const int rc=llama_decode(ctx,b);llama_batch_free(b);
        check(rc==0,flag?"longpos lookahead chunk rc=0":"longpos serial chunk rc=0");
        if(rc!=0){llama_free(ctx);return false;}
    }
    auto t1=std::chrono::steady_clock::now();ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    const float*lp=llama_get_logits_ith(ctx,511);if(!lp){llama_free(ctx);return false;}
    out.assign(lp,lp+llama_vocab_n_tokens(v));llama_free(ctx);return true;
}
}
int main(){
    const char*path=std::getenv("N256_SMALL_MODEL");if(!path||!*path)path=kDefaultModel;
    if(!file_ok(path)){std::printf("FAIL: model missing\n");return 1;}
    llama_backend_init();ggml_backend_dev_t cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[]={cpu,nullptr};llama_model_params mp=llama_model_default_params();mp.n_gpu_layers=0;mp.devices=devices;
    llama_model*model=llama_model_load_from_file(path,mp);if(!model){llama_backend_free();return 1;}
    const char * se = std::getenv("N256_LONG_START");
    const int start = se && *se ? std::atoi(se) : 102400;
    std::printf("longpos start=%d\n", start);
    std::vector<float>a,b;double ma=0,mb=0;
    check(run(model,false,a,ma),"longpos serial completed");
    check(run(model,true,b,mb),"longpos lookahead completed");
    check(a.size()==b.size(),"longpos logits size");
    int mismatch=0;if(a.size()==b.size())for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(float))!=0)++mismatch;
    std::printf("longpos logits compared=%zu mismatches=%d serial_ms=%.1f lookahead_ms=%.1f\n",a.size(),mismatch,ma,mb);
    check(mismatch==0,"longpos serial vs lookahead bitwise");
    llama_model_free(model);llama_backend_free();
    std::printf("%s: %d failure(s)\n",failures==0?"OK":"FAILED",failures);
    return failures==0?0:1;
}
