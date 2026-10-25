// Context/backend sampler fallback: configure a real greedy sampler chain in the
// context. The opt-in 2xN256 path must fall back to the normal serial decode;
// the sampled token must match flag-OFF. No GPU/RPC/benchmark.
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
namespace {
const char * kDefaultModel="/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";
int failures=0;
void check(bool ok,const char*n){std::printf("%s: %s\n",n,ok?"OK":"FAIL");if(!ok)++failures;}
bool file_ok(const char*p){struct stat st;return stat(p,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0;}
bool run(llama_model*model,bool flag,bool dist,llama_token&tok){
    llama_context_params cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=512;cp.n_ubatch=256;cp.n_threads=1;cp.n_threads_batch=1;cp.n_seq_max=1;
    llama_context*ctx=llama_new_context_with_model(model,cp);if(!ctx)return false;
    llama_sampler_chain_params sp=llama_sampler_chain_default_params();
    llama_sampler*chain=llama_sampler_chain_init(sp);
    llama_sampler_chain_add(chain, dist ? llama_sampler_init_dist(12345) : llama_sampler_init_greedy());
    check(llama_set_sampler(ctx,0,chain), dist ? "context dist sampler accepted" : "context greedy sampler accepted");
    const llama_vocab*v=llama_model_get_vocab(model);llama_token bos=llama_vocab_bos(v);if(bos<0)bos=0;
    llama_batch b=llama_batch_init(512,0,1);b.n_tokens=512;
    for(int i=0;i<512;++i){b.token[i]=bos;b.pos[i]=i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=0;}
    b.logits[511]=1;
    setenv("LLAMA_N256_LOOKAHEAD",flag?"1":"0",1);
    const int rc=llama_decode(ctx,b);llama_batch_free(b);
    check(rc==0,flag ? (dist?"context dist flag-on rc=0":"context greedy flag-on rc=0")
                     : (dist?"context dist flag-off rc=0":"context greedy flag-off rc=0"));
    if(rc==0)tok=llama_get_sampled_token_ith(ctx,511);
    llama_free(ctx);
    llama_sampler_free(chain);
    return rc==0;
}
}
int main(){
    const char*path=std::getenv("N256_SMALL_MODEL");if(!path||!*path)path=kDefaultModel;
    if(!file_ok(path)){std::printf("FAIL: model missing\n");return 1;}
    llama_backend_init();ggml_backend_dev_t cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[]={cpu,nullptr};llama_model_params mp=llama_model_default_params();mp.n_gpu_layers=0;mp.devices=devices;
    llama_model*model=llama_model_load_from_file(path,mp);if(!model){llama_backend_free();return 1;}
    llama_token a=-1,b=-1;
    check(run(model,false,false,a),"flag-off context greedy completed");
    check(run(model,true,false,b),"flag-on context greedy completed");
    std::printf("context greedy token flag-off=%d flag-on=%d\n",(int)a,(int)b);
    check(a==b && a>=0,"context greedy token equal");

    llama_token c=-1,d=-1;
    check(run(model,false,true,c),"flag-off context dist completed");
    check(run(model,true,true,d),"flag-on context dist completed");
    std::printf("context dist token flag-off=%d flag-on=%d\n",(int)c,(int)d);
    check(c==d && c>=0,"context dist token equal (seed)");
    llama_model_free(model);llama_backend_free();
    std::printf("%s: %d failure(s)\n",failures==0?"OK":"FAILED",failures);
    return failures==0?0:1;
}
