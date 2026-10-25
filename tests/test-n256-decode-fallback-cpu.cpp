// CPU fallback test: an ineligible batch (one N256 ubatch) with the LLAMA_N256
// experiment flag on must use the normal serial path and match flag-off logits.
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <vector>

namespace {
const char * kDefaultModel = "/home/hjotha/models/Qwen3.5-4B-Q4_K_M.gguf";
int failures = 0;
void check(bool ok, const char * name) { std::printf("%s: %s\n", name, ok ? "OK" : "FAIL"); if (!ok) ++failures; }
bool file_ok(const char * p) { struct stat st; return stat(p,&st)==0 && S_ISREG(st.st_mode) && st.st_size>0; }
bool run(llama_model * model, bool flag, std::vector<float> & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=256; cp.n_threads=1; cp.n_threads_batch=1; cp.n_seq_max=1;
    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) return false;
    const llama_vocab * v = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(v); if (bos<0) bos=0;
    llama_batch b = llama_batch_init(256,0,1);
    b.n_tokens=256;
    for (int i=0;i<256;++i) { b.token[i]=bos; b.pos[i]=i; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=0; }
    b.logits[255]=1;
    setenv("LLAMA_N256_LOOKAHEAD", flag?"1":"0", 1);
    const int rc = llama_decode(ctx,b);
    llama_batch_free(b);
    check(rc==0, flag?"ineligible flag-on rc=0":"ineligible flag-off rc=0");
    if (rc!=0) { llama_free(ctx); return false; }
    const float * lp = llama_get_logits_ith(ctx,255);
    if (!lp) { llama_free(ctx); return false; }
    out.assign(lp, lp + llama_vocab_n_tokens(v));
    llama_free(ctx);
    return true;
}
}
int main() {
    const char * path = std::getenv("N256_SMALL_MODEL"); if (!path||!*path) path=kDefaultModel;
    if (!file_ok(path)) { std::printf("FAIL: model missing\n"); return 1; }
    llama_backend_init();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = {cpu,nullptr};
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers=0; mp.devices=devices;
    llama_model * model = llama_model_load_from_file(path,mp);
    if (!model) { llama_backend_free(); return 1; }
    std::vector<float> a,b;
    check(run(model,false,a), "flag-off fallback completed");
    check(run(model,true,b), "flag-on fallback completed");
    check(a.size()==b.size(), "fallback logits size");
    int mismatch=0;
    if (a.size()==b.size()) for (size_t i=0;i<a.size();++i) if (std::memcmp(&a[i],&b[i],sizeof(float))!=0) ++mismatch;
    std::printf("fallback logits compared=%zu mismatches=%d\n",a.size(),mismatch);
    check(mismatch==0,"fallback logits bitwise");
    llama_model_free(model); llama_backend_free();
    std::printf("%s: %d failure(s)\n", failures==0?"OK":"FAILED", failures);
    return failures==0?0:1;
}
