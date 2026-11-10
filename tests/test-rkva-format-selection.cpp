// Local-only format identity regression. No model loading or network calls.
#include "common.h"
#include "ggml-remote-attn.h"
#include <stdexcept>
#include <string>
#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
    for (const char * requested : {"kvarn2", "kvarn3", "kvarn4", "kvarn5", "kvarn6", "kvarn8", "kvarn"}) {
        for (const char * upload : {"kvarn2", "kvarn3", "kvarn4", "kvarn5", "kvarn6", "kvarn8", "kvarn"}) {
            common_params params;
            params.remote_attn_host = "127.0.0.1";
            params.kv_position_split_p = 256;
            params.remote_attn_native_cache_type_k = requested;
            params.remote_attn_native_cache_type_v = requested;
            params.remote_attn_format = upload;
            const std::string a = std::string(requested) == "kvarn" ? "kvarn4" : requested;
            const std::string b = std::string(upload) == "kvarn" ? "kvarn4" : upload;
            bool accepted = false;
            try {
                accepted = common_resolve_remote_native_companion(params) == LLAMA_REMOTE_COMPANION_KVARN4;
            } catch (const std::invalid_argument &) {}
            assert(accepted == (a == b));
        }
    }
    for (const char * format : {"kvarn2", "kvarn3", "kvarn4", "kvarn5", "kvarn6", "kvarn8"}) {
        assert(ggml_remote_attn_companion_format_matches(format, GGML_TYPE_F32));
        for (auto type : {GGML_TYPE_Q2_0S, GGML_TYPE_Q3_0, GGML_TYPE_Q4_0,
                          GGML_TYPE_Q5_0, GGML_TYPE_Q6_0, GGML_TYPE_Q8_0})
            assert(!ggml_remote_attn_companion_format_matches(format, type));
    }
    std::puts("PASS: 49 explicit KVarN pairs and F32-only native companion; no network");
}
