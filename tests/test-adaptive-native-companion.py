#!/usr/bin/env python3
"""Exercise real resolver and ahead selection after local tier overrides are cleared."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
common = (root / 'common/common.cpp').read_text()
server = (root / 'tools/server/server-context.cpp').read_text()
backend = (root / 'ggml/src/ggml-remote-attn/ggml-remote-attn.cpp').read_text()

def function(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 0
    for end in range(brace, len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[start:end + 1]
    raise RuntimeError('unterminated function')

resolver = function(common, 'enum llama_remote_native_companion common_resolve_remote_native_companion(')
policy = function(backend, 'bool ggml_remote_attn_companion_format_matches(')
start = server.index('        const auto native_selection = adaptive_remote_native_companion;')
end = server.index(';', server.index('adaptive_remote_attn_cache_type_v == GGML_TYPE_Q4_0', start)) + 1
ahead = server[start:end]
assert 'adaptive_remote_native_companion = common_resolve_remote_native_companion(params);' in server
program = r'''
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
enum ggml_type { GGML_TYPE_COUNT, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32 };
enum llama_remote_native_companion { LLAMA_REMOTE_COMPANION_AUTO, LLAMA_REMOTE_COMPANION_NONE,
    LLAMA_REMOTE_COMPANION_KVARN4, LLAMA_REMOTE_COMPANION_SAW_INT4, LLAMA_REMOTE_COMPANION_TURBOQUANT_NC };
struct common_params {
    ggml_type remote_attn_cache_type_k = GGML_TYPE_COUNT, remote_attn_cache_type_v = GGML_TYPE_COUNT;
    std::string remote_attn_native_cache_type_k, remote_attn_native_cache_type_v, remote_attn_format, remote_attn_host;
    unsigned kv_position_split_p = 1024;
};
''' + policy + r'''
bool common_remote_attn_is_local_vulkan(const std::string &host) { return host.rfind("vulkan:", 0) == 0; }
''' + resolver + r'''
bool select_ahead(llama_remote_native_companion adaptive_remote_native_companion,
        ggml_type adaptive_remote_attn_cache_type_k, ggml_type adaptive_remote_attn_cache_type_v) {
''' + ahead + r'''
    return matching_companion;
}
int main() {
    unsetenv("GGML_RKVA_NATIVE_COMPANION");
    for (const char *format : {"kvarn4", "kvarn", "saw_int4", "saw-int4", "saw4", "bdr_int4",
                               "turboquant_nc", "turboquant-nc", "tqnc", "turboquant_4bit_nc"}) {
        assert(ggml_remote_attn_companion_format_matches(format, GGML_TYPE_F32));
        assert(!ggml_remote_attn_companion_format_matches(format, GGML_TYPE_Q4_0));
        assert(!ggml_remote_attn_companion_format_matches(format, GGML_TYPE_Q8_0));
        assert(!ggml_remote_attn_companion_format_matches(format, GGML_TYPE_F16));
    }
    assert(ggml_remote_attn_companion_format_matches("q4_0", GGML_TYPE_Q4_0));
    assert(!ggml_remote_attn_companion_format_matches("q4_0", GGML_TYPE_Q8_0));
    assert(ggml_remote_attn_companion_format_matches("q8_0", GGML_TYPE_Q8_0));
    assert(!ggml_remote_attn_companion_format_matches("q8_0", GGML_TYPE_Q4_0));
    assert(ggml_remote_attn_companion_format_matches("f16", GGML_TYPE_F16));
    assert(!ggml_remote_attn_companion_format_matches("f16", GGML_TYPE_Q4_0));
    assert(ggml_remote_attn_companion_format_matches("auto", GGML_TYPE_Q4_0));
    assert(ggml_remote_attn_companion_format_matches("auto", GGML_TYPE_Q8_0));
    assert(ggml_remote_attn_companion_format_matches("auto", GGML_TYPE_F16));
    assert(!ggml_remote_attn_companion_format_matches("auto", GGML_TYPE_F32));
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16})
        assert(ggml_remote_attn_companion_format_matches("f32", type)); // explicit diagnostic transport
    assert(!ggml_remote_attn_companion_format_matches("f32", GGML_TYPE_F32));
    assert(!ggml_remote_attn_companion_format_matches("oscar_int2", GGML_TYPE_F32));
    for (const auto &entry : {std::pair<const char *, ggml_type>{"q4_1", GGML_TYPE_Q4_1},
                              {"iq4_nl", GGML_TYPE_IQ4_NL}}) {
        assert(ggml_remote_attn_companion_format_matches(entry.first, entry.second));
        for (auto other : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32})
            assert(!ggml_remote_attn_companion_format_matches(entry.first, other));
        assert(!ggml_remote_attn_companion_format_matches("auto", entry.second));
        assert(!ggml_remote_attn_companion_format_matches("f32", entry.second));
    }
    assert(!ggml_remote_attn_companion_format_matches("q4_1", GGML_TYPE_IQ4_NL));
    assert(!ggml_remote_attn_companion_format_matches("iq4_nl", GGML_TYPE_Q4_1));
    for (const auto &entry : {std::pair<const char *, ggml_type>{"q4_1", GGML_TYPE_Q4_1},
                              {"iq4_nl", GGML_TYPE_IQ4_NL}}) {
        common_params low4;
        low4.remote_attn_host = "127.0.0.1";
        low4.remote_attn_format = entry.first;
        low4.remote_attn_cache_type_k = low4.remote_attn_cache_type_v = entry.second;
        assert(common_resolve_remote_native_companion(low4) == LLAMA_REMOTE_COMPANION_NONE);
        low4.remote_attn_cache_type_v = GGML_TYPE_Q4_0;
        bool mismatch = false;
        try { (void) common_resolve_remote_native_companion(low4); } catch (const std::invalid_argument &) { mismatch = true; }
        assert(mismatch);
    }
    common_params original;
    original.remote_attn_format = "kvarn4";
    original.remote_attn_cache_type_k = GGML_TYPE_Q4_0;
    original.remote_attn_cache_type_v = GGML_TYPE_Q4_0;
    bool refused = false;
    try { (void) common_resolve_remote_native_companion(original); } catch (const std::invalid_argument &) { refused = true; }
    assert(refused); // native KVarN cannot allocate a Q4 companion
    original.remote_attn_format = "q4_0";
    auto captured = common_resolve_remote_native_companion(original);
    assert(captured == LLAMA_REMOTE_COMPANION_NONE);
    common_params local = original;
    local.remote_attn_cache_type_k = local.remote_attn_cache_type_v = GGML_TYPE_COUNT;
    local.remote_attn_format = "kvarn4"; // changed profile cannot reinterpret the captured original
    assert(common_resolve_remote_native_companion(local) == LLAMA_REMOTE_COMPANION_KVARN4);
    assert(select_ahead(captured, original.remote_attn_cache_type_k, original.remote_attn_cache_type_v));
    assert(!select_ahead(common_resolve_remote_native_companion(local),
                         original.remote_attn_cache_type_k, original.remote_attn_cache_type_v));
    common_params native = local;
    native.remote_attn_native_cache_type_k = native.remote_attn_native_cache_type_v = "kvarn4";
    assert(select_ahead(common_resolve_remote_native_companion(native), GGML_TYPE_COUNT, GGML_TYPE_COUNT));
}
'''
with tempfile.TemporaryDirectory(prefix='beellama-adaptive-native-') as directory:
    source = Path(directory) / 'test.cpp'
    binary = Path(directory) / 'test'
    source.write_text(program)
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Werror', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print('PASS: ahead retains original Q4 companion selection across local-tier override clearing')
