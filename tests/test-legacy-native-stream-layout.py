#!/usr/bin/env python3
"""Exercise the real legacy-native streaming layout admission helper."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'tools/server/server-context.cpp').read_text()
start = source.index('    static bool legacy_native_stream_layout_compatible(')
brace = source.index('{', start)
depth = 0
for end in range(brace, len(source)):
    if source[end] == '{': depth += 1
    elif source[end] == '}':
        depth -= 1
        if depth == 0: break
else:
    raise RuntimeError('unterminated layout helper')
method = source[start:end + 1]
program = r'''
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_set>
using common_json = nlohmann::json;
enum { GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q8_0 = 8, LLAMA_KVARN_K4V4_G128 = 9 };
struct server {
''' + method + r'''
};
int main() {
    common_json source = {
        {"cache_layout_version", 2}, {"rope_type", 1}, {"ctx_type", 0}, {"kv_layout_known", true},
        {"type_k", 9}, {"type_v", 9}, {"type_k_aux", 2}, {"type_v_aux", 2},
        {"native_global_tail_tokens", 128}, {"mixed_kv", true}, {"mixed_kvarn_type", 9},
        {"mixed_kvarn_bits_k", 4}, {"mixed_kvarn_bits_v", 4}, {"mixed_layer_count", 4},
        {"mixed_remote_layers", common_json::array({{{"layer", 0}, {"type_k", 2}, {"type_v", 2}},
                                                     {{"layer", 2}, {"type_k", 2}, {"type_v", 2}}})}
    };
    common_json target = {{"cache_layout_version", 2}, {"rope_type", 1}, {"ctx_type", 0},
        {"kv_layout_known", true}, {"type_k", 9}, {"type_v", 9}, {"type_k_aux", 0}, {"type_v_aux", 0},
        {"native_global_tail_tokens", 128}, {"native_companion_format", 0}, {"native_companion_split_p", 512},
        {"native_companion_kvarn_type", 9}, {"native_companion_kvarn_bits_k", 4}, {"native_companion_kvarn_bits_v", 4}};
    auto accept = [&](const common_json &a, const common_json &b, const char *model = "model", bool marker = true,
                      uint32_t ctx = 1024, uint32_t p = 512) {
        return server::legacy_native_stream_layout_compatible(a.dump(), b.dump(), model, "model", marker,
                                                               ctx, 1024, p, 512);
    };
    assert(accept(source, target));
    auto q8 = source; for (auto &layer : q8["mixed_remote_layers"]) layer["type_k"] = layer["type_v"] = 8;
    assert(accept(q8, target));
    assert(!accept(source, target, "wrong")); assert(!accept(source, target, "model", false));
    assert(!accept(source, target, "model", true, 2048)); assert(!accept(source, target, "model", true, 1024, 256));
    auto bad = source; bad["mixed_kvarn_bits_v"] = 5; assert(!accept(bad, target));
    bad = target; bad["native_companion_kvarn_bits_k"] = 5; assert(!accept(source, bad));
    bad = target; bad["native_global_tail_tokens"] = 0; assert(!accept(source, bad));
    bad = target; bad["native_companion_split_p"] = 256; assert(!accept(source, bad));
    bad = target; bad["native_companion_format"] = 4; assert(!accept(source, bad));
    bad = source; bad["mixed_remote_layers"][0]["type_v"] = 8; assert(!accept(bad, target));
    bad = source; bad["mixed_remote_layers"][1]["type_k"] = 8; bad["mixed_remote_layers"][1]["type_v"] = 8; assert(!accept(bad, target));
    bad = source; bad["mixed_remote_layers"][1]["layer"] = 0; assert(!accept(bad, target));
    bad = source; bad["mixed_remote_layers"] = common_json::array(); assert(!accept(bad, target));
    bad = source; bad["rope_type"] = 2; assert(!accept(bad, target));
    bad = source; bad["extra"] = true; assert(!accept(bad, target));
}
'''
with tempfile.TemporaryDirectory(prefix='beellama-legacy-native-layout-') as directory:
    cpp = Path(directory) / 'test.cpp'
    binary = Path(directory) / 'test'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Werror', '-I', str(root / 'vendor'),
                    str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print('PASS: legacy Q4/Q8 native-stream layout admission and negative controls')
