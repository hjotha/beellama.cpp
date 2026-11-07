#!/usr/bin/env python3
"""Exercise the actual C-context positional-companion admission guard."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
backend = (root / 'ggml/src/ggml-remote-attn/ggml-remote-attn.cpp').read_text()
context = (root / 'src/llama-context.cpp').read_text()

def function(text, signature):
    start = text.index(signature)
    brace = text.index('{', start)
    depth = 0
    for end in range(brace, len(text)):
        if text[end] == '{': depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0: return text[start:end + 1]
    raise RuntimeError('unterminated function')

policy = function(backend, 'bool ggml_remote_attn_companion_format_matches(')
start = context.index('    const bool remote_type_k = cparams.remote_attn_cache_type_k != GGML_TYPE_COUNT;')
end = context.index('    if (remote_type_k) {', start)
guard = context[start:end]
program = r'''
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
enum ggml_type { GGML_TYPE_COUNT, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32 };
enum companion { LLAMA_REMOTE_COMPANION_NONE, LLAMA_REMOTE_COMPANION_KVARN4,
                 LLAMA_REMOTE_COMPANION_SAW_INT4, LLAMA_REMOTE_COMPANION_TURBOQUANT_NC };
struct params { companion remote_attn_native_companion; ggml_type remote_attn_cache_type_k, remote_attn_cache_type_v;
                std::string remote_attn_upload_format; };
''' + policy + r'''
bool permits(bool tcp_position_split, const params &cparams) {
    try {
''' + guard + r'''
        return true;
    } catch (const std::invalid_argument &) { return false; }
}
int main() {
    for (auto format : {"kvarn4", "saw_int4", "turboquant_nc"}) {
        auto native = !std::strcmp(format, "kvarn4") ? LLAMA_REMOTE_COMPANION_KVARN4 :
                      !std::strcmp(format, "saw_int4") ? LLAMA_REMOTE_COMPANION_SAW_INT4 :
                                                          LLAMA_REMOTE_COMPANION_TURBOQUANT_NC;
        assert(permits(true, {native, GGML_TYPE_COUNT, GGML_TYPE_COUNT, format}));
        for (auto q : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
            assert(!permits(true, {native, q, q, format}));
            assert(!permits(true, {LLAMA_REMOTE_COMPANION_NONE, q, q, format}));
        }
    }
    assert(permits(true, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "q4_0"}));
    assert(permits(true, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "q8_0"}));
    assert(permits(true, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_F16, GGML_TYPE_F16, "f16"}));
    assert(permits(true, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_F16, GGML_TYPE_F16, "f32"}));
    assert(!permits(true, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "q8_0"}));
    assert(!permits(true, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, "auto"}));
    assert(permits(false, {LLAMA_REMOTE_COMPANION_NONE, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "kvarn4"}));
}
'''
with tempfile.TemporaryDirectory(prefix='beellama-companion-c-context-') as directory:
    cpp = Path(directory) / 'test.cpp'
    binary = Path(directory) / 'test'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Werror', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print('PASS: C-context companion guard matrix (isolated constructor branch)')
