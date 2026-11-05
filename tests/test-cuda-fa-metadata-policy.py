#!/usr/bin/env python3
"""Exercise the production CUDA FA guard with a small CPU-only tensor fixture."""

from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
start = source.index("static bool ggml_cuda_flash_attn_validation_ok(")
end = source.index("\n}\n", start) + 2
guard = source[start:end]

program = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr int GGML_MAX_SRC = 12;
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_F16 };
struct ggml_tensor {
    ggml_tensor * src[GGML_MAX_SRC] = {};
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {};
    void * data = nullptr;
    bool contiguous = true;
    bool kvarn_view = false;
};
static bool support = true;
static int support_calls = 0;
static bool ggml_is_contiguous(const ggml_tensor * t) { return t->contiguous; }
static bool ggml_cuda_flash_attn_ext_kvarn_uses_views(const ggml_tensor * t) {
    return t->src[1]->kvarn_view || t->src[2]->kvarn_view;
}
static bool ggml_cuda_flash_attn_ext_supported(int, const ggml_tensor * t) {
    ++support_calls;
    const bool metadata_only = t->src[8] && !t->src[5] && !t->src[6] &&
        !t->src[7] && !t->src[9] && !t->src[10] && !t->src[11];
    const bool current_pair = (t->src[10] == nullptr) == (t->src[11] == nullptr);
    return support && current_pair &&
        (!metadata_only || (t->src[1]->kvarn_view && t->src[2]->kvarn_view));
}
''' + guard + r'''

static void check(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}
int main() {
    ggml_tensor q, k, v, meta, extra, node;
    q.ne[0] = 256; q.ne[1] = 3; q.ne[2] = 6; q.ne[3] = 2;
    meta.ne[0] = 2; meta.ne[1] = 6; meta.ne[2] = 3; meta.ne[3] = 2;
    meta.data = &meta;
    k.kvarn_view = v.kvarn_view = true;
    node.src[0] = &q; node.src[1] = &k; node.src[2] = &v; node.src[8] = &meta;

    check(ggml_cuda_flash_attn_validation_ok(0, &node), "valid metadata rejected");
    check(support_calls == 1, "valid metadata bypassed backend support");
    for (int dim = 0; dim < 4; ++dim) {
        ++meta.ne[dim];
        check(!ggml_cuda_flash_attn_validation_ok(0, &node), "invalid metadata shape accepted");
        --meta.ne[dim];
    }
    meta.type = GGML_TYPE_F16;
    check(!ggml_cuda_flash_attn_validation_ok(0, &node), "invalid metadata type accepted");
    meta.type = GGML_TYPE_F32;
    meta.contiguous = false;
    check(!ggml_cuda_flash_attn_validation_ok(0, &node), "non-contiguous metadata accepted");
    meta.contiguous = true;
    meta.data = nullptr;
    check(!ggml_cuda_flash_attn_validation_ok(0, &node), "unallocated metadata accepted");
    meta.data = &meta;

    k.kvarn_view = v.kvarn_view = false;
    check(!ggml_cuda_flash_attn_validation_ok(0, &node), "non-KVarN metadata accepted");
    k.kvarn_view = true;
    check(!ggml_cuda_flash_attn_validation_ok(0, &node), "one-sided KVarN accepted");
    v.kvarn_view = true;
    support = false;
    check(!ggml_cuda_flash_attn_validation_ok(0, &node), "backend rejection bypassed");
    support = true;

    // Every incomplete tail combination must fail, apart from valid src[8] metadata alone.
    for (unsigned mask = 0; mask < (1u << (GGML_MAX_SRC - 5)); ++mask) {
        for (int i = 5; i < GGML_MAX_SRC; ++i) {
            node.src[i] = mask & (1u << (i - 5)) ? (i == 8 ? &meta : &extra) : nullptr;
        }
        const unsigned required_tail = (1u << 5) - 1;
        const bool current_pair = bool(mask & (1u << 5)) == bool(mask & (1u << 6));
        const bool complete = (mask & required_tail) == required_tail && current_pair;
        const bool valid = mask == 0 || mask == (1u << 3) || complete;
        support_calls = 0;
        check(ggml_cuda_flash_attn_validation_ok(0, &node) == valid, "incorrect tail/metadata decision");
        check(support_calls == int(valid), "unexpected backend support call");
    }
    for (int current = 10; current <= 11; ++current) {
        for (int i = 5; i <= 9; ++i) node.src[i] = i == 8 ? &meta : &extra;
        node.src[10] = node.src[11] = nullptr;
        node.src[current] = &extra;
        support_calls = 0;
        check(!ggml_cuda_flash_attn_validation_ok(0, &node), "one-sided current tail accepted");
        check(support_calls == 0, "incomplete current tail reached backend support");
    }
    for (int i = 5; i < GGML_MAX_SRC; ++i) node.src[i] = nullptr;
    node.src[8] = &meta;
    check(ggml_cuda_flash_attn_validation_ok(0, &node), "valid metadata fixture not restored");
    for (int i = 0; i < 3; ++i) {
        auto * saved = node.src[i]; node.src[i] = nullptr;
        check(!ggml_cuda_flash_attn_validation_ok(0, &node), "missing Q/K/V accepted");
        node.src[i] = saved;
    }
    std::puts("CUDA FA metadata-only guard: positive and negative cases passed");
}
'''

with tempfile.TemporaryDirectory(prefix="cuda-fa-metadata-policy-") as tmp:
    cpp = Path(tmp) / "test.cpp"
    exe = Path(tmp) / "test"
    cpp.write_text(program)
    subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
