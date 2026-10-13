// Parity test v2: GGML_OP_KVARN_WHT across CPU / CUDA / Vulkan,
// grid over production-like 4D tail shapes [D, n_kv_heads, n_tail] and F32/F16/BF16.
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float load_at(const unsigned char * p, enum ggml_type type) {
    if (type == GGML_TYPE_F32) { float v; memcpy(&v, p, 4); return v; }
    uint16_t h; memcpy(&h, p, 2);
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(h);
    uint32_t u = ((uint32_t)h) << 16; float v; memcpy(&v, &u, 4); return v; // bf16
}

static int run_one(ggml_backend_t be, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                   int hw, const void * src, void * dst, size_t nbytes) {
    struct ggml_init_params p = { 128u << 20, NULL, true };
    struct ggml_context * ctx = ggml_init(p);
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    struct ggml_tensor * t = ggml_kvarn_wht(ctx, a, hw);
    if (!ggml_backend_alloc_ctx_tensors(ctx, be)) { ggml_free(ctx); return -1; }
    ggml_backend_tensor_set(a, src, 0, nbytes);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, t);
    if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return -2; }
    ggml_backend_tensor_get(t, dst, 0, nbytes);
    ggml_free(ctx);
    return 0;
}

static ggml_backend_t find_dev(const char * key) {
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(d), key)) return ggml_backend_dev_init(d, NULL);
    }
    return NULL;
}

static int report(const char * tag, enum ggml_type type,
                  const void * ref, const void * got, size_t nbytes) {
    size_t esz = ggml_type_size(type);
    int64_t n = (int64_t)(nbytes / esz), diff = 0;
    double maxabs = 0.0, maxrel = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const unsigned char * pr = (const unsigned char *)ref + i * esz;
        const unsigned char * pg = (const unsigned char *)got + i * esz;
        if (memcmp(pr, pg, esz) != 0) {
            diff++;
            double a = load_at(pr, type), b = load_at(pg, type);
            double d = fabs(a - b);
            if (d > maxabs) maxabs = d;
            double r = d / (fabs(a) + 1e-30);
            if (r > maxrel) maxrel = r;
        }
    }
    printf("%-44s bitdiff=%lld/%lld maxabs=%.3e maxrel=%.3e %s\n", tag,
           (long long)diff, (long long)n, maxabs, maxrel,
           diff == 0 ? "BIT_IDENTICAL" : "DIFFERS");
    return diff == 0 ? 0 : 1;
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_t be_cpu  = find_dev("CPU");
    ggml_backend_t be_cuda = find_dev("CUDA");
    ggml_backend_t be_vk   = find_dev("Vulkan");
    if (!be_cpu) { printf("FAIL no cpu\n"); return 3; }
    printf("backends: cuda=%s vulkan=%s\n", be_cuda ? "yes" : "no", be_vk ? "yes" : "no");

    const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
    const char * tnames[] = { "f32", "f16", "bf16" };
    struct { int64_t ne0, ne1, ne2; } shapes[] = {
        { 256, 4, 127 }, { 256, 4, 31 }, { 256, 4, 1 },
        { 256, 1, 127 }, { 128, 4, 127 }, { 256, 4, 128 },
    };
    int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));
    int rc = 0;
    for (int ti = 0; ti < 3; ti++) {
        enum ggml_type type = types[ti];
        size_t esz = ggml_type_size(type);
        for (int si = 0; si < nshape; si++) {
            int64_t ne0 = shapes[si].ne0, ne1 = shapes[si].ne1, ne2 = shapes[si].ne2;
            int hw = (int)ne0;
            int64_t nel = ne0 * ne1 * ne2;
            if (nel % hw != 0) continue;
            size_t nbytes = (size_t)nel * esz;
            unsigned char * src = malloc(nbytes);
            unsigned char * o_cpu = malloc(nbytes);
            unsigned char * o_oth = malloc(nbytes);
            for (int64_t i = 0; i < nel; i++) {
                float v = (float)((i % 97) - 48) * 0.01f + (float)((i / 97) % 31) * 0.0017f;
                if (type == GGML_TYPE_F32) { memcpy(src + i * 4, &v, 4); }
                else if (type == GGML_TYPE_F16) { uint16_t h = ggml_fp32_to_fp16(v); memcpy(src + i * 2, &h, 2); }
                else { uint32_t u; memcpy(&u, &v, 4); uint16_t h = (uint16_t)(u >> 16); memcpy(src + i * 2, &h, 2); }
            }
            char tag[128];
            int r0 = run_one(be_cpu, type, ne0, ne1, ne2, hw, src, o_cpu, nbytes);
            if (r0 != 0) {
                printf("FAIL cpu %s [%lld,%lld,%lld] rc=%d\n", tnames[ti],
                       (long long)ne0, (long long)ne1, (long long)ne2, r0);
                rc = 3; goto done;
            }
            if (be_cuda) {
                int r1 = run_one(be_cuda, type, ne0, ne1, ne2, hw, src, o_oth, nbytes);
                snprintf(tag, sizeof(tag), "%s [%lld,%lld,%lld] CUDA", tnames[ti],
                         (long long)ne0, (long long)ne1, (long long)ne2);
                if (r1 != 0) { printf("%s run_fail rc=%d\n", tag, r1); rc = 1; }
                else if (report(tag, type, o_cpu, o_oth, nbytes)) rc = 1;
            }
            if (be_vk) {
                int r2 = run_one(be_vk, type, ne0, ne1, ne2, hw, src, o_oth, nbytes);
                snprintf(tag, sizeof(tag), "%s [%lld,%lld,%lld] VULKAN", tnames[ti],
                         (long long)ne0, (long long)ne1, (long long)ne2);
                if (r2 != 0) { printf("%s run_fail rc=%d\n", tag, r2); rc = 1; }
                else if (report(tag, type, o_cpu, o_oth, nbytes)) rc = 1;
            }
            free(src); free(o_cpu); free(o_oth);
        }
    }
done:
    if (be_cuda) ggml_backend_free(be_cuda);
    if (be_vk) ggml_backend_free(be_vk);
    ggml_backend_free(be_cpu);
    printf("PARITY_RC=%d\n", rc);
    return rc;
}
