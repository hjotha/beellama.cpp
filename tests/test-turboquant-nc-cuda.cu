#include "../ggml/src/ggml-cuda/turboquant-nc.cuh"
#include "../ggml/src/ggml-cuda/turboquant-nc-launch.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Only the absence of a usable CUDA driver/device is a skip. Unexpected
// runtime errors and failures after a device is detected remain test errors.
static constexpr int cuda_test_availability(cudaError_t status, int count) {
    return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ? 77 :
           status != cudaSuccess || count < 0 ? 1 : count == 0 ? 77 : 0;
}
static_assert(cuda_test_availability(cudaSuccess, 1) == 0 &&
              cuda_test_availability(cudaSuccess, 0) == 77 &&
              cuda_test_availability(cudaErrorNoDevice, 0) == 77 &&
              cuda_test_availability(cudaErrorInsufficientDriver, 0) == 77 &&
              cuda_test_availability(cudaErrorUnknown, 0) == 1 &&
              cuda_test_availability(cudaSuccess, -1) == 1, "CUDA test availability policy");

#define CUDA_CHECK_TEST(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err_)); \
        std::exit(2); \
    } \
} while (0)

__global__ static void encode(const float * keys, const float * values,
                              ggml_tqnc_key * out_k, ggml_tqnc_value * out_v, int * status) {
    const int row = blockIdx.x;
    const bool k = tqnc_cuda_encode_key(keys + row*GGML_TQNC_DIM, out_k + row);
    const bool v = tqnc_cuda_encode_value(values + row*GGML_TQNC_DIM, out_v + row);
    if (threadIdx.x == 0) {
        status[2*row] = k;
        status[2*row + 1] = v;
    }
}

template<typename T>
__global__ static void decode(const ggml_tqnc_key * keys, const ggml_tqnc_value * values,
                              T * out_k, T * out_v, int * status) {
    const int row = blockIdx.x;
    const bool k = tqnc_cuda_decode_key(keys + row, out_k + row*GGML_TQNC_DIM);
    const bool v = tqnc_cuda_decode_value(values + row, out_v + row*GGML_TQNC_DIM);
    if (threadIdx.x == 0) {
        status[2*row] = k;
        status[2*row + 1] = v;
    }
}

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

__global__ static void test_launch_write(int * dst) { if (threadIdx.x == 0) *dst = 42; }

static void test_launch_error_recovery() {
    int * dst = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&dst, sizeof(int)));
    CUDA_CHECK_TEST(cudaMemset(dst, 0, sizeof(int)));
    const cudaError_t invalid = ggml_cuda_tqnc_launch(test_launch_write, dim3(0), dim3(32), nullptr, dst);
    require(invalid != cudaSuccess, "invalid launch must return an error without aborting");
    int out = -1;
    CUDA_CHECK_TEST(cudaMemcpy(&out, dst, sizeof(out), cudaMemcpyDeviceToHost));
    require(out == 0, "failed launch must not write");
    CUDA_CHECK_TEST(ggml_cuda_tqnc_launch(test_launch_write, dim3(1), dim3(32), nullptr, dst));
    CUDA_CHECK_TEST(cudaMemcpy(&out, dst, sizeof(out), cudaMemcpyDeviceToHost));
    require(out == 42, "CUDA remains usable after rejected launch");
    CUDA_CHECK_TEST(cudaFree(dst));
}

static const float reference_centroids[16] = {
    -1.7068263888e-01f, -1.2927794456e-01f, -1.0111761093e-01f, -7.8516095877e-02f,
    -5.8903016150e-02f, -4.1054993868e-02f, -2.4258611724e-02f, -8.0267284065e-03f,
     8.0267284065e-03f,  2.4258611724e-02f,  4.1054993868e-02f,  5.8903016150e-02f,
     7.8516095877e-02f,  1.0111761093e-01f,  1.2927794456e-01f,  1.7068263888e-01f,
};

static unsigned parity(unsigned x) {
    x ^= x >> 16; x ^= x >> 8; x ^= x >> 4;
    return (0x6996u >> (x & 15u)) & 1u;
}

static double rotated_at(const float * row, int dim) {
    double norm2 = 0, sum = 0;
    for (int i = 0; i < GGML_TQNC_DIM; ++i) {
        norm2 += double(row[i])*row[i];
        sum += parity(unsigned(i & dim)) ? -double(row[i]) : double(row[i]);
    }
    return sum/(16.0*(std::sqrt(norm2) + 1e-8));
}

int main() {
    int devices = 0;
    const cudaError_t probe_status = cudaGetDeviceCount(&devices);
    const int availability = cuda_test_availability(probe_status, devices);
    if (availability != 0) {
        std::fprintf(stderr, "%s: CUDA device probe: %s (count=%d)\n",
                     availability == 77 ? "SKIP" : "FAIL", cudaGetErrorString(probe_status), devices);
        return availability;
    }
    test_launch_error_recovery();
    constexpr int random_rows = 64, boundary_rows = 15*5, n = random_rows + 2*boundary_rows, d = GGML_TQNC_DIM;
    std::mt19937 rng(61743);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> k(n*d), v(n*d);
    for (float & x : k) x = dist(rng);
    for (float & x : v) x = dist(rng);
    std::fill(k.begin(), k.begin() + d, 0.0f);
    std::fill(v.begin(), v.begin() + d, 1.25f);
    std::fill(k.begin() + d, k.begin() + 2*d, 0.0f);
    k[d] = 1.0f;
    v[2*d + 17] = 125.0f;
    std::normal_distribution<double> normal(0.0, 1.0);
    const double offsets[5] = {-1e-5, -1e-7, 0.0, 1e-7, 1e-5};
    for (int b = 0; b < 15; ++b) for (int j = 0; j < 5; ++j) {
        const int kr = random_rows + b*5 + j;
        const int vr = random_rows + boundary_rows + b*5 + j;
        double rotated[d];
        rotated[0] = (double(reference_centroids[b]) + reference_centroids[b + 1])/2 + offsets[j];
        double rest2 = 0;
        for (int i = 1; i < d; ++i) { rotated[i] = normal(rng); rest2 += rotated[i]*rotated[i]; }
        const double rest_scale = std::sqrt((1.0 - rotated[0]*rotated[0])/rest2);
        for (int i = 1; i < d; ++i) rotated[i] *= rest_scale;
        for (int i = 0; i < d; ++i) {
            double sum = 0;
            for (int x = 0; x < d; ++x) sum += parity(unsigned(i & x)) ? -rotated[x] : rotated[x];
            k[kr*d + i] = float(sum/16.0);
        }
        std::fill(v.begin() + vr*d, v.begin() + (vr + 1)*d, -1.0f);
        v[vr*d + 1] = 1.0f;
        v[vr*d + 2] = float(-1.0 + (b + 0.5)*2.0/15.0 + offsets[j]);
    }

    std::vector<ggml_tqnc_key> expected_k(n), gpu_k(n);
    std::vector<ggml_tqnc_value> expected_v(n), gpu_v(n);
    std::vector<float> reference_k(n*d), reference_v(n*d), gpu_k32(n*d), gpu_v32(n*d);
    for (int i = 0; i < n; ++i) {
        require(ggml_tqnc_quantize_key(k.data() + i*d, &expected_k[i]), "CPU key encode");
        require(ggml_tqnc_quantize_value(v.data() + i*d, &expected_v[i]), "CPU value encode");
        require(ggml_tqnc_dequantize_key(&expected_k[i], reference_k.data() + i*d), "CPU key decode");
        require(ggml_tqnc_dequantize_value(&expected_v[i], reference_v.data() + i*d), "CPU value decode");
    }

    float * dk = nullptr, * dv = nullptr, * dk32 = nullptr, * dv32 = nullptr;
    half * dk16 = nullptr, * dv16 = nullptr;
    ggml_tqnc_key * dpk = nullptr;
    ggml_tqnc_value * dpv = nullptr;
    int * dstatus = nullptr;
    CUDA_CHECK_TEST(cudaMalloc(&dk, k.size()*sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&dv, v.size()*sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&dk32, k.size()*sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&dv32, v.size()*sizeof(float)));
    CUDA_CHECK_TEST(cudaMalloc(&dk16, k.size()*sizeof(half)));
    CUDA_CHECK_TEST(cudaMalloc(&dv16, v.size()*sizeof(half)));
    CUDA_CHECK_TEST(cudaMalloc(&dpk, n*sizeof(ggml_tqnc_key)));
    CUDA_CHECK_TEST(cudaMalloc(&dpv, n*sizeof(ggml_tqnc_value)));
    CUDA_CHECK_TEST(cudaMalloc(&dstatus, 2*n*sizeof(int)));
    CUDA_CHECK_TEST(cudaMemcpy(dk, k.data(), k.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dv, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice));
    encode<<<n, 32>>>(dk, dv, dpk, dpv, dstatus);
    CUDA_CHECK_TEST(cudaGetLastError());
    std::vector<int> status(2*n);
    CUDA_CHECK_TEST(cudaMemcpy(status.data(), dstatus, status.size()*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(gpu_k.data(), dpk, n*sizeof(ggml_tqnc_key), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(gpu_v.data(), dpv, n*sizeof(ggml_tqnc_value), cudaMemcpyDeviceToHost));
    require(std::all_of(status.begin(), status.end(), [](int x) { return x == 1; }), "GPU encode status");
    size_t key_bytes = 0, value_bytes = 0, key_codes = 0, value_codes = 0, norm_differences = 0;
    size_t unexpected = 0;
    double max_k_boundary_distance = 0, max_v_boundary_distance = 0;
    float max_key_reconstruction_delta = 0, max_value_reconstruction_delta = 0;
    std::vector<float> decoded_cpu(d), decoded_gpu(d);
    for (int i = 0; i < n; ++i) {
        const auto * a = reinterpret_cast<const uint8_t *>(&gpu_k[i]);
        const auto * b = reinterpret_cast<const uint8_t *>(&expected_k[i]);
        for (size_t j = 0; j < sizeof(gpu_k[i]); ++j) key_bytes += a[j] != b[j];
        a = reinterpret_cast<const uint8_t *>(&gpu_v[i]);
        b = reinterpret_cast<const uint8_t *>(&expected_v[i]);
        for (size_t j = 0; j < sizeof(gpu_v[i]); ++j) value_bytes += a[j] != b[j];
        if (gpu_k[i].norm != expected_k[i].norm) {
            ++norm_differences;
            const float left = ggml_fp16_to_fp32(gpu_k[i].norm);
            const float right = ggml_fp16_to_fp32(expected_k[i].norm);
            if (std::abs(left - right) > 0.001f*std::max(1.0f, std::abs(right))) ++unexpected;
        }
        if (gpu_v[i].scale != expected_v[i].scale || gpu_v[i].minimum != expected_v[i].minimum) ++unexpected;
        const float * vin = v.data() + i*d;
        const float vmin = *std::min_element(vin, vin + d);
        const float vmax = *std::max_element(vin, vin + d);
        const float vscale = std::max((vmax - vmin)/15.0f, 1e-8f);
        for (int j = 0; j < d; ++j) {
            const int kc = (expected_k[i].qs[j/2] >> (4*(j % 2))) & 15;
            const int kg = (gpu_k[i].qs[j/2] >> (4*(j % 2))) & 15;
            if (kc != kg) {
                ++key_codes;
                if (std::abs(kc - kg) != 1) ++unexpected;
                const int low = std::min(kc, kg);
                const double boundary = (double(reference_centroids[low]) + reference_centroids[low + 1])/2;
                const double distance = std::abs(rotated_at(k.data() + i*d, j) - boundary);
                max_k_boundary_distance = std::max(max_k_boundary_distance, distance);
                if (distance > 2e-6) ++unexpected;
            }
            const int vc = (expected_v[i].qs[j/2] >> (4*(j % 2))) & 15;
            const int vg = (gpu_v[i].qs[j/2] >> (4*(j % 2))) & 15;
            if (vc != vg) {
                ++value_codes;
                if (std::abs(vc - vg) != 1) ++unexpected;
                const double ratio = (double(vin[j]) - vmin)/vscale;
                const double distance = std::abs(ratio - (std::min(vc, vg) + 0.5));
                max_v_boundary_distance = std::max(max_v_boundary_distance, distance);
                if (distance > 2e-5) ++unexpected;
            }
        }
        require(ggml_tqnc_dequantize_key(&expected_k[i], decoded_cpu.data()), "CPU decode reference key");
        require(ggml_tqnc_dequantize_key(&gpu_k[i], decoded_gpu.data()), "CPU decode GPU key");
        for (int j = 0; j < d; ++j) max_key_reconstruction_delta = std::max(max_key_reconstruction_delta, std::abs(decoded_cpu[j] - decoded_gpu[j]));
        require(ggml_tqnc_dequantize_value(&expected_v[i], decoded_cpu.data()), "CPU decode reference value");
        require(ggml_tqnc_dequantize_value(&gpu_v[i], decoded_gpu.data()), "CPU decode GPU value");
        for (int j = 0; j < d; ++j) max_value_reconstruction_delta = std::max(max_value_reconstruction_delta, std::abs(decoded_cpu[j] - decoded_gpu[j]));
    }
    std::printf("encode K/V byte differences=%zu/%zu, K/V code differences=%zu/%zu, norm differences=%zu\n",
                key_bytes, value_bytes, key_codes, value_codes, norm_differences);
    std::printf("max K/V distance to boundary=%g/%g, max decoded K/V delta=%g/%g, unexpected=%zu\n",
                max_k_boundary_distance, max_v_boundary_distance, max_key_reconstruction_delta, max_value_reconstruction_delta, unexpected);

    // Decode the identical CPU-produced bytes, separating decoder correctness
    // from possible encoding differences near centroid boundaries.
    CUDA_CHECK_TEST(cudaMemcpy(dpk, expected_k.data(), n*sizeof(ggml_tqnc_key), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dpv, expected_v.data(), n*sizeof(ggml_tqnc_value), cudaMemcpyHostToDevice));
    decode<float><<<n, 32>>>(dpk, dpv, dk32, dv32, dstatus);
    decode<half><<<n, 32>>>(dpk, dpv, dk16, dv16, dstatus);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaMemcpy(gpu_k32.data(), dk32, k.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(gpu_v32.data(), dv32, v.size()*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<half> gpu_k16(n*d), gpu_v16(n*d);
    CUDA_CHECK_TEST(cudaMemcpy(gpu_k16.data(), dk16, k.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(gpu_v16.data(), dv16, v.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(status.data(), dstatus, status.size()*sizeof(int), cudaMemcpyDeviceToHost));
    require(std::all_of(status.begin(), status.end(), [](int x) { return x == 1; }), "GPU decode status");
    float max_k_error = 0, max_v_error = 0;
    size_t half_mismatches = 0;
    for (size_t i = 0; i < k.size(); ++i) {
        max_k_error = std::max(max_k_error, std::abs(gpu_k32[i] - reference_k[i]));
        max_v_error = std::max(max_v_error, std::abs(gpu_v32[i] - reference_v[i]));
        half_mismatches += __half2float(gpu_k16[i]) != __half2float(__float2half_rn(gpu_k32[i]));
        half_mismatches += __half2float(gpu_v16[i]) != __half2float(__float2half_rn(gpu_v32[i]));
    }
    std::printf("decode max K=%g V=%g F16 mismatches=%zu\n", max_k_error, max_v_error, half_mismatches);
    require(max_k_error < 2e-5f && max_v_error < 2e-6f && half_mismatches == 0, "GPU/CPU decode");
    require(unexpected == 0, "GPU/CPU differences away from boundaries");

    std::vector<float> invalid(d, 1.0f), untouched(d, 123.0f), checked(d);
    for (float bad : {NAN, INFINITY, -INFINITY, 1e30f}) {
        invalid[17] = bad;
        CUDA_CHECK_TEST(cudaMemcpy(dk, invalid.data(), d*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_TEST(cudaMemcpy(dv, invalid.data(), d*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_TEST(cudaMemset(dpk, 0xa5, sizeof(ggml_tqnc_key)));
        CUDA_CHECK_TEST(cudaMemset(dpv, 0xa5, sizeof(ggml_tqnc_value)));
        encode<<<1, 32>>>(dk, dv, dpk, dpv, dstatus);
        CUDA_CHECK_TEST(cudaGetLastError());
        CUDA_CHECK_TEST(cudaMemcpy(status.data(), dstatus, 2*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_TEST(cudaMemcpy(gpu_k.data(), dpk, sizeof(ggml_tqnc_key), cudaMemcpyDeviceToHost));
        CUDA_CHECK_TEST(cudaMemcpy(gpu_v.data(), dpv, sizeof(ggml_tqnc_value), cudaMemcpyDeviceToHost));
        require(status[0] == 0 && status[1] == 0, "GPU rejects invalid input");
        const auto * kbytes = reinterpret_cast<const uint8_t *>(&gpu_k[0]);
        const auto * vbytes = reinterpret_cast<const uint8_t *>(&gpu_v[0]);
        require(std::all_of(kbytes, kbytes + sizeof(gpu_k[0]), [](uint8_t b) { return b == 0xa5; }),
                "invalid key leaves destination untouched");
        require(std::all_of(vbytes, vbytes + sizeof(gpu_v[0]), [](uint8_t b) { return b == 0xa5; }),
                "invalid value leaves destination untouched");
    }
    std::fill(invalid.begin(), invalid.end(), -65504.0f);
    invalid[0] = 65504.0f;
    CUDA_CHECK_TEST(cudaMemcpy(dk, invalid.data(), d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dv, invalid.data(), d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(dpk, 0xa5, sizeof(ggml_tqnc_key)));
    CUDA_CHECK_TEST(cudaMemset(dpv, 0xa5, sizeof(ggml_tqnc_value)));
    encode<<<1, 32>>>(dk, dv, dpk, dpv, dstatus);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaMemcpy(status.data(), dstatus, 2*sizeof(int), cudaMemcpyDeviceToHost));
    require(status[0] == 0 && status[1] == 0, "GPU rejects FP16 overflow");
    expected_k[0].norm = 0x7c00; // +inf
    expected_v[0].scale = 0xbc00; // -1
    CUDA_CHECK_TEST(cudaMemcpy(dpk, expected_k.data(), sizeof(ggml_tqnc_key), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dpv, expected_v.data(), sizeof(ggml_tqnc_value), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dk32, untouched.data(), d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(dv32, untouched.data(), d*sizeof(float), cudaMemcpyHostToDevice));
    decode<float><<<1, 32>>>(dpk, dpv, dk32, dv32, dstatus);
    CUDA_CHECK_TEST(cudaGetLastError());
    CUDA_CHECK_TEST(cudaMemcpy(status.data(), dstatus, 2*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_TEST(cudaMemcpy(checked.data(), dk32, d*sizeof(float), cudaMemcpyDeviceToHost));
    require(status[0] == 0 && status[1] == 0 && checked == untouched, "GPU rejects invalid metadata");
    CUDA_CHECK_TEST(cudaMemcpy(checked.data(), dv32, d*sizeof(float), cudaMemcpyDeviceToHost));
    require(checked == untouched, "invalid value metadata leaves output untouched");

    CUDA_CHECK_TEST(cudaFree(dk)); CUDA_CHECK_TEST(cudaFree(dv));
    CUDA_CHECK_TEST(cudaFree(dk32)); CUDA_CHECK_TEST(cudaFree(dv32));
    CUDA_CHECK_TEST(cudaFree(dk16)); CUDA_CHECK_TEST(cudaFree(dv16));
    CUDA_CHECK_TEST(cudaFree(dpk)); CUDA_CHECK_TEST(cudaFree(dpv)); CUDA_CHECK_TEST(cudaFree(dstatus));
    std::puts("TurboQuant NC CUDA codec checks passed");
    return 0;
}
