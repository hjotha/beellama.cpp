// Included by the existing POSIX loopback harness. Reply values are synthetic:
// this tests real host codec/transport, not Xbox attention numerics.
static void test_low4_negotiation() {
    storage_env upload(nullptr, "GGML_RKVA_UPLOAD_FORMAT"), legacy(nullptr, "GGML_RKVA_FORMAT");
    storage_env wire("fp8", "GGML_RKVA_WIRE_FORMAT"), kernel("reference", "GGML_RKVA_Q4_KERNEL");
    storage_env profile("0", "GGML_RKVA_PROFILE");
    for (auto type : {GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) for (uint32_t failure = 0; failure < 5; ++failure) {
        const uint32_t storage = type == GGML_TYPE_Q4_1 ? RKVA_RANGE_STORAGE_Q4_1 : RKVA_RANGE_STORAGE_IQ4_NL;
        storage_env format(ggml_type_name(type)), experimental(failure == 1 ? "0" : "1", "GGML_RKVA_EXPERIMENTAL_LOW4");
        storage_env wrong_wire(failure == 4 ? "f16" : "fp8", "GGML_RKVA_WIRE_FORMAT");
        loopback server(false, true, false, false, false, false, 0, 0, true, 8 | 32,
                        failure == 0 ? 0 : storage);
        auto* backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
        ggml_remote_attn_geometry geo{1, failure == 3 ? 2u : 6u, 1, 256, 512, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.f/16};
        assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
        assert(ggml_backend_remote_attn_connect(backend));
        assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, 512));
        ggml_backend_remote_attn_set_active_session(backend, 0);
        assert(!ggml_backend_remote_attn_configure_range(backend, 256, 512,
                failure == 2 ? GGML_TYPE_Q4_0 : type));
        {
            std::lock_guard<std::mutex> lock(server.mutex);
            assert(server.native_descriptor.layout_version == 0);
            assert(server.range.capacity == 0); // no F16/Q4_0 fallback
        }
        ggml_backend_free(backend);
    }
}

static void test_low4_loopback(ggml_type type, bool async, uint32_t nq, bool e4 = false) {
    storage_env upload(nullptr, "GGML_RKVA_UPLOAD_FORMAT"), legacy(nullptr, "GGML_RKVA_FORMAT");
    storage_env format(ggml_type_name(type)), experimental("1", "GGML_RKVA_EXPERIMENTAL_LOW4");
    storage_env wire(e4 ? "fp8_e4m3fn" : "fp8", "GGML_RKVA_WIRE_FORMAT"), bulk("auto", "GGML_RKVA_NATIVE_BULK");
    storage_env overlap(async ? "1" : "0", "GGML_RKVA_ASYNC_LOCAL_REMOTE"), cuda("0", "GGML_RKVA_CUDA_FP8");
    storage_env profile("0", "GGML_RKVA_PROFILE"), kernel("reference", "GGML_RKVA_Q4_KERNEL");
    storage_env kernel_alias(nullptr, "GGML_RKVA_RANGE_KERNEL"), prefetch("1", "GGML_RKVA_TCP_PREFETCH");
    constexpr uint32_t origin = 256, capacity = 512, heads = 6, dim = 256;
    const uint32_t storage = type == GGML_TYPE_Q4_1 ? RKVA_RANGE_STORAGE_Q4_1 : RKVA_RANGE_STORAGE_IQ4_NL;
    loopback server(false, true, false, false, false, false, 0, 0, true, 8 | 32 | (e4 ? 16384 : 0), storage);
    server.fp8_output_byte = e4 ? 0x38 : 0x3c;
    auto* backend = ggml_backend_remote_attn_init("127.0.0.1", server.port);
    ggml_remote_attn_geometry geo{1, heads, 1, dim, capacity, 0, 0, 128, 3, 128, 2, 0, 0, false, false, 1.f/16};
    assert(ggml_backend_remote_attn_set_geometry(backend, &geo));
    assert(ggml_backend_remote_attn_connect(backend));
    assert(ggml_backend_remote_attn_create_range_session(backend, 0, 0, capacity));
    ggml_backend_remote_attn_set_active_session(backend, 0);
    assert(ggml_backend_remote_attn_configure_range(backend, origin, capacity, type));
    assert(ggml_backend_remote_attn_native_kvarn_enabled(backend));
    assert(!ggml_backend_remote_attn_native_kvarn_records_enabled(backend));
    assert(!ggml_backend_remote_attn_native_kvarn_snapshot_supported(backend));
    assert(ggml_backend_remote_attn_native_kvarn_can_rewind(backend));
    ggml_remote_attn_set_active(backend);
    auto* ctx = ggml_init({2u << 20, nullptr, true});
    auto* k = ggml_new_tensor_2d(ctx, type, dim, capacity);
    auto* v = ggml_dup_tensor(ctx, k);
    struct call { ggml_tensor *q, *ids, *pos, *ok, *ov, *node, *wait; uint32_t count; };
    auto make = [&](uint32_t count) {
        call c{}; c.count = count;
        c.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, count);
        c.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, capacity);
        c.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
        c.ok = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, count); c.ov = ggml_dup_tensor(ctx, c.ok);
        c.node = ggml_remote_attn_range(ctx, c.q, k, v, c.ids, c.pos, 0, origin, nq, geo.kq_scale);
        ggml_remote_attn_range_set_native_sources(c.node, c.ok, c.ov);
        if (async) ggml_remote_attn_range_set_async(c.node);
        c.wait = async ? ggml_remote_attn_range_wait(ctx, c.node) : c.node;
        return c;
    };
    auto first = make(nq), suffix = make(1);
    auto* cpu = ggml_backend_cpu_init(); auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buffer); ggml_backend_buffer_clear(buffer, 0);
    auto run = [&](const call& c, uint32_t first_row) {
        std::vector<float> query(size_t(c.count)*heads*dim, 1.f), key(size_t(c.count)*dim), value(key.size());
        for (size_t i = 0; i < key.size(); ++i) { key[i] = 2.f + std::sin(float(i)*.031f); value[i] = std::cos(float(i)*.047f) - 2.75f; }
        std::vector<int32_t> cells(capacity, -1), pos(c.count);
        for (uint32_t i = 0; i < nq; ++i) cells[i] = origin + i;
        for (uint32_t i = 0; i < c.count; ++i) pos[i] = origin + first_row + i;
        ggml_backend_tensor_set(c.q, query.data(), 0, query.size()*4);
        ggml_backend_tensor_set(c.ids, cells.data(), 0, cells.size()*4);
        ggml_backend_tensor_set(c.pos, pos.data(), 0, pos.size()*4);
        ggml_backend_tensor_set(c.ok, key.data(), 0, key.size()*4);
        ggml_backend_tensor_set(c.ov, value.data(), 0, value.size()*4);
        size_t prior;
        { std::lock_guard<std::mutex> lock(server.mutex); prior = server.low4_upload_headers.size(); }
        assert(ggml_remote_attn_exec(c.node));
        if (async) assert(ggml_remote_attn_exec(c.wait));
        const auto* out = static_cast<const float*>(c.wait->data);
        for (uint32_t t = 0; t < c.count; ++t) {
            for (uint32_t d = 0; d < heads*dim; ++d) assert(out[t*(heads*dim+heads)+d] == 1.f);
            for (uint32_t h = 0; h < heads; ++h)
                assert(std::fabs(out[t*(heads*dim+heads)+heads*dim+h] - std::log(float(first_row+t+1))) < 1e-6f);
        }
        const size_t row_bytes = ggml_row_size(type, dim);
        std::vector<uint8_t> expected_k(c.count*row_bytes), expected_v(expected_k.size());
        assert(ggml_quantize_chunk(type, key.data(), expected_k.data(), 0, c.count, dim, nullptr) == expected_k.size());
        assert(ggml_quantize_chunk(type, value.data(), expected_v.data(), 0, c.count, dim, nullptr) == expected_v.size());
        std::lock_guard<std::mutex> lock(server.mutex);
        uint32_t offset = 0;
        for (size_t i = prior; i < server.low4_upload_headers.size(); ++i) {
            const auto& u = server.low4_upload_headers[i]; const auto& packed = server.low4_upload_planes[i];
            assert(u.append_begin == first_row + offset);
            assert(u.flags == (u.append_begin == 0 ? RKVA_KVARN_UPLOAD_RESET : 0));
            size_t plane = u.append_rows * row_bytes;
            assert(packed.size() == plane*2);
            assert(std::memcmp(packed.data(), expected_k.data()+offset*row_bytes, plane) == 0);
            assert(std::memcmp(packed.data()+plane, expected_v.data()+offset*row_bytes, plane) == 0);
            offset += u.append_rows;
        }
        assert(offset == c.count);
        assert(server.fp8_query_positions == pos);
        assert(server.fp8_query_data.size() == query.size());
        assert(std::all_of(server.fp8_query_data.begin(), server.fp8_query_data.end(), [e4](uint8_t b) { return b == (e4 ? 0x38 : 0x3c); }));
    };
    run(first, 0);
    assert(ggml_backend_remote_attn_trim(backend, 0, origin+nq-1));
    run(suffix, nq-1);
    const std::string stats = ggml_backend_remote_attn_stats_json(backend);
    assert(stats.find(ggml_type_name(type)) != std::string::npos);
    if (async) assert(stats.find(nq >= 10 ? "\"tcp_prefetch_completed\":1" :
                                            "\"tcp_prefetch_completed\":0") != std::string::npos);
    ggml_remote_attn_set_active(nullptr);
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(cpu); ggml_backend_free(backend);
}
