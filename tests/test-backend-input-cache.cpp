// Hardware-free scheduler residency tests. The non-host backend below executes
// ordinary CPU kernels in malloc-backed "device" storage; no GPU is loaded.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

// Keep the registry object linked for remote-backend dependencies in static
// builds, without calling it or initializing any registered device.
static auto volatile registry_link = &ggml_backend_reg_by_name;
static uint8_t * pending_source_write = nullptr;
static uint8_t pending_source_value = 0;
static size_t source_synchronizes = 0;

struct fake_device {
    ggml_backend_device device = {};
    ggml_backend_buffer_type buft = {};
    ggml_backend backend = {};
    ggml_backend_t cpu;
    size_t copies = 0, synchronizes = 0, event_syncs = 0, async_copies = 0;
    bool fail_allocate = false, fail_compute = false;

    explicit fake_device(ggml_backend_t cpu) : cpu(cpu) {
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "CACHE_TEST_DEVICE"; };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_memory = [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 1u << 28; };
        device.iface.get_props = [](ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
            *props = {};
            props->name = dev->iface.get_name(dev);
            props->description = props->name;
            props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
            props->caps.async = true;
            props->caps.events = true;
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) { return &static_cast<fake_device *>(dev->context)->buft; };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            return type == &static_cast<fake_device *>(dev->context)->buft;
        };
        device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * t) {
            return t->op == GGML_OP_NONE || t->op == GGML_OP_SCALE || t->op == GGML_OP_ADD ||
                t->op == GGML_OP_DUP || t->op == GGML_OP_CONT || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                t->op == GGML_OP_TRANSPOSE;
        };
        device.iface.offload_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
        device.iface.event_new = [](ggml_backend_dev_t dev) { return new ggml_backend_event { dev, nullptr }; };
        device.iface.event_free = [](ggml_backend_dev_t, ggml_backend_event_t event) { delete event; };
        device.iface.event_synchronize = [](ggml_backend_dev_t dev, ggml_backend_event_t) {
            ++static_cast<fake_device *>(dev->context)->event_syncs;
        };
        buft.device = &device;
        buft.context = this;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "CACHE_TEST_STORAGE"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 64; };
        buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t bytes) -> ggml_backend_buffer_t {
            auto * fake = static_cast<fake_device *>(type->context);
            if (fake->fail_allocate) return nullptr;
            void * raw = std::malloc(bytes + 64);
            if (!raw) return nullptr;
            ggml_backend_buffer_i iface = {};
            iface.free_buffer = [](ggml_backend_buffer_t buffer) { std::free(buffer->context); };
            iface.get_base = [](ggml_backend_buffer_t buffer) -> void * {
                return reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(buffer->context) + 63) & ~uintptr_t(63));
            };
            iface.set_tensor = [](ggml_backend_buffer_t buffer, ggml_tensor * t, const void * src, size_t offset, size_t bytes) {
                ++static_cast<fake_device *>(buffer->buft->context)->copies;
                std::memcpy(static_cast<uint8_t *>(t->data) + offset, src, bytes);
            };
            iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * t, void * dst, size_t offset, size_t bytes) {
                std::memcpy(dst, static_cast<const uint8_t *>(t->data) + offset, bytes);
            };
            iface.clear = [](ggml_backend_buffer_t buffer, uint8_t value) {
                std::memset(buffer->iface.get_base(buffer), value, buffer->size);
            };
            return ggml_backend_buffer_init(type, iface, raw, bytes);
        };
        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "CACHE_TEST_DEVICE"; };
        backend.iface.free = [](ggml_backend_t) {};
        backend.iface.synchronize = [](ggml_backend_t b) { ++static_cast<fake_device *>(b->context)->synchronizes; };
        backend.iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * graph) {
            auto * fake = static_cast<fake_device *>(b->context);
            if (fake->fail_compute) return GGML_STATUS_ABORTED;
            return ggml_backend_graph_compute(fake->cpu, graph);
        };
        backend.iface.cpy_tensor_async = [](ggml_backend_t, ggml_backend_t dst, const ggml_tensor * src, ggml_tensor * t) {
            ++static_cast<fake_device *>(dst->context)->async_copies;
            std::memcpy(t->data, src->data, ggml_nbytes(src));
            return true;
        };
        backend.iface.event_record = [](ggml_backend_t, ggml_backend_event_t) {};
        backend.iface.event_wait = [](ggml_backend_t, ggml_backend_event_t) {};
    }
};

struct test_graph {
    ggml_context * ctx;
    ggml_backend_buffer_t host;
    ggml_tensor * root;
    ggml_tensor * input;
    ggml_tensor * second = nullptr;
    ggml_tensor * result;
    ggml_tensor * consumer;
    ggml_cgraph * graph;

    test_graph(ggml_backend_t cpu, void * arena, size_t arena_bytes, int64_t n = 128,
            bool views = false, bool mark = true, bool noncontiguous = false, int scratch_steps = 12) {
        ctx = ggml_init({arena_bytes, arena, true});
        CHECK(ctx);
        root = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, views || noncontiguous ? 2 : 1);
        ggml_set_input(root);
        host = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        CHECK(host);
        auto * data = static_cast<float *>(root->data);
        for (int64_t i = 0; i < ggml_nelements(root); ++i) data[i] = float(i % 31 + 1);
        input = views ? ggml_view_1d(ctx, root, n, 0) : root;
        if (views) second = ggml_view_1d(ctx, root, n, n*sizeof(float));
        if (noncontiguous) input = ggml_transpose(ctx, root);
        if (mark) {
            ggml_set_input_cacheable(input);
            if (second) ggml_set_input_cacheable(second);
        }
        // Scratch before/after the consumer must never alias its retained copy.
        auto * scratch = ggml_scale(ctx, root, 3.0f);
        scratch = ggml_scale(ctx, scratch, 5.0f);
        consumer = noncontiguous ? ggml_cont(ctx, input) : ggml_scale(ctx, input, 2.0f);
        result = second ? ggml_add(ctx, consumer, ggml_scale(ctx, second, 4.0f)) : consumer;
        graph = ggml_new_graph_custom(ctx, 2048, false);
        ggml_build_forward_expand(graph, scratch);
        ggml_build_forward_expand(graph, result);
        scratch = ggml_scale(ctx, result, 7.0f);
        for (int i = 0; i < scratch_steps; ++i) scratch = ggml_scale(ctx, scratch, 0.5f);
        ggml_build_forward_expand(graph, scratch);
        ggml_set_output(result);
    }
    ~test_graph() { ggml_backend_buffer_free(host); ggml_free(ctx); }

    void check_values() const {
        std::vector<float> values(ggml_nelements(result));
        ggml_backend_tensor_get(result, values.data(), 0, values.size()*sizeof(float));
        const auto * data = static_cast<const float *>(root->data);
        for (size_t i = 0; i < values.size(); ++i) {
            const float expected = second ? 2.0f*data[i] + 4.0f*data[i + values.size()] : 2.0f*data[i];
            CHECK(values[i] == expected);
        }
    }
};

struct fixture {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    fake_device fake {cpu};
    ggml_backend_sched_t sched;
    bool allocated = false;
    std::vector<uint8_t> arena = std::vector<uint8_t>(2u << 20);

    explicit fixture(bool parallel = false) {
        CHECK(cpu);
        // CPU work is synchronous; emulate a pending host producer to prove
        // that cache comparison cannot move ahead of its source fence.
        cpu->iface.synchronize = [](ggml_backend_t) {
            ++source_synchronizes;
            if (pending_source_write) {
                *pending_source_write = pending_source_value;
                pending_source_write = nullptr;
            }
        };
        ggml_backend_t backends[] = {&fake.backend, cpu};
        sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, parallel, true);
        CHECK(sched);
    }
    ~fixture() { ggml_backend_sched_free(sched); ggml_backend_free(cpu); }
    void place(test_graph & g) {
        allocated = false;
        for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
            auto * node = ggml_graph_node(g.graph, i);
            if (node->op != GGML_OP_VIEW && node->op != GGML_OP_TRANSPOSE && node->op != GGML_OP_RESHAPE) {
                ggml_backend_sched_set_tensor_backend(sched, node, &fake.backend);
            }
        }
    }
    void run(test_graph & g) {
        if (!allocated) { CHECK(ggml_backend_sched_alloc_graph(sched, g.graph)); allocated = true; }
        CHECK(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS);
        g.check_values();
    }
    ggml_backend_sched_input_cache_stats stats() const {
        ggml_backend_sched_input_cache_stats stats;
        ggml_backend_sched_get_input_cache_stats(sched, &stats);
        return stats;
    }
};

static void test_bytes_and_lifetime(bool views) {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, views);
    f.place(g);
    // Tensor intent survives a reset before explicit allocation.
    ggml_backend_sched_set_tensor_input_cacheable(f.sched, g.input);
    ggml_backend_sched_reset(f.sched);
    f.place(g);
    f.run(g);
    const auto first = f.stats();
    CHECK(first.misses == (views ? 2u : 1u) && first.hits == 0);
    auto * retained = g.consumer->src[0];
    CHECK((retained->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) ==
            (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT));
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
        auto * node = ggml_graph_node(g.graph, i);
        if (node->view_src || node == retained || !node->data || node->buffer == g.host) continue;
        const uintptr_t a = reinterpret_cast<uintptr_t>(node->data), b = reinterpret_cast<uintptr_t>(retained->data);
        CHECK(a + ggml_nbytes(node) <= b || b + ggml_nbytes(retained) <= a);
    }
    const size_t syncs = f.fake.synchronizes;
    f.run(g);
    CHECK(f.stats().hits == (views ? 2u : 1u));
    CHECK(f.fake.synchronizes > syncs); // a hit does not skip the ownership fence
    CHECK(f.fake.async_copies == 0); // marked inputs never use unsafe async source ownership
    const auto original = static_cast<uint8_t *>(g.root->data)[0];
    static_cast<uint8_t *>(g.root->data)[0] ^= 1; // change exactly one byte
    f.run(g);
    CHECK(f.stats().misses == first.misses + 1);
    static_cast<uint8_t *>(g.root->data)[0] = original; // A -> B -> A still needs a copy
    f.run(g);
    CHECK(f.stats().misses == first.misses + 2);
    f.run(g);
    CHECK(f.stats().skipped_bytes > 0 && f.stats().copied_bytes > 0);
    const auto before_source_write = f.stats();
    const size_t source_syncs = source_synchronizes;
    pending_source_write = static_cast<uint8_t *>(g.root->data);
    pending_source_value = original ^ 1;
    f.run(g);
    CHECK(!pending_source_write && source_synchronizes > source_syncs);
    CHECK(f.stats().misses == before_source_write.misses + 1);
    const auto old = f.stats();
    ggml_backend_sched_reset_input_cache_stats(f.sched);
    CHECK(f.stats().hits == 0 && f.stats().copied_bytes == 0);
    f.run(g);
    CHECK(f.stats().hits == (views ? 2u : 1u) && f.stats().misses == 0);
    CHECK(old.misses > 0);
    f.fake.fail_compute = true;
    CHECK(ggml_backend_sched_graph_compute(f.sched, g.graph) == GGML_STATUS_ABORTED);
    f.fake.fail_compute = false;
    const auto failed = f.stats();
    f.run(g);
    CHECK(f.stats().misses == failed.misses + (views ? 2u : 1u));
}

static void test_unmarked_write_invalidates(bool views, bool callback) {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, views);
    size_t callback_copies = 0;
    if (callback) {
        CHECK(views);
        ggml_backend_buffer_set_usage(g.host, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_sched_set_copy_callback(f.sched,
                [](ggml_backend_t, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph *, void * user) {
                    ++*static_cast<size_t *>(user);
                    ggml_backend_tensor_copy(src, dst);
                    return true;
                }, &callback_copies);
    }
    f.place(g);
    f.run(g); // cached A
    const auto cached = f.stats();
    const float original = static_cast<float *>(g.root->data)[0];
    g.input->flags &= ~GGML_TENSOR_FLAG_INPUT_CACHEABLE;
    if (callback) ggml_backend_buffer_set_usage(g.host, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    static_cast<float *>(g.root->data)[0] = 61.0f;
    f.run(g); // ordinary fallback writes B into the same retained destination
    CHECK(f.stats().misses == cached.misses);
    ggml_set_input_cacheable(g.input);
    if (callback) ggml_backend_buffer_set_usage(g.host, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    static_cast<float *>(g.root->data)[0] = original;
    f.run(g); // A must be copied again, never hit snapshot A over destination B
    CHECK(f.stats().misses == cached.misses + (callback ? 2u : 1u));
    if (callback) CHECK(callback_copies > 0);
}

static void test_weight_callback_state(bool weights_from_allocation) {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, true);
    struct transform_state { size_t calls = 0; float bias = 5.0f; } state;
    ggml_backend_buffer_set_usage(g.host, weights_from_allocation ?
            GGML_BACKEND_BUFFER_USAGE_WEIGHTS : GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_sched_set_copy_callback(f.sched,
            [](ggml_backend_t, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph *, void * user) {
                auto & state = *static_cast<transform_state *>(user);
                ++state.calls;
                std::vector<float> transformed(ggml_nelements(src));
                const auto * source = static_cast<const float *>(src->data);
                for (size_t i = 0; i < transformed.size(); ++i) transformed[i] = source[i] + state.bias;
                ggml_backend_tensor_set(dst, transformed.data(), 0, transformed.size()*sizeof(float));
                return true;
            }, &state);
    f.place(g);
    const float original = static_cast<float *>(g.root->data)[0];
    if (!weights_from_allocation) {
        f.run(g); // establish cache A while source usage is COMPUTE
        CHECK(f.stats().misses == 2 && state.calls == 0);
    } else {
        CHECK(ggml_backend_sched_alloc_graph(f.sched, g.graph));
    }
    const auto cached = f.stats();
    ggml_backend_buffer_set_usage(g.host, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    static_cast<float *>(g.root->data)[0] = 61.0f;
    auto run_transformed = [&] {
        CHECK(ggml_backend_sched_graph_compute(f.sched, g.graph) == GGML_STATUS_SUCCESS);
        std::vector<float> values(ggml_nelements(g.result));
        ggml_backend_tensor_get(g.result, values.data(), 0, values.size()*sizeof(float));
        const auto * source = static_cast<const float *>(g.root->data);
        for (size_t i = 0; i < values.size(); ++i) {
            CHECK(values[i] == 2.0f*(source[i] + state.bias) +
                    4.0f*(source[i + values.size()] + state.bias));
        }
    };
    run_transformed();
    CHECK(state.calls == 2);
    // Identical source bytes, different callback userdata: the transformed
    // destination must change even though the cacheable flag stays set.
    state.bias = 7.0f;
    run_transformed();
    CHECK(state.calls == 4);
    CHECK(f.stats().hits == cached.hits && f.stats().misses == cached.misses);
    if (!weights_from_allocation) {
        ggml_backend_buffer_set_usage(g.host, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        static_cast<float *>(g.root->data)[0] = original;
        f.run(g);
        CHECK(f.stats().misses == cached.misses + 2); // transformed B invalidated BOTH views
        f.run(g);
        CHECK(f.stats().hits == cached.hits + 2);
    }
}

static void test_destination_identity() {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size());
    f.place(g);
    f.run(g);
    auto * retained = g.consumer->src[0];
    const auto original_data = retained->data;
    const auto original_buffer = retained->buffer;
    auto alternate = ggml_backend_alloc_buffer(&f.fake.backend, ggml_nbytes(retained));
    CHECK(alternate);
    retained->buffer = alternate;
    retained->data = ggml_backend_buffer_get_base(alternate);
    const auto before = f.stats();
    f.run(g);
    CHECK(f.stats().misses == before.misses + 1);
    retained->buffer = original_buffer;
    retained->data = original_data;
    f.run(g);
    CHECK(f.stats().misses == before.misses + 2);
    ggml_backend_buffer_free(alternate);
}

static void test_reset_replan_slots() {
    fixture f(true);
    ggml_tensor * reused_header = nullptr;
    const int slots = ggml_backend_sched_get_n_copies(f.sched);
    CHECK(slots > 1);
    for (int slot = 0; slot < slots*2; ++slot) {
        ggml_backend_sched_reset(f.sched);
        // Reuse the exact metadata arena: a new tensor must not inherit an
        // old pointer registration or a previous allocation's residency.
        // Change the allocation plan as well as the copy slot. Inactive
        // parallel input leaves must be reserved afresh for that slot.
        test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, false, true, false, 12 + slot);
        if (reused_header) CHECK(g.root == reused_header);
        reused_header = g.root;
        f.place(g);
        CHECK(ggml_backend_sched_alloc_graph(f.sched, g.graph));
        f.allocated = true;
        const auto before = f.stats();
        f.run(g);
        CHECK(f.stats().misses == before.misses + 1);
        auto * retained = g.consumer->src[0];
        const char * suffix = std::strrchr(retained->name, '#');
        CHECK(suffix && std::atoi(suffix + 1) == slot % slots);
        const size_t syncs = f.fake.event_syncs;
        f.run(g);
        CHECK(f.stats().hits == before.hits + 1 && f.fake.event_syncs > syncs);
    }
    ggml_backend_sched_reset(f.sched);
    {
        test_graph unmarked(f.cpu, f.arena.data(), f.arena.size(), 128, false, false);
        CHECK(!(unmarked.root->flags & GGML_TENSOR_FLAG_INPUT_CACHEABLE));
        f.place(unmarked);
        const auto before = f.stats();
        f.run(unmarked);
        f.run(unmarked);
        CHECK(f.stats().hits == before.hits && f.stats().misses == before.misses);
    }
}

static void test_failed_allocation_recovery() {
    fixture f;
    {
        test_graph original(f.cpu, f.arena.data(), f.arena.size());
        f.place(original);
        f.run(original);
        f.run(original);
        CHECK(f.stats().hits == 1 && f.stats().misses == 1);
        ggml_backend_sched_reset(f.sched);
    }
    {
        test_graph huge(f.cpu, f.arena.data(), f.arena.size(), 65536);
        f.place(huge);
        f.fake.fail_allocate = true;
        CHECK(!ggml_backend_sched_alloc_graph(f.sched, huge.graph));
        f.fake.fail_allocate = false;
        ggml_backend_sched_reset(f.sched);
    }
    // A failed gallocr reserve needs a fresh reservation before reuse. Use a
    // disposable measure graph, as production sched_reserve() does.
    {
        test_graph measure(f.cpu, f.arena.data(), f.arena.size(), 128, false, true, false, 13);
        f.place(measure);
        CHECK(ggml_backend_sched_reserve(f.sched, measure.graph));
    }
    {
        test_graph restored(f.cpu, f.arena.data(), f.arena.size(), 128, false, true, false, 13);
        f.place(restored);
        const auto before = f.stats();
        f.run(restored);
        CHECK(f.stats().misses == before.misses + 1);
        f.run(restored);
        CHECK(f.stats().hits == before.hits + 1);
    }
}

static void test_computed_input_is_not_cacheable() {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, false, false);
    f.place(g);
    // Explicitly marking an ordinary computed CPU intermediate must not turn
    // it into an immutable input. Graph writers to INPUT roots or aliases are
    // separately forbidden by the public opt-in contract.
    ggml_backend_sched_set_tensor_backend(f.sched, g.consumer, f.cpu);
    ggml_set_input_cacheable(g.consumer);
    auto * destination = ggml_scale(g.ctx, g.consumer, 3.0f);
    ggml_backend_sched_set_tensor_backend(f.sched, destination, &f.fake.backend);
    ggml_build_forward_expand(g.graph, destination);
    CHECK(ggml_backend_sched_alloc_graph(f.sched, g.graph));
    CHECK(ggml_backend_sched_graph_compute(f.sched, g.graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_sched_graph_compute(f.sched, g.graph) == GGML_STATUS_SUCCESS);
    CHECK(f.stats().hits == 0 && f.stats().misses == 0);
}

static void test_late_mark_falls_back() {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, false, false);
    f.place(g);
    f.run(g);
    ggml_backend_sched_set_tensor_input_cacheable(f.sched, g.root);
    // Its already planned single-copy destination was not pinned. Marking
    // intent late must never admit that storage into persistent residency.
    f.run(g);
    f.run(g);
    CHECK(f.stats().hits == 0 && f.stats().misses == 0);
}

static void test_entry_budget_fallback() {
    fixture f;
    auto * ctx = ggml_init({f.arena.size(), f.arena.data(), true});
    CHECK(ctx);
    std::vector<ggml_tensor *> inputs;
    for (int i = 0; i < 129; ++i) {
        auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(input);
        ggml_set_input_cacheable(input);
        inputs.push_back(input);
    }
    auto host = ggml_backend_alloc_ctx_tensors(ctx, f.cpu);
    CHECK(host);
    auto * graph = ggml_new_graph_custom(ctx, 2048, false);
    for (auto * input : inputs) {
        *static_cast<float *>(input->data) = 3.0f;
        auto * result = ggml_scale(ctx, input, 2.0f);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        ggml_backend_sched_set_tensor_backend(f.sched, result, &f.fake.backend);
    }
    CHECK(ggml_backend_sched_alloc_graph(f.sched, graph));
    CHECK(ggml_backend_sched_graph_compute(f.sched, graph) == GGML_STATUS_SUCCESS);
    CHECK(f.stats().misses == 129 && f.stats().hits == 0);
    CHECK(ggml_backend_sched_graph_compute(f.sched, graph) == GGML_STATUS_SUCCESS);
    // The 129th entry exceeds the bounded sidecar count and keeps copying.
    CHECK(f.stats().misses == 130 && f.stats().hits == 128);
    CHECK(f.stats().copied_bytes == 130*sizeof(float));
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        float result;
        ggml_backend_tensor_get(ggml_graph_node(graph, i), &result, 0, sizeof(result));
        CHECK(result == 6.0f);
    }
    ggml_backend_sched_reset(f.sched);
    ggml_backend_buffer_free(host);
    ggml_free(ctx);
}

static void test_noncontiguous_fallback() {
    fixture f;
    test_graph g(f.cpu, f.arena.data(), f.arena.size(), 128, false, true, true);
    f.place(g);
    CHECK(ggml_backend_sched_alloc_graph(f.sched, g.graph));
    CHECK(ggml_backend_sched_graph_compute(f.sched, g.graph) == GGML_STATUS_SUCCESS);
    CHECK(f.stats().hits == 0 && f.stats().misses == 0);
    std::vector<float> values(ggml_nelements(g.result));
    ggml_backend_tensor_get(g.result, values.data(), 0, values.size()*sizeof(float));
    const auto * data = static_cast<const float *>(g.root->data);
    for (size_t i = 0; i < values.size(); ++i) CHECK(values[i] == data[i/2 + (i%2)*128]);
}

int main() {
    CHECK(registry_link != nullptr);
    test_bytes_and_lifetime(false);
    test_bytes_and_lifetime(true);
    test_destination_identity();
    test_unmarked_write_invalidates(false, false);
    test_unmarked_write_invalidates(true, false);
    test_unmarked_write_invalidates(true, true);
    test_weight_callback_state(false);
    test_weight_callback_state(true);
    test_reset_replan_slots();
    test_failed_allocation_recovery();
    test_noncontiguous_fallback();
    test_computed_input_is_not_cacheable();
    test_late_mark_falls_back();
    test_entry_budget_fallback();
    std::puts("scheduler input cache: PASS (CPU fake device, no GPU)");
}
