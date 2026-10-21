#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opt-in, bounded in-memory trace. GGML_RKVA_TRACE_FILE is a path prefix;
// a process-id suffix prevents router/model processes overwriting each other.
// Host intervals use ggml_time_us. GPU durations are event elapsed times and
// must not be added to overlapping host intervals or interpreted as wire time.
struct ggml_remote_attn_trace_event {
    uint64_t begin_us, end_us;
    uint64_t request_id, graph_run, graph_uid, cycle, ticket, epoch;
    uint64_t bytes, value;
    uint32_t session, layer, nq, flags;
    int32_t first_pos, last_pos, model_layer;
    double gpu_us;
    char phase[40];
    char node[64];
};

GGML_API bool ggml_remote_attn_trace_enabled(void);
GGML_API void ggml_remote_attn_trace_request(uint64_t request_id);
GGML_API void ggml_remote_attn_trace_graph(uint64_t graph_uid);
GGML_API struct ggml_remote_attn_trace_event ggml_remote_attn_trace_identity(
        const char * phase, const struct ggml_tensor * node);
GGML_API struct ggml_remote_attn_trace_event ggml_remote_attn_trace_context(
        const struct ggml_remote_attn_trace_event * context);
GGML_API void ggml_remote_attn_trace_record(const struct ggml_remote_attn_trace_event * event);
GGML_API void ggml_remote_attn_trace_flush(void);

#ifdef __cplusplus
}

// No timestamps, locks or allocation when tracing is disabled. Events are
// buffered until an existing terminal synchronization/destruction boundary.
struct ggml_remote_attn_trace_scope {
    ggml_remote_attn_trace_event event{};
    bool enabled = false;
    ggml_remote_attn_trace_scope(const char * phase, const ggml_tensor * node = nullptr)
        : enabled(ggml_remote_attn_trace_enabled()) {
        if (enabled) { event = ggml_remote_attn_trace_identity(phase, node); event.begin_us = ggml_time_us(); }
    }
    ~ggml_remote_attn_trace_scope() {
        if (enabled) { event.end_us = ggml_time_us(); ggml_remote_attn_trace_record(&event); }
    }
};

struct ggml_remote_attn_trace_context_scope {
    ggml_remote_attn_trace_event previous{};
    bool enabled;
    explicit ggml_remote_attn_trace_context_scope(const ggml_remote_attn_trace_event & context)
        : enabled(ggml_remote_attn_trace_enabled()) {
        if (enabled) previous = ggml_remote_attn_trace_context(&context);
    }
    ~ggml_remote_attn_trace_context_scope() {
        if (enabled) ggml_remote_attn_trace_context(&previous);
    }
};
#endif
