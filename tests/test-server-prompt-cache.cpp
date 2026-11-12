#include "common.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory.h"
#include "../src/llama-model.h"
#include "server-task.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "cache check failed at %d: %s\n", __LINE__, #expr); std::abort(); } } while (0)

// Same ceiling as the planned production scenario; buffers are allocated only as used.
// Failure/eviction cases below still derive their smaller bounds from actual snapshot sizes.
static constexpr int32_t cache_budget_mib = 2048;

static void test_eagle_capture(llama_model * model, llama_context * target, llama_context * draft,
                               const common_params_speculative & params) {
    // Synthetic EAGLE driver: test only deferred-boundary ownership, not inference.
    auto saved_layers = model->target_layer_ids;
    model->target_layer_ids = {0, 1, 2};
    {
        auto eagle_params = params;
        eagle_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 };
        eagle_params.draft.ctx_tgt = target;
        eagle_params.draft.ctx_dft = draft;
        common_speculative_ptr eagle(common_speculative_init(eagle_params, 1));
        CHECK(eagle);
        std::vector<uint8_t> captured;
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured) == common_speculative_capture_status::empty);
        CHECK(common_speculative_capture_state(eagle.get(), -1, captured) == common_speculative_capture_status::error);
        CHECK(common_speculative_capture_state(eagle.get(), 1, captured) == common_speculative_capture_status::error);
        const llama_pos position = 3;
        std::vector<uint8_t> state(sizeof(position) + size_t(llama_model_n_embd_out(model))*sizeof(float));
        std::memcpy(state.data(), &position, sizeof(position));
        CHECK(common_speculative_set_state(eagle.get(), 0, state, position));
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured, position) == common_speculative_capture_status::present);
        CHECK(captured == state);
        const auto valid_state = state;
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured, position + 1) == common_speculative_capture_status::error);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(state.data() + sizeof(position), &nan, sizeof(nan));
        CHECK(!common_speculative_set_state(eagle.get(), 0, state, position));
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured, position) == common_speculative_capture_status::present && captured == valid_state);
        const float inf = std::numeric_limits<float>::infinity();
        std::memcpy(state.data() + sizeof(position), &inf, sizeof(inf));
        CHECK(!common_speculative_set_state(eagle.get(), 0, state, position));
        state = valid_state;
        state.pop_back();
        CHECK(!common_speculative_set_state(eagle.get(), 0, state, position));
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured, position) == common_speculative_capture_status::present && captured == valid_state);
        CHECK(common_speculative_clear_state(eagle.get(), 0));
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured) == common_speculative_capture_status::empty);
        const auto saved_arch = model->arch;
        model->arch = LLM_ARCH_LLAMA;
        CHECK(common_speculative_capture_state(eagle.get(), 0, captured) == common_speculative_capture_status::empty);
        CHECK(common_speculative_clear_state(eagle.get(), 0));
        model->arch = saved_arch;
    }
    model->target_layer_ids = std::move(saved_layers);
}

static void test_native_corrupt_entry_retry(llama_context * ctx) {
    llama_batch batch = llama_batch_init(8, 0, 1);
    for (int i = 0; i < 4; ++i) common_batch_add(batch, i + 1, i, {0}, true);
    CHECK(llama_decode(ctx, batch) == 0);
    server_prompt saved;
    saved.tokens = server_tokens(llama_tokens {1, 2, 3, 4}, false);
    server_prompt_cache cache(cache_budget_mib, 0);
    CHECK(cache.save(saved, ctx, nullptr, nullptr, 0));
    auto & good = cache.states.front();

    cache.states.emplace_back();
    auto & bad = cache.states.back();
    bad.model = good.model;
    bad.model_instance = good.model_instance;
    bad.layout_tgt = good.layout_tgt;
    bad.pos_tgt = 4;
    bad.native_ram = true;
    bad.prompt.tokens = server_tokens(llama_tokens {1, 2, 3, 4, 5}, false);
    bad.data.main = good.data.main;
    bad.checksum = bad.digest();
    CHECK(!bad.data.main.empty());
    bad.data.main[0] ^= 1;

    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1));
    common_batch_clear(batch);
    common_batch_add(batch, 9, 0, {0}, true);
    CHECK(llama_decode(ctx, batch) == 0);
    server_prompt live;
    live.tokens = server_tokens(llama_tokens {9}, false);
    server_tokens request(llama_tokens {1, 2, 3, 4, 5, 6}, false);
    const size_t bytes = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> before(bytes), after(bytes);
    CHECK(bytes && llama_state_seq_get_data(ctx, before.data(), bytes, 0) == bytes);

    // With no usable alternative, quarantine must leave the live sequence intact.
    good.quarantined = true;
    CHECK(cache.load(live, request, ctx, nullptr, nullptr, 0) == server_prompt_cache_result::unchanged);
    CHECK(bad.quarantined && live.tokens.get_tokens() == llama_tokens({9}));
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == 0);
    CHECK(llama_state_seq_get_data(ctx, after.data(), bytes, 0) == bytes && before == after);

    // The same corrupt first choice must not hide the valid shorter entry.
    good.quarantined = false;
    bad.quarantined = false;
    CHECK(cache.load(live, request, ctx, nullptr, nullptr, 0) == server_prompt_cache_result::hit);
    CHECK(bad.quarantined && live.tokens.get_tokens() == llama_tokens({1, 2, 3, 4}));
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == 3);

    server_prompt_cache clear_failure(cache_budget_mib, 0);
    CHECK(clear_failure.save(live, ctx, nullptr, nullptr, 0));
    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1));
    common_batch_clear(batch);
    common_batch_add(batch, 9, 0, {0}, true);
    CHECK(llama_decode(ctx, batch) == 0);
    live.tokens = server_tokens(llama_tokens {9}, false);
    int clear_calls = 0;
    llama_get_memory(ctx)->set_on_seq_rm([&](llama_seq_id, llama_pos, llama_pos) {
        ++clear_calls;
        return false;
    });
    CHECK(clear_failure.restore(live, clear_failure.states.front(), ctx, nullptr, nullptr, 0) == server_prompt_cache_result::unchanged);
    CHECK(clear_calls == 1 && clear_failure.native_context_invalidated && live.tokens.get_tokens() == llama_tokens({9}));

    // A failed cleanup during rollback also invalidates the shared context.
    llama_get_memory(ctx)->set_on_seq_rm({});
    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1));
    CHECK(llama_decode(ctx, batch) == 0);
    clear_failure.native_context_invalidated = false;
    auto & invalid = clear_failure.states.front();
    invalid.data.main.pop_back();
    invalid.checksum = invalid.digest();
    clear_calls = 0;
    llama_get_memory(ctx)->set_on_seq_rm([&](llama_seq_id, llama_pos, llama_pos) { return ++clear_calls == 1; });
    CHECK(clear_failure.restore(live, invalid, ctx, nullptr, nullptr, 0) == server_prompt_cache_result::miss);
    CHECK(clear_calls == 2 && clear_failure.native_context_invalidated && live.tokens.get_tokens() == llama_tokens({9}));
    llama_get_memory(ctx)->set_on_seq_rm({});
    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1));
    llama_batch_free(batch);
}

static server_tokens media_tokens(const std::string & id) {
    // Real MTMD v2 placeholder metadata: one 3x2 M-RoPE image, six tokens and three positions.
    std::vector<char> bytes;
    auto put = [&](auto value) {
        const auto * data = reinterpret_cast<const char *>(&value);
        bytes.insert(bytes.end(), data, data + sizeof(value));
    };
    put(uint64_t(2)); put(uint32_t(MTMD_INPUT_CHUNK_TYPE_IMAGE)); put(uint64_t(0)); put(uint8_t(1));
    put(uint32_t(3)); put(uint32_t(2)); put(uint32_t(1)); put(uint32_t(0)); put(uint32_t(1));
    put(uint64_t(id.size())); bytes.insert(bytes.end(), id.begin(), id.end());
    put(uint8_t(0)); put(uint64_t(1));
    put(uint8_t(0)); put(uint8_t(0)); put(int32_t(0)); put(int32_t(3)); put(int32_t(2));
    put(uint8_t(0));
    mtmd::input_chunk_ptr chunk(mtmd_input_chunk_load(bytes.data(), bytes.size()));
    CHECK(chunk && mtmd_input_chunk_get_n_tokens(chunk.get()) == 6 && mtmd_input_chunk_get_n_pos(chunk.get()) == 3);
    server_tokens tokens;
    tokens.has_mtmd = true;
    tokens.push_back(1); tokens.push_back(2);
    tokens.push_back_placeholder(chunk.get());
    return tokens;
}

static void test_media_cache(llama_context * ctx) {
    server_prompt prompt;
    prompt.tokens = media_tokens("cache-image");
    CHECK(prompt.tokens.size() == 8 && prompt.tokens.pos_next() == 5);
    CHECK(prompt.tokens.size_up_to_pos(4) == 8 && prompt.tokens.pos_next(8) == 5);
    CHECK(prompt.tokens.size_up_to_pos(5) == 8);
    CHECK(prompt.tokens.cache_size() > 8*sizeof(llama_token));
    llama_batch batch = llama_batch_init(8, 0, 1);
    auto decode = [&](int count) {
        CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1));
        common_batch_clear(batch);
        const llama_pos positions[] = {0, 1, 2, 2, 3, 3, 4, 4};
        for (int i = 0; i < count; ++i) { common_batch_add(batch, i + 1, positions[i], {0}, i + 1 == count); }
        CHECK(llama_decode(ctx, batch) == 0);
    };
    decode(8);
    server_prompt_cache cache(cache_budget_mib, 0);
    CHECK(cache.save(prompt, ctx, nullptr, nullptr, 0));
    CHECK(cache.states.front().prompt.n_tokens() == 8);
    const uint64_t checksum = cache.states.front().checksum;
    decode(6);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == 3);
    CHECK(!cache.save(prompt, ctx, nullptr, nullptr, 0));
    CHECK(cache.states.size() == 1 && cache.states.front().checksum == checksum);
    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1));
    server_prompt restored;
    CHECK(cache.restore(restored, cache.states.front(), ctx, nullptr, nullptr, 0) == server_prompt_cache_result::hit);
    CHECK(restored.n_tokens() == 8 && restored.tokens.pos_next() == 5);
    cache.states.front().prompt.tokens = media_tokens("changed-image");
    CHECK(cache.restore(restored, cache.states.front(), ctx, nullptr, nullptr, 0) == server_prompt_cache_result::miss);
    CHECK(cache.last_reason == "snapshot integrity check failed");
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == 4);
    llama_batch_free(batch);
    fprintf(stderr, "cache media: 8 tokens/5 positions accepted, partial chunk refused, metadata corruption refused\n");
}

static void test_common_memory_full_cleanup(llama_context * target, llama_context * draft) {
    auto * target_memory = llama_get_memory(target);
    auto * draft_memory = llama_get_memory(draft);
    CHECK(target_memory && draft_memory);
    CHECK(llama_memory_seq_pos_max(target_memory, 0) >= 0);
    CHECK(llama_memory_seq_pos_max(draft_memory, 0) >= 0);
    int target_resets = 0, draft_resets = 0;
    bool target_poisoned = false, draft_poisoned = false;
    target_memory->set_on_seq_rm([&](llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        CHECK(seq_id == 0 && p0 == -1 && p1 == -1);
        ++target_resets;
        return !target_poisoned;
    });
    draft_memory->set_on_seq_rm([&](llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
        CHECK(seq_id == 0 && p0 == -1 && p1 == -1);
        ++draft_resets;
        return !draft_poisoned;
    });
    common_memory memory;
    memory.init(target, draft);
    memory.seq_rm(0, -1, -1);
    CHECK(target_resets == 1 && draft_resets == 1);
    CHECK(llama_memory_seq_pos_min(target_memory, 0) == -1 && llama_memory_seq_pos_max(target_memory, 0) == -1);
    CHECK(llama_memory_seq_pos_min(draft_memory, 0) == -1 && llama_memory_seq_pos_max(draft_memory, 0) == -1);

    // Healthy full cleanup must still execute both RESET callbacks when empty.
    memory.seq_rm(0, -1, -1);
    CHECK(target_resets == 2 && draft_resets == 2);
    target_poisoned = draft_poisoned = true;
    memory.seq_rm(0, -1, -1);
    CHECK(target_resets == 3 && draft_resets == 3);
    CHECK(target_poisoned && draft_poisoned);
    CHECK(llama_memory_seq_pos_max(target_memory, 0) == -1);
    CHECK(llama_memory_seq_pos_max(draft_memory, 0) == -1);
    // Avoid callbacks capturing this fixture's stack after it returns.
    target_memory->set_on_seq_rm({});
    draft_memory->set_on_seq_rm({});
    fprintf(stderr, "cache cleanup CPU: populated RESET, healthy empty RESET, refused empty cleanup continues target/draft without unpoisoning passed\n");
}

static void run_cleanup_refusal(llama_context * ctx, const std::string & mode) {
    auto * mem = llama_get_memory(ctx);
    CHECK(mem);
    if (mode == "cleanup-fail-nonempty") {
        llama_tokens tokens {1, 2, 3, 4};
        auto batch = llama_batch_get_one(tokens.data(), tokens.size());
        CHECK(llama_decode(ctx, batch) == 0);
        CHECK(llama_memory_seq_pos_max(mem, 0) == 3);
    } else {
        CHECK(llama_memory_seq_pos_min(mem, 0) == -1 && llama_memory_seq_pos_max(mem, 0) == -1);
    }
    mem->set_on_seq_rm([&](llama_seq_id, llama_pos, llama_pos) {
        fprintf(stderr, "cleanup CPU fixture: RESET refusal reached (%s)\n", mode.c_str());
        fflush(stderr);
        return false;
    });
    common_memory memory;
    memory.init(ctx, nullptr);
    // Wildcard -1 is supported by the memory API but is deliberately outside
    // the common helper's already-empty, valid-single-sequence exception.
    const auto seq_id = mode == "cleanup-fail-wildcard" ? -1 : 0;
    CHECK(!memory.seq_rm(seq_id, mode == "cleanup-fail-partial" ? 0 : -1,
                         mode == "cleanup-fail-partial" ? 1 : -1));
    mem->set_on_seq_rm({});
}

static void test_cleanup_refusals(const char * executable, const char * model) {
#if !defined(_WIN32)
    for (const char * mode : {"cleanup-fail-nonempty", "cleanup-fail-partial", "cleanup-fail-wildcard"}) {
        int output[2];
        CHECK(pipe(output) == 0);
        const pid_t child = fork();
        CHECK(child >= 0);
        if (child == 0) {
            close(output[0]);
            if (dup2(output[1], STDERR_FILENO) < 0) { _exit(126); }
            close(output[1]);
            const rlimit no_core {0, 0};
            if (setrlimit(RLIMIT_CORE, &no_core) != 0) { _exit(125); }
            // Fresh process: never run a memory API on inherited thread pools.
            execl(executable, executable, model, mode, (char *) nullptr);
            _exit(127);
        }
        close(output[1]);
        std::string captured;
        char chunk[4096];
        ssize_t bytes;
        while ((bytes = read(output[0], chunk, sizeof(chunk))) > 0 || (bytes < 0 && errno == EINTR)) {
            if (bytes > 0) { captured.append(chunk, size_t(bytes)); }
        }
        close(output[0]);
        int status = 0;
        pid_t waited;
        do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
        CHECK(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        CHECK(captured.find(std::string("RESET refusal reached (") + mode + ")") != std::string::npos);
        CHECK(captured.find("propagating failure without abort") != std::string::npos);
        fprintf(stderr, "cache cleanup CPU: %s propagates refused non-noop removal\n", mode);
    }
#else
    (void) executable;
    (void) model;
    fprintf(stderr, "cache cleanup CPU: POSIX fork/exec cases skipped on Windows\n");
#endif
}

static void test_cleanup_only(common_init_result & init, const common_params & params) {
    // The existing architecture fixtures have no tokenizer. This regression
    // needs real context/memory/compute, but no speculative sampler or text.
    llama_context_ptr draft(llama_init_from_model(init.model(), common_context_params_to_llama(params)));
    CHECK(draft);
    llama_tokens tokens {1, 2, 3, 4};
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    CHECK(llama_decode(init.context(), batch) == 0);
    CHECK(llama_decode(draft.get(), batch) == 0);
    test_common_memory_full_cleanup(init.context(), draft.get());
}

static void test_legacy_no_carry(common_init_result & init, const common_params & params) {
    CHECK(!llama_model_mtp_weights_get_info(init.model()).managed);
    llama_context_ptr draft(llama_init_from_model(init.model(), common_context_params_to_llama(params)));
    CHECK(draft);
    auto sp = params.speculative;
    sp.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    sp.draft.ctx_tgt = init.context();
    sp.draft.ctx_dft = draft.get();
    common_speculative_ptr spec(common_speculative_init(sp, 1));
    CHECK(spec);
    llama_tokens tokens {1, 2, 3, 4};
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    CHECK(llama_decode(init.context(), batch) == 0);
    CHECK(llama_decode(draft.get(), batch) == 0);
    common_prompt_checkpoint plain;
    plain.update_pos(4, 0, 3);
    plain.update_tgt(init.context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    plain.update_dft(draft.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    CHECK(plain.retained_count_tgt == 1 && plain.retained_count_dft == 1);
    CHECK(llama_memory_seq_rm(llama_get_memory(init.context()), 0, -1, -1));
    CHECK(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, -1, -1));
    CHECK(!plain.restore_tgt(init.context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    CHECK(plain.restore_dft(draft.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::missing_base);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init.context()), 0) == -1);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == -1);
    CHECK(llama_decode(init.context(), batch) == 0);
    CHECK(llama_decode(draft.get(), batch) == 0);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init.context()), 0) == 3);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == 3);
    fprintf(stderr, "cache plain KV: PARTIAL_ONLY refuses missing retained target/draft attention\n");
    std::vector<uint8_t> carry;
    CHECK(!common_speculative_get_state(spec.get(), 0, carry) && carry.empty());
    CHECK(common_speculative_capture_state(spec.get(), 0, carry) == common_speculative_capture_status::empty);
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    server_prompt_cache cache(cache_budget_mib, 0);
    CHECK(cache.save(prompt, init.context(), draft.get(), spec.get(), 0));
    CHECK(cache.states.front().data.spec.empty() && cache.states.front().has_draft());
    CHECK(llama_memory_seq_rm(llama_get_memory(init.context()), 0, -1, -1));
    CHECK(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, -1, -1));
    prompt.clear();
    server_tokens request(llama_tokens {1, 2, 3, 4, 5}, false);
    CHECK(cache.load(prompt, request, init.context(), draft.get(), spec.get(), 0) == server_prompt_cache_result::hit);
    CHECK(prompt.n_tokens() == 4 && cache.states.empty());
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init.context()), 0) == 3);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == 3);
    llama_token next = 5;
    batch = llama_batch_get_one(&next, 1);
    CHECK(llama_decode(init.context(), batch) == 0);
    CHECK(llama_decode(draft.get(), batch) == 0);
    CHECK(!common_speculative_get_state(spec.get(), 0, carry));
    test_common_memory_full_cleanup(init.context(), draft.get());
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init.context()), 0) == -1);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == -1);
    test_media_cache(init.context());
    llama_synchronize(draft.get());
    fprintf(stderr, "cache legacy draft-simple: no carry, target/draft restored, successful entry consumed\n");
}

static void test_iswa(common_init_result & init, const common_params & params) {
    auto * target = init.context();
    const uint64_t target_id = llama_get_prompt_cache_profile(target).context_instance;
    llama_context_ptr draft(llama_init_from_model(init.model(), common_context_params_to_llama(params)));
    CHECK(draft && llama_get_prompt_cache_profile(draft.get()).context_instance > target_id);
    llama_batch batch = llama_batch_init(32, 0, 1);
    auto decode = [&](llama_context * ctx, int first, int end) {
        for (int begin = first; begin < end; begin += 32) {
            common_batch_clear(batch);
            for (int pos = begin; pos < std::min(end, begin + 32); ++pos) {
                common_batch_add(batch, 1 + pos % 61, pos, {0}, pos + 1 == end);
            }
            CHECK(llama_decode(ctx, batch) == 0);
        }
    };
    decode(target, 0, 64);
    decode(draft.get(), 0, 64);
    common_prompt_checkpoint checkpoint;
    checkpoint.update_pos(64, llama_memory_seq_pos_min(llama_get_memory(target), 0), 63);
    checkpoint.update_tgt(target, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    checkpoint.update_dft(draft.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto saved = llama_get_prompt_cache_profile(target, 0);
    CHECK(saved.attn_min == 0 && saved.attn_max == 63 && saved.attn_aux_max == 63);
    CHECK(checkpoint.retained_count_tgt == 1 && checkpoint.retained_count_dft == 1);
    decode(target, 64, 384);
    decode(draft.get(), 64, 384);
    const auto advanced = llama_get_prompt_cache_profile(target, 0);
    CHECK(advanced.attn_min == 0 && advanced.attn_max == 383);
    CHECK(advanced.attn_aux_min > saved.attn_aux_max); // Actual physical window recycling, not a mock.
    CHECK(checkpoint.restore_tgt(target, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    CHECK(checkpoint.restore_dft(draft.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::restored);
    const auto rewound = llama_get_prompt_cache_profile(target, 0);
    // Native SWA serialization omits cells already masked by the saved window.
    const llama_pos useful_min = saved.attn_aux_max + 1 - llama_model_n_swa(init.model());
    fprintf(stderr, "cache iSWA bounds: saved base=[%d,%d] swa=[%d,%d], advanced swa=[%d,%d], restored base=[%d,%d] swa=[%d,%d]\n",
        saved.attn_min, saved.attn_max, saved.attn_aux_min, saved.attn_aux_max,
        advanced.attn_aux_min, advanced.attn_aux_max,
        rewound.attn_min, rewound.attn_max, rewound.attn_aux_min, rewound.attn_aux_max);
    CHECK(rewound.attn_max == 383 && rewound.attn_aux_min == useful_min && rewound.attn_aux_max == 63);
    // Match the server ordering: restore the SWA component, then trim the future base KV.
    CHECK(llama_memory_seq_rm(llama_get_memory(target), 0, 64, -1));
    CHECK(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 64, -1));
    decode(target, 64, 65);
    decode(draft.get(), 64, 65);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(init.model()));
    const float * logits = llama_get_logits_ith(target, -1);
    const std::vector<float> restored(logits, logits + n_vocab);
    CHECK(llama_memory_seq_rm(llama_get_memory(target), 0, -1, -1));
    CHECK(!checkpoint.restore_tgt(target, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    CHECK(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, -1, -1));
    CHECK(checkpoint.restore_dft(draft.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::missing_base);
    decode(target, 0, 65);
    logits = llama_get_logits_ith(target, -1);
    float max_diff = 0;
    for (int i = 0; i < n_vocab; ++i) {
        CHECK(std::isfinite(restored[i]) && std::isfinite(logits[i]));
        max_diff = std::max(max_diff, std::fabs(restored[i] - logits[i]));
    }
    CHECK(max_diff < 1e-5f);
    const uint64_t old_id = llama_get_prompt_cache_profile(draft.get()).context_instance;
    draft.reset();
    draft.reset(llama_init_from_model(init.model(), common_context_params_to_llama(params)));
    CHECK(draft && llama_get_prompt_cache_profile(draft.get()).context_instance > old_id);
    llama_batch_free(batch);
    fprintf(stderr, "cache iSWA: auxiliary min %d -> %d -> %d, retained base max %d, target/draft rewind, missing base refused, hit=1 cold=65 max_logit_diff=%g, context IDs monotonic\n",
        saved.attn_aux_min, advanced.attn_aux_min, rewound.attn_aux_min, rewound.attn_max, max_diff);
}

int main(int argc, char ** argv) {
    const std::string mode = argc == 3 ? argv[2] : "";
    const bool cleanup_refusal = mode == "cleanup-fail-nonempty" || mode == "cleanup-fail-partial" || mode == "cleanup-fail-wildcard";
    const bool cleanup_only = mode == "cleanup";
    const bool native_integrity = mode == "native-integrity";
    CHECK(argc == 2 || (argc == 3 && (mode == "legacy" || mode == "iswa" || cleanup_only || cleanup_refusal || native_integrity)));
    const bool iswa = mode == "iswa";
    const bool legacy = argc == 3 && !native_integrity;
    llama_backend_init();
    common_params params;
    params.model.path = argv[1];
    params.n_ctx = iswa ? 512 : 256;
    params.n_batch = params.n_ubatch = 32;
    params.n_parallel = 1;
    params.n_gpu_layers = 0;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 2;
    params.fit_params = false;
    params.warmup = false;
    params.split_mtp_weights = !legacy;
    if (!legacy) { params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP }; }
    params.speculative.draft.n_max = 2;
    params.speculative.draft.p_min = 0.0f;
    params.sampling.temp = 0.0f;
    params.n_outputs_max = params.n_outputs_max_per_seq = 32;
    auto init = common_init_from_params(params);
    CHECK(init && init->context());
    if (native_integrity) {
        test_native_corrupt_entry_retry(init->context());
        init.reset();
        llama_backend_free();
        return 0;
    }
    if (cleanup_refusal) {
        run_cleanup_refusal(init->context(), mode);
        init.reset();
        llama_backend_free();
        return 0;
    }
    if (legacy) {
        if (cleanup_only) { test_cleanup_only(*init, params); }
        else if (iswa) { test_iswa(*init, params); }
        else { test_legacy_no_carry(*init, params); }
        init.reset();
        llama_backend_free();
        if (!iswa) { test_cleanup_refusals(argv[0], argv[1]); }
        return 0;
    }
    auto * model = init->model();
    common_speculative_init_result_ptr draft;
    common_speculative_ptr spec;
    auto make_draft = [&]() {
        auto p = common_base_params_to_speculative(params);
        draft = common_speculative_init_from_params(p, model, init->context());
        CHECK(draft && draft->context());
        auto sp = params.speculative;
        sp.draft.ctx_tgt = init->context();
        sp.draft.ctx_dft = draft->context();
        spec.reset(common_speculative_init(sp, 1));
        CHECK(spec);
    };
    make_draft();
    std::vector<uint8_t> capture_probe;
    CHECK(common_speculative_capture_state(nullptr, 0, capture_probe) == common_speculative_capture_status::error);
    CHECK(common_speculative_capture_state(spec.get(), -1, capture_probe) == common_speculative_capture_status::error);
    CHECK(common_speculative_capture_state(spec.get(), 0, capture_probe) == common_speculative_capture_status::empty && capture_probe.empty());
    {
        common_params_speculative simple_params;
        simple_params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
        common_speculative_ptr simple(common_speculative_init(simple_params, 1));
        CHECK(simple);
        CHECK(common_speculative_capture_state(simple.get(), 0, capture_probe) == common_speculative_capture_status::empty);
        CHECK(common_speculative_capture_state(simple.get(), -1, capture_probe) == common_speculative_capture_status::error);
        CHECK(common_speculative_clear_state(simple.get(), 0));
    }
    {
        // Exercise chain-head policy without running a synthetic multihead graph.
        auto chain_params = params.speculative;
        chain_params.draft.ctx_tgt = init->context();
        chain_params.draft.ctx_dft = draft->context();
        const auto layers = model->hparams.n_layer_nextn;
        model->hparams.n_layer_nextn = 2;
        common_speculative_ptr chain(common_speculative_init(chain_params, 1));
        model->hparams.n_layer_nextn = layers;
        CHECK(chain);
        CHECK(common_speculative_capture_state(chain.get(), 0, capture_probe) == common_speculative_capture_status::error);
        CHECK(!common_speculative_validate_state(chain.get(), 0, {}));
        CHECK(!common_speculative_prepare_state(chain.get(), 0, nullptr, 0));
        CHECK(!common_speculative_clear_state(chain.get(), 0));
        CHECK(!common_speculative_set_state(chain.get(), 0, {}, -1));
        CHECK(!common_speculative_set_state(chain.get(), 0, {}, 3));
    }
    llama_batch batch = llama_batch_init(32, 0, 1);
    int evaluated_tokens = 0;
    auto decode = [&](int first, int end) {
        common_batch_clear(batch);
        for (int pos = first; pos < end; ++pos) { common_batch_add(batch, pos + 1, pos, {0}, true); }
        CHECK(llama_decode(init->context(), batch) == 0);
        evaluated_tokens += end - first;
        if (spec) { CHECK(common_speculative_process(spec.get(), batch)); }
    };
    decode(0, 4);
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens {1, 2, 3, 4, 5}, false); // token 5 is not evaluated
    auto checkpoint = std::make_shared<common_prompt_checkpoint>();
    checkpoint->update_pos(4, llama_memory_seq_pos_min(llama_get_memory(init->context()), 0), 3);
    checkpoint->update_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    checkpoint->update_dft(draft->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    checkpoint->update_spec(spec.get(), 0);
    CHECK(!checkpoint->data_spec.empty() && checkpoint->draft_base_valid);
    CHECK(common_speculative_capture_state(spec.get(), 0, capture_probe) == common_speculative_capture_status::present && !capture_probe.empty());
    CHECK(common_speculative_capture_state(spec.get(), 0, capture_probe, checkpoint->pos_max + 1) == common_speculative_capture_status::error);
    auto nonfinite_carry = checkpoint->data_spec;
    auto * finite_plan = common_speculative_prepare_state(spec.get(), 0, nonfinite_carry.data(), nonfinite_carry.size());
    CHECK(finite_plan);
    common_speculative_state_restore_plan_free(finite_plan);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(nonfinite_carry.data() + 4*sizeof(uint32_t), &nan, sizeof(nan));
    CHECK(!common_speculative_prepare_state(spec.get(), 0, nonfinite_carry.data(), nonfinite_carry.size()));
    {
        auto mixed_params = params.speculative;
        mixed_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
        mixed_params.draft.ctx_tgt = init->context();
        mixed_params.draft.ctx_dft = draft->context();
        common_speculative_ptr mixed(common_speculative_init(mixed_params, 1));
        CHECK(mixed);
        CHECK(common_speculative_capture_state(mixed.get(), 0, capture_probe) == common_speculative_capture_status::empty);
        CHECK(common_speculative_capture_state(mixed.get(), -1, capture_probe) == common_speculative_capture_status::error);
        CHECK(common_speculative_set_state(mixed.get(), 0, checkpoint->data_spec, checkpoint->pos_max));
        CHECK(common_speculative_capture_state(mixed.get(), 0, capture_probe) == common_speculative_capture_status::present);
        CHECK(common_speculative_clear_state(mixed.get(), 0));
        CHECK(common_speculative_capture_state(mixed.get(), 0, capture_probe) == common_speculative_capture_status::empty);
    }
    prompt.checkpoints.push_back(checkpoint);
    // Restore recurrent/carry state at its checkpoint position while attention still contains future KV.
    decode(4, 6);
    CHECK(checkpoint->restore_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    CHECK(checkpoint->restore_dft(draft->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::restored);
    CHECK(common_speculative_set_state(spec.get(), 0, checkpoint->data_spec, checkpoint->pos_max));
    CHECK(llama_memory_seq_rm(llama_get_memory(init->context()), 0, 4, -1));
    CHECK(llama_memory_seq_rm(llama_get_memory(draft->context()), 0, 4, -1));
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == 3);
    auto device_checkpoint = std::make_shared<common_prompt_checkpoint>();
    device_checkpoint->update_pos(4, 3, 3);
    device_checkpoint->update_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    CHECK(!device_checkpoint->host_only() && !device_checkpoint->data_tgt.empty());
    prompt.checkpoints.push_back(device_checkpoint);
    server_prompt_cache cache(cache_budget_mib, 0);
    CHECK(cache.save(prompt, init->context(), draft->context(), spec.get(), 0));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().prompt.n_tokens() == 4);
    CHECK(cache.states.front().prompt.checkpoints.size() == 1);
    CHECK(cache.states.front().prompt.checkpoints.front().get() == checkpoint.get());
    CHECK(cache.states.front().data.spec == checkpoint->data_spec);
    prompt.checkpoints.remove(device_checkpoint);
    device_checkpoint.reset();
    server_prompt no_prefix;
    server_tokens divergent(llama_tokens {1, 2, 8, 9}, false);
    CHECK(cache.load(no_prefix, divergent, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::unchanged);
    CHECK(cache.last_reason == "matching recurrent prefix has no usable checkpoint");
    const size_t saved_bytes = cache.size();
    const auto initial_weights = llama_model_mtp_weights_get_info(model);
    CHECK(initial_weights.model_load_count == 1);
    const auto original_checksum = cache.states.front().checksum;
    server_prompt restored = prompt.clone();
    const auto first_restore = cache.restore(restored, cache.states.front(), init->context(), draft->context(), spec.get(), 0);
    if (first_restore != server_prompt_cache_result::hit) fprintf(stderr, "initial cache restore: %s\n", cache.last_reason.c_str());
    CHECK(first_restore == server_prompt_cache_result::hit);
    CHECK(restored.n_tokens() == 4 && cache.states.front().checksum == original_checksum);

    // A corrupt source is rejected before live mutation.
    auto & source = cache.states.front();
    {
        common_params_speculative unknown_params;
        unknown_params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K };
        common_speculative_ptr unknown(common_speculative_init(unknown_params, 1));
        CHECK(unknown);
        CHECK(common_speculative_capture_state(unknown.get(), 0, capture_probe) == common_speculative_capture_status::error);
        CHECK(llama_memory_seq_rm(llama_get_memory(init->context()), 0, -1, -1));
        CHECK(llama_memory_seq_rm(llama_get_memory(draft->context()), 0, -1, -1));
        server_prompt empty;
        CHECK(cache.restore(empty, source, init->context(), draft->context(), unknown.get(), 0) == server_prompt_cache_result::miss);
        CHECK(cache.last_reason == "destination state could not be captured for rollback; refusing before mutation");
        CHECK(empty.tokens.empty() && llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == -1 &&
                llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0) == -1);
        CHECK(common_speculative_clear_state(spec.get(), 0));
        CHECK(cache.restore(empty, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
        restored = std::move(empty);
    }
    source.data.main[0] ^= 1;
    CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == 3);
    source.data.main[0] ^= 1;
    CHECK(source.digest() == original_checksum);

    // Recomputed integrity metadata lets each malformed component reach its actual import path.
    std::vector<uint8_t> * components[] = { &source.data.main, &source.data.drft, &source.data.spec };
    const char * failures[] = { "target restore failed", "draft restore failed", "MTP carry restore failed" };
    const auto resident_bytes = [](llama_context * ctx) {
        const size_t size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> bytes(size);
        CHECK(size && llama_state_seq_get_data(ctx, bytes.data(), size, 0) == size);
        return bytes;
    };
    for (size_t i = 0; i < 3; ++i) {
        const auto target_before = resident_bytes(init->context());
        const auto draft_before = resident_bytes(draft->context());
        const auto prompt_before = restored.tokens.get_tokens();
        std::vector<uint8_t> carry_before;
        CHECK(common_speculative_get_state(spec.get(), 0, carry_before) && !carry_before.empty());
        const uint8_t last_byte = components[i]->back();
        components[i]->pop_back();
        source.checksum = source.digest();
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
        CHECK(cache.last_reason == failures[i]);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == 3);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0) == 3);
        CHECK(restored.tokens.get_tokens() == prompt_before);
        CHECK(resident_bytes(init->context()) == target_before);
        CHECK(resident_bytes(draft->context()) == draft_before);
        std::vector<uint8_t> after_failure;
        CHECK(common_speculative_get_state(spec.get(), 0, after_failure) && after_failure == carry_before);
        components[i]->push_back(last_byte);
        source.checksum = original_checksum;
        CHECK(source.digest() == original_checksum);
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
    }

    {
        server_prompt_cache variants(cache_budget_mib, 0);
        CHECK(variants.save(restored, init->context(), draft->context(), spec.get(), 0));
        CHECK(common_speculative_set_state(spec.get(), 0, {}));
        CHECK(!variants.save(restored, init->context(), draft->context(), spec.get(), 0));
        CHECK(variants.states.size() == 1 && !variants.states.front().data.spec.empty());
        CHECK(common_speculative_capture_state(spec.get(), 0, capture_probe) == common_speculative_capture_status::empty);
        const auto target_before = resident_bytes(init->context());
        const auto draft_before = resident_bytes(draft->context());
        const auto removed = source.data.main.back();
        source.data.main.pop_back();
        source.checksum = source.digest();
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
        CHECK(cache.last_reason == "target restore failed");
        CHECK(resident_bytes(init->context()) == target_before && resident_bytes(draft->context()) == draft_before);
        CHECK(common_speculative_capture_state(spec.get(), 0, capture_probe) == common_speculative_capture_status::empty);
        source.data.main.push_back(removed);
        source.checksum = original_checksum;
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
    }

    {
        const auto layout = checkpoint->layout_tgt;
        checkpoint->layout_tgt += "incompatible";
        CHECK(!checkpoint->restore_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        CHECK(llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == 3);
        checkpoint->layout_tgt = layout;
        CHECK(llama_memory_seq_rm(llama_get_memory(draft->context()), 0, -1, -1));
        CHECK(checkpoint->restore_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        const auto before = llama_memory_seq_pos_max(llama_get_memory(init->context()), 0);
        // Plain draft KV is omitted by PARTIAL_ONLY and must still be resident.
        CHECK(checkpoint->retained_count_tgt == 1 && checkpoint->retained_count_dft == 1);
        CHECK(checkpoint->restore_dft(draft->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::missing_base);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0) == -1);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == before);
        auto unknown = *checkpoint;
        unknown.retained_count_tgt = UINT32_MAX;
        CHECK(!unknown.restore_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        fprintf(stderr, "cache Qwen MTP: missing retained draft attention and unknown capability refused\n");
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
    }
    {
        server_prompt_cache poison(cache_budget_mib, 0);
        CHECK(poison.save(restored, init->context(), draft->context(), spec.get(), 0));
        auto & bad = poison.states.front();
        const auto checksum = bad.checksum;
        const auto byte = bad.data.main.back();
        bad.data.main.pop_back();
        bad.checksum = bad.digest();
        CHECK(llama_memory_seq_rm(llama_get_memory(init->context()), 0, -1, -1));
        CHECK(llama_memory_seq_rm(llama_get_memory(draft->context()), 0, -1, -1));
        CHECK(common_speculative_clear_state(spec.get(), 0));
        server_prompt empty;
        server_tokens next(llama_tokens {1, 2, 3, 4, 5}, false);
        CHECK(poison.load(empty, next, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
        CHECK(bad.quarantined);
        CHECK(poison.load(empty, next, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::unchanged);
        CHECK(poison.last_reason == "snapshot quarantined after failed restore");
        bad.data.main.push_back(byte);
        bad.checksum = checksum;
        CHECK(poison.restore(restored, bad, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
    }
    {
        decode(4, 5);
        server_prompt five;
        five.tokens = server_tokens(llama_tokens {1, 2, 3, 4, 5}, false);
        five.checkpoints = prompt.checkpoints;
        server_prompt_cache valid(cache_budget_mib, 0);
        CHECK(valid.save(five, init->context(), draft->context(), spec.get(), 0));
        const auto checksum = valid.states.front().checksum;
        restored.tokens = five.tokens.clone_for_cache();
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
        common_batch_clear(batch);
        common_batch_add(batch, 5, 4, {0}, true);
        CHECK(llama_decode(init->context(), batch) == 0);
        CHECK(!common_speculative_is_ready(spec.get(), 0, 5));
        CHECK(!valid.save(five, init->context(), draft->context(), spec.get(), 0));
        CHECK(valid.states.size() == 1 && valid.states.front().checksum == checksum);
        restored.tokens = five.tokens.clone_for_cache();
        CHECK(valid.restore(restored, valid.states.front(), init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
        CHECK(valid.last_reason == "destination state could not be captured for rollback; refusing before mutation");
        CHECK(common_speculative_clear_state(spec.get(), 0));
        CHECK(valid.restore(restored, valid.states.front(), init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
        common_batch_clear(batch);
        common_batch_add(batch, 5, 4, {0}, true);
        CHECK(llama_decode(init->context(), batch) == 0);
        CHECK(common_speculative_bootstrap(spec.get(), batch, llama_get_nextn_decode_id(init->context())));
        common_prompt_checkpoint seed;
        seed.update_pos(5, 4, 4);
        seed.update_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        seed.update_dft(draft->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        seed.update_spec(spec.get(), 0);
        CHECK(seed.draft_base_valid && llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0) == -1);
        CHECK(seed.restore_dft(draft->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::restored);
        CHECK(common_speculative_is_ready(spec.get(), 0, 5));
        decode(5, 6);
        restored.tokens = server_tokens(llama_tokens {1, 2, 3, 4, 5, 6}, false);
        CHECK(llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0) == 5);
        CHECK(seed.restore_tgt(init->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        CHECK(seed.restore_dft(draft->context(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == common_checkpoint_restore::restored);
        CHECK(common_speculative_set_state(spec.get(), 0, seed.data_spec, 4));
        CHECK(llama_memory_seq_rm(llama_get_memory(init->context()), 0, 5, -1));
        CHECK(common_speculative_is_ready(spec.get(), 0, 5));
        server_prompt_cache seeded(cache_budget_mib, 0);
        CHECK(seeded.save(five, init->context(), draft->context(), spec.get(), 0));
        CHECK(seeded.restore(restored, seeded.states.front(), init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
        CHECK(common_speculative_is_ready(spec.get(), 0, 5));
        CHECK(cache.restore(restored, source, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
    }

    // Complete short state survives actual context destruction and target-only long saves.
    llama_synchronize(draft->context());
    spec.reset(); draft.reset(); init->release_context();
    CHECK(llama_model_mtp_weights_set_resident(model, false));
    auto long_params = params;
    long_params.n_ctx = 512;
    long_params.speculative.types.clear();
    CHECK(init->recreate_context(long_params));
    restored.clear();
    CHECK(cache.restore(restored, source, init->context(), nullptr, nullptr, 0) == server_prompt_cache_result::hit);
    CHECK(cache.save(restored, init->context(), nullptr, nullptr, 0));
    CHECK(cache.states.size() == 2 && cache.states.front().has_draft() && !cache.states.back().has_draft());
    CHECK(cache.size() == cache.states.front().private_size() + cache.states.back().private_size() + checkpoint->size() + 2*sizeof(void *));
    const auto original_budget = cache.limit_size;
    cache.limit_size = cache.size();
    const auto preserved_checksum = cache.states.front().checksum;
    CHECK(cache.restore(restored, cache.states.front(), init->context(), nullptr, nullptr, 0) == server_prompt_cache_result::hit);
    CHECK(cache.states.size() == 1 && cache.states.front().checksum == preserved_checksum);
    cache.limit_size = original_budget;
    CHECK(cache.save(restored, init->context(), nullptr, nullptr, 0));
    CHECK(cache.states.size() == 2);
    const auto target_only_checksum = cache.states.back().checksum;
    init->release_context();
    CHECK(llama_model_mtp_weights_set_resident(model, true));
    CHECK(init->recreate_context(params));
    make_draft();
    restored.clear();
    CHECK(cache.restore(restored, cache.states.back(), init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::needs_bootstrap);
    CHECK(cache.states.back().checksum == target_only_checksum);
    server_tokens request(llama_tokens {1, 2, 3, 4, 5}, false);
    CHECK(cache.load(restored, request, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::hit);
    CHECK(restored.n_tokens() == 4 && cache.states.size() == 2);
    CHECK(cache.states.back().has_draft());
    const int evaluated_before_hit = evaluated_tokens;
    decode(4, 5);
    CHECK(evaluated_tokens - evaluated_before_hit == 1);
    CHECK(llama_memory_seq_pos_max(llama_get_memory(init->context()), 0) == 4);
    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const auto * logits = llama_get_logits_ith(init->context(), -1);
    std::vector<float> restored_logits(logits, logits + n_vocab);
    const llama_token selected = std::max_element(restored_logits.begin(), restored_logits.end()) - restored_logits.begin();
    auto propose = [&]() {
        llama_tokens result;
        auto history = request.get_text_tokens();
        auto & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true;
        dp.n_max = 2;
        dp.pos0 = 5;
        dp.id_last = selected;
        dp.temperature = 0.0f;
        dp.seed = 1;
        dp.prompt = &history;
        dp.result = &result;
        common_speculative_draft(spec.get());
        llama_synchronize(draft->context());
        dp.drafting = false;
        dp.prompt = nullptr;
        dp.result = nullptr;
        return result;
    };
    const auto restored_drafts = propose();
    CHECK(!restored_drafts.empty());
    CHECK(llama_memory_seq_rm(llama_get_memory(init->context()), 0, -1, -1));
    CHECK(llama_memory_seq_rm(llama_get_memory(draft->context()), 0, -1, -1));
    CHECK(common_speculative_set_state(spec.get(), 0, {}));
    const int evaluated_before_cold = evaluated_tokens;
    decode(0, 5);
    CHECK(evaluated_tokens - evaluated_before_cold == 5);
    logits = llama_get_logits_ith(init->context(), -1);
    CHECK(std::max_element(logits, logits + n_vocab) - logits == selected);
    float max_logit_diff = 0;
    for (size_t i = 0; i < n_vocab; ++i) {
        CHECK(std::isfinite(logits[i]) && std::isfinite(restored_logits[i]));
        max_logit_diff = std::max(max_logit_diff, std::fabs(logits[i] - restored_logits[i]));
    }
    CHECK(max_logit_diff < 0.001f);
    CHECK(propose() == restored_drafts);
    CHECK(llama_memory_seq_rm(llama_get_memory(draft->context()), 0, 5, -1));
    CHECK(common_speculative_is_ready(spec.get(), 0, 5));
    fprintf(stderr, "cache: evaluated hit=1 cold=5, matching greedy token=%d and MTP proposals=%zu, max_logit_diff=%g\n",
            selected, restored_drafts.size(), max_logit_diff);

    // Preflight rejects incompatible layouts and capacity without consuming a valid source.
    auto & full = cache.states.back();
    ++full.model_instance;
    CHECK(cache.restore(restored, full, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
    CHECK(cache.last_reason == "snapshot model differs");
    --full.model_instance;
    const auto saved_pos = full.pos_tgt;
    full.pos_tgt = llama_n_ctx_seq(init->context());
    CHECK(cache.restore(restored, full, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
    CHECK(cache.last_reason == "snapshot does not fit destination context");
    full.pos_tgt = saved_pos;
    const auto layout = full.layout_tgt;
    full.layout_tgt += "invalid";
    CHECK(cache.restore(restored, full, init->context(), draft->context(), spec.get(), 0) == server_prompt_cache_result::miss);
    full.layout_tgt = layout;
    CHECK(full.digest() == full.checksum);

    server_prompt_cache too_small(1, 0);
    too_small.limit_size = saved_bytes - 1;
    CHECK(!too_small.save(prompt, init->context(), draft->context(), spec.get(), 0));
    CHECK(too_small.states.empty());
    cache.limit_size = cache.size();
    CHECK(cache.save(prompt, init->context(), draft->context(), spec.get(), 0));
    CHECK(cache.size() <= cache.limit_size && cache.states.size() == 1);
    // `update()` enforces the payload-only accounting limit.  `state.size()`
    // also includes container/checkpoint ownership overhead, so using it here
    // can leave a payload below the configured limit and make this eviction
    // assertion depend on allocator details.
    cache.limit_size = cache.accounted_size() - 1;
    cache.update();
    CHECK(cache.states.empty());

    server_prompt_cache transient(1, 0);
    CHECK(transient.reserve_transient(512 * 1024));
    CHECK(transient.size() == 512 * 1024);
    CHECK(!transient.reserve_transient(600 * 1024));
    transient.release_transient(512 * 1024);
    CHECK(transient.size() == 0);
    fprintf(stderr, "cache: host snapshot, shared checkpoints, checksum, partial rollback, short-long-short, MTP preference, capacity, eviction passed; initial_bytes=%zu\n", saved_bytes);
    CHECK(init->model() == model);
    CHECK(llama_model_mtp_weights_get_info(model).model_instance == initial_weights.model_instance);
    CHECK(llama_model_mtp_weights_get_info(model).model_load_count == 1);
    fprintf(stderr, "cache: main model loads=1, additional MTP CPU backing=%zu bytes\n", initial_weights.host_bytes);
    llama_synchronize(draft->context());
    spec.reset(); draft.reset();
    {
        llama_context_ptr scratch(llama_init_from_model(model, common_context_params_to_llama(params)));
        CHECK(scratch);
        test_native_corrupt_entry_retry(scratch.get());
        llama_context_ptr scratch_draft(llama_init_from_model(model, common_context_params_to_llama(params)));
        CHECK(scratch_draft);
        test_eagle_capture(model, scratch.get(), scratch_draft.get(), params.speculative);
    }
    init.reset();
    llama_batch_free(batch);
    llama_backend_free();
}
