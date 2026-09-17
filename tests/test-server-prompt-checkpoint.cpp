#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>
#include <memory>

static constexpr size_t KIB = 1024;

static void speculative_rollback_checkpoint_boundary() {
    constexpr uint32_t reserve = 8;

    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_NO, reserve, reserve + 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_PART, reserve, reserve + 1));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, reserve, 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, 1));
    assert(!server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, reserve));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, reserve, reserve + 1));
    assert( server_speculative_rollback_requires_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, 0, 1));

    assert(!server_prompt_reuse_requires_checkpoint_search(false, 0, 528));
    assert( server_prompt_reuse_requires_checkpoint_search(false, 512, 512));
    assert( server_prompt_reuse_requires_checkpoint_search(true, 0, 528));

    assert(server_prompt_reuse_alignment(0) == 1);
    assert(server_prompt_reuse_alignment(128) == 128);
    assert(server_prompt_reuse_alignment(192) == 192);

    assert(server_prompt_checkpoint_boundary(795, 132,   1) == 663);
    assert(server_prompt_checkpoint_boundary(795, 132, 128) == 640);
    assert(server_prompt_checkpoint_boundary(795,   4, 128) == 768);
    assert(server_prompt_checkpoint_boundary(3,     4, 128) == 0);

    assert(!server_draft_context_owns_state(false, false));
    assert( server_draft_context_owns_state(true,  false));
    assert(!server_draft_context_owns_state(true,  true));
}

static server_prompt make_prompt(const llama_tokens & tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(tokens, false);
    return prompt;
}

static std::shared_ptr<common_prompt_checkpoint> make_checkpoint(
        int64_t n_tokens, llama_pos pos_max, size_t data_tgt_bytes) {
    auto ckpt = std::make_shared<common_prompt_checkpoint>();
    ckpt->n_tokens = n_tokens;
    ckpt->pos_max = pos_max;
    ckpt->data_tgt.resize(data_tgt_bytes);
    return ckpt;
}

static common_memory_seq_rm_result test_seq_rm_suffix(
        llama_seq_id seq_id,
        llama_pos requested_p0,
        const server_tokens & prompt_tokens,
        const common_memory_seq_rm_io & io,
        llama_pos & planned_p0) {
    const auto normalize_p0 = [&](llama_pos value) {
        return value > 0 ? prompt_tokens.pos_next(prompt_tokens.size_up_to_pos(value)) : value;
    };
    return common_memory_seq_rm_suffix(seq_id, requested_p0, io, normalize_p0, planned_p0);
}

static void restore_transaction_validation_failures_are_atomic() {
    const server_prompt_state_view states[] = {
        { reinterpret_cast<const uint8_t *>("target"), 6 },
        { reinterpret_cast<const uint8_t *>("draft"), 5 },
        { reinterpret_cast<const uint8_t *>("spec"), 4 },
    };
    const server_prompt_state_kind kinds[] = {
        SERVER_PROMPT_STATE_MAIN,
        SERVER_PROMPT_STATE_DRAFT,
        SERVER_PROMPT_STATE_SPECULATIVE,
    };

    for (const auto failed_kind : kinds) {
        int prepared = 0;
        int committed = 0;
        server_prompt_restore_transaction_io io {
            /*.restore_target =*/ true,
            /*.restore_draft =*/ true,
            /*.restore_speculative =*/ true,
            /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view) {
                ++prepared;
                return kind != failed_kind;
            },
            /*.commit =*/ [&](server_prompt_state_kind) { ++committed; },
        };
        assert(!server_prompt_restore_transaction(states[0], states[1], states[2], io));
        assert(prepared >= 1 && prepared <= 3);
        assert(committed == 0);
    }
}

static void restore_transaction_validation_failure_identifies_prepare_leg() {
    const server_prompt_state_view states[] = {
        { reinterpret_cast<const uint8_t *>("target"), 6 },
        { reinterpret_cast<const uint8_t *>("draft"), 5 },
        { reinterpret_cast<const uint8_t *>("spec"), 4 },
    };
    const server_prompt_state_kind kinds[] = {
        SERVER_PROMPT_STATE_MAIN,
        SERVER_PROMPT_STATE_DRAFT,
        SERVER_PROMPT_STATE_SPECULATIVE,
    };

    for (const auto failed_kind : kinds) {
        int committed = 0;
        server_prompt_restore_transaction_io io {
            /*.restore_target =*/ true,
            /*.restore_draft =*/ true,
            /*.restore_speculative =*/ true,
            /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view) {
                return kind != failed_kind;
            },
            /*.commit =*/ [&](server_prompt_state_kind) { ++committed; },
        };
        const auto result = server_prompt_restore_transaction_diagnostic(
                states[0], states[1], states[2], io);
        assert(!result.success);
        assert(result.component == failed_kind);
        assert(result.reason == SERVER_PROMPT_RESTORE_PREPARE_REJECTED);
        assert(committed == 0);
    }
}

static void restore_transaction_draft_failure_commits_nothing() {
    const server_prompt_state_view main_state = { reinterpret_cast<const uint8_t *>("main"), 6 };
    const server_prompt_state_view draft_state = { reinterpret_cast<const uint8_t *>("drft"), 8 };

    bool restored_main = false;
    bool restored_draft = false;
    server_prompt_restore_transaction_io io {
        /*.restore_target =*/ true,
        /*.restore_draft =*/ true,
        /*.restore_speculative =*/ false,
        /*.prepare =*/ [&](server_prompt_state_kind kind, server_prompt_state_view state) {
            if (kind == SERVER_PROMPT_STATE_MAIN) {
                return state.data == main_state.data && state.size == main_state.size;
            }
            return state.data != draft_state.data || state.size != draft_state.size;
        },
        /*.commit =*/ [&](server_prompt_state_kind kind) {
            if (kind == SERVER_PROMPT_STATE_MAIN) {
                restored_main = true;
            } else {
                restored_draft = true;
            }
        },
    };

    assert(!server_prompt_restore_transaction(main_state, draft_state, {}, io));
    assert(!restored_main && !restored_draft);
}

static void prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp() {
    // The live slot has the larger lexical prefix (3), but no durable checkpoint
    // at the divergent boundary.  A checkpoint that covers the whole prefix (2)
    // is restorable and must win despite its shorter lexical LCP.
    server_prompt current = make_prompt({1, 2, 3, 4});
    current.checkpoints.push_back(make_checkpoint(2, 2, 16));
    server_tokens requested(llama_tokens {1, 2, 3, 9}, false);

    const auto plan = server_prompt_plan_reuse(current, requested, 1, 0, false);
    assert(plan.lexical_tokens == 3);
    assert(plan.restorable_tokens == 2);
    assert(plan.reason == SERVER_PROMPT_REUSE_CHECKPOINT);
}

static void prompt_cache_self_contained_wins_over_native() {
    server_prompt current = make_prompt({1, 2, 3, 4});
    server_tokens requested(llama_tokens {1, 2, 3, 4, 9}, false);

    // A large native restorable prefix exists, but the RAM prompt fully covers
    // the requested prefix, so the self-contained prefix wins.
    const auto plan = server_prompt_plan_reuse(current, requested, 1, 3, true);
    assert(plan.lexical_tokens == 4);
    assert(plan.restorable_tokens == 4);
    assert(plan.reason == SERVER_PROMPT_REUSE_SELF_CONTAINED);
}

static void prompt_cache_checkpoint_must_respect_alignment_and_position() {
    server_prompt current = make_prompt({1, 2, 3, 4, 5, 6});
    current.checkpoints.push_back(make_checkpoint(2, 2, 16));     // misaligned (alignment 4): rejected
    current.checkpoints.push_back(make_checkpoint(4, 4, 16));     // aligned, pos_max in range: accepted
    current.checkpoints.push_back(make_checkpoint(6, 8, 16));     // aligned but pos_max past the requested boundary: rejected
    server_tokens requested(llama_tokens {1, 2, 3, 4, 5, 9}, false);

    const auto plan = server_prompt_plan_reuse(current, requested, 4, 0, false);
    assert(plan.lexical_tokens == 5);
    assert(plan.restorable_tokens == 4);
    assert(plan.reason == SERVER_PROMPT_REUSE_CHECKPOINT);
}

static void checkpoint_failed_target_save_cannot_reuse_stale_bytes() {
    common_prompt_checkpoint checkpoint;
    checkpoint.data_tgt.resize(32, 0x5a);

    checkpoint.update_tgt(nullptr, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    assert(checkpoint.data_tgt.empty());
    assert(checkpoint.empty());
}

static void speculative_draft_rollback_uses_draft_axis_and_recovers() {
    assert(server_speculative_draft_rollback_p0(4096, 63) == 4096);
    assert(server_speculative_draft_rollback_p0(0, 63) == 64);

    std::vector<std::pair<llama_pos, llama_pos>> removals;
    server_speculative_draft_rollback_io exact_io {
        /*.plan =*/ {},
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return true;
        },
    };
    llama_pos applied_p0 = -1;
    assert(server_speculative_draft_rollback(4096, exact_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_EXACT);
    const std::vector<std::pair<llama_pos, llama_pos>> exact_expected = {{4096, -1}};
    assert(applied_p0 == 4096 && removals == exact_expected);

    removals.clear();
    server_speculative_draft_rollback_io widened_io {
        /*.plan =*/ [](llama_pos p0, llama_pos p1, llama_pos & planned_p0, llama_pos & planned_p1) {
            assert(p0 == 4096 && p1 == -1);
            planned_p0 = 3968;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return p0 == 3968 && p1 == -1;
        },
    };
    assert(server_speculative_draft_rollback(4096, widened_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_WIDENED);
    assert(applied_p0 == 3968);
    const std::vector<std::pair<llama_pos, llama_pos>> widened_expected = {{4096, -1}, {3968, -1}};
    assert(removals == widened_expected);

    removals.clear();
    server_speculative_draft_rollback_io clear_io {
        /*.plan =*/ [](llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = -1;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [&](llama_pos p0, llama_pos p1) {
            removals.emplace_back(p0, p1);
            return p0 == -1 && p1 == -1;
        },
    };
    assert(server_speculative_draft_rollback(4096, clear_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_CLEARED);
    assert(applied_p0 == -1);
    const std::vector<std::pair<llama_pos, llama_pos>> clear_expected = {{4096, -1}, {-1, -1}};
    assert(removals == clear_expected);

    server_speculative_draft_rollback_io failed_io {
        /*.plan =*/ [](llama_pos, llama_pos, llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = 3968;
            planned_p1 = -1;
            return true;
        },
        /*.remove =*/ [](llama_pos, llama_pos) { return false; },
    };
    assert(server_speculative_draft_rollback(4096, failed_io, applied_p0) ==
            SERVER_SPECULATIVE_DRAFT_ROLLBACK_FAILED);
}

static void server_unsupported_removal_falls_back_to_full_reprocess() {
    server_tokens prompt_tokens(llama_tokens(5626, 1), false);
    int partial_removals = 0;
    int full_clears = 0;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](common_memory_context_kind kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            if (kind == COMMON_MEMORY_CONTEXT_DRAFT) {
                return false;
            }
            planned_p0 = 5504;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                ++full_clears;
            } else {
                ++partial_removals;
            }
            return true;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, 5626, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_FULL_REPROCESS);
    assert(planned_p0 == 0);
    assert(partial_removals == 0);
    assert(full_clears == 2);
}

static void server_post_preflight_mutation_failure_clears_both_contexts() {
    server_tokens prompt_tokens(llama_tokens(5626, 1), false);
    bool main_partial = false;
    bool draft_partial = false;
    bool main_cleared = false;
    bool draft_cleared = false;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ true,
        /*.plan =*/ [](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos p1,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = p0;
            planned_p1 = p1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind kind, llama_seq_id, llama_pos p0, llama_pos p1) {
            if (p0 == -1 && p1 == -1) {
                (kind == COMMON_MEMORY_CONTEXT_TARGET ? main_cleared : draft_cleared) = true;
                return true;
            }
            if (kind == COMMON_MEMORY_CONTEXT_TARGET) {
                main_partial = true;
                return true;
            }
            draft_partial = true;
            return false;
        },
    };
    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, 5626, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_MUTATION_FAILED);
    assert(main_partial && draft_partial);
    assert(main_cleared && draft_cleared);
}

static void server_planned_removal_preserves_atomic_media_chunks() {
    mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
    server_tokens prompt_tokens(chunks, true);

    const llama_pos requested_p0 = prompt_tokens.pos_next();
    const llama_pos inside_media = 6;
    const llama_pos media_end = prompt_tokens.pos_next(prompt_tokens.size_up_to_pos(inside_media));
    assert(media_end > inside_media);

    llama_pos removed_p0 = -1;
    common_memory_seq_rm_io io {
        /*.has_draft =*/ false,
        /*.plan =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos,
                       llama_pos & planned_p0, llama_pos & planned_p1) {
            planned_p0 = inside_media;
            planned_p1 = -1;
            return true;
        },
        /*.can_remove =*/ [](common_memory_context_kind, llama_seq_id, llama_pos, llama_pos) { return true; },
        /*.remove =*/ [&](common_memory_context_kind, llama_seq_id, llama_pos p0, llama_pos) {
            removed_p0 = p0;
            return true;
        },
    };

    llama_pos planned_p0 = -1;
    const auto result = test_seq_rm_suffix(0, requested_p0, prompt_tokens, io, planned_p0);
    assert(result == COMMON_MEMORY_SEQ_RM_APPLIED);
    assert(planned_p0 == media_end);
    assert(removed_p0 == media_end);
}

static void prompt_cache_accounting_shares_checkpoint_bytes() {
    server_prompt source = make_prompt({7, 8, 9});
    source.checkpoints.push_back(make_checkpoint(3, 3, 200*KIB));
    server_prompt_cache cache(0, 0);
    server_prompt_cache_state first;
    first.prompt = source.clone();
    first.data.main.resize(10*KIB);
    server_prompt_cache_state second;
    second.prompt = source.clone();
    second.data.main.resize(20*KIB);
    cache.states.push_back(std::move(first));
    cache.states.push_back(std::move(second));
    // 200 KiB checkpoint counted once (shared), plus 10 + 20 KiB payloads.
    assert(cache.accounted_size() == 230*KIB);
}

static void prompt_cache_make_room_evicts_until_budget_fits() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state existing;
    existing.prompt = make_prompt({1, 2});
    existing.data.main.resize(700*KIB);
    cache.states.push_back(std::move(existing));

    assert(cache.make_room(600*KIB));
    assert(cache.states.empty());
    assert(cache.accounted_size() == 0);
}

static void prompt_cache_make_room_keeps_kept_state() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state victim;
    victim.prompt = make_prompt({1, 2});
    victim.data.main.resize(900*KIB);
    cache.states.push_back(std::move(victim));

    server_prompt_cache_state existing;
    existing.prompt = make_prompt({3, 4});
    existing.data.main.resize(200*KIB);
    cache.states.push_back(std::move(existing));

    server_prompt_cache_state incoming;
    incoming.prompt = make_prompt({5, 6});
    incoming.data.main.resize(100*KIB);

    // 1.2 MiB live + incoming exceed the 1 MiB budget; the evictable victim
    // goes, the kept state survives and the incoming fits.
    assert(cache.make_room(100*KIB, &incoming, &cache.states.back()));
    assert(cache.states.size() == 1);
    assert(cache.accounted_size() == 200*KIB);
}

static void prompt_cache_make_room_fails_when_single_state_exceeds_budget() {
    server_prompt_cache cache(1, 0);
    server_prompt_cache_state existing;
    existing.prompt = make_prompt({1, 2});
    existing.data.main.resize(100*KIB);
    cache.states.push_back(std::move(existing));

    server_prompt current = make_prompt({3, 4});
    current.checkpoints.push_back(make_checkpoint(4, 4, 200*KIB));

    assert(!cache.make_room(1000*KIB, nullptr, &cache.states.back()));
    assert(cache.states.size() == 1);
    assert(cache.accounted_size() == 100*KIB);
}

int main() {
    prompt_cache_ranks_safe_restorable_prefix_before_lexical_lcp();
    prompt_cache_self_contained_wins_over_native();
    prompt_cache_checkpoint_must_respect_alignment_and_position();
    restore_transaction_validation_failures_are_atomic();
    restore_transaction_draft_failure_commits_nothing();
    restore_transaction_validation_failure_identifies_prepare_leg();
    speculative_rollback_checkpoint_boundary();
    checkpoint_failed_target_save_cannot_reuse_stale_bytes();
    speculative_draft_rollback_uses_draft_axis_and_recovers();
    server_unsupported_removal_falls_back_to_full_reprocess();
    server_post_preflight_mutation_failure_clears_both_contexts();
    server_planned_removal_preserves_atomic_media_chunks();
    prompt_cache_accounting_shares_checkpoint_bytes();
    prompt_cache_make_room_evicts_until_budget_fits();
    prompt_cache_make_room_keeps_kept_state();
    prompt_cache_make_room_fails_when_single_state_exceeds_budget();
    {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = 3;
        ckpt.pos_min = 1;
        ckpt.pos_max = 2;
        ckpt.data_tgt.resize(128);
        ckpt.data_dft.resize(64);
        ckpt.data_spec.resize(32);
        assert(ckpt.size() == sizeof(ckpt) + ckpt.data_tgt.capacity() +
                ckpt.data_dft.capacity() + ckpt.data_spec.capacity() +
                ckpt.layout_tgt.capacity() + ckpt.layout_dft.capacity());

        ckpt.clear();
        assert(ckpt.n_tokens == 0);
        assert(ckpt.pos_min == -1);
        assert(ckpt.pos_max == -1);
        assert(ckpt.empty());
        assert(ckpt.data_tgt.empty() && ckpt.data_dft.empty() && ckpt.data_spec.empty());
    }

    {
        server_prompt prompt = make_prompt({1, 2, 3});
        prompt.checkpoints.push_back(make_checkpoint(3, 3, 16));

        const server_prompt clone = prompt.clone();
        assert(clone.n_tokens() == 3);
        assert(clone.checkpoints.size() == 1);
        assert(clone.checkpoints.front().get() == prompt.checkpoints.front().get());

        server_prompt_cache_state state {
            /*.prompt =*/ std::move(prompt),
            /*.data =*/ {
                /*.main =*/ std::vector<uint8_t>(64),
                /*.drft =*/ std::vector<uint8_t>(32),
                /*.spec =*/ { },
            },
        };
        const auto & ckpt_ptr = state.prompt.checkpoints.front();
        assert(state.accounted_size() == 96 + sizeof(common_prompt_checkpoint) +
                ckpt_ptr->data_tgt.capacity() + ckpt_ptr->data_dft.capacity() +
                ckpt_ptr->layout_tgt.capacity() + ckpt_ptr->layout_dft.capacity() +
                2*sizeof(void *));
    }

    return 0;
}