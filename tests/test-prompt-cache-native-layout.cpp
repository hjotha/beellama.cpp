#include "common.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <stdexcept>

using json = nlohmann::json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

int main() {
    try {
        const json pure = {
            {"cache_layout_version", 2}, {"ctx_type", 9}, {"state_version", 4},
            {"session_version", 9}, {"nextn_layer", 0}, {"causal", true},
            {"kv_unified", false}, {"kv_paged", false}, {"flash_attn", true},
            {"rope_type", 2}, {"rope_base", 1000000}, {"rope_scale", 1},
            {"kv_layout_known", false}, {"source_context", 123}, {"source_capacity", 4096},
            {"type_k", 1}, {"type_v", 1}, {"type_k_aux", 1}, {"type_v_aux", 1},
            {"rotation_k", true}, {"rotation_v", true},
        };
        json split = pure;
        split.erase("source_context"); split.erase("source_capacity");
        split["kv_layout_known"] = true;
        split["mixed_kv"] = true;
        split["mixed_layer_count"] = 64;
        split["mixed_kvarn_type"] = 1;
        split["mixed_kvarn_bits_k"] = 4;
        split["mixed_kvarn_bits_v"] = 4;
        split["mixed_remote_layers"] = json::array({{{"layer", 3}, {"type_k", 2}, {"type_v", 2}}});
        require(common_prompt_cache_layout_pure_to_split(pure.dump(), split.dump()), "forward eligibility rejected");
        require(common_prompt_cache_layout_split_to_pure(split.dump(), pure.dump()), "inverse eligibility rejected");
        require(!common_prompt_cache_layout_split_to_pure(pure.dump(), split.dump()), "wrong direction accepted");
        require(!common_prompt_cache_layout_split_to_pure(pure.dump(), pure.dump()), "pure source accepted as split");
        require(!common_prompt_cache_layout_split_to_pure(split.dump(), split.dump()), "mixed destination accepted as pure");
        json pure_zero = pure;
        json split_zero = split;
        pure_zero["kvarn_zero_exact_tail"] = true;
        split_zero["kvarn_zero_exact_tail"] = true;
        require(common_prompt_cache_layout_pure_to_split(pure_zero.dump(), split_zero.dump()),
                "matching zero-tail pure/split layouts rejected");
        require(!common_prompt_cache_layout_pure_to_split(pure.dump(), split_zero.dump()) &&
                !common_prompt_cache_layout_split_to_pure(split.dump(), pure_zero.dump()),
                "cross-tail handoff must not reuse a different exact-tail representation");
        require(!common_prompt_cache_layout_reusable(pure_zero.dump(), pure.dump()) &&
                !common_prompt_cache_layout_convertible(pure_zero.dump(), pure.dump()),
                "zero-tail cache must not select or convert an ordinary-tail snapshot");

        // Native overflow has no standard mixed-cache payload. Its format and
        // geometry must remain part of the identity across contexts.
        json native = pure;
        native.erase("source_context"); native.erase("source_capacity");
        native["kv_layout_known"] = true;
        native["native_companion_format"] = 0;
        native["native_companion_split_p"] = 1024;
        native["native_companion_kvarn_type"] = 1;
        native["native_companion_kvarn_bits_k"] = 4;
        native["native_companion_kvarn_bits_v"] = 4;
        native["native_global_tail_tokens"] = 128;
        require(common_prompt_cache_layout_reusable(native.dump(), native.dump()),
                "identical native companion rejected");
        require(common_prompt_cache_layout_convertible(native.dump(), native.dump()),
                "identical native companion conversion rejected");
        for (const char * key : {"native_companion_format", "native_companion_split_p",
                                 "native_companion_kvarn_type", "native_companion_kvarn_bits_k",
                                 "native_companion_kvarn_bits_v", "native_global_tail_tokens"}) {
            json changed = native;
            changed[key] = native[key].get<int>() + 1;
            require(!common_prompt_cache_layout_reusable(native.dump(), changed.dump()) &&
                    !common_prompt_cache_layout_reusable(changed.dump(), native.dump()),
                    "different native companion identities reused");
            require(!common_prompt_cache_layout_convertible(native.dump(), changed.dump()) &&
                    !common_prompt_cache_layout_convertible(changed.dump(), native.dump()),
                    "different native companion identities converted");
        }
        for (const json * other : {&pure, static_cast<const json *>(&split)}) {
            require(!common_prompt_cache_layout_reusable(native.dump(), other->dump()) &&
                    !common_prompt_cache_layout_reusable(other->dump(), native.dump()),
                    "native companion reused as pure or standard mixed cache");
            require(!common_prompt_cache_layout_convertible(native.dump(), other->dump()) &&
                    !common_prompt_cache_layout_convertible(other->dump(), native.dump()),
                    "native companion converted as pure or standard mixed cache");
            require(!common_prompt_cache_layout_pure_to_split(other->dump(), native.dump()) &&
                    !common_prompt_cache_layout_split_to_pure(native.dump(), other->dump()),
                    "native companion accepted by standard mixed handoff");
        }
        for (const char * key : {"rope_type", "rope_base", "rope_scale", "state_version", "session_version",
                                "ctx_type", "nextn_layer", "cache_layout_version"}) {
            json mismatch = pure;
            mismatch[key] = pure[key].get<int>() + 1;
            require(!common_prompt_cache_layout_split_to_pure(split.dump(), mismatch.dump()), "semantic mismatch accepted");
        }
        require(!common_prompt_cache_layout_split_to_pure("{broken", pure.dump()), "malformed layout accepted");
        require(!common_prompt_cache_layout_split_to_pure("[]", pure.dump()), "non-object layout accepted");
        std::puts("native KVarN layout eligibility: OK (payload validation belongs to the transactional reader)");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
