// Pure policy test for GGML_RKVA_LOCAL_INPUT_REUSE, including decode N1.
// No model, backend, or hardware is initialized.
#include "llama-rkva-local-input-reuse.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

using llama_rkva_local_input_reuse::enabled;
using llama_rkva_local_input_reuse::mode;
using llama_rkva_local_input_reuse::mode_from_value;

static const std::int64_t kQueries[] = { 1, 2, 9, 10, 128, 256, 512 };

static void check_legacy_off_values() {
    // Unset, empty, "0", and unknown values must keep the original OFF policy
    // for every remote setting and every batch size.
    const char * values[] = { nullptr, "", "0", "false", "true", "2", "decode ", "DECODE", "1x", "01" };
    for (const char * value : values) {
        CHECK(mode_from_value(value) == mode::off);
        for (bool remote : { false, true }) {
            for (std::int64_t queries : kQueries) {
                CHECK(!enabled(value, remote, queries));
            }
        }
    }
}

static void check_legacy_one() {
    CHECK(mode_from_value("1") == mode::all);
    for (bool remote : { false, true }) {
        for (std::int64_t queries : kQueries) {
            CHECK(enabled("1", remote, queries) == remote);
        }
    }
}

static void check_decode_n1() {
    CHECK(mode_from_value("decode") == mode::decode);
    for (bool remote : { false, true }) {
        for (std::int64_t queries : kQueries) {
            CHECK(enabled("decode", remote, queries) == (remote && queries == 1));
        }
    }
    // N1 is exact; neighboring small counts must not opt in.
    CHECK(enabled("decode", true, 1));
    CHECK(!enabled("decode", true, 0));
    CHECK(!enabled("decode", true, 2));
    CHECK(!enabled("decode", true, 9));
}

static void check_wavefront_parent_queries() {
    // Wavefront tile views must consult the parent batch, never the subview
    // size. A prefill parent that happens to contain a one-token tile keeps
    // the decode policy OFF when the parent count is passed.
    auto view_is_cacheable = [](const char * value, bool remote,
            std::int64_t parent_queries, std::int64_t subview_queries) {
        (void) subview_queries; // deliberately ignored by the policy input
        return enabled(value, remote, parent_queries);
    };
    CHECK(!view_is_cacheable("decode", true, 128, 1));
    CHECK(!view_is_cacheable("decode", true, 256, 1));
    for (std::int64_t parent_queries : { std::int64_t(64), std::int64_t(128), std::int64_t(256),
            std::int64_t(512), std::int64_t(1024) }) {
        CHECK(!enabled("decode", true, parent_queries));
    }
    // The legacy all-batch mode is unchanged at this layer; the call sites keep
    // their existing 512-query four-tile guard, which the pure policy does not
    // replace.
    CHECK(enabled("1", true, 128));
    CHECK(enabled("1", true, 1024));
    CHECK(!enabled("1", false, 1));
    // A real N1 parent (which is also a one-token tile) opts in.
    CHECK(view_is_cacheable("decode", true, 1, 1));
}

int main() {
    check_legacy_off_values();
    check_legacy_one();
    check_decode_n1();
    check_wavefront_parent_queries();
    std::puts("rkva local input reuse policy: PASS (unset/0/unknown/1/decode, remote off/on, N1/prefill)");
    return 0;
}
