#include "../ggml/src/ggml-remote-attn/rkva-managed-auth.h"
#undef NDEBUG
#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace rkva_managed_auth;
using namespace rkva_registry_wire;

// Test fixture only: not durable and must never be used as a runtime guard.
struct memory_fixture final : durable_nonce_guard {
    std::mutex mutex;
    std::set<nonce> seen;
    size_t calls = 0;
    bool unavailable = false, throws = false;
    nonce_result record(const id &, const nonce & n) override {
        std::lock_guard<std::mutex> lock(mutex); ++calls;
        if (throws) throw std::runtime_error("injected");
        if (unavailable) return nonce_result::unavailable;
        return seen.insert(n).second ? nonce_result::recorded : nonce_result::replay;
    }
};
static void vector_test(const std::vector<uint8_t> & key, const std::string & data, const char * expected) {
    digest out{};
    assert(hmac_sha256(key.data(), key.size(), reinterpret_cast<const uint8_t *>(data.data()), data.size(), out));
    assert(rkva_snapshot::hex(out.data()) == expected);
}
int main() {
    // RFC 4231 test cases 1, 2 and 6, including a key longer than the SHA-256 block.
    vector_test(std::vector<uint8_t>(20, 0x0b), "Hi There", "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    vector_test(std::vector<uint8_t>{'J','e','f','e'}, "what do ya want for nothing?", "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    vector_test(std::vector<uint8_t>(131, 0xaa), "Test Using Larger Than Block-Size Key - Hash Key First", "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    digest d{}; assert(!hmac_sha256(nullptr, 1, nullptr, 0, d)); assert(!hmac_sha256(nullptr, 0, nullptr, 1, d));
    secret key{}; key.fill(0x9a); id owner{}, tx{}, ref{}, object{};
    owner.fill(0x12); tx.fill(0x34); ref.fill(0x56); object.fill(0x78);
    request q{owner, tx, ref, object, 0}, decoded{};
    nonce n{}; n.fill(0xab);
    rkva_request h{}; h.magic = RKVA_MAGIC; h.version = RKVA_VERSION; h.op = rkva_registry_wire::op::prepare;
    h.cycle_id = 0x123456789abcdef0ull; h.session_id = RKVA_SESSION_NONE; h.payload_bytes = rkva_managed_auth::request_bytes;
    wire signed_request{}; assert(encode(key, h, q, n, signed_request));
    // Cross-checked with Python hmac/hashlib and struct little-endian packing.
    assert(rkva_snapshot::hex(signed_request.data() + rkva_registry_wire::request_bytes + 32) ==
           "14d524c0823d2acded4e3fef30ed1aa4b3859605e940386d0dcc72cf6d63f31c");
    memory_fixture guard;
    // Every single payload byte is authenticated; even a structurally invalid
    // mutation must not consume a nonce or publish parsed output.
    for (size_t i = 0; i < signed_request.size(); ++i) {
        auto altered = signed_request; altered[i] ^= 1;
        request unchanged{}; unchanged.tx.fill(0xed);
        assert(verify(key, owner, h, altered.data(), altered.size(), guard, unchanged) == result::invalid);
        assert(unchanged.tx[0] == 0xed && guard.calls == 0);
    }
    for (size_t nbytes = 0; nbytes < signed_request.size(); ++nbytes)
        assert(verify(key, owner, h, signed_request.data(), nbytes, guard, decoded) == result::invalid);
    assert(verify(key, owner, h, nullptr, signed_request.size(), guard, decoded) == result::invalid);
    auto wrong_owner = owner; wrong_owner[0] ^= 1;
    assert(verify(key, wrong_owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::invalid);
    auto wrong_key = key; wrong_key[0] ^= 1;
    assert(verify(wrong_key, owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::invalid);
    const auto rejected_header = [&](rkva_request changed) {
        return verify(key, owner, changed, signed_request.data(), signed_request.size(), guard, decoded) == result::invalid;
    };
    auto altered_header = h; altered_header.cycle_id++; assert(rejected_header(altered_header));
    altered_header = h; altered_header.version++; assert(rejected_header(altered_header));
    altered_header = h; altered_header.magic++; assert(rejected_header(altered_header));
    altered_header = h; altered_header.op = rkva_registry_wire::op::commit; assert(rejected_header(altered_header));
    altered_header = h; altered_header.session_id = 0; assert(rejected_header(altered_header));
    altered_header = h; altered_header.layer_id = 1; assert(rejected_header(altered_header));
    altered_header = h; altered_header.n_tokens = 1; assert(rejected_header(altered_header));
    altered_header = h; altered_header.payload_bytes--; assert(rejected_header(altered_header));
    assert(guard.calls == 0);
    assert(verify(key, owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::accepted);
    assert(decoded.owner == owner && decoded.tx == tx && decoded.reference == ref && decoded.key == object);
    assert(verify(key, owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::replay);
    // A retry uses a fresh nonce; the underlying tx remains unchanged.
    n[0] ^= 1; assert(encode(key, h, q, n, signed_request));
    guard.unavailable = true; assert(verify(key, owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::unavailable);
    guard.unavailable = false; guard.throws = true; assert(verify(key, owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::unavailable);
    guard.throws = false;
    // Simultaneous verification is accepted at most once if the guard is atomic.
    std::array<result, 8> results{}; std::vector<std::thread> threads;
    for (size_t i = 0; i < results.size(); ++i) threads.emplace_back([&,i] {
        request local{}; results[i] = verify(key, owner, h, signed_request.data(), signed_request.size(), guard, local);
    });
    for (auto & t : threads) t.join();
    assert(std::count(results.begin(), results.end(), result::accepted) == 1);
    assert(std::count(results.begin(), results.end(), result::replay) == 7);
    nonce zero{}; assert(!encode(key, h, q, zero, signed_request));
    secret zero_key{}; assert(!encode(zero_key, h, q, n, signed_request));
    // RELEASE authenticates its nonzero version too.
    h.op = rkva_registry_wire::op::release; q.tx = {}; q.key = {}; q.committed_version = 7; n[1] ^= 1;
    assert(encode(key, h, q, n, signed_request));
    auto version_flip = signed_request; version_flip[136] ^= 1;
    const auto before = guard.calls;
    assert(verify(key, owner, h, version_flip.data(), version_flip.size(), guard, decoded) == result::invalid);
    assert(guard.calls == before);
    assert(verify(key, owner, h, signed_request.data(), signed_request.size(), guard, decoded) == result::accepted);
    assert(decoded.committed_version == 7);
    std::cout << "managed authentication contract tests passed (fixture replay guard is not durable)\n";
}
