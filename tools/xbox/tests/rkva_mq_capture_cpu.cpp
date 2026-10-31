#include "../uwp/rkva/rkva-mq-capture.h"
#include <cassert>
#include <cstring>
#include <sstream>

int main() {
    using namespace rkva_mq_capture;
    const Selection sel{7, 128, 3};
    assert(sel.matches(7, 128, 3));
    assert(!sel.matches(8, 128, 3) && !sel.matches(7, 129, 3) && !sel.matches(7, 128, 2));
    Extents e{};
    assert(extents(20, 40, 80, 12, e) && e.total == 152 && e.partial == 80);
    assert(extents(20, 40, 0, 12, e) && e.total == 72 && e.partial == 0);
    assert(!extents(0, 40, 0, 12, e));
    assert(!extents(max_bytes, 1, 0, 12, e));
    assert(!extents(UINT64_MAX, 1, 0, 12, e));
    assert(extents(4, max_bytes - 8, 0, 4, e) && e.total == max_bytes);
    std::ostringstream out(std::ios::binary);
    assert(write(out, "a", 1, "bc", 2));
    const auto bytes = out.str();
    static const unsigned char sha256_abc[] = {
        0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
        0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
    assert(bytes.size() == 35 && bytes.substr(0, 3) == "abc");
    assert(std::memcmp(bytes.data() + 3, sha256_abc, sizeof(sha256_abc)) == 0);
    assert(!write(out, "a", 1, nullptr, 2));
}
