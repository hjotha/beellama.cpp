#pragma once
#include <cstdint>

// A checkpoint is captured before the verification anchor. Replaying it may
// therefore remove anchor+draft rows even when the rejected draft count is N.
// Keep the first native MTP contract safe for both RS and checkpoint paths.
inline bool llama_native_kvarn_mtp_depth_supported(uint32_t drafts) {
    return uint64_t(drafts) + 1 <= 128;
}
