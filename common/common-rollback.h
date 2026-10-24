#pragma once

#include "llama.h"

// Cleanup policy for a refused memory-sequence removal. A removed range needs
// no cleanup; an already empty full sequence and a full clear (p0<0, p1<0) are
// tolerated so a failed compute/poisoned backend cannot abort the server.
// Partial removals keep the hard failure (programming/state error).
inline bool common_seq_rm_cleanup_tolerated(bool removed, bool was_empty_full) {
    // A refused total/partial clear must propagate; only a successful removal
    // or an already-empty full sequence is safe to tolerate.
    return removed || was_empty_full;
}
