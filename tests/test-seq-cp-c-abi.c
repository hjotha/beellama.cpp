#include "llama.h"

// Called by the CPU memory fixture so the checked API crosses a real C frame.
enum llama_memory_seq_cp_result test_seq_cp_from_c(
        llama_memory_t mem, llama_seq_id source, llama_seq_id destination, llama_pos p0, llama_pos p1) {
    return llama_memory_seq_cp_checked(mem, source, destination, p0, p1);
}

void test_seq_cp_legacy_from_c(
        llama_memory_t mem, llama_seq_id source, llama_seq_id destination, llama_pos p0, llama_pos p1) {
    llama_memory_seq_cp(mem, source, destination, p0, p1);
}
