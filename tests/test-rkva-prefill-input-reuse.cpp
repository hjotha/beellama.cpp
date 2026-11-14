#include "../src/llama-rkva-local-input-reuse.h"

#include <cstdlib>
#include <initializer_list>

int main() {
    const char * values[] = {nullptr, "", "0", "prefill ", "decode"};
    for (const char * value : values) {
        if (llama_rkva_local_input_reuse(value, true, 256)) return EXIT_FAILURE;
    }
    for (int n : {1, 2, 256, 512, 1024}) {
        if (!llama_rkva_local_input_reuse("1", true, n)) return EXIT_FAILURE;
        if (llama_rkva_local_input_reuse("prefill", true, n) != (n > 1)) return EXIT_FAILURE;
        if (llama_rkva_local_input_reuse("prefill", false, n)) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
