// Generate a small CPU-only Qwen3.5 fixture with the 256-wide attention heads
// required by the native KVarN peer. No GPU or external service is needed.
#define main llama_archs_existing_main
#include "test-llama-archs.cpp"
#undef main

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto gguf = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    gguf_set_val_u32(gguf.get(), "qwen35.context_length", 512);
    gguf_set_val_u32(gguf.get(), "qwen35.attention.head_count", 1);
    gguf_set_val_u32(gguf.get(), "qwen35.attention.head_count_kv", 1);
    gguf_set_val_u32(gguf.get(), "qwen35.attention.key_length", 256);
    gguf_set_val_u32(gguf.get(), "qwen35.attention.value_length", 256);
    gguf_set_val_u32(gguf.get(), "qwen35.attention.key_length_swa", 256);
    gguf_set_val_u32(gguf.get(), "qwen35.attention.value_length_swa", 256);
    auto model = get_model_and_ctx(gguf.get(), nullptr, 123, 0.05f, {});
    llama_model_save_to_file(model.first.get(), argv[1]);
    return 0;
}
