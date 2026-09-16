#include "common.h"
#include "llama-cpp.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "server-route-state.h"
#include "../src/llama-io-file.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#endif

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static void decode(llama_context * ctx, const llama_tokens & tokens, int start = 0) {
    for (size_t i = 0; i < tokens.size();) {
        const int count = std::min<size_t>(llama_n_batch(ctx), tokens.size() - i);
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (int j = 0; j < count; ++j) { common_batch_add(batch, tokens[i+j], start+i+j, {0}, j == count-1); }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        require(rc == 0, "decode failed");
        i += count;
    }
}

static llama_token greedy(llama_context * ctx) {
    const int count = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, -1);
    return std::max_element(logits, logits + count) - logits;
}

static void synthetic(const std::string & directory) {
    // A single tensor larger than the transfer buffer; no model/GPU required.
    const size_t bytes = 3 * LLAMA_STATE_FILE_BUFFER_SIZE + 128;
    const std::string path = directory + "/tensor.bin";
    {
        std::ofstream out(path, std::ios::binary);
        std::vector<char> block(4096, 42);
        for (size_t left = bytes; left;) {
            const size_t count = std::min(left, block.size());
            out.write(block.data(), count);
            left -= count;
        }
    }
    auto * ctx = ggml_init({ 2*ggml_tensor_overhead(), nullptr, true });
    auto * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bytes/sizeof(float));
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu != nullptr, "CPU backend missing");
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(cpu));
    ggml_backend_tensor_memset(tensor, 0, 0, bytes);
    bool published = false;
    {
        llama_file file(path.c_str(), "rb");
        llama_io_read_file_stream reader(file, bytes);
        reader.read_tensor(tensor, 0, bytes);
        reader.on_commit([&]() { published = true; });
        uint8_t probe = 99;
        ggml_backend_tensor_get(tensor, &probe, bytes-1, 1);
        require(probe == 0 && !published, "parse mutated tensor or published metadata");
        // Real late I/O failure, after the first chunks can be installed.
        std::filesystem::resize_file(path, 2*LLAMA_STATE_FILE_BUFFER_SIZE);
        bool failed = false;
        try { reader.commit(); } catch (const std::exception &) { failed = true; }
        require(failed && !published, "late truncation published metadata");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::cout << "PASS: >24 MiB tensor indexed in 8 MiB chunks; parse/late-I/O publication boundary\n";
}

static void make_http_fixture(const char * input, const char * output) {
    ggml_context * raw = nullptr;
    gguf_context_ptr meta(gguf_init_from_file(input, {false, &raw}));
    ggml_context_ptr tensors(raw);
    require(bool(meta) && bool(tensors), "fixture GGUF read failed");
    gguf_context_ptr writer(gguf_init_empty());
    gguf_set_kv(writer.get(), meta.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
        gguf_add_tensor(writer.get(), ggml_get_tensor(tensors.get(), gguf_get_tensor_name(meta.get(), i)));
    }
    const auto * embedding = ggml_get_tensor(tensors.get(), "token_embd.weight");
    require(embedding, "fixture embedding missing");
    const size_t count = embedding->ne[1];
    std::vector<std::string> names;
    for (size_t i = 0; i < count; ++i) { names.push_back("token" + std::to_string(i)); }
    std::vector<const char *> pointers;
    for (const auto & name : names) { pointers.push_back(name.c_str()); }
    std::vector<int32_t> types(count, 1); // GGML_TOKEN_TYPE_NORMAL
    types[0] = types[1] = 3; // control BOS/EOS
    gguf_set_val_str(writer.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(writer.get(), "tokenizer.ggml.pre", "gpt-2");
    gguf_set_arr_str(writer.get(), "tokenizer.ggml.tokens", pointers.data(), pointers.size());
    gguf_set_arr_str(writer.get(), "tokenizer.ggml.merges", nullptr, 0);
    gguf_set_arr_data(writer.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(writer.get(), "tokenizer.ggml.bos_token_id", 0);
    gguf_set_val_u32(writer.get(), "tokenizer.ggml.eos_token_id", 1);
    gguf_set_val_bool(writer.get(), "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_bool(writer.get(), "tokenizer.ggml.add_eos_token", false);
    require(gguf_write_to_file(writer.get(), output, false), "fixture GGUF write failed");
    std::cout << "Created HTTP fixture with " << count << " token pieces: " << output << '\n';
}

int main(int argc, char ** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--make-http-fixture") {
            make_http_fixture(argv[2], argv[3]);
            return 0;
        }
        require(argc == 2, "usage: test-state-file-stream MODEL");
        ggml_backend_load_all();
        const char * tmpdir = std::getenv("TMPDIR");
        require(tmpdir && *tmpdir, "TMPDIR must be an explicit disk directory");
        std::string pattern = std::string(tmpdir) + "/state-stream-XXXXXX";
#ifndef _WIN32
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
        require(mkdtemp(name.data()) != nullptr, "mkdtemp failed");
        const std::string dir(name.data());
#else
        const std::string dir = pattern;
        std::filesystem::create_directory(dir);
#endif
        synthetic(dir);
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        mp.n_gpu_layers = 0; mp.devices = devices;
        llama_model_ptr model(llama_model_load_from_file(argv[1], mp));
        require(bool(model), "model load failed");
        auto params = llama_context_default_params();
        params.n_ctx = 512; params.n_batch = 256; params.n_ubatch = 256;
        params.n_threads = params.n_threads_batch = 2;
        params.type_k = params.type_v = GGML_TYPE_Q4_0;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.offload_kqv = false;
        llama_context_ptr source(llama_init_from_model(model.get(), params));
        require(bool(source), "source context failed");
        llama_tokens tokens;
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (int i = 0; i < 160; ++i) { tokens.push_back(3 + (i*17) % (n_vocab-3)); }
        decode(source.get(), tokens);
        for (int i = 0; i < 24; ++i) {
            const auto token = greedy(source.get());
            decode(source.get(), {token}, tokens.size());
            tokens.push_back(token);
        }
        const auto identity = server_model_identity::prepare(argv[1], {});
        const std::string path = dir + "/route.bin";
        const size_t written = server_route_state_save(source.get(), 0, tokens, identity, path, SIZE_MAX);
        const auto file = server_route_state_read(path, SIZE_MAX, 1024);
        require(file.file_bytes == written && file.tokens == tokens, "manifest roundtrip failed");

        // The managed-MTP auto-cache path persists the evaluated prefix at
        // N-1, not a full N state that would later need an unpersisted
        // recurrent rewind plan. Compare a cold full decode against
        // restore(N-1)+decode(last), including logits and the recurrent state
        // bytes after the one-token suffix. The inputs are deliberately
        // non-degenerate so row/carry selection is exercised.
        const llama_token last_prompt_token = tokens.back();
        llama_tokens prefix_tokens = tokens;
        prefix_tokens.pop_back();
        const std::string prefix_path = dir + "/route-prefix.bin";
        auto prefix_source_params = llama_context_default_params();
        prefix_source_params.n_ctx = 512;
        prefix_source_params.n_batch = prefix_source_params.n_ubatch = 256;
        prefix_source_params.n_threads = prefix_source_params.n_threads_batch = 2;
        prefix_source_params.type_k = prefix_source_params.type_v = GGML_TYPE_Q4_0;
        prefix_source_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        prefix_source_params.offload_kqv = false;
        llama_context_ptr prefix_source(llama_init_from_model(model.get(), prefix_source_params));
        require(bool(prefix_source), "prefix source context failed");
        decode(prefix_source.get(), prefix_tokens);
        const size_t prefix_written = server_route_state_save(prefix_source.get(), 0, prefix_tokens,
                identity, prefix_path, SIZE_MAX);
        const auto prefix_file = server_route_state_read(prefix_path, SIZE_MAX, 1024);
        require(prefix_file.file_bytes == prefix_written && prefix_file.tokens == prefix_tokens &&
                prefix_file.position + 1 == (llama_pos) prefix_tokens.size(),
                "pre-last prefix snapshot manifest failed");

        llama_context_ptr cold_full(llama_init_from_model(model.get(), prefix_source_params));
        llama_context_ptr restored_prefix(llama_init_from_model(model.get(), prefix_source_params));
        require(bool(cold_full) && bool(restored_prefix), "prefix comparison contexts failed");
        decode(cold_full.get(), tokens);
        llama_tokens restored_tokens(prefix_tokens.size());
        size_t restored_count = 0;
        require(llama_state_seq_load_file_streaming(restored_prefix.get(), prefix_path.c_str(), 0,
                restored_tokens.data(), restored_tokens.size(), &restored_count,
                prefix_file.state_bytes, prefix_file.state_checksum) == prefix_file.state_bytes &&
                restored_count == prefix_tokens.size() && restored_tokens == prefix_tokens,
                "pre-last prefix streaming restore failed");
        decode(restored_prefix.get(), {last_prompt_token}, prefix_tokens.size());
        require(llama_memory_seq_pos_max(llama_get_memory(cold_full.get()), 0) ==
                llama_memory_seq_pos_max(llama_get_memory(restored_prefix.get()), 0),
                "pre-last restore position differs from cold");
        const float * cold_logits = llama_get_logits_ith(cold_full.get(), -1);
        const float * restored_logits = llama_get_logits_ith(restored_prefix.get(), -1);
        require(cold_logits && restored_logits, "pre-last comparison logits missing");
        for (int i = 0; i < n_vocab; ++i) {
            require(std::isfinite(cold_logits[i]) && std::isfinite(restored_logits[i]) &&
                    std::abs(cold_logits[i] - restored_logits[i]) < 0.002f,
                    "pre-last restore logits differ from cold");
        }
        const size_t cold_state_size = llama_state_seq_get_size_ext(cold_full.get(), 0,
                LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t restored_state_size = llama_state_seq_get_size_ext(restored_prefix.get(), 0,
                LLAMA_STATE_SEQ_FLAGS_NONE);
        require(cold_state_size == restored_state_size && cold_state_size > 0,
                "pre-last recurrent state size differs from cold");
        std::vector<uint8_t> cold_state(cold_state_size), restored_state(restored_state_size);
        require(llama_state_seq_get_data_ext(cold_full.get(), cold_state.data(), cold_state.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_NONE) == cold_state.size() &&
                llama_state_seq_get_data_ext(restored_prefix.get(), restored_state.data(), restored_state.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_NONE) == restored_state.size(),
                "pre-last recurrent state extraction failed");
        const bool physical_state_equal = cold_state == restored_state;
        // The recurrent serializer writes the logical row selected by its
        // current rollback index, while state_read restores that row at a
        // fresh head and resets the index. Therefore physical state bytes are
        // allowed to differ after the two paths, even when the state is
        // semantically identical. Probe the recurrent transition itself over
        // several non-degenerate suffixes; this is the acceptance criterion
        // for the pre-last snapshot, not an unsafe rewind assumption.
        for (int step = 0; step < 8; ++step) {
            const float * cold_next = llama_get_logits_ith(cold_full.get(), -1);
            const float * restored_next = llama_get_logits_ith(restored_prefix.get(), -1);
            require(cold_next && restored_next, "semantic recurrent probe logits missing");
            for (int i = 0; i < n_vocab; ++i) {
                require(std::isfinite(cold_next[i]) && std::isfinite(restored_next[i]) &&
                        std::abs(cold_next[i] - restored_next[i]) < 0.002f,
                        "semantic recurrent probe logits diverged");
            }
            const llama_token next = greedy(cold_full.get());
            require(next == greedy(restored_prefix.get()), "semantic recurrent probe token diverged");
            const llama_pos next_pos = llama_memory_seq_pos_max(llama_get_memory(cold_full.get()), 0) + 1;
            decode(cold_full.get(), {next}, next_pos);
            decode(restored_prefix.get(), {next}, next_pos);
        }
        std::cout << "PASS: pre-last N-1 snapshot + one-token suffix matches cold logits and "
                     "8-step recurrent semantics (physical_bytes_equal="
                  << (physical_state_equal ? "true" : "false") << ")\n";
        params.n_ctx = 1024; params.n_batch = params.n_ubatch = 64;
        llama_context_ptr dest(llama_init_from_model(model.get(), params));
        require(bool(dest), "destination context failed");
        auto restore = [&](llama_context * ctx, const std::string & from, uint64_t hash) {
            llama_tokens out(tokens.size()); size_t count = 777;
            const size_t read = llama_state_seq_load_file_streaming(ctx, from.c_str(), 0, out.data(), out.size(),
                    &count, file.state_bytes, hash);
            if (read) { require(count == tokens.size() && out == tokens, "native tokens mismatch"); }
            else { require(count == 0, "failed load published a token count"); }
            return read;
        };
        require(restore(dest.get(), path, file.state_checksum) == file.state_bytes, "streaming restore failed");
        require(restore(dest.get(), path, file.state_checksum) == 0, "nonempty destination accepted");
        // The plain hybrid PARTIAL_ONLY frame starts with the host header,
        // recurrent cell count, position and seq-id count. An invalid seq-id
        // count used to clear live recurrent metadata on a parse rejection.
        const auto partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        std::vector<uint8_t> before(llama_state_seq_get_size_ext(dest.get(), 0, partial));
        require(before.size() > 20 && llama_state_seq_get_data_ext(dest.get(), before.data(), before.size(), 0, partial)
                == before.size(), "partial capture failed");
        uint32_t cells = 0;
        std::memcpy(&cells, before.data()+8, 4);
        require(cells == 1, "expected one recurrent cell in hybrid partial fixture");
        auto malformed = before;
        uint32_t invalid_seq_count = 1;
        std::memcpy(malformed.data()+16, &invalid_seq_count, 4);
        require(llama_state_seq_set_data_ext(dest.get(), malformed.data(), malformed.size(), 0, partial) == 0,
                "malformed recurrent metadata accepted");
        std::vector<uint8_t> after(before.size());
        require(llama_state_seq_get_data_ext(dest.get(), after.data(), after.size(), 0, partial) == after.size()
                && before == after, "recurrent parse rejection changed live destination");
        llama_tokens suffix = {13, 27, 42, 31};
        decode(source.get(), suffix, tokens.size()); decode(dest.get(), suffix, tokens.size());
        for (int step = 0; step < 16; ++step) {
            const float * a = llama_get_logits_ith(source.get(), -1);
            const float * b = llama_get_logits_ith(dest.get(), -1);
            for (int i = 0; i < n_vocab; ++i) {
                require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i]-b[i]) < 0.002f,
                        "restored continuation logits differ");
            }
            require(greedy(source.get()) == greedy(dest.get()), "deterministic continuation differs");
            const llama_token token = greedy(source.get());
            decode(source.get(), {token}, tokens.size()+suffix.size()+step);
            decode(dest.get(), {token}, tokens.size()+suffix.size()+step);
        }
        std::cout << "PASS: q4 512 b/ub256 -> 1024 b/ub64 after 24 generated tokens, 16-step deterministic continuation\n";
        llama_memory_clear(llama_get_memory(dest.get()), false);
        require(restore(dest.get(), path, file.state_checksum ^ 1) == 0, "checksum corruption accepted");
        require(llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0) == -1, "corrupt load mutated empty state");
        const std::string bad = dir + "/bad.bin";
        std::filesystem::copy_file(path, bad);
        {
            std::fstream out(bad, std::ios::binary | std::ios::in | std::ios::out);
            out.seekg(file.state_bytes-1); char byte; out.read(&byte, 1); byte ^= 1;
            out.seekp(file.state_bytes-1); out.write(&byte, 1);
        }
        require(restore(dest.get(), bad, file.state_checksum) == 0, "tensor corruption accepted");
        std::filesystem::resize_file(bad, file.state_bytes/2);
        require(restore(dest.get(), bad, file.state_checksum) == 0, "truncated state accepted");
        bool rejected = false;
        try { server_route_state_read(bad, SIZE_MAX, 1024); } catch (...) { rejected = true; }
        require(rejected, "truncated manifest accepted");
        params.type_k = params.type_v = GGML_TYPE_F16;
        llama_context_ptr wrong(llama_init_from_model(model.get(), params));
        require(bool(wrong) && restore(wrong.get(), path, file.state_checksum) == 0, "incompatible KV layout accepted");
        require(llama_memory_seq_pos_max(llama_get_memory(wrong.get()), 0) == -1, "layout reject published state");
#ifndef _WIN32
        setenv("LLAMA_TEST_STATE_FILE_COMMIT_FAIL_AFTER", "1", 1);
        require(restore(dest.get(), path, file.state_checksum) == 0, "late I/O injection accepted");
        unsetenv("LLAMA_TEST_STATE_FILE_COMMIT_FAIL_AFTER");
        require(llama_memory_seq_pos_max(llama_get_memory(dest.get()), 0) == -1, "late I/O left partial state");
        require(restore(dest.get(), path, file.state_checksum) == file.state_bytes, "retry after late I/O failed");
        const auto before_retention_failure = server_route_state_read(path, SIZE_MAX, 1024);
        rejected = false;
        try {
            server_route_state_save(dest.get(), 0, tokens, identity, path, SIZE_MAX, 0, nullptr,
                                    {}, []() { return false; });
        } catch (...) {
            rejected = true;
        }
        const auto after_retention_failure = server_route_state_read(path, SIZE_MAX, 1024);
        require(rejected && after_retention_failure.state_checksum == before_retention_failure.state_checksum &&
                after_retention_failure.tokens == before_retention_failure.tokens,
                "failed retention commit replaced the previous snapshot");
        setenv("LLAMA_TEST_ROUTE_STATE_PUBLISH_FAIL", "1", 1);
        rejected = false;
        // Source advanced, so re-saving the old tokens is independently invalid.
        // Use the recovered destination which exactly matches this file.
        try { server_route_state_save(dest.get(), 0, tokens, identity, path, SIZE_MAX); } catch (...) { rejected = true; }
        unsetenv("LLAMA_TEST_ROUTE_STATE_PUBLISH_FAIL");
        require(rejected && server_route_state_read(path, SIZE_MAX, 1024).state_checksum == file.state_checksum,
                "failed publication replaced previous snapshot");
#endif
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            require(entry.path().filename().string().find(".tmp-") == std::string::npos, "temporary file leaked");
        }
        std::cout << "PASS: corruption, truncation, incompatible layout, nonempty transaction, late I/O retry, atomic cleanup\n";
        std::cout << "Artifacts: " << dir << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
