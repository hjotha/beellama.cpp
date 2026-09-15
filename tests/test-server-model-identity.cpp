#include "common.h"
#include "server-model-identity.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "hash/hash.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>

#ifndef _WIN32
#include <unistd.h>
#endif

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "identity check failed at %d: %s\n", __LINE__, #expr); std::abort(); } } while (0)

static void refused(const std::function<void()> & action) {
    try { action(); } catch (const std::exception & e) {
        fprintf(stderr, "identity refusal: %s\n", e.what());
        return;
    }
    CHECK(false);
}

static llama_model_kv_override integer(const char * key, int64_t value) {
    llama_model_kv_override result = {};
    result.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    std::snprintf(result.key, sizeof(result.key), "%s", key);
    result.val_i64 = value;
    return result;
}

static void write_shards(const std::filesystem::path & model, const std::filesystem::path & first,
        const std::filesystem::path & second) {
    ggml_context * raw = nullptr;
    gguf_context_ptr original(gguf_init_from_file(model.string().c_str(), {false, &raw}));
    ggml_context_ptr tensors(raw);
    CHECK(original && tensors);
    gguf_context_ptr shards[] = {gguf_context_ptr(gguf_init_empty()), gguf_context_ptr(gguf_init_empty())};
    for (uint16_t i = 0; i < 2; ++i) {
        gguf_set_kv(shards[i].get(), original.get());
        gguf_set_val_u16(shards[i].get(), "split.count", 2);
        gguf_set_val_u16(shards[i].get(), "split.no", i);
        gguf_set_val_i32(shards[i].get(), "split.tensors.count", gguf_get_n_tensors(original.get()));
    }
    for (int64_t index = 0; index < gguf_get_n_tensors(original.get()); ++index) {
        auto * tensor = ggml_get_tensor(tensors.get(), gguf_get_tensor_name(original.get(), index));
        CHECK(tensor);
        gguf_add_tensor(shards[index%2].get(), tensor);
    }
    CHECK(gguf_write_to_file(shards[0].get(), first.string().c_str(), false));
    CHECK(gguf_write_to_file(shards[1].get(), second.string().c_str(), false));
}

static void change_last_byte(const std::filesystem::path & path) {
    std::unique_ptr<FILE, int (*)(FILE *)> file(ggml_fopen(path.string().c_str(), "r+b"), &std::fclose);
    CHECK(file && std::fseek(file.get(), -1, SEEK_END) == 0);
    const int byte = std::fgetc(file.get());
    CHECK(byte != EOF && std::fseek(file.get(), -1, SEEK_END) == 0 && std::fputc(byte ^ 1, file.get()) != EOF);
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    (void) argc; (void) argv;
    refused([] { server_model_identity::prepare("unsupported.gguf", {}); });
    return 0;
#else
    CHECK(argc == 2);
    namespace fs = std::filesystem;
    std::string pattern = (fs::current_path()/"llama-identity-XXXXXX").string();
    std::vector<char> folder(pattern.begin(), pattern.end()); folder.push_back(0);
    CHECK(mkdtemp(folder.data()));
    const fs::path dir(folder.data());
    // This directory was exclusively created by this test; no external files are removed.
    struct cleanup { fs::path path; ~cleanup() { std::error_code ec; fs::remove_all(path, ec); } } owned{dir};
    const auto model = dir/"model.gguf";
    fs::copy_file(argv[1], model);
    const auto identity = server_model_identity::prepare(model.string(), {});
    CHECK(identity.source_count() == 1 && identity.fingerprint().size() == 64);
    identity.verify_sources();

    const auto copy = dir/"same-bytes.gguf";
    fs::copy_file(model, copy);
    CHECK(identity.fingerprint() == server_model_identity::prepare(copy.string(), {}).fingerprint());
    fs::create_symlink(model.filename(), dir/"alias.gguf");
    const auto alias = server_model_identity::prepare((dir/"alias.gguf").string(), {});
    CHECK(alias.fingerprint() == identity.fingerprint());

    auto rope = integer("gemma2.rope.freq_base", 10000);
    auto norm = integer("gemma2.context_length", 512);
    CHECK(server_model_identity::prepare(model.string(), {rope, norm}).fingerprint() ==
          server_model_identity::prepare(model.string(), {norm, rope, integer(rope.key, 20000)}).fingerprint());
    const auto changed_override = server_model_identity::prepare(model.string(), {integer(norm.key, 1024)});
    CHECK(changed_override.fingerprint() != identity.fingerprint());
    CHECK(server_model_identity::prepare(model.string(), {llama_model_kv_override{}, norm}).fingerprint() == identity.fingerprint());
    auto different_type = norm; different_type.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT; different_type.val_f64 = 512;
    CHECK(server_model_identity::prepare(model.string(), {norm}).fingerprint() !=
          server_model_identity::prepare(model.string(), {different_type}).fingerprint());

    // The exact pre-load identity survives a real model load; no second hash is required.
    llama_backend_init();
    common_params params;
    params.model.path = model.string(); params.n_ctx = 512; params.n_batch = params.n_ubatch = 32;
    params.n_gpu_layers = 0; params.fit_params = false; params.warmup = false;
    params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 2;
    auto loaded = common_init_from_params(params);
    CHECK(loaded && loaded->model());
    identity.verify_sources();
    // Replace pathname entries, never modify a GGUF backing a live mapping.
    fs::rename(model, dir/"original.gguf");
    fs::rename(copy, model);
    refused([&] { identity.verify_sources(); });
    loaded.reset();

    const auto current = server_model_identity::prepare(model.string(), {});
    CHECK(current.fingerprint() == identity.fingerprint());
    params.kv_overrides = {integer("gemma2.context_length", 1024), llama_model_kv_override{}};
    const auto semantic = server_model_identity::prepare(model.string(), params.kv_overrides);
    CHECK(semantic.fingerprint() != current.fingerprint());
    loaded = common_init_from_params(params);
    CHECK(loaded && llama_model_n_ctx_train(loaded->model()) == 1024);
    semantic.verify_sources();
    loaded.reset(); params.kv_overrides.clear();
    fs::remove(dir/"alias.gguf"); fs::create_symlink("original.gguf", dir/"alias.gguf");
    refused([&] { alias.verify_sources(); });
    fs::remove(dir/"alias.gguf"); fs::create_symlink("model.gguf", dir/"alias.gguf");
    refused([&] { alias.verify_sources(); }); // Same destination again, different alias object.
    fs::copy_file(model, dir/"different.gguf");
    change_last_byte(dir/"different.gguf");
    CHECK(server_model_identity::prepare((dir/"different.gguf").string(), {}).fingerprint() != current.fingerprint());

    fs::create_directory(dir/"real"); fs::create_directory(dir/"visible");
    const auto first = dir/"real"/"weights-00001-of-00002.gguf";
    const auto second = dir/"real"/"weights-00002-of-00002.gguf";
    write_shards(model, first, second);
    const auto split = server_model_identity::prepare(first.string(), {});
    CHECK(split.source_count() == 2);
    params.model.path = first.string(); loaded = common_init_from_params(params);
    CHECK(loaded && loaded->model()); split.verify_sources(); loaded.reset();
    const auto visible_first = dir/"visible"/"named-00001-of-00002.gguf";
    fs::create_symlink(first, visible_first);
    // Canonical-first discovery would wrongly find real/weights-00002 here.
    refused([&] { server_model_identity::prepare(visible_first.string(), {}); });
    fs::create_symlink(second, dir/"visible"/"named-00002-of-00002.gguf");
    CHECK(server_model_identity::prepare(visible_first.string(), {}).fingerprint() == split.fingerprint());
    refused([&] { server_model_identity::prepare(second.string(), {}); });
    const auto count_one = server_model_identity::prepare(first.string(), {integer("split.count", 1), integer("split.count", 2)});
    CHECK(count_one.source_count() == 1);
    auto wrong_count_type = integer("split.count", 0); wrong_count_type.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL; wrong_count_type.val_bool = false;
    CHECK(server_model_identity::prepare(first.string(), {wrong_count_type}).source_count() == 2);
    fs::rename(second, dir/"second-original.gguf");
    fs::copy_file(dir/"second-original.gguf", second);
    change_last_byte(second);
    refused([&] { split.verify_sources(); });
    CHECK(server_model_identity::prepare(first.string(), {}).fingerprint() != split.fingerprint());

    std::unique_ptr<FILE, int (*)(FILE *)> file(std::tmpfile(), &std::fclose);
    CHECK(file && std::fwrite("abc", 1, 3, file.get()) == 3 && std::fseek(file.get(), 0, SEEK_SET) == 0);
    CHECK(hash_sha256_hex(file.get()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    llama_backend_free();
    fprintf(stderr, "identity: real single/sharded loads, SHA256 vector, path aliases, typed first-wins/sentinel overrides, source replacement/retarget and original-name shard discovery passed\n");
    return 0;
#endif
}
