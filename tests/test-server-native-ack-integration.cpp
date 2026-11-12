// Server + session-isolated RKVA peer. The peer can close after RESET but
// before its ACK; this uses no Xbox, production service or sleeps.
#ifndef _WIN32
#define main remote_range_existing_main
#include "test-remote-attn-range.cpp"
#undef main
#include "server-context.h"
#include "arg.h"

static void test_require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

static server_task_result_ptr receive(server_response_reader & reader) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
    auto result = reader.next([&] { return std::chrono::steady_clock::now() >= deadline; });
    test_require(bool(result), "timed out waiting for a server result");
    return result;
}

static server_task completion(server_response_reader & reader, int slot, int predicted,
                              int variant = -1, bool cache = false, bool suffix = false) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = reader.get_new_id();
    task.id_slot = slot;
    llama_tokens tokens(257);
    const size_t salt = size_t(variant < 0 ? slot*11 : variant);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = llama_token(1 + (i + salt) % 100);
    if (suffix) tokens.push_back(101);
    task.tokens = server_tokens(tokens, false);
    task.params.n_predict = predicted;
    task.params.sampling.temp = 0;
    task.params.sampling.seed = 42;
    task.params.sampling.ignore_eos = true;
    task.params.cache_prompt = cache;
    return task;
}

static bool slot_is_processing(server_context & server, int id) {
    auto reader = server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_SLOT_GET);
    task.id = reader.get_new_id();
    reader.post_task(std::move(task));
    auto result = receive(reader);
    auto * slots = dynamic_cast<server_task_result_slots *>(result.get());
    test_require(slots && slots->slots_data.is_array() && slots->slots_data.size() == 2, "invalid slots result");
    for (const auto & slot : slots->slots_data) {
        if (slot.at("id").get<int>() == id) return slot.at("is_processing").get<bool>();
    }
    throw std::runtime_error("missing slot in slots result");
}

static void run_case(const char * model, bool drop_ack, bool ram_restore) {
    char log_name[] = "/tmp/beellama-native-ack-XXXXXX";
    const int log_fd = mkstemp(log_name);
    test_require(log_fd >= 0, "cannot create test log");
    close(log_fd);
    struct log_guard {
        const char * path;
        ~log_guard() { common_log_flush(common_log_main()); common_log_set_file(common_log_main(), nullptr); unlink(path); }
    } log_cleanup{log_name};
    common_log_set_file(common_log_main(), log_name);
    storage_env native("kvarn4");
    storage_env producer("cpu", "GGML_RKVA_NATIVE_PRODUCER");
    storage_env experimental("1", "GGML_RKVA_NATIVE_KVARN4_EXPERIMENTAL");
    storage_env companion("1", "GGML_RKVA_NATIVE_COMPANION");
    storage_env snapshot("1", "GGML_RKVA_DISK_SNAPSHOTS");
    storage_env bulk("0", "GGML_RKVA_NATIVE_BULK");
    storage_env wire(nullptr, "GGML_RKVA_WIRE_FORMAT");
    storage_env bulk_wire(nullptr, "GGML_RKVA_NATIVE_BULK_WIRE");
    storage_env async("0", "GGML_RKVA_ASYNC_LOCAL_REMOTE");
    storage_env fp8("0", "GGML_RKVA_CUDA_FP8");
    storage_env tail("128", "GGML_RKVA_NATIVE_TAIL_TOKENS");
    storage_env wavefront("0", "GGML_RKVA_WAVEFRONT_TILE");
    loopback peer(false, true, false, false, false, false, 0, 0, true, 3u | 4u | 512u);
    peer.isolate_native_sessions = true;

    common_params params;
    const char * gpu_option = std::getenv("GGML_NATIVE_ACK_TEST_GPU");
    const bool use_gpu = gpu_option && std::strcmp(gpu_option, "1") == 0;
    std::vector<std::string> arguments = {
        "native-ack-test", "-m", model, "--parallel", "2", "--ctx-size", "1024",
        "--batch-size", "128", "--ubatch-size", "128", "--threads", "1", "--gpu-layers", use_gpu ? "99" : "0",
        "--fit", "off", "--no-warmup", "--no-context-shift", "--no-cache-idle-slots",
        "--ctx-checkpoints", "0", "--cache-ram", "128", "--slot-prompt-similarity", "0.1", "--cache-type-k", "kvarn4",
        "--cache-type-v", "kvarn4", "--remote-attn", "127.0.0.1:" + std::to_string(peer.port),
        "--remote-attn-layers", "full", "--kv-position-split-p", "256",
        "--remote-attn-format", "kvarn4", "--rkva-disk-snapshots" };
    if (use_gpu) { arguments.push_back("--device"); arguments.push_back("CUDA0"); }
    std::vector<char *> argv;
    for (auto & arg : arguments) argv.push_back(arg.data());
    test_require(common_params_parse(int(argv.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER), "server argument parse failed");
    common_log_set_verbosity_thold(4);
    server_context server;
    test_require(server.load_model(params), "CPU server model load failed");
    std::exception_ptr worker_error;
    std::thread worker([&] { try { server.start_loop(); } catch (...) { worker_error = std::current_exception(); } });
    struct shutdown_guard {
        loopback & peer; server_context & server; std::thread & worker;
        ~shutdown_guard() { peer.unpause(); server.terminate(); if (worker.joinable()) worker.join(); }
    } shutdown{peer, server, worker};

    auto first = server.get_response_reader();
    first.post_task(completion(first, 0, 1, 0, ram_restore));
    auto first_result = receive(first);
    test_require(!first_result->is_error() && first_result->id_slot == 0, "slot 0 did not establish its overflow owner");

    if (ram_restore) {
        auto replacement = server.get_response_reader();
        replacement.post_task(completion(replacement, 0, 1, 11, true));
        test_require(!receive(replacement)->is_error(), "could not establish B after caching A");
    }

    peer.pause_session_filter = 1;
    peer.set_pause(true);
    auto second = server.get_response_reader();
    second.post_task(completion(second, 1, 16, 31));
    {
        std::unique_lock<std::mutex> lock(peer.pause_mutex);
        test_require(peer.pause_cv.wait_for(lock, std::chrono::seconds(45), [&] { return peer.paused_replies > 0; }),
                     "slot 1 did not reach the peer attention barrier");
        test_require(peer.paused_session == 1, "paused attention did not belong to slot 1");
    }
    test_require(slot_is_processing(server, 1), "slot 1 was not active when the reset was queued");
    const uint64_t reset_before = peer.slot0_reset_count.load();
    auto action = server.get_response_reader();
    if (ram_restore) {
        action.post_task(completion(action, 0, 1, 0, true, true), true);
    } else {
        server_task erase_task(SERVER_TASK_TYPE_SLOT_ERASE);
        erase_task.id = action.get_new_id();
        erase_task.slot_action.id_slot = 0;
        action.post_task(std::move(erase_task), true);
    }
    if (ram_restore && drop_ack) peer.drop_slot0_reset_at = reset_before + 2;
    else peer.drop_reset_ack = drop_ack;
    peer.grant_reply(); // allow slot 1's current attention, keeping the next one paused
    if (ram_restore && !drop_ack) {
        std::unique_lock<std::mutex> lock(peer.pause_mutex);
        test_require(peer.pause_cv.wait_for(lock, std::chrono::seconds(45), [&] {
            return peer.slot0_reset_count.load() >= reset_before + 2;
        }), "RAM restore did not reach its second RESET");
        lock.unlock();
        test_require(slot_is_processing(server, 1), "other slot stopped during control RAM restore");
        peer.unpause();
    }
    auto action_result = receive(action);
    test_require(peer.last_reset_session.load() == 0, "fault did not target slot 0 RESET");
    if (drop_ack) {
        test_require(action_result->is_error(), "lost RESET ACK was reported as a successful action");
        if (ram_restore) {
            common_log_flush(common_log_main());
            std::ifstream input(log_name);
            const std::string log((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            test_require(log.find("prompt cache load: native RAM transaction invalidated context") != std::string::npos,
                         "lost ACK did not occur inside the native RAM restore transaction");
        }
        {
            std::lock_guard<std::mutex> lock(peer.mutex);
            const auto other = peer.native_sessions.find(1);
            test_require(peer.native_active_session == 0 && !peer.native_counts.empty() && other != peer.native_sessions.end() &&
                    !other->second.counts.empty() &&
                    std::all_of(other->second.counts.begin(), other->second.counts.end(), [](uint32_t rows) { return rows > 0; }) &&
                    std::all_of(peer.native_counts.begin(), peer.native_counts.end(), [](uint32_t rows) { return rows == 0; }),
                    "peer did not retain the other session while applying slot 0 RESET");
        }
        auto second_result = receive(second);
        test_require(second_result->is_error(), "active slot 1 was not cancelled after shared transport failure");
        auto retry = server.get_response_reader();
        retry.post_task(completion(retry, 0, 1));
        test_require(receive(retry)->is_error(), "unavailable shared context accepted another inference");
    } else {
        test_require(!action_result->is_error(), "control RESET failed");
        if (ram_restore) {
            const auto * result = dynamic_cast<const server_task_result_cmpl_final *>(action_result.get());
            test_require(result && result->stats.cache_source == "ram" && result->n_prompt_tokens_cache > 256,
                         "control did not reuse the overflow prefix from RAM");
        }
        if (!ram_restore) test_require(slot_is_processing(server, 1), "control RESET interrupted the other slot");
        peer.unpause();
        test_require(!receive(second)->is_error(), "control continuation failed");
    }
    peer.unpause();
    server.terminate();
    worker.join();
    if (worker_error) std::rethrow_exception(worker_error);
}

int main(int argc, char ** argv) {
    try {
        test_require(argc == 2, "expected generated Qwen3.5 model path");
        common_init();
        ggml_backend_load_all();
        llama_backend_init();
        run_case(argv[1], false, false);
        run_case(argv[1], true, false);
        run_case(argv[1], false, true);
        run_case(argv[1], true, true);
        llama_backend_free();
        std::puts("PASS: erase and RAM-restore control/lost RESET ACK with an active second slot");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: native ACK integration: %s\n", e.what());
        return 1;
    }
}
#else
int main() { return 0; }
#endif
