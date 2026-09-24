#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-stream.h"
#include "server-route-state.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "http.h"
#include "subproc.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>
#include <limits>

#ifndef _WIN32
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

// note: SIGPIPE is ignored by the server
static void request_child_exit(server_subproc & proc) {
    FILE * stdin_file = proc.sproc.stdin_file();
    if (stdin_file) {
        fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
        fflush(stdin_file);
    }
}

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

// single-threaded, watching all child processes at once
struct server_monitor {
    server_monitor(server_models & models) : models(models) {
        th = std::thread([this]() { run(); });
    }

    ~server_monitor() {
        push({ cmd_t::QUIT, {}, "", 0, false });
        th.join();
    }

    // thread-safe
    void watch(const std::string & name, std::shared_ptr<server_subproc> proc, server_child_mode mode, int port) {
        child_t c;
        c.name = name;
        c.proc = std::move(proc);
        c.mode = mode;
        c.port = port;
        if (!c.proc->has_output()) {
            SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            c.eof = true;
        }
        push({ cmd_t::WATCH, std::move(c), "", 0, false });
    }

    // thread-safe
    void stop(const std::string & name, int stop_timeout, bool send_exit) {
        push({ cmd_t::STOP, {}, name, stop_timeout, send_exit });
    }

private:
    struct child_t {
        std::string name;
        std::shared_ptr<server_subproc> proc;
        server_child_mode mode = SERVER_CHILD_MODE_NORMAL;
        int port = 0;
        std::string buf;      // partial line
        bool eof = false;     // output closed, waiting for the process to be reaped
        int64_t deadline = 0; // force-kill time in ms, 0 when no stop is pending
    };

    struct cmd_t {
        enum { WATCH, STOP, QUIT } type;
        child_t child;
        std::string name;
        int  stop_timeout;
        bool send_exit;
    };

    void push(cmd_t && cmd) {
        {
            std::lock_guard<std::mutex> lk(mu);
            cmds.push_back(std::move(cmd));
        }
        waiter.wake();
    }

    // returns true if the loop should exit
    bool handle_commands() {
        std::deque<cmd_t> batch;
        {
            std::lock_guard<std::mutex> lk(mu);
            batch.swap(cmds);
        }
        for (auto & cmd : batch) {
            switch (cmd.type) {
                case cmd_t::WATCH:
                    children.push_back(std::move(cmd.child));
                    break;
                case cmd_t::STOP:
                    // the newest child with this name is the one the registry knows
                    for (auto it = children.rbegin(); it != children.rend(); ++it) {
                        if (it->name != cmd.name) {
                            continue;
                        }
                        if (cmd.send_exit && !it->eof) {
                            request_child_exit(*it->proc);
                        }
                        it->deadline = ggml_time_ms() + (int64_t) cmd.stop_timeout * 1000;
                        break;
                    }
                    break;
                case cmd_t::QUIT:
                    return true;
            }
        }
        return false;
    }

    // read what the child wrote, forward complete lines
    void read_output(child_t & c) {
        char chunk[4096];
        while (!c.eof) {
            int n = c.proc->read_output(chunk, sizeof(chunk));
            if (n < 0) {
                c.eof = true;
                break;
            }
            if (n == 0) {
                break;
            }
            c.buf.append(chunk, (size_t) n);
            size_t start = 0;
            while (true) {
                size_t nl = c.buf.find('\n', start);
                if (nl == std::string::npos) {
                    break;
                }
                std::string line = c.buf.substr(start, nl + 1 - start);
                start = nl + 1;
                on_line(c, line);
            }
            c.buf.erase(0, start);
            if (c.buf.size() > max_line) {
                c.buf.clear(); // a child that never writes a newline must not grow this without bound
            }
        }
        if (c.eof && !c.buf.empty()) {
            on_line(c, c.buf);
            c.buf.clear();
        }
    }

    void on_line(child_t & c, const std::string & line) {
        if (string_starts_with(line, CMD_CHILD_TO_ROUTER_STATE)) {
            LOG_DBG("[%5d] %s", c.port, line.c_str()); // prevent spamming the log
            models.handle_child_state(c.name, line);
        } else {
            LOG("[%5d] %s", c.port, line.c_str()); // forward log
        }
    }

    void run() {
        while (true) {
            if (handle_commands()) {
                return;
            }

            // wait for output, a wakeup, or the next deadline;
            // a child whose output closed is polled for its exit every 50 ms
            int64_t now     = ggml_time_ms();
            int64_t timeout = -1;
            for (const auto & c : children) {
                if (c.eof) {
                    timeout = timeout < 0 ? 50 : std::min<int64_t>(timeout, 50);
                }
                if (c.deadline) {
                    int64_t d = std::max<int64_t>(0, c.deadline - now);
                    timeout = timeout < 0 ? d : std::min(timeout, d);
                }
            }
            std::vector<server_subproc *> procs;
            std::vector<child_t *>        owners;
            for (auto & c : children) {
                if (!c.eof) {
                    procs.push_back(c.proc.get());
                    owners.push_back(&c);
                }
            }
            std::vector<bool> ready;
            waiter.wait(procs, ready, timeout);
            for (size_t i = 0; i < owners.size(); i++) {
                if (ready[i]) {
                    read_output(*owners[i]);
                }
            }

            // deadlines and exits
            now = ggml_time_ms();
            for (auto it = children.begin(); it != children.end();) {
                if (it->deadline && now >= it->deadline && !it->proc->stopped.load(std::memory_order_acquire)) {
                    SRV_WRN("force-killing model instance name=%s after timeout\n", it->name.c_str());
                    it->proc->terminate();
                    it->deadline = 0;
                }
                if (it->eof && !it->proc->is_alive()) {
                    int exit_code = it->proc->join();
                    it->proc->stopped.store(true, std::memory_order_release);
                    models.on_child_exit(it->name, it->proc, it->mode, exit_code);
                    SRV_INF("instance name=%s exited with status %d\n", it->name.c_str(), exit_code);
                    it = children.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    static constexpr size_t max_line = 1024 * 1024;

    server_models & models;
    std::mutex mu;
    std::deque<cmd_t> cmds;
    std::vector<child_t> children; // monitor thread only
    server_subproc::waiter waiter;
    std::thread th;
};

struct server_lru_sched {
    server_lru_sched(server_models & models) : models(models) {}

    bool has_capacity(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return models.base_params.models_max <= 0
            || count_running() < (size_t) models.base_params.models_max;
    }

    // returns "" if no model can be given up
    std::string pick_victim(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        std::string victim;
        int64_t victim_last_used = 0;
        for (const auto & m : models.mapping) {
            // a busy model is mid-request, one still coming up has no request to finish
            if (m.second.req_count != 0 || !m.second.meta.is_ready_or_sleep()) {
                continue;
            }
            // already on its way out, or a queued request wants it
            if (models.stopping_models.count(m.first) || find(m.first)) {
                continue;
            }
            if (victim.empty() || m.second.meta.last_used < victim_last_used) {
                victim           = m.first;
                victim_last_used = m.second.meta.last_used;
            }
        }
        return victim;
    }

    // requests wanting the same model share one entry, so they all need only one slot
    // and all get unblocked by the single load that entry performs
    void join(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (entry_t * e = find(model_id)) {
            e->n_waiters++;
            SRV_INF("request for name=%s joined the queue, %d waiting\n", model_id.c_str(), e->n_waiters);
            return;
        }
        queue.push_back({ model_id, 1, false });
        SRV_INF("models_max reached, request for name=%s queued at position %zu\n",
                model_id.c_str(), queue.size());
    }

    void leave(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                if (--it->n_waiters <= 0) {
                    queue.erase(it); // last one waiting for this model went away
                }
                return;
            }
        }
    }

    // true if it is this model's turn to load, and nobody is loading it yet
    bool try_claim(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (queue.empty() || queue.front().model_id != model_id || queue.front().loading) {
            return false;
        }
        if (!has_capacity(lk)) {
            return false;
        }
        queue.front().loading = true;
        return true;
    }

    // on failure the entry is back in line; on success it stays until its waiters leave,
    // so the model coming up is never picked as a victim before they use it
    void claim_done(std::unique_lock<std::mutex> & lk, const std::string & model_id, bool ok) {
        check_lock(lk);
        if (ok) {
            return;
        }
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                it->loading = false;
                return;
            }
        }
    }

    // evict idle models while queued requests outnumber the slots that are free or being freed
    // caller must hold models.mutex; never blocks, so it is safe from any thread
    void tick(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        if (models.base_params.models_max <= 0 || queue.empty()) {
            return;
        }
        int n_running  = 0;
        int n_stopping = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running()) {
                n_running++;
                if (models.stopping_models.count(m.first)) {
                    n_stopping++;
                }
            }
        }
        int n_needed  = 0;
        int n_claimed = 0; // claimed the slot, but load() has not spawned yet
        for (const auto & e : queue) {
            if (!e.loading) {
                n_needed++;
                continue;
            }
            auto it = models.mapping.find(e.model_id);
            if (it != models.mapping.end() && !it->second.meta.is_running()) {
                n_claimed++;
            }
        }
        int n_free = models.base_params.models_max - n_running + n_stopping - n_claimed;
        while (n_free < n_needed) {
            std::string victim = pick_victim(lk);
            if (victim.empty()) {
                return; // all remaining models are busy, wait for a request to end
            }
            SRV_INF("evicting idle LRU name=%s for a queued request\n", victim.c_str());
            models.request_stop(victim);
            n_free++;
        }
    }

  private:
    struct entry_t {
        std::string model_id;
        int  n_waiters; // requests waiting for this model
        bool loading;   // one of the waiters is doing the load right now
    };

    entry_t * find(const std::string & model_id) {
        for (auto & e : queue) {
            if (e.model_id == model_id) {
                return &e;
            }
        }
        return nullptr;
    }

    void check_lock(std::unique_lock<std::mutex> & lk) {
        GGML_ASSERT(lk.owns_lock() && lk.mutex() == &models.mutex);
    }

    size_t count_running() {
        size_t count = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running()) {
                count++;
            }
        }
        return count;
    }

    server_models & models;
    std::deque<entry_t> queue;
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

// bounds for the router's estimator round-trips to a loaded child. tokenizing an 80k prompt
// takes milliseconds and the child's HTTP thread answers while a generation is in flight, but
// a wedged child must not stall routing
static constexpr int ROUTE_ESTIMATE_TIMEOUT_MS = 5000;

// output reserve assumed when a request omits max_tokens/n_predict and the router itself was
// not started with a positive -n: the deployment contract is 4096 output tokens. never 0,
// otherwise an 80k prompt with an implicit output lands on the capped tier and dies at the
// boundary
static constexpr int64_t ROUTE_DEFAULT_OUTPUT_RESERVE = 4096;

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.remove_sensitive_options();
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

static void set_environment_value(
        std::vector<std::string> & environment,
        const std::string & key,
        const std::string & value) {
    const std::string prefix = key + "=";
    auto matches = [&](const std::string & entry) {
        if (entry.size() < prefix.size()) {
            return false;
        }
#ifdef _WIN32
        return _strnicmp(entry.c_str(), prefix.c_str(), prefix.size()) == 0;
#else
        return entry.compare(0, prefix.size(), prefix) == 0;
#endif
    };

    for (std::string & entry : environment) {
        if (matches(entry)) {
            entry = prefix + value;
            return;
        }
    }
    environment.push_back(prefix + value);
}

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    // a child echoes its own alias back in every response's "model" field. a route-group member
    // must echo the public group name instead: otherwise the tier leaks to the client, and a
    // client that sends response["model"] back on the next turn would address the tier directly
    // and skip routing. the router keeps addressing the child by its own name, this alias only
    // shapes what the child calls itself
    std::string route_group;
    const bool in_group = preset.get_option(COMMON_ARG_PRESET_ROUTE_GROUP, route_group) && !route_group.empty();
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", in_group ? route_group : name);
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    try {
        common_params params;
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_REPO_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params); // note: this won't download the model because offline=true
        if (params.no_mmproj || params.mmproj.path.empty()) {
            multimodal = { false, false };
        } else {
            multimodal = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", e.what());
        multimodal = { false, false };
    }
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              hf_token(params.hf_token),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)),
              sched(std::make_unique<server_lru_sched>(*this)),
              monitor(std::make_unique<server_monitor>(*this)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);
    base_params.hf_token.clear();
    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
    debug_fake_timing = !common_get_env("LLAMA_SERVER_DEBUG_FAKE_TIMING").empty();
}

server_models::~server_models() {
    cleanup_route_state_dir();
}

static bool is_route_member(const server_model_meta & meta) {
    std::string group;
    return meta.preset.get_option(COMMON_ARG_PRESET_ROUTE_GROUP, group) && !group.empty();
}

static std::string route_state_model_identity(const server_model_meta & meta) {
    static constexpr const char * keys[] = {
        "LLAMA_ARG_MODEL",
        "LLAMA_ARG_MODEL_URL",
        "LLAMA_ARG_DOCKER_REPO",
        "LLAMA_ARG_HF_REPO",
        "LLAMA_ARG_HF_FILE",
        "LLAMA_ARG_MMPROJ",
        "LLAMA_ARG_MMPROJ_URL",
    };

    std::string identity;
    bool has_identity = false;
    for (const char * key : keys) {
        std::string value;
        identity += key;
        identity += '=';
        if (meta.preset.get_option(key, value)) {
            identity += value;
            has_identity = true;
        }
        identity += '\n';
    }
    return has_identity ? identity : std::string();
}

static bool route_state_weights_match(const server_model_meta & source, const server_model_meta & target) {
    const std::string source_identity = route_state_model_identity(source);
    return !source_identity.empty() && source_identity == route_state_model_identity(target);
}

static int route_member_rank(const std::vector<route_group_member> & members, const std::string & name) {
    for (size_t i = 0; i < members.size(); ++i) {
        if (members[i].name == name) {
            return (int) i;
        }
    }
    return -1;
}

static std::string route_state_filename(const std::string & conv_id) {
    const size_t first = std::hash<std::string>{}(conv_id);
    const size_t second = std::hash<std::string>{}(std::string("llama-route-state:") + conv_id);
    static std::atomic<uint64_t> nonce{0};
    return "slot-" + std::to_string(first) + "-" + std::to_string(second) + "-" +
        std::to_string(nonce.fetch_add(1, std::memory_order_relaxed)) + ".bin";
}

void server_models::ensure_route_state_dir(const std::string & configured_store) {
    if (!route_state_dir.empty()) {
        std::error_code ec;
        if (std::filesystem::is_directory(route_state_dir, ec)) {
            return;
        }
    }

    std::error_code ec;
    const bool has_persistent_store = !base_params.slot_save_path.empty() || !configured_store.empty();
    const std::filesystem::path configured_path = !base_params.slot_save_path.empty()
        ? std::filesystem::path(base_params.slot_save_path)
        : std::filesystem::path(configured_store);
    if (has_persistent_store) {
        ec.clear();
        std::filesystem::create_directories(configured_path, ec);
        if (!ec && std::filesystem::is_directory(configured_path, ec) && !ec) {
            route_state_dir = configured_path.string();
            route_state_dir_owned = false;
            SRV_INF("unified snapshot store: %s\n", route_state_dir.c_str());
            return;
        }
        SRV_WRN("unable to use persistent snapshot store %s: %s\n", configured_path.string().c_str(), ec.message().c_str());
        ec.clear();
    }

    const std::filesystem::path root = std::filesystem::temp_directory_path(ec);
    if (ec || root.empty()) {
        SRV_WRN("%s\n", "unable to create route state directory: no temporary directory is available");
        return;
    }

    const std::string suffix = std::to_string(ggml_time_ms()) + "-" +
        std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    for (int attempt = 0; attempt < 4; ++attempt) {
        const std::filesystem::path dir = root / ("llama-router-state-" + suffix + "-" + std::to_string(attempt));
        ec.clear();
        if (!std::filesystem::create_directory(dir, ec)) {
            if (!ec) {
                continue;
            }
            break;
        }

        std::error_code perm_ec;
        std::filesystem::permissions(
            dir,
            std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace,
            perm_ec);
        if (perm_ec) {
            SRV_WRN("unable to restrict route state directory permissions: %s\n", perm_ec.message().c_str());
        }
        route_state_dir = dir.string();
        route_state_dir_owned = true;
        SRV_INF("route state directory: %s\n", route_state_dir.c_str());
        return;
    }

    SRV_WRN("unable to create route state directory under %s: %s\n", root.string().c_str(), ec.message().c_str());
}

void server_models::cleanup_route_state_dir() {
    std::lock_guard<std::mutex> route_lock(route_mutex);
    if (route_state_dir_owned && !route_state_dir.empty()) {
        std::error_code ec;
        for (std::filesystem::directory_iterator it(route_state_dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.find(".tmp") != std::string::npos ||
                    (name.size() >= 6 && name.compare(name.size() - 6, 6, ".lease") == 0)) {
                std::filesystem::remove(it->path(), ec);
                ec.clear();
            }
        }
        if (ec) {
            SRV_WRN("unable to clean temporary route state files in %s: %s\n", route_state_dir.c_str(), ec.message().c_str());
        }
    }
    // Published unified snapshots are deliberately retained. They belong to
    // the persistent store and are reclaimed only by the configured LRU.
    route_state_dir.clear();
    route_state_dir_owned = false;
    route_states.clear();
}

void server_models::render_child_args(server_model_meta & meta) {
    meta.update_args(ctx_preset, bin_path);
    if (!is_route_member(meta)) {
        return;
    }

    std::string configured_store;
    for (size_t i = 0; i + 1 < meta.args.size(); ++i) {
        if (meta.args[i] == "--slot-save-path") {
            configured_store = meta.args[i + 1];
            break;
        }
    }
    ensure_route_state_dir(configured_store);

    // Group children must expose the native slot action endpoint. POSIX route
    // handoff keeps the child's configured slot-save path intact: --slot-save-
    // auto owns that persistent store, while route_target_no_mtp selects
    // route_state_dir via the private environment variable below.
#ifndef _WIN32
    if (configured_store.empty() && !route_state_dir.empty()) {
        // Preserve the historical public /slots contract for route members
        // that have no explicit store, without overriding an explicit
        // persistent --slot-save-path.
        meta.args.push_back("--slot-save-path");
        std::string child_slot_path = route_state_dir;
        if (child_slot_path.back() != std::filesystem::path::preferred_separator) {
            child_slot_path += std::filesystem::path::preferred_separator;
        }
        meta.args.push_back(std::move(child_slot_path));
    }
    if (std::find(meta.args.begin(), meta.args.end(), "--slots") == meta.args.end()) {
        meta.args.push_back("--slots");
    }
#else
    // The native temporary/publication path is POSIX-only. Preserve the old
    // Windows route contract by pointing the child slot store at route_state_dir.
    if (route_state_dir.empty()) {
        SRV_WRN("route member %s has no slot state directory; KV transfer is disabled\n", meta.name.c_str());
        return;
    }
    for (auto it = meta.args.begin(); it != meta.args.end();) {
        if (*it == "--slot-save-path") {
            it = meta.args.erase(it);
            if (it != meta.args.end()) {
                it = meta.args.erase(it);
            }
        } else {
            ++it;
        }
    }
    if (std::find(meta.args.begin(), meta.args.end(), "--slots") == meta.args.end()) {
        meta.args.push_back("--slots");
    }
    meta.args.push_back("--slot-save-path");
    std::string child_slot_path = route_state_dir;
    if (child_slot_path.empty() || child_slot_path.back() != std::filesystem::path::preferred_separator) {
        child_slot_path += std::filesystem::path::preferred_separator;
    }
    meta.args.push_back(std::move(child_slot_path));
#endif
    if (route_state_dir.empty()) {
        SRV_WRN("route member %s has no slot state directory; KV transfer is disabled\n", meta.name.c_str());
    }
}

std::optional<json> server_models::route_slot_action(
        const server_model_meta & meta,
        const char * action,
        int id_slot,
        const std::string & filename,
        bool target_no_mtp,
        bool reuse) {
    if (meta.port <= 0 || route_state_dir.empty()) {
        return std::nullopt;
    }

    httplib::Client cli(CHILD_ADDR, meta.port);
    if (base_params.timeout_read > 0) {
        cli.set_connection_timeout(base_params.timeout_read, 0);
        cli.set_write_timeout(base_params.timeout_read, 0);
    }
    if (base_params.timeout_write > 0) {
        cli.set_read_timeout(base_params.timeout_write, 0);
    }

    const std::string path = "/slots/" + std::to_string(id_slot) + "?action=" + action;
    json body_json = {{"filename", filename}};
#ifndef _WIN32
    body_json["route_state_transfer"] = true;
    if (target_no_mtp) {
        body_json["route_target_no_mtp"] = true;
    }
    if (reuse) {
        body_json["route_state_reuse"] = true;
    }
#else
    GGML_UNUSED(target_no_mtp);
    GGML_UNUSED(reuse);
#endif
    const std::string body = body_json.dump();
    auto response = cli.Post(path, body, "application/json");
    if (!response) {
        SRV_WRN("route slot %s failed for model %s: no response\n", action, meta.name.c_str());
        return std::nullopt;
    }
    if (response->status != 200) {
        SRV_WRN("route slot %s failed for model %s: HTTP %d\n", action, meta.name.c_str(), response->status);
        return std::nullopt;
    }

    try {
        json result = json::parse(response->body);
        const char * count_key = std::strcmp(action, "save") == 0 ? "n_saved" : "n_restored";
        if (!result.is_object() || !result.contains(count_key) || !result.at(count_key).is_number_integer()) {
            SRV_WRN("route slot %s returned an invalid response for model %s\n", action, meta.name.c_str());
            return std::nullopt;
        }
        return result;
    } catch (const std::exception & e) {
        SRV_WRN("route slot %s returned invalid JSON for model %s: %s\n", action, meta.name.c_str(), e.what());
        return std::nullopt;
    }
}

bool server_models::save_route_state(
        const std::string & group,
        const std::string & conv_id,
        const std::string & target,
        int id_slot) {
    if (group.empty() || conv_id.empty() || target.empty()) {
        return false;
    }

    const auto pinned = conv_models.lookup(conv_id);
    if (!pinned.has_value() || *pinned == target) {
        return false;
    }

    const auto groups = get_route_groups();
    const auto group_it = groups.find(group);
    if (group_it == groups.end()) {
        return false;
    }
    const int source_rank = route_member_rank(group_it->second, *pinned);
    const int target_rank = route_member_rank(group_it->second, target);
    if (source_rank < 0 || target_rank <= source_rank) {
        return false;
    }

    const auto source_meta = get_meta(*pinned);
    const auto target_meta = get_meta(target);
    if (!source_meta.has_value() || !target_meta.has_value() || !source_meta->is_ready_or_sleep()) {
        return false;
    }
    if (!route_state_weights_match(*source_meta, *target_meta)) {
        SRV_WRN("route state transfer disabled for %s -> %s: model identity differs\n",
                pinned->c_str(), target.c_str());
        return false;
    }
    const std::string model_identity = route_state_model_identity(*source_meta);

    ensure_route_state_dir();
    if (route_state_dir.empty()) {
        return false;
    }

    const std::string filename = route_state_filename(conv_id);
    const std::string filepath = (std::filesystem::path(route_state_dir) / filename).string();
    // Reserve the reference stripe before the child publishes. This blocks
    // eviction without blocking the child's separate publication lock.
    auto lease = std::make_shared<server_route_state_lease>(filepath,
            server_route_state_lock_mode::reservation);
    if (!lease->acquired()) {
        SRV_WRN("route state save deferred for conversation %s: snapshot is already in use\n", conv_id.c_str());
        return false;
    }
    bool target_no_mtp = true;
    for (size_t i = 0; i + 1 < target_meta->args.size(); ++i) {
        if ((target_meta->args[i] == "--spec-type" && target_meta->args[i + 1] != "none") ||
                (target_meta->args[i] == "--ctx-size-mtp" && target_meta->args[i + 1] != "0")) {
            target_no_mtp = false;
        }
    }
#ifdef _WIN32
    // Content identity/atomic temporary-file support is currently POSIX-only.
    // Keep conventional Windows route groups on their existing file contract.
    target_no_mtp = false;
#endif
    SRV_INF("route state hand-off %s -> %s: target_no_mtp=%d\n",
            pinned->c_str(), target.c_str(), target_no_mtp ? 1 : 0);
    const auto result = route_slot_action(*source_meta, "save", id_slot, filename, target_no_mtp, true);
    if (!result.has_value()) {
        return false;
    }

    const int64_t n_tokens = result->at("n_saved").get<int64_t>();
    if (n_tokens <= 0) { return false; }
    std::string effective_filename = filename;
    if (result->contains("filename") && result->at("filename").is_string()) {
        effective_filename = result->at("filename").get<std::string>();
    }
    if (!fs_validate_filename(effective_filename)) {
        SRV_WRN("route state save returned an invalid snapshot filename for conversation %s\n", conv_id.c_str());
        return false;
    }
    if (effective_filename != filename) {
        lease->release();
        const std::string effective_path = (std::filesystem::path(route_state_dir) / effective_filename).string();
        lease = std::make_shared<server_route_state_lease>(effective_path,
                server_route_state_lock_mode::reservation);
        if (!lease->acquired()) {
            SRV_WRN("route state reference could not be acquired for conversation %s\n", conv_id.c_str());
            return false;
        }
    }
    bool unified = false;
#ifndef _WIN32
    unified = true;
#endif
    route_states[conv_id] = { *pinned, target, model_identity, effective_filename, id_slot, n_tokens,
        target_no_mtp, unified, std::move(lease) };
    SRV_INF("saved route state for conversation %s: %s -> %s, slot=%d, tokens=%lld\n",
            conv_id.c_str(), pinned->c_str(), target.c_str(), id_slot, (long long) n_tokens);
    return true;
}

void server_models::discard_route_state(const std::string & conv_id) {
    auto it = route_states.find(conv_id);
    if (it == route_states.end()) {
        return;
    }
    SRV_DBG("retaining unified route snapshot for conversation %s: %s\n", conv_id.c_str(), it->second.filename.c_str());
    route_states.erase(it);
}

bool server_models::restore_route_state(const std::string & conv_id, const std::string & target) {
    auto it = route_states.find(conv_id);
    if (it == route_states.end()) {
        return false;
    }
    if (it->second.target != target) {
        discard_route_state(conv_id);
        return false;
    }

    const auto target_meta = get_meta(target);
    if (!target_meta.has_value() || !target_meta->is_ready_or_sleep()) {
        return false;
    }

    const route_state state = it->second;
    if (route_state_model_identity(*target_meta) != state.model_identity) {
        SRV_WRN("route state restore skipped for conversation %s: target model identity changed\n", conv_id.c_str());
        discard_route_state(conv_id);
        return false;
    }
    const auto result = route_slot_action(*target_meta, "restore", state.id_slot, state.filename,
            state.target_no_mtp, state.unified);
    if (!result.has_value()) {
        discard_route_state(conv_id);
        return false;
    }

    const int64_t n_tokens = result->at("n_restored").get<int64_t>();
    if (state.n_tokens >= 0 && n_tokens != state.n_tokens) {
        SRV_WRN("route state restore count mismatch for conversation %s: saved=%lld restored=%lld\n",
                conv_id.c_str(), (long long) state.n_tokens, (long long) n_tokens);
        route_slot_action(*target_meta, "erase", state.id_slot, state.filename, state.target_no_mtp);
        discard_route_state(conv_id);
        return false;
    }

    SRV_INF("restored route state for conversation %s: %s, slot=%d, tokens=%lld\n",
            conv_id.c_str(), target.c_str(), state.id_slot, (long long) n_tokens);
    discard_route_state(conv_id);
    return true;
}

void server_models::instance_t::request_exit() const {
    request_child_exit(*subproc);
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    render_child_args(meta);
    meta.update_caps();
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* meta    */ std::move(meta)
    };
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {
    std::lock_guard<std::mutex> reload_lk(reload_mutex);
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
    std::set<std::string> hidden_models;
    {
        std::set<std::string> preset_paths;
        for (const auto & [name, preset] : custom_presets) {
            std::string val;
            if (!preset.get_option(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS, val) || !common_arg_utils::is_truthy(val)) {
                continue;
            }
            std::string hf_repo;
            if (!preset.get_option("LLAMA_ARG_HF_REPO", hf_repo) || hf_repo.empty()) {
                continue;
            }
            std::string hf_file;
            preset.get_option("LLAMA_ARG_HF_FILE", hf_file);
            std::string path = common_download_resolve_path(hf_repo, hf_file);
            if (!path.empty()) {
                preset_paths.insert(path);
            }
        }
        if (!preset_paths.empty()) {
            for (const auto & [name, preset] : cached_models) {
                if (get_source(name) != SERVER_MODEL_SOURCE_CACHE) {
                    continue; // merged with another source, not a pure cache entry
                }
                std::string path = common_download_resolve_path(name);
                if (!path.empty() && preset_paths.count(path)) {
                    SRV_INF("hiding cache model name=%s (deduplicated by a preset)\n", name.c_str());
                    hidden_models.insert(name);
                }
            }
        }
    }

    // Helpers that read `mapping` - must be called while holding the lock.
    std::unordered_set<std::string> custom_names;
    for (const auto & [name, preset] : custom_presets) custom_names.insert(name);
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
            std::string info;
            if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
            if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";
            SRV_INF("  %c %s%s\n", has_custom ? '*' : ' ', name.c_str(), info.c_str());
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    auto apply_hidden = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string group;
            // members of a routing group are hidden from the model list: only the group name is
            // advertised, the tiers are still reachable by their own names
            const bool group_member = inst.meta.preset.get_option(COMMON_ARG_PRESET_ROUTE_GROUP, group) && !group.empty();
            inst.meta.hidden = hidden_models.count(name) > 0 || group_member;
        }
    };

    // build the routing groups from the (already reconciled) mapping. a public name maps to an
    // ordered ladder of members, ascending max_tokens with the uncapped member last, so the
    // first tier whose cap covers a request's budget wins and the uncapped one is the fallback
    auto rebuild_route_groups = [&]() {
        route_groups.clear();
        if (mapping.empty()) {
            return;
        }
        std::unordered_map<std::string, std::map<std::string, int64_t>> raw;
        for (const auto & [name, inst] : mapping) {
            std::string group;
            if (!inst.meta.preset.get_option(COMMON_ARG_PRESET_ROUTE_GROUP, group) || group.empty()) {
                continue;
            }
            if (mapping.find(group) != mapping.end()) {
                SRV_WRN("route group '%s' collides with the model of the same name; requests for '%s' will be served by the group\n",
                        group.c_str(), group.c_str());
            }
            int64_t cap = -1;
            std::string cap_str;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_ROUTE_MAX_TOKENS, cap_str)) {
                try {
                    cap = std::stoll(cap_str);
                    if (cap <= 0) {
                        SRV_WRN("non-positive route-max-tokens value '%s' for model '%s', treating it as an uncapped member\n",
                                cap_str.c_str(), name.c_str());
                        cap = -1;
                    }
                } catch (...) {
                    SRV_WRN("invalid route-max-tokens value '%s' for model '%s', treating it as an uncapped member\n",
                            cap_str.c_str(), name.c_str());
                    cap = -1;
                }
            }
            raw[group][name] = cap;
        }
        for (auto & [group, members] : raw) {
            auto & ladder = route_groups[group];
            for (const auto & [member_name, cap] : members) {
                ladder.push_back({member_name, cap});
            }
            // stable: raw is keyed by member name, so equal caps keep a deterministic (alphabetical)
            // order across reloads instead of whatever the sort happens to produce
            std::stable_sort(ladder.begin(), ladder.end(), [](const route_group_member & a, const route_group_member & b) {
                const int64_t ac = a.max_tokens >= 0 ? a.max_tokens : std::numeric_limits<int64_t>::max();
                const int64_t bc = b.max_tokens >= 0 ? b.max_tokens : std::numeric_limits<int64_t>::max();
                return ac < bc;
            });
            std::string ladder_str;
            int uncapped = 0;
            for (const auto & m : ladder) {
                if (!ladder_str.empty()) ladder_str += " -> ";
                ladder_str += m.name;
                ladder_str += m.max_tokens < 0 ? " (uncapped)" : " (<= " + std::to_string(m.max_tokens) + ")";
                uncapped += m.max_tokens < 0 ? 1 : 0;
            }
            if (uncapped > 1) {
                SRV_WRN("route group '%s' has %d uncapped members; only the last one is reachable, give the others a route-max-tokens\n",
                        group.c_str(), uncapped);
            }
            SRV_INF("route group '%s': %s\n", group.c_str(), ladder_str.c_str());
        }
    };
    // update_args() injects HOST/PORT/ALIAS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally (unload)
    std::unique_lock<std::mutex> lk(mutex);

    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                /* multimodal    */ mtmd_caps{false, false},
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        apply_hidden();
        rebuild_route_groups();
        log_available_models();

        // skipped on reload, see startup_models
        if (startup_models.has_value()) {
            std::vector<std::string> models_to_load;
            for (const auto & [name, inst] : mapping) {
                std::string val;
                if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    models_to_load.push_back(name);
                }
            }
            if ((int)models_to_load.size() > base_params.models_max) {
                throw std::runtime_error(string_format(
                    "number of models to load on startup (%zu) exceeds models_max (%d)",
                    models_to_load.size(), base_params.models_max));
            }

            // to be lazy-loaded after main() setup phase is completed
            startup_models = std::move(models_to_load);
        }

        lk.unlock();
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, safe to erase
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            render_child_args(inst.meta);
            inst.meta.update_caps();
        }

        // add models that are new in this reload, load-on-startup is not honored here since a
        // reload never spawns an instance
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    /* multimodal    */ mtmd_caps{false, false},
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
            }
        }

        apply_stop_timeout();
        apply_hidden();
        rebuild_route_groups();

        // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        lk.unlock();

        notify_sse("models_reload", "*");
    }
}

void server_models::load_startup_models() {
    std::vector<std::string> to_load;
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (!startup_models.has_value()) {
            return; // already drained
        }
        to_load = std::move(*startup_models);
        startup_models.reset();
    }
    for (const auto & name : to_load) {
        SRV_INF("(startup) loading model %s\n", name.c_str());
        load(name);
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::lock_guard<std::mutex> lk(mutex);

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

std::unordered_map<std::string, std::vector<route_group_member>> server_models::get_route_groups() {
    std::lock_guard<std::mutex> lk(mutex);
    return route_groups;
}

bool server_models::is_route_group(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = route_groups.find(name);
    return it != route_groups.end() && !it->second.empty();
}

std::unique_lock<std::mutex> server_models::lock_route_requests() {
    return std::unique_lock<std::mutex>(route_mutex);
}

// does this request carry a prompt whose token budget matters for routing? control, slots,
// detokenize, lora and similar routes have no prompt to measure: they always land on the
// loaded member (or the capped tier when nothing is loaded), regardless of body size
static bool route_request_has_prompt(const std::string & path, const json & body) {
    // metadata routes carry a prompt-shaped body but never generate: tokenizing or counting an
    // 80k text must not boot the wide tier, and every member answers them identically. checked
    // before the field probes because their paths contain the generation paths as substrings
    // (/v1/chat/completions/input_tokens, /v1/messages/count_tokens)
    if (path.find("/tokenize")       != std::string::npos ||   // also /detokenize
        path.find("/apply-template") != std::string::npos ||
        path.find("input_tokens")    != std::string::npos ||
        path.find("count_tokens")    != std::string::npos) {
        return false;
    }
    if (body.contains("messages") || body.contains("prompt") || body.contains("input")) {
        return true;
    }
    return path.find("/chat/completions") != std::string::npos
        || path.find("/responses")     != std::string::npos
        || path.find("/messages")      != std::string::npos // anthropic
        || path.find("/infill")        != std::string::npos
        || path.find("/completions")   != std::string::npos
        || path.find("embedding")      != std::string::npos;
}

// try to learn the exact prompt token count from a loaded child, mirroring the tokenization
// the request itself will later go through on that child: raw prompts go to /tokenize (the
// plain completions path tokenizes with add_special=true; infill also contributes its suffix),
// template-shaped bodies go to the input-token counting endpoints, which apply the chat
// template exactly as the generation would. returns the count, or -1 when the child does not
// answer or the request shape has no countable prompt
static int64_t route_count_prompt_tokens(const std::string & path, const json & body, int port) {
    const bool has_messages = body.contains("messages");
    const bool has_prompt   = body.contains("prompt");

    // Native completions may already contain token IDs. Count them exactly,
    // without trying to tokenize numeric JSON through the text-only endpoint.
    if (has_prompt && body.at("prompt").is_array() && path.find("infill") == std::string::npos) {
        bool token_ids = true;
        for (const auto & token : body.at("prompt")) { token_ids = token_ids && token.is_number_integer(); }
        if (token_ids) { return body.at("prompt").size(); }
    }
    if (port <= 0) { return -1; }

    std::string count_path;
    json count_body = body;
    if (path.find("/chat/completions") != std::string::npos && has_messages) {
        count_path = "/v1/chat/completions/input_tokens";
    } else if (path.find("/responses") != std::string::npos && body.contains("input")) {
        count_path = "/v1/responses/input_tokens";
    } else if (path.find("/messages") != std::string::npos && has_messages) { // anthropic
        count_path = "/v1/messages/count_tokens";
    } else if (has_messages) {
        // template-shaped body on a route the classifier above did not name (apply-template)
        count_path = "/v1/chat/completions/input_tokens";
    } else if (has_prompt || path.find("/infill") != std::string::npos || path.find("/completions") != std::string::npos) {
        if (!has_prompt) {
            return -1;
        }
        count_path = "/tokenize";
        if (path.find("/infill") != std::string::npos && body.contains("suffix")) {
            // the suffix goes into the KV cache too, count both sides of the infill prompt
            count_body = json{{"content", json::array({body.at("prompt"), body.at("suffix")})}, {"add_special", true}};
        } else {
            count_body = json{{"content", body.at("prompt")}, {"add_special", true}};
        }
    } else if (path.find("embedding") != std::string::npos && body.contains("input")) {
        count_path = "/tokenize";
        count_body = json{{"content", body.at("input")}};
    } else {
        return -1;
    }

    try {
        httplib::Client cli(CHILD_ADDR, port);
        cli.set_connection_timeout(0, ROUTE_ESTIMATE_TIMEOUT_MS * 1000);
        cli.set_read_timeout(0, ROUTE_ESTIMATE_TIMEOUT_MS * 1000);
        cli.set_write_timeout(0, ROUTE_ESTIMATE_TIMEOUT_MS * 1000);
        auto resp = cli.Post(count_path, count_body.dump(), "application/json");
        if (!resp || resp->status != 200) {
            return -1;
        }
        const json out = json::parse(resp->body);
        if (count_path == "/tokenize") {
            return out.contains("tokens") && out.at("tokens").is_array()
                ? (int64_t) out.at("tokens").size() : -1;
        }
        return out.is_object() && out.contains("input_tokens") && out.at("input_tokens").is_number_integer()
            ? out.at("input_tokens").get<int64_t>() : -1;
    } catch (const std::exception &) {
        return -1;
    }
}

std::string server_models::resolve_route_target(const std::string & name, const server_http_req & req, const json & body) {
    // snapshot the groups under the lock; everything past this point locks on its own
    std::unordered_map<std::string, std::vector<route_group_member>> groups;
    {
        std::lock_guard<std::mutex> lk(mutex);
        groups = route_groups;
    }
    auto git = groups.find(name);
    if (git == groups.end() || git->second.empty()) {
        return name; // not a routing group: all existing behaviour is preserved
    }
    const std::vector<route_group_member> & members = git->second;

    // a non-object body (e.g. a bare JSON array) is malformed for every proxied route; treat
    // it as a prompt-less request rather than letting the field probes below throw
    const bool prompt_relevant = body.is_object() && route_request_has_prompt(req.path, body);
    if (!prompt_relevant) {
        // body-less or non-prompt requests (GET /props, control, slots, ...) must not get to
        // pick which child boots: go to the loaded member, or the first (capped) tier cold
        for (const auto & m : members) {
            auto meta = get_meta(m.name);
            if (meta.has_value() && meta->is_ready()) {
                return m.name;
            }
        }
        return members.front().name;
    }

    // exact counting needs a loaded child; any member works, they share the same tokenizer
    int counting_port = -1;
    for (const auto & m : members) {
        auto meta = get_meta(m.name);
        if (meta.has_value() && meta->is_ready()) {
            counting_port = meta->port;
            break;
        }
    }
    int64_t prompt_tokens = route_count_prompt_tokens(req.path, body, counting_port);
    if (prompt_tokens < 0) {
        // cold start (no member loaded) or the child did not answer: bytes/token heuristic,
        // biased upward in practice because the raw body carries JSON overhead
        prompt_tokens = (int64_t) (req.body.size() / 3.5);
    }

    // output reserve: what the request asked to generate; when omitted, the router's -n, else
    // the deployment default. never 0 on omission, an implicit output must not be under-budgeted
    int64_t reserve = -1;
    for (const auto & key : {"max_tokens", "max_completion_tokens", "max_output_tokens", "n_predict"}) {
        if (body.contains(key) && body.at(key).is_number_integer()) {
            reserve = body.at(key).get<int64_t>();
            break;
        }
    }
    if (reserve < 0) {
        // an embedding request generates nothing, so reserving output for it would push prompts
        // near the boundary onto the wide tier for no reason
        reserve = req.path.find("embedding") != std::string::npos
            ? 0
            : (base_params.n_predict > 0 ? base_params.n_predict : ROUTE_DEFAULT_OUTPUT_RESERVE);
    }
    const int64_t budget = prompt_tokens + reserve;

    // first tier whose cap covers the budget, else the group's fallback (the last member: the
    // uncapped one, or the largest cap when the config declared none)
    std::string chosen = members.back().name;
    for (size_t i = 0; i < members.size(); i++) {
        const auto & m = members[i];
        if (m.max_tokens < 0 || m.max_tokens >= budget) {
            chosen = m.name;
            break;
        }
    }

    SRV_INF("route group '%s': %lld prompt + %lld output = %lld tokens -> %s\n",
            name.c_str(), (long long) prompt_tokens, (long long) reserve, (long long) budget, chosen.c_str());
    return chosen;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name;
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (sched->has_capacity(lk)) {
            return;
        }
        lru_model_name = sched->pick_victim(lk);
    }
    if (!lru_model_name.empty()) {
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        unload(lru_model_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

void server_models::load(const std::string & name, const load_options & opts) {
    if (debug_fake_timing) {
        // do not hold the mutex here, other requests must keep making progress
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        unload_lru();
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    // Download workers do not use models_max slots.
    if (opts.mode == SERVER_CHILD_MODE_NORMAL && base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw std::runtime_error("model limit reached, try again later");
        }
    }

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        render_child_args(inst.meta);

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));
#ifndef _WIN32
        if (is_route_member(inst.meta)) {
            set_environment_value(child_env, "LLAMA_SERVER_ROUTER_STATE", "1");
            if (!route_state_dir.empty()) {
                set_environment_value(child_env, "LLAMA_SERVER_ROUTER_STATE_DIR", route_state_dir);
            }
        }
#endif
        if (!hf_token.empty()) {
            set_environment_value(child_env, "HF_TOKEN", hf_token);
        }

        if (opts.mode == SERVER_CHILD_MODE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back("LLAMA_SERVER_CHILD_MODE=download");
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        // TODO @ngxson : maybe separate stdout and stderr in the future
        //                so that we can use stdout for commands and stderr for logging
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        if (!inst.subproc->sproc.create(child_args, options, child_env)) {
            throw std::runtime_error("failed to spawn server instance");
        }
    }

    // old process should have exited already, but just in case, we clean it up here
    {
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.subproc && it->second.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            it->second.subproc->terminate(); // force kill
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    inst.req_count = mapping[name].req_count;
    auto proc = inst.subproc;
    int  port = inst.meta.port;
    mapping[name] = std::move(inst);
    monitor->watch(name, proc, opts.mode, port);
    cv.notify_all();
}

void server_models::request_stop(const std::string & name, bool send_exit) {
    auto it = mapping.find(name);
    if (it == mapping.end() || stopping_models.count(name)) {
        return;
    }
    stopping_models.insert(name);
    monitor->stop(name, it->second.meta.stop_timeout, send_exit);
}

void server_models::on_child_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, server_child_mode mode, int exit_code) {
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end() || it->second.subproc != proc) {
            return; // entry erased, or a newer instance took the name
        }
    }
    if (mode == SERVER_CHILD_MODE_DOWNLOAD) {
        // instance will be cleaned up on next load_models() call
        std::lock_guard<std::mutex> lk(mutex);
        stopping_models.erase(name);
        cv.notify_all();
    } else {
        update_status(name, {
            SERVER_MODEL_STATUS_UNLOADED,
            exit_code
        });
    }
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            request_stop(name, !loading);
            // status change will be handled by the monitor
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::unique_lock<std::mutex> lk(mutex);
    for (auto & [name, inst] : mapping) {
        if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            inst.request_exit();
        } else if (inst.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = inst.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                inst.subproc->terminate();
            }
            request_stop(name, !loading);
        }
    }
    // wait for every child to exit, the monitor force-kills the ones that ignore the exit command
    cv.wait(lk, [this]() {
        for (const auto & [name, inst] : mapping) {
            if (inst.meta.is_running() || inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                return false;
            }
        }
        return true;
    });
    cleanup_route_state_dir();
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            stopping_models.erase(name);
        }
        // a model that comes up idle or goes down changes the slot count for queued requests
        sched->tick(lk);
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    cv.notify_all();
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
        if (loading) {
            it->second.subproc->terminate();
        }
        request_stop(name, !loading);
    }

    // wait until the child is gone
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // re-find after wait - load_models() may have erased the entry during the wait
    it = mapping.find(name);
    if (it == mapping.end()) {
        // load_models() already erased the entry; we just need to clean up the cached files on disk
        lk.unlock();
        bool ok = common_download_remove(name);
        SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
        notify_sse("model_remove", name, {});
        return true;
    }

    // remove from disk (best-effort: cancelled downloads may have no cached files)
    bool ok = common_download_remove(name);
    mapping.erase(name);
    if (!ok) {
        SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", name.c_str());
    }
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop) {
    if (!get_meta(name).has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }

    bool queued   = false;
    bool did_load = false;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        const bool stopping = stopping_models.count(name) != 0;
        if (it != mapping.end() && !stopping && it->second.meta.is_ready_or_sleep()) {
            return false; // ready for taking requests
        }
        if (it != mapping.end() && (it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED || stopping)) {
            sched->join(lk, name);
            sched->tick(lk);
            queued = true;
        }
    }

    // while queued, this is also where the load happens: the head of the queue does it
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    std::unique_lock<std::mutex> lk(mutex);
    auto leave_queue = [this, &queued, &lk, &name]() {
        if (queued) {
            sched->leave(lk, name);
            queued = false;
        }
    };

    try {
        bool saw_loading = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                break; // removed by another code path, nothing to wait for
            }
            const server_model_status status = it->second.meta.status;
            const bool stopping = stopping_models.count(name) != 0;

            if (!stopping && (status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING)) {
                break;
            }
            if (status == SERVER_MODEL_STATUS_DOWNLOADING || status == SERVER_MODEL_STATUS_DOWNLOADED) {
                break; // do not wait on a download child
            }
            if (status == SERVER_MODEL_STATUS_LOADING) {
                saw_loading = true;
            } else if (status == SERVER_MODEL_STATUS_UNLOADED) {
                if (did_load || saw_loading) {
                    // a spawn happened and the instance came back down
                    if (it->second.meta.is_failed()) {
                        throw std::runtime_error("model name=" + name + " failed to load");
                    }
                    break; // unloaded by another code path, caller reports "not running"
                }
                if (!queued) {
                    break; // not queued, and the load someone else started fell over
                }
            }

            if (should_stop && should_stop()) {
                // if a model was evicted for us, the free slot goes to the next waiter
                throw std::runtime_error("request cancelled while waiting for model name=" + name);
            }

            // our turn: our model is at the head, and a slot really did free up
            if (status == SERVER_MODEL_STATUS_UNLOADED && sched->try_claim(lk, name)) {
                lk.unlock();
                bool ok = true;
                try {
                    SRV_INF("slot available, loading queued model name=%s\n", name.c_str());
                    load(name);
                    did_load = true;
                } catch (const std::exception & e) {
                    // lost a race for the slot, stay in line and retry
                    SRV_WRN("queued load of name=%s did not go through: %s\n", name.c_str(), e.what());
                    ok = false;
                }
                lk.lock();
                sched->claim_done(lk, name, ok);
                sched->tick(lk);
                continue;
            }

            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    } catch (...) {
        leave_queue();
        sched->tick(lk); // a slot freed for this waiter goes to the next one
        throw;
    }
    leave_queue();

    return true;
}

std::shared_ptr<void> server_models::reserve_request(const std::string & name) {
    auto release = [this, name](void *) { release_request(name); };
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end()) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        it->second.req_count++;
    }
    return std::shared_ptr<void>(this, std::move(release));
}

void server_models::release_request(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end() && it->second.req_count > 0) {
        it->second.req_count--;
        if (it->second.req_count == 0) {
            sched->tick(lk);
        }
    }
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (update_last_used) {
            mapping[name].meta.last_used = ggml_time_ms();
        }
        mapping[name].req_count++;
    }
    if (debug_fake_timing) {
        // sleep after req_count++, so the model counts as busy while we wait here
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            // a detached request belongs to a replay session
            detached
                ? std::function<bool()>([]() { return false; })
                : req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );

    proxy->cleanup = [this, name]() { release_request(name); };

    return proxy;
}

void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_READY:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    payload.size() > 0 ? payload : nullptr,
                    {}, // reset progress info
                });
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}

//
// server_child
//

bool server_child::is_child() {
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}

server_child_mode server_child::get_mode() {
    const char * mode = std::getenv("LLAMA_SERVER_CHILD_MODE");
    std::string mode_str(mode ? mode : "");
    if (mode_str == "download") {
        return SERVER_CHILD_MODE_DOWNLOAD;
    } else {
        return SERVER_CHILD_MODE_NORMAL;
    }
}

struct server_download_state : public common_download_callback {
    server_child * self;
    std::function<bool()> should_stop;
    std::atomic<int64_t> last_progress_time{0}; // multiple files downloading in different threads
    bool is_ok = false;

    server_download_state(server_child * s) : self(s) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        self->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - last_progress_time.load(std::memory_order_relaxed) >= 100) {
            on_progress(p);
            last_progress_time.store(now, std::memory_order_relaxed);
        }
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

int server_child::run_download(common_params & params) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(this);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

std::thread server_child::setup(const std::function<void(int)> & shutdown_handler) {
    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_child::notify_to_router(const std::string & state, const json & payload) {
    json data = {
        {"state", state},
        {"payload", payload},
    };
    std::lock_guard<std::mutex> lk(mtx_stdout);
    common_log_pause(common_log_main());
    fflush(stdout);
    // the router matches the command on a line prefix, so the leading newline
    // closes whatever the logger left open on the shared pipe, down to the
    // trailing color reset that carries no newline of its own
    fprintf(stdout, "\n%s%s\n", CMD_CHILD_TO_ROUTER_STATE, safe_json_to_str(data).c_str());
    fflush(stdout);
    common_log_resume(common_log_main());
}


//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (!models_autoload && !meta->is_running()) {
        res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    if (!common_subproc::is_supported()) {
        throw std::runtime_error("subprocess is not enabled on this build");
    }

    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        std::unique_lock<std::mutex> route_lock;
        if (models.is_route_group(name)) {
            route_lock = models.lock_route_requests();
        }
        // body-less requests resolve to the loaded member of a group, or the capped tier cold
        name = models.resolve_route_target(name, req, json::object());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        auto reservation = models.reserve_request(name);
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop);
        }
        if (route_lock.owns_lock()) {
            route_lock.unlock();
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string requested_name = json_value(body, "model", std::string());
        // A route group must resolve to a child before model validation.
        std::string conv_id = server_stream_conv_id_from_headers(req.headers);
        std::unique_lock<std::mutex> route_lock;
        if (models.is_route_group(requested_name)) {
            route_lock = models.lock_route_requests();
        }
        std::string name = models.resolve_route_target(requested_name, req, body);
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        auto reservation = models.reserve_request(name);
        int id_slot = json_value(body, "id_slot", 0);
        if (id_slot < 0) {
            id_slot = 0;
        }
        const bool state_saved = models.save_route_state(requested_name, conv_id, name, id_slot);
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header. registered before
        // the load wait so a stop issued while the model loads can erase the entry and cancel
        // this request instead of leaving an orphan generation
        uint64_t ticket = models.conv_models.remember(conv_id, name);
        // a dead socket must not cancel a session request, only a stop does (checked right below)
        auto should_stop = ticket == 0 ? req.should_stop : nullptr;
        bool waited = false;
        try {
            waited = autoload && models.ensure_model_ready(name, should_stop);
        } catch (...) {
            if (state_saved) {
                models.discard_route_state(conv_id);
            }
            throw;
        }
        if (ticket != 0 && !models.conv_models.alive(conv_id, ticket)) {
            SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n",
                    conv_id.c_str(), name.c_str());
            if (state_saved) {
                models.discard_route_state(conv_id);
            }
            res_err(error_res, format_error_response(
                    "request cancelled by a stop while the model was loading", ERROR_TYPE_INVALID_REQUEST));
            return error_res;
        }
        if (state_saved) {
            models.restore_route_state(conv_id, name);
        }
        if (route_lock.owns_lock()) {
            route_lock.unlock();
        }
        // a session request that waited for a load detaches from the client socket: the
        // client may have dropped during the wait (page reload) and the session buffer must
        // still receive the generation for a later resume
        return models.proxy_request(req, method, name, true, waited && ticket != 0); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        GGML_UNUSED(req);
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        // the "models" array a child also returns here: Codex reads its context_window from it, so
        // the router must produce it too, for names it has not loaded yet
        json codex_json  = json::array();
        auto all_models = models.get_all_meta();
        // Widest configured adaptive tier, or 0 if the preset declares none.
        // Unloaded children cannot report their runtime metadata yet.
        auto preset_n_ctx = [](const server_model_meta & m) {
            int64_t result = 0;
            for (const char * option : {
                    "LLAMA_ARG_CTX_SIZE",
                    "LLAMA_ARG_CTX_SIZE_MTP_SHORT",
                    "LLAMA_ARG_CTX_SIZE_MTP",
                    "LLAMA_ARG_CTX_SIZE_XLONG",
                    "LLAMA_ARG_CTX_SIZE_XXLONG"}) {
                std::string ctx_size;
                if (!m.preset.get_option(option, ctx_size)) {
                    continue;
                }
                try {
                    result = std::max<int64_t>(result, std::stoll(ctx_size));
                } catch (...) {
                    // Ignore malformed optional metadata; argument parsing will
                    // still reject it if this preset is loaded.
                }
            }
            return result;
        };
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            if (meta.hidden) {
                continue; // cache model deduplicated by a preset
            }
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }

            // pi coding agent multimodal compatibility
            json input_modalities = json::array({"text"});
            if (meta.multimodal.inp_vision) {
                input_modalities.push_back("image");
            }
            if (meta.multimodal.inp_audio) {
                input_modalities.push_back("audio");
            }
            json architecture {
                {"input_modalities",  input_modalities},
                {"output_modalities", json::array({"text"})},
            };

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields, may require reading GGUF metadata
            };

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);

            // a loaded child reports its real n_ctx; before that, the configured one is all we have
            int64_t n_ctx = model_info.value("meta", json::object()).value("n_ctx", (int64_t) 0);
            if (n_ctx <= 0) {
                n_ctx = preset_n_ctx(meta);
            }
            codex_json.push_back(format_codex_model_entry(meta.name, n_ctx,
                meta.multimodal.inp_vision || meta.multimodal.inp_audio));
        }

        // routing groups: the members are hidden above, advertise the group itself so a client
        // that inspects the list sees one entry per public name. the advertised n_ctx is the
        // widest member's configured --ctx-size (what the group can actually serve), read from
        // the preset so it is known before any member loads; omitted if no member declares one
        for (const auto & [group, members] : models.get_route_groups()) {
            json group_info = {
                {"id",            group},
                {"aliases",       json::array()},
                {"tags",          json::array()},
                {"object",        "model"},
                {"owned_by",      "llamacpp"},
                {"created",       t},
                {"status",        json{{"value", "unloaded"}, {"args", json::array()}}},
                {"architecture",  json{{"input_modalities", json::array({"text"})}, {"output_modalities", json::array({"text"})}}},
                {"source",        "route-group"},
                {"can_remove",    false},
                {"meta",          {
                    {"route_members",  json::array()},
                }},
            };
            int64_t group_n_ctx = 0;
            bool group_multimodal = false;
            for (const auto & m : members) {
                auto member_meta = models.get_meta(m.name);
                if (member_meta.has_value()) {
                    group_n_ctx = std::max<int64_t>(group_n_ctx, preset_n_ctx(*member_meta));
                    group_multimodal = group_multimodal
                        || member_meta->multimodal.inp_vision || member_meta->multimodal.inp_audio;
                }
                if (member_meta.has_value() && member_meta->is_running() && member_meta->loaded_info.is_object()) {
                    // mirror the child's own model info (same weights, same shape) like a
                    // running member does; our n_ctx/meta and id already exist, so they win
                    for (auto it = member_meta->loaded_info.begin(); it != member_meta->loaded_info.end(); ++it) {
                        if (!group_info.contains(it.key())) {
                            group_info[it.key()] = it.value();
                        }
                    }
                    group_info["status"]["value"] = "loaded";
                }
                group_info["meta"]["route_members"].push_back({
                    {"name",       m.name},
                    {"max_tokens", m.max_tokens},
                });
            }
            if (group_n_ctx > 0) {
                group_info["meta"]["n_ctx"] = group_n_ctx;
            }
            models_json.push_back(std::move(group_info));
            codex_json.push_back(format_codex_model_entry(group, group_n_ctx, group_multimodal));
        }
        res_ok(res, {
            {"models", codex_json},
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_reload = [this](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        models.load_models();
        res_ok(res, {{"success", true}});
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        common_params p;
        p.model.hf_repo  = name;
        p.hf_token       = models.hf_token;

        // validate by fetching metadata
        bool ok = false;
        try {
            common_models_handler_init(p, LLAMA_EXAMPLE_SERVER);
            ok = true;
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        if (!ok) {
            throw std::invalid_argument("model validation failed, unable to download");
        }

        // reject if model already exists
        if (models.has_model(name)) {
            throw std::invalid_argument("model '" + name + "' already exists");
        }

        // then, proceed with the actual download
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.mode = SERVER_CHILD_MODE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            // a registered conv whose model is still loading earns a retry: the session appears
            // once the load ends and the pending request reaches the child
            auto tracked = models.conv_models.lookup(conv_id);
            auto meta = tracked.has_value() ? models.get_meta(*tracked) : std::nullopt;
            bool transient = meta.has_value() && (meta->status == SERVER_MODEL_STATUS_LOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADED);
            if (transient) {
                res_err(res, format_error_response("Stream owner model is loading, retry later", ERROR_TYPE_UNAVAILABLE));
            } else {
                res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            }
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "&from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else if (auto tracked = models.conv_models.lookup(conv_id); tracked.has_value()) {
            // the entry exists but its model is still loading: the forget below erases it,
            // which cancels the request parked in proxy_post before the generation starts
            SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
                    conv_id.c_str(), tracked->c_str());
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}



//
// server_http_proxy
//

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & scheme,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::map<std::string, uploaded_file> & files,
        const std::function<bool()> should_stop,
        int32_t timeout_read,
        int32_t timeout_write
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<server_pipe<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup_pipes = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // build the header message forwarded to the reader thread, stripping internal proxy headers
    auto make_header_msg = [](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return msg;
    };

    // true once response_handler has already forwarded the headers
    auto headers_sent = std::make_shared<std::atomic<bool>>(false);

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, headers_sent, make_header_msg](const httplib::Response & response) {
        headers_sent->store(true);
        return pipe->write(make_header_msg(response)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse_no_throw(body);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                const std::string url_host = common_http_format_host(host);
                req.set_header(key, is_default_port ? url_host : url_host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req, headers_sent, make_header_msg]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        } else if (!headers_sent->load()) {
            // httplib skips response_handler for bodyless statuses like 204, send headers here instead
            pipe->write(make_header_msg(*result));
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
