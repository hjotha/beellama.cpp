#include "arg.h"

#include "build-info.h"
#include "common.h"
#include "gguf.h"
#include "log.h"
#include "download.h"
#include "file-lock.h"
#include "hf-cache.h"
#include "json.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

#include "http.h"

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

//
// downloader
//

// the suffix of the file a download writes until it completes
static const std::string PARTIAL_SUFFIX = ".downloadInProgress";

static void write_etag(const std::string & path, const std::string & etag) {
    const std::string etag_path = path + ".etag";
    fs_write_atomic(std::filesystem::u8path(etag_path), etag);
    LOG_DBG("%s: file etag saved: %s\n", __func__, etag_path.c_str());
}

static std::string read_etag(const std::string & path) {
    const std::string etag_path = path + ".etag";
    if (!std::filesystem::exists(etag_path)) {
        return {};
    }
    std::ifstream etag_in(etag_path);
    if (!etag_in) {
        LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, etag_path.c_str());
        return {};
    }
    std::string etag;
    std::getline(etag_in, etag);
    return etag;
}

static bool is_http_status_ok(int status) {
    return status >= 200 && status < 400;
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto parts = string_split<std::string>(hf_repo_with_tag, ':');
    std::string tag = parts.size() > 1 ? parts.back() : "";
    std::string hf_repo = parts[0];
    if (string_split<std::string>(hf_repo, '/').size() != 2) {
        throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
    }
    return {hf_repo, tag};
}

// file name from url, without the query string
static std::string url_get_filename(const std::string & url) {
    std::string filename = url;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }
    if (auto pos = filename.find('?'); pos != std::string::npos) {
        filename = filename.substr(0, pos);
    }
    return filename;
}

class ProgressBar : public common_download_callback {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

    std::string filename;
    size_t len = 0;

    static void cleanup(const ProgressBar * line) {
        lines.erase(line);
        if (lines.empty()) {
            max_line = 0;
        }
    }

    static bool is_output_a_tty() {
        return common_is_tty(stdout);
    }

public:
    ProgressBar() = default;

    void on_start(const common_download_progress & p) override {
        filename = url_get_filename(p.url);

        for (size_t i = 0; i < filename.size(); ++i) {
            if ((filename[i] & 0xC0) != 0x80) {
                if (len++ == 39) {
                    filename.resize(i);
                    filename += "…";
                    break;
                }
            }
        }
    }

    void on_done(const common_download_progress &, bool) override {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void on_update(const common_download_progress & p) override {
        if (!p.total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = (55 - len) * 2;
        size_t pct = (100 * p.downloaded) / p.total;
        size_t pos = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; i += 2) {
            std::cout << (i + 1 < pos ? "─" : (i < pos ? "╴" : " "));
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (p.downloaded == p.total) {
            cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

// resolve redirects manually so parallel Range workers can hit the final CDN URL directly
// and never leak Authorization: Bearer across origins.
struct common_resolved_remote_file {
    std::string final_url;
    std::string etag;
    size_t      total_size = 0;
    bool        supports_ranges = false;
    bool        same_origin = true;
    int         status = -1;
};

static bool pwrite_all_fd(int fd, const void * data, size_t len, size_t offset) {
#ifdef _WIN32
    static std::mutex win_io_mtx;
    std::lock_guard<std::mutex> lock(win_io_mtx);
    if (_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) < 0) {
        return false;
    }
    const char * p = static_cast<const char *>(data);
    size_t rem = len;
    while (rem > 0) {
        int n = _write(fd, p, static_cast<unsigned int>(std::min<size_t>(rem, size_t(1) << 30)));
        if (n <= 0) {
            return false;
        }
        p += n;
        rem -= static_cast<size_t>(n);
    }
    return true;
#else
    const char * p = static_cast<const char *>(data);
    size_t rem = len;
    size_t off = offset;
    while (rem > 0) {
        ssize_t n = ::pwrite(fd, p, rem, static_cast<off_t>(off));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        p   += static_cast<size_t>(n);
        off += static_cast<size_t>(n);
        rem -= static_cast<size_t>(n);
    }
    return true;
#endif
}

static bool ftruncate_fd(int fd, size_t size) {
#ifdef _WIN32
    return _chsize_s(fd, static_cast<__int64>(size)) == 0;
#else
    return ::ftruncate(fd, static_cast<off_t>(size)) == 0;
#endif
}

static int open_rw_fd(const std::string & path) {
#ifdef _WIN32
    return _open(path.c_str(), _O_CREAT | _O_RDWR | _O_BINARY, 0644);
#else
    return ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
#endif
}

static void close_rw_fd(int & fd) {
    if (fd >= 0) {
#ifdef _WIN32
        _close(fd);
#else
        ::close(fd);
#endif
        fd = -1;
    }
}

static httplib::Headers build_common_headers(const common_download_opts & opts, bool include_auth) {
    httplib::Headers headers;
    for (const auto & h : opts.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }
    if (include_auth && !opts.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + opts.bearer_token);
    }
    return headers;
}

static bool common_resolve_remote_file(
        const std::string & url,
        const common_download_opts & opts,
        common_resolved_remote_file & out) {
    const common_http_url initial_parts = common_http_parse_url(url);
    std::string cur_url = url;
    out = {};

    for (int hop = 0; hop < 8; ++hop) {
        auto [cli, parts] = common_http_client(cur_url);
        cli.set_follow_location(false);
        const bool same_origin = (parts.host == initial_parts.host && parts.port == initial_parts.port);
        httplib::Headers hdrs = build_common_headers(opts, same_origin);

        auto res = cli.Head(parts.path, hdrs);
        if (!res) {
            out.status = -1;
            return false;
        }
        out.status = res->status;

        if ((res->status == 301 || res->status == 302 || res->status == 303 || res->status == 307 || res->status == 308) &&
            res->has_header("Location")) {
            std::string loc = res->get_header_value("Location");
            if (!loc.empty() && loc[0] == '/') {
                cur_url = parts.scheme + "://" + common_http_format_host(parts.host) + ":" + std::to_string(parts.port) + loc;
            } else {
                cur_url = loc;
            }
            continue;
        }

        if (res->status >= 200 && res->status < 300) {
            out.final_url = cur_url;
            out.same_origin = same_origin;
            if (res->has_header("ETag")) {
                out.etag = res->get_header_value("ETag");
            }
            if (res->has_header("Content-Length")) {
                try {
                    out.total_size = std::stoull(res->get_header_value("Content-Length"));
                } catch (...) {}
            }
            if (res->has_header("Accept-Ranges")) {
                out.supports_ranges = res->get_header_value("Accept-Ranges") != "none";
            }
            return true;
        }

        // Some S3/CDN backends disallow HEAD or require Range GET 0-0
        httplib::Headers rhdrs = hdrs;
        rhdrs.emplace("Range", "bytes=0-0");
        auto gres = cli.Get(parts.path, rhdrs, [](const char *, size_t) { return true; });
        if (gres && (gres->status == 301 || gres->status == 302 || gres->status == 303 || gres->status == 307 || gres->status == 308) &&
            gres->has_header("Location")) {
            std::string loc = gres->get_header_value("Location");
            if (!loc.empty() && loc[0] == '/') {
                cur_url = parts.scheme + "://" + common_http_format_host(parts.host) + ":" + std::to_string(parts.port) + loc;
            } else {
                cur_url = loc;
            }
            continue;
        }
        if (gres && (gres->status == 206 || gres->status == 200)) {
            out.status = gres->status;
            out.final_url = cur_url;
            out.same_origin = same_origin;
            if (gres->has_header("ETag")) {
                out.etag = gres->get_header_value("ETag");
            }
            if (gres->status == 206 && gres->has_header("Content-Range")) {
                std::string cr = gres->get_header_value("Content-Range");
                auto slash = cr.rfind('/');
                if (slash != std::string::npos) {
                    try {
                        out.total_size = std::stoull(cr.substr(slash + 1));
                        out.supports_ranges = true;
                    } catch (...) {}
                }
            } else if (gres->has_header("Content-Length")) {
                try {
                    out.total_size = std::stoull(gres->get_header_value("Content-Length"));
                } catch (...) {}
            }
            return true;
        }
        return false;
    }
    return false;
}

static bool http_fetch_range_into(
        httplib::Client & cli,
        const std::string & req_path,
        const httplib::Headers & base_hdrs,
        size_t offset,
        size_t len,
        uint8_t * dst,
        size_t * out_received) {
    httplib::Headers hdrs = base_hdrs;
    hdrs.emplace("Range", "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + len - 1));
    size_t received = 0;
    bool overflow = false;
    int status = -1;
    auto res = cli.Get(req_path, hdrs,
        [&](const httplib::Response & resp) {
            status = resp.status;
            return resp.status == 206 || (offset == 0 && resp.status == 200);
        },
        [&](const char * data, size_t n) {
            if (received + n > len) {
                overflow = true;
                return false;
            }
            std::memcpy(dst + received, data, n);
            received += n;
            return true;
        });
    if (out_received) {
        *out_received = received;
    }
    if (!res || overflow || (status != 206 && !(offset == 0 && status == 200)) || received != len) {
        LOG_WRN("%s: failed url_path=%s offset=%zu len=%zu status=%d received=%zu overflow=%d err=%s\n",
                __func__, req_path.c_str(), offset, len, status, received, (int)overflow,
                res ? "none" : httplib::to_string(res.error()).c_str());
        return false;
    }
    return true;
}

static bool common_pull_file_parallel(
        const common_resolved_remote_file & resolved,
        const common_download_opts & opts,
        const std::string & path_tmp,
        common_download_progress & p) {
    const size_t total = resolved.total_size;
    if (!resolved.supports_ranges || total < size_t(16) * 1024 * 1024 || opts.n_threads <= 1) {
        return false;
    }
    int fd = open_rw_fd(path_tmp);
    if (fd < 0) {
        return false;
    }
    if (!ftruncate_fd(fd, total)) {
        close_rw_fd(fd);
        return false;
    }

    const size_t chunk_sz = size_t(8) * 1024 * 1024;
    const size_t n_chunks = (total + chunk_sz - 1) / chunk_sz;
    const int n_workers = std::min<int>(std::max(2, opts.n_threads), static_cast<int>(std::min<size_t>(n_chunks, 32)));

    std::atomic<size_t> next_chunk{0};
    std::atomic<size_t> done_bytes{0};
    std::atomic<bool> failed{false};
    std::mutex cb_mtx;

    auto parts = common_http_parse_url(resolved.final_url);
    httplib::Headers base_hdrs = build_common_headers(opts, resolved.same_origin);

    std::vector<std::thread> threads;
    threads.reserve(n_workers);
    for (int w = 0; w < n_workers; ++w) {
        threads.emplace_back([&]() {
            std::vector<uint8_t> buf(chunk_sz);
            auto [cli, _] = common_http_client(resolved.final_url);
            cli.set_follow_location(false);
            cli.set_keep_alive(true);
            cli.set_read_timeout(60, 0);
            while (!failed.load(std::memory_order_relaxed)) {
                if (opts.callback && opts.callback->is_cancelled()) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                const size_t idx = next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (idx >= n_chunks) {
                    break;
                }
                const size_t off = idx * chunk_sz;
                const size_t len = std::min(chunk_sz, total - off);
                bool ok = false;
                for (int attempt = 0; attempt < 4; ++attempt) {
                    if (attempt > 0) {
                        std::this_thread::sleep_for(std::chrono::seconds(1 << (attempt - 1)));
                        auto [fresh_cli, __] = common_http_client(resolved.final_url);
                        fresh_cli.set_follow_location(false);
                        fresh_cli.set_keep_alive(true);
                        fresh_cli.set_read_timeout(60, 0);
                        cli = std::move(fresh_cli);
                    }
                    if (http_fetch_range_into(cli, parts.path, base_hdrs, off, len, buf.data(), nullptr)) {
                        ok = true;
                        break;
                    }
                }
                if (!ok || !pwrite_all_fd(fd, buf.data(), len, off)) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                const size_t cur_done = done_bytes.fetch_add(len, std::memory_order_relaxed) + len;
                if (opts.callback) {
                    std::lock_guard<std::mutex> lock(cb_mtx);
                    p.downloaded = cur_done;
                    opts.callback->on_update(p);
                }
            }
        });
    }
    for (auto & t : threads) {
        t.join();
    }
    close_rw_fd(fd);
    return !failed.load();
}

static bool common_pull_file(httplib::Client & cli,
                             const std::string & resolve_path,
                             const std::string & path_tmp,
                             bool supports_ranges,
                             common_download_progress & p,
                             common_download_callback * callback) {
    std::ofstream ofs(path_tmp, std::ios::binary | std::ios::app);
    if (!ofs.is_open()) {
        LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    httplib::Headers headers;
    if (supports_ranges && p.downloaded > 0) {
        headers.emplace("Range", "bytes=" + std::to_string(p.downloaded) + "-");
    }

    const char * func = __func__; // avoid __func__ inside a lambda
    size_t progress_step = 0;

    auto res = cli.Get(resolve_path, headers,
        [&](const httplib::Response &response) {
            if (p.downloaded > 0 && response.status != 206) {
                LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", func, response.status);
                if (response.status == 416 || response.status == 200) {
                    // the kept partial cannot be resumed: 416 for a complete or stale one,
                    // 200 from a server that ignores the range; drop it, so the next
                    // attempt starts over instead of wedging on the same answer
                    ofs.close();
                    remove(path_tmp.c_str());
                }
                return false;
            }
            if (p.downloaded == 0 && response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", func, response.status);
                return false;
            }
            if (p.total == 0 && response.has_header("Content-Length")) {
                try {
                    size_t content_length = std::stoull(response.get_header_value("Content-Length"));
                    p.total = p.downloaded + content_length;
                } catch (const std::exception &e) {
                    LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                }
            }
            return true;
        },
        [&](const char *data, size_t len) {
            ofs.write(data, len);
            if (!ofs) {
                LOG_ERR("%s: error writing to file: %s\n", func, path_tmp.c_str());
                return false;
            }
            p.downloaded += len;
            progress_step += len;

            if (progress_step >= p.total / 1000 || p.downloaded == p.total) {
                if (callback) {
                    callback->on_update(p);
                    if (callback->is_cancelled() && (p.total == 0 || p.downloaded != p.total)) {
                        return false;
                    }
                }
                progress_step = 0;
            }
            return true;
        },
        nullptr
    );

    if (!res) {
        LOG_ERR("%s: download failed: %s (status: %d)\n",
                __func__,
                httplib::to_string(res.error()).c_str(),
                res ? res->status : -1);
        return false;
    }

    ofs.close();
    if (!ofs) {
        LOG_ERR("%s: error closing file: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    return true;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int common_download_file_single_online(const std::string & url,
                                              const std::string & path,
                                              const common_download_opts & opts,
                                              bool skip_etag,
                                              const std::string & lock_path,
                                              const std::string & alt_path) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    const std::string path_temporary = path + PARTIAL_SUFFIX;
    const std::string path_lock = lock_path.empty() ? path + ".lock" : lock_path;

    common_download_progress p;
    p.url = url;

    if (opts.callback) {
        opts.callback->on_start(p);
    }

    // the completion callback may itself take the same lock (the HF cache
    // finalizes blobs under it), so every exit goes through finish()
    common_file_lock lock(path_lock);
    auto finish = [&](int status) {
        lock.close();
        if (opts.callback) {
            opts.callback->on_done(p, is_http_status_ok(status));
        }
        return status;
    };

    // alt_path is the HF snapshot: the blob may have been moved there when symlinks are unavailable
    auto cache_hit = [&] {
        return std::filesystem::exists(std::filesystem::u8path(path)) ||
               (!alt_path.empty() && std::filesystem::exists(std::filesystem::u8path(alt_path)));
    };

    if (skip_etag && cache_hit()) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return finish(304); // 304 Not Modified - fake cached response
    }

    // take the lock before touching the cache, so two processes cannot race on the same file
    bool waited = false;
    if (!lock.acquire([&] {
            if (!waited) {
                LOG_INF("%s: file '%s' is being downloaded by another process, waiting...\n", __func__, url_get_filename(url).c_str());
                waited = true;
            }
            return !(opts.callback && opts.callback->is_cancelled());
        })) {
        return finish(-1);
    }

    // under the lock: another process may have finished while we waited
    bool file_exists = cache_hit();

    if (file_exists && skip_etag) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return finish(304); // 304 Not Modified - fake cached response
    }

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : opts.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }
    if (!opts.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + opts.bearer_token);
    }
    cli.set_default_headers(headers);

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_DBG("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    auto head = cli.Head(parts.path);
    if (!head || head->status < 200 || head->status >= 300) {
        LOG_TRC("%s: HEAD failed, status: %d\n", __func__, head ? head->status : -1);
        if (file_exists) {
            LOG_TRC("%s: using cached file (HEAD failed): %s\n", __func__, path.c_str());
            return finish(304); // 304 Not Modified - fake cached response
        }
        return finish(head ? head->status : -1);
    }

    std::string etag;
    if (head->has_header("ETag")) {
        etag = head->get_header_value("ETag");
    }

    if (head->has_header("Content-Length")) {
        try {
            p.total = std::stoull(head->get_header_value("Content-Length"));
        } catch (const std::exception& e) {
            LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, e.what());
        }
    }

    bool supports_ranges = false;
    if (head->has_header("Accept-Ranges")) {
        supports_ranges = head->get_header_value("Accept-Ranges") != "none";
    }

    if (file_exists) {
        if (etag.empty()) {
            LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, path.c_str());
            return finish(304); // 304 Not Modified - fake cached response
        }
        if (!last_etag.empty() && last_etag == etag) {
            LOG_DBG("%s: using cached file (same etag): %s\n", __func__, path.c_str());
            return finish(304); // 304 Not Modified - fake cached response
        }
        // pass this point, the file exists but is different from the server version, so we need to redownload it
        if (remove(path.c_str()) != 0) {
            LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
            return finish(-1);
        }
    }

    { // silent
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    }

    bool success = false;
    int delay = retry_delay_seconds;

    if (!skip_etag && !etag.empty()) {
        if (std::filesystem::exists(path_temporary) && read_etag(path_temporary) != etag) {
            LOG_WRN("%s: partial file does not match the server etag, discarding it: %s\n", __func__, path_temporary.c_str());
            if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                return finish(-1);
            }
        }
        write_etag(path_temporary, etag);
    }

    if (opts.callback) {
        // report the resume offset, so a resumed file does not start at 0
        p.downloaded = supports_ranges && std::filesystem::exists(path_temporary) ?
                       std::filesystem::file_size(path_temporary) : 0;
        opts.callback->on_update(p);
    }

    for (int i = 0; i < max_attempts; ++i) {
        if (opts.callback && opts.callback->is_cancelled()) {
            break;
        }
        if (i) {
            LOG_WRN("%s: retrying after %d seconds...\n", __func__, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
            delay *= retry_delay_seconds;
        }

        size_t existing_size = 0;

        if (std::filesystem::exists(path_temporary)) {
            if (supports_ranges) {
                existing_size = std::filesystem::file_size(path_temporary);
            } else if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                break;
            }
        }

        p.downloaded = existing_size;

        LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(),
                path_temporary.c_str(), etag.c_str());

        if (existing_size == 0 && supports_ranges && opts.n_threads > 1 && p.total >= size_t(16) * 1024 * 1024) {
            common_resolved_remote_file resolved;
            if (common_resolve_remote_file(url, opts, resolved) &&
                resolved.supports_ranges && resolved.total_size >= size_t(16) * 1024 * 1024 &&
                common_pull_file_parallel(resolved, opts, path_temporary, p)) {
                if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
                    LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
                    break;
                }
                if (!etag.empty() && !skip_etag) {
                    write_etag(path, etag);
                    remove((path_temporary + ".etag").c_str());
                }
                success = true;
                break;
            }
            if (opts.callback && opts.callback->is_cancelled()) {
                break;
            }
            LOG_WRN("%s: parallel range download failed, falling back to single-stream\n", __func__);
            remove(path_temporary.c_str());
            p.downloaded = 0;
        }

        if (common_pull_file(cli, parts.path, path_temporary, supports_ranges, p, opts.callback)) {
            if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
                LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
                break;
            }
            if (!etag.empty() && !skip_etag) {
                write_etag(path, etag);
                remove((path_temporary + ".etag").c_str());
            }
            success = true;
            break;
        }
    }

    // a stopped download keeps path_temporary, so the next attempt resumes from it
    if (!success) {
        LOG_ERR("%s: download failed after %d attempts\n", __func__, max_attempts);
    }

    return finish(success ? head->status : -1);
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : params.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }

    if (params.timeout > 0) {
        cli.set_read_timeout(params.timeout, 0);
        cli.set_write_timeout(params.timeout, 0);
    }

    std::vector<char> buf;
    auto res = cli.Get(parts.path, headers,
        [&](const char *data, size_t len) {
            buf.insert(buf.end(), data, data + len);
            return params.max_size == 0 ||
                   buf.size() <= static_cast<size_t>(params.max_size);
        },
        nullptr
    );

    if (!res) {
        throw std::runtime_error("error: cannot make GET request");
    }

    return { res->status, std::move(buf) };
}

int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts,
                                bool skip_etag,
                                const std::string & lock_path,
                                const std::string & alt_path) {
    if (!opts.offline) {
        ProgressBar tty_cb;
        common_download_opts online_opts = opts;
        if (!online_opts.callback) {
            online_opts.callback = &tty_cb;
        }
        return common_download_file_single_online(url, path, online_opts, skip_etag, lock_path, alt_path);
    }

    if (!std::filesystem::exists(std::filesystem::u8path(path))) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, path.c_str());

    // notify the callback that the file was cached
    if (opts.callback) {
        common_download_progress p;
        p.url = url;
        p.cached = true;
        opts.callback->on_start(p);
        opts.callback->on_done(p, true);
    }

    return 304; // Not Modified - fake cached response
}

struct gguf_split_info {
    std::string prefix; // tag included
    std::string tag;
    int index;
    int count;
};

static gguf_split_info get_gguf_split_info(const std::string & path) {
    static const std::regex re_split("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
    static const std::regex re_tag("[-.]([A-Z0-9_]+)$", std::regex::icase);
    std::smatch m;

    std::string prefix = path;
    if (!string_remove_suffix(prefix, ".gguf")) {
        return {};
    }

    int index = 1;
    int count = 1;

    if (std::regex_match(prefix, m, re_split)) {
        index = std::stoi(m[2].str());
        count = std::stoi(m[3].str());
        prefix = m[1].str();
    }

    std::string tag;
    if (std::regex_search(prefix, m, re_tag)) {
        tag = m[1].str();
        for (char & c : tag) {
            c = std::toupper((unsigned char)c);
        }
    }

    return {std::move(prefix), std::move(tag), index, count};
}

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
static int extract_quant_bits(const std::string & filename) {
    auto split = get_gguf_split_info(filename);

    auto pos = split.tag.find_first_of("0123456789");
    if (pos == std::string::npos) {
        return 0;
    }

    return std::stoi(split.tag.substr(pos));
}

static hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                          const hf_cache::hf_file  & file) {
    auto split = get_gguf_split_info(file.path);

    if (split.count <= 1) {
        return {file};
    }
    hf_cache::hf_files result;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            result.push_back(f);
        }
    }
    return result;
}

// pick the best sibling GGUF whose filename contains `keyword` (e.g. "mmproj" / "mtp"),
// preferring deeper shared directory prefix with the model, then exact `tag` match,
// then closest quantization to the tag when given, or to the model otherwise
static hf_cache::hf_file find_best_sibling(const hf_cache::hf_files & files,
                                           const std::string        & model,
                                           const std::string        & keyword,
                                           const std::string        & tag = "") {
    hf_cache::hf_file best;
    size_t best_depth = 0;
    int best_diff = 0;
    bool best_exact = false;
    bool found = false;

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    int model_bits = 0;
    if (!tag_upper.empty()) {
        auto pos = tag_upper.find_first_of("0123456789");
        model_bits = pos == std::string::npos ? 0 : std::stoi(tag_upper.substr(pos));
    } else {
        model_bits = extract_quant_bits(model);
    }
    auto model_parts = string_split<std::string>(model, '/');
    auto model_dir = model_parts.end() - 1;

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf") ||
            f.path.find(keyword) == std::string::npos) {
            continue;
        }

        auto sib_parts = string_split<std::string>(f.path, '/');
        auto sib_dir = sib_parts.end() - 1;

        auto [_, dir] = std::mismatch(model_parts.begin(), model_dir,
                                      sib_parts.begin(), sib_dir);
        if (dir != sib_dir) {
            continue;
        }

        size_t depth = dir - sib_parts.begin();
        auto bits = extract_quant_bits(f.path);
        auto diff = std::abs(bits - model_bits);

        std::string path_upper = f.path;
        for (char & c : path_upper) {
            c = (char) std::toupper((unsigned char) c);
        }
        bool exact = !tag_upper.empty() && path_upper.find("-" + tag_upper + ".") != std::string::npos;

        if (!found || depth > best_depth ||
            (depth == best_depth && exact && !best_exact) ||
            (depth == best_depth && exact == best_exact && diff < best_diff)) {
            best = f;
            best_depth = depth;
            best_diff = diff;
            best_exact = exact;
            found = true;
        }
    }
    return best;
}

static const std::vector<common_download_sidecar> sidecars = {
    { "mmproj", "mmproj",  false, &common_download_opts::download_mmproj, &common_download_hf_plan::mmproj },
    { "mtp",    "mtp-",    true,  &common_download_opts::download_mtp,    &common_download_hf_plan::mtp    },
    { "dflash", "dflash-", true,  &common_download_opts::download_dflash, &common_download_hf_plan::dflash },
    { "eagle3", "eagle3-", true,  &common_download_opts::download_eagle3, &common_download_hf_plan::eagle3 },
    { "dspark", "dspark-", true,  &common_download_opts::download_dspark, &common_download_hf_plan::dspark },
};

const std::vector<common_download_sidecar> & common_download_sidecars() {
    return sidecars;
}

bool common_download_opts_add_sidecar(common_download_opts & opts, const std::string & kind) {
    for (const auto & s : sidecars) {
        if (kind == s.name) {
            opts.*s.opt = true;
            return true;
        }
    }
    return false;
}

std::vector<std::string> common_download_plan_sidecars(const common_download_hf_plan & plan) {
    std::vector<std::string> out;
    for (const auto & s : sidecars) {
        if (!(plan.*s.file).path.empty()) {
            out.emplace_back(s.name);
        }
    }
    return out;
}

std::vector<common_download_partial> common_download_plan_partials(const common_download_hf_plan & plan) {
    std::vector<common_download_partial> out;

    // a file in the cache resolves to its snapshot entry, a file still to download to its blob
    auto add = [&](const std::string & kind, const hf_cache::hf_file & f) {
        if (!f.local_path.empty() && f.local_path != f.final_path) {
            out.push_back({ kind, f.local_path + PARTIAL_SUFFIX });
        }
    };

    for (const auto & f : plan.model_files) {
        add("", f);
    }
    for (const auto & s : sidecars) {
        add(s.name, plan.*s.file);
    }
    return out;
}

// true when `name` carries the keyword of a sidecar kind
static bool name_is_sidecar(const std::string & name) {
    for (const auto & s : sidecars) {
        if (name.find(s.keyword) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool gguf_filename_is_model(const std::string & filepath) {
    if (!string_ends_with(filepath, ".gguf")) {
        return false;
    }

    std::string filename = filepath;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }

    return filename.find("imatrix") == std::string::npos && !name_is_sidecar(filename);
}

static hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                         const std::string        & tag) {
    std::vector<std::string> tags;

    if (!tag.empty()) {
        tags.push_back(tag);
    } else {
        tags = {"Q4_K_M", "Q8_0"};
    }

    for (const auto & t : tags) {
        std::regex pattern(t + "[.-]", std::regex::icase);
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path) &&
                std::regex_search(f.path, pattern)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.empty()) {
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return {};
}

static void list_available_gguf_files(const hf_cache::hf_files & files) {
    LOG_INF("Available GGUF files:\n");
    for (const auto & f : files) {
        if (string_ends_with(f.path, ".gguf")) {
            LOG_INF(" - %s\n", f.path.c_str());
        }
    }
}

common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts) {
    common_download_hf_plan plan;
    hf_cache::hf_files all;

    auto [repo, tag] = common_download_split_repo_tag(model.hf_repo);

    if (!opts.offline) {
        all = hf_cache::get_repo_files(repo, opts.bearer_token);
    }
    if (all.empty()) {
        all = hf_cache::get_cached_files(repo);
    }
    if (all.empty()) {
        return plan;
    }

    // if preset.ini exists in the repo root, download only that file
    for (const auto & f : all) {
        if (f.path == "preset.ini") {
            plan.preset = f;
            return plan;
        }
    }

    hf_cache::hf_file primary;

    if (!model.hf_file.empty()) {
        for (const auto & f : all) {
            if (f.path == model.hf_file) {
                primary = f;
                break;
            }
        }
        if (primary.path.empty()) {
            LOG_ERR("%s: file '%s' not found in repository\n", __func__, model.hf_file.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    } else {
        primary = find_best_model(all, tag);
        // a requested draft can resolve on its own, without a full model of the same tag
        const bool want_draft = std::any_of(sidecars.begin(), sidecars.end(), [&](const common_download_sidecar & s) {
            return s.draft && opts.*s.opt;
        });
        if (primary.path.empty() && !want_draft) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    }

    if (!primary.path.empty()) {
        plan.primary = primary;
        plan.model_files = get_split_files(all, primary);
    }

    // a draft resolves at the tag, the mmproj pairs with the model
    for (const auto & s : sidecars) {
        if (!(opts.*s.opt)) {
            continue;
        }
        if (s.draft) {
            plan.*s.file = find_best_sibling(all, primary.path, s.keyword, tag);
        } else if (!primary.path.empty()) {
            plan.*s.file = find_best_sibling(all, primary.path, s.keyword);
        }
    }

    const bool has_draft = std::any_of(sidecars.begin(), sidecars.end(), [&](const common_download_sidecar & s) {
        return s.draft && !(plan.*s.file).local_path.empty();
    });
    if (primary.path.empty() && !has_draft) {
        LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
        list_available_gguf_files(all);
    }

    return plan;
}

void common_download_run_tasks(const std::vector<common_download_task> & tasks) {
    std::vector<std::future<int>> futures;
    for (const auto & task : tasks) {
        futures.push_back(std::async(std::launch::async,
            [&task]() {
                return common_download_file_single(task.url, task.local_path, task.opts, task.is_hf, task.lock_path, task.alt_path);
            }
        ));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        std::string url = tasks[i].url;
        int status = futures[i].get();
        bool is_ok = is_http_status_ok(status);
        if (!is_ok) {
            throw std::runtime_error(string_format("Download '%s' failed with status code: %d", url.c_str(), status));
        }
    }
}

std::vector<std::string> common_download_get_all_parts(const std::string & url) {
    auto split = get_gguf_split_info(url);

    if (split.count <= 1) {
        return {url};
    }

    std::vector<std::string> parts;
    for (int i = 1; i <= split.count; i++) {
        auto suffix = string_format("-%05d-of-%05d.gguf", i, split.count);
        parts.push_back(split.prefix + suffix);
    }
    return parts;
}

//
// Docker registry functions
//

static std::string common_docker_get_token(const std::string & repo) {
    std::string url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repo + ":pull";

    common_remote_params params;
    auto                 res = common_remote_get_content(url, params);

    if (res.first != 200) {
        throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(res.first));
    }

    std::string response_str(res.second.begin(), res.second.end());
    common_json response = common_json::parse(response_str);

    if (!response.contains("token")) {
        throw std::runtime_error("Docker registry token response missing 'token' field");
    }

    return response["token"].get<std::string>();
}

std::string common_docker_resolve_model(const std::string & docker) {
    // Parse ai/smollm2:135M-Q4_0
    size_t      colon_pos = docker.find(':');
    std::string repo, tag;
    if (colon_pos != std::string::npos) {
        repo = docker.substr(0, colon_pos);
        tag  = docker.substr(colon_pos + 1);
    } else {
        repo = docker;
        tag  = "latest";
    }

    // ai/ is the default
    size_t      slash_pos = docker.find('/');
    if (slash_pos == std::string::npos) {
        repo.insert(0, "ai/");
    }

    LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, repo.c_str(), tag.c_str());
    try {
        // --- helper: digest validation ---
        auto validate_oci_digest = [](const std::string & digest) -> std::string {
            // Expected: algo:hex ; start with sha256 (64 hex chars)
            // You can extend this map if supporting other algorithms in future.
            static const std::regex re("^sha256:([a-fA-F0-9]{64})$");
            std::smatch m;
            if (!std::regex_match(digest, m, re)) {
                throw std::runtime_error("Invalid OCI digest format received in manifest: " + digest);
            }
            // normalize hex to lowercase
            std::string normalized = digest;
            std::transform(normalized.begin()+7, normalized.end(), normalized.begin()+7, [](unsigned char c){
                return std::tolower(c);
            });
            return normalized;
        };

        std::string token = common_docker_get_token(repo);  // Get authentication token

        // Get manifest
        // TODO: cache the manifest response so that it appears in the model list
        const std::string    url_prefix = "https://registry-1.docker.io/v2/" + repo;
        std::string          manifest_url = url_prefix + "/manifests/" + tag;
        common_remote_params manifest_params;
        manifest_params.headers.push_back({"Authorization", "Bearer " + token});
        manifest_params.headers.push_back({"Accept",
            "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
        });
        auto manifest_res = common_remote_get_content(manifest_url, manifest_params);
        if (manifest_res.first != 200) {
            throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(manifest_res.first));
        }

        std::string manifest_str(manifest_res.second.begin(), manifest_res.second.end());
        common_json manifest = common_json::parse(manifest_str);
        std::string gguf_digest;  // Find the GGUF layer
        if (manifest.contains("layers")) {
            for (const auto & layer : manifest["layers"]) {
                if (layer.contains("mediaType")) {
                    std::string media_type = layer["mediaType"].get<std::string>();
                    if (media_type == "application/vnd.docker.ai.gguf.v3" ||
                        media_type.find("gguf") != std::string::npos) {
                        gguf_digest = layer["digest"].get<std::string>();
                        break;
                    }
                }
            }
        }

        if (gguf_digest.empty()) {
            throw std::runtime_error("No GGUF layer found in Docker manifest");
        }

        // Validate & normalize digest
        gguf_digest = validate_oci_digest(gguf_digest);
        LOG_DBG("%s: Using validated digest: %s\n", __func__, gguf_digest.c_str());

        // Prepare local filename
        std::string model_filename = repo;
        std::replace(model_filename.begin(), model_filename.end(), '/', '_');
        model_filename += "_" + tag + ".gguf";
        std::string local_path = fs_path_to_utf8(fs_get_cache_file(model_filename));

        const std::string blob_url = url_prefix + "/blobs/" + gguf_digest;
        common_download_opts opts;
        opts.bearer_token = token;
        const int http_status = common_download_file_single(blob_url, local_path, opts);
        if (!is_http_status_ok(http_status)) {
            throw std::runtime_error("Failed to download Docker Model");
        }

        LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, local_path.c_str());
        return local_path;
    } catch (const std::exception & e) {
        LOG_ERR("%s: Docker Model download failed: %s\n", __func__, e.what());
        throw;
    }
}

// a cached file that lists a model under its tag: the first split of a model, not a sidecar
static bool is_listed_model(const gguf_split_info & split) {
    return split.index == 1 && !split.tag.empty() && !name_is_sidecar(split.prefix);
}

std::vector<common_cached_model_info> common_list_cached_models() {
    std::unordered_set<std::string> seen;
    std::vector<common_cached_model_info> result;

    auto files = hf_cache::get_cached_files();

    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (!is_listed_model(split)) {
            continue;
        }
        if (seen.insert(f.repo_id + ":" + split.tag).second) {
            result.push_back({f.repo_id, split.tag});
        }
    }

    return result;
}

std::string common_download_resolve_path(const std::string & hf_repo_with_tag, const std::string & hf_file) {
    auto [repo, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    auto files = hf_cache::get_cached_files(repo);
    if (files.empty()) {
        return "";
    }

    if (!hf_file.empty()) {
        for (const auto & f : files) {
            if (f.path == hf_file) {
                return f.local_path;
            }
        }
        return "";
    }

    return find_best_model(files, tag).local_path;
}

// the files -hf loads from the cache, the model with every sidecar, for each cached
// tag of `repo_id` other than `tag`
static std::unordered_set<std::string> files_in_use(const std::string & repo_id, const std::string & tag) {
    common_download_opts opts;
    opts.offline = true;
    for (const auto & s : sidecars) {
        opts.*s.opt = true;
    }

    std::unordered_set<std::string> tags;
    std::unordered_set<std::string> used;

    for (const auto & f : hf_cache::get_cached_files(repo_id)) {
        auto split = get_gguf_split_info(f.path);
        if (!is_listed_model(split) || split.tag == tag || !tags.insert(split.tag).second) {
            continue;
        }

        common_params_model model;
        model.hf_repo = repo_id + ":" + split.tag;

        auto plan = common_download_get_hf_plan(model, opts);

        for (const auto & m : plan.model_files) {
            used.insert(m.local_path);
        }
        for (const auto & s : sidecars) {
            const auto & sf = plan.*s.file;
            if (!sf.local_path.empty()) {
                used.insert(sf.local_path);
            }
        }
    }
    return used;
}

static bool remove_snapshot_entries(const std::string & repo_id, const std::vector<std::filesystem::path> & to_remove);

bool common_download_remove(const std::string & hf_repo_with_tag) {
    namespace fs = std::filesystem;

    auto [repo_id, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    if (tag.empty()) {
        return hf_cache::remove_cached_repo(repo_id);
    }

    std::string tag_upper = tag;
    for (char & c : tag_upper) {
        c = (char) std::toupper((unsigned char) c);
    }

    auto files = hf_cache::get_cached_files(repo_id);
    if (files.empty()) {
        return false;
    }

    // collect snapshot entries whose tag matches, but a file another tag loads stays
    const auto used = files_in_use(repo_id, tag_upper);

    std::vector<fs::path> to_remove;
    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (split.tag == tag_upper && used.find(f.local_path) == used.end()) {
            to_remove.emplace_back(f.local_path);
        }
    }

    if (to_remove.empty()) {
        return false;
    }

    return remove_snapshot_entries(repo_id, to_remove);
}

// remove snapshot entries of `repo_id`, then the blobs no remaining entry references
static bool remove_snapshot_entries(const std::string & repo_id, const std::vector<std::filesystem::path> & to_remove) {
    namespace fs = std::filesystem;

    // resolve blob paths from symlinks before deleting snapshot entries
    std::vector<fs::path> blobs_to_check;
    for (const auto & p : to_remove) {
        std::error_code ec;
        if (fs::is_symlink(p, ec)) {
            auto target = fs::read_symlink(p, ec);
            if (!ec) {
                blobs_to_check.push_back((p.parent_path() / target).lexically_normal());
            }
        }
    }

    // remove snapshot entries
    for (const auto & p : to_remove) {
        std::error_code ec;
        fs::remove(p, ec);
        if (ec) {
            LOG_WRN("%s: failed to remove %s: %s\n", __func__, p.string().c_str(), ec.message().c_str());
        }
    }

    if (blobs_to_check.empty()) {
        return true;
    }

    // collect blobs still referenced by remaining snapshot entries
    std::unordered_set<std::string> still_referenced;
    for (const auto & f : hf_cache::get_cached_files(repo_id)) {
        fs::path p(f.local_path);
        std::error_code ec;
        if (fs::is_symlink(p, ec)) {
            auto target = fs::read_symlink(p, ec);
            if (!ec) {
                still_referenced.insert((p.parent_path() / target).lexically_normal().string());
            }
        }
    }

    // remove orphaned blobs
    for (const auto & blob : blobs_to_check) {
        if (still_referenced.find(blob.string()) == still_referenced.end()) {
            std::error_code ec;
            fs::remove(blob, ec);
            if (ec) {
                LOG_WRN("%s: failed to remove blob %s: %s\n", __func__, blob.string().c_str(), ec.message().c_str());
            }
        }
    }

    return true;
}

bool common_download_remove_sidecars(const std::string & hf_repo_with_tag, const common_download_opts & opts) {
    common_params_model model;
    model.hf_repo = hf_repo_with_tag;

    common_download_opts cached = opts;
    cached.offline = true;

    auto plan = common_download_get_hf_plan(model, cached);

    std::vector<std::filesystem::path> to_remove;
    for (const auto & s : sidecars) {
        const auto & f = plan.*s.file;
        if (!f.local_path.empty()) {
            to_remove.emplace_back(std::filesystem::u8path(f.local_path));
        }
    }
    if (to_remove.empty()) {
        return false;
    }

    return remove_snapshot_entries(common_download_split_repo_tag(hf_repo_with_tag).first, to_remove);
}

//
// Direct HTTP Range -> VRAM streaming (+ Disk Tee)
//

namespace {

struct stream_file_state {
    common_download_task        task;
    common_resolved_remote_file resolved;
    std::string                 temp_path;
    std::string                 lock_path;
    std::unique_ptr<common_file_lock> lock;
    int                         fd = -1;
    size_t                      header_end = 0;
};

struct stream_session {
    common_download_stream_opts              opts;
    std::vector<std::unique_ptr<stream_file_state>> files;
    bool                                     consumed = false;
};

struct bg_write_item {
    int fd = -1;
    size_t offset = 0;
    std::vector<uint8_t> data;
};

std::mutex g_stream_mtx;
std::unordered_map<std::string, std::shared_ptr<stream_session>> g_stream_sessions;
std::vector<std::thread> g_bg_threads;

static void finalize_stream_file(stream_file_state & fs, enum common_hf_stream_disk_mode disk_mode) {
    close_rw_fd(fs.fd);
    bool renamed_ok = false;
    if (disk_mode == COMMON_HF_STREAM_DISK_NONE) {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(fs.temp_path), ec);
    } else {
        if (std::rename(fs.temp_path.c_str(), fs.task.local_path.c_str()) != 0) {
            LOG_ERR("%s: unable to rename streamed file '%s' -> '%s'\n",
                    __func__, fs.temp_path.c_str(), fs.task.local_path.c_str());
        } else {
            if (!fs.task.is_hf && !fs.resolved.etag.empty()) {
                write_etag(fs.task.local_path, fs.resolved.etag);
            }
            renamed_ok = true;
        }
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(fs.temp_path + ".etag"), ec);
    }
    // Release stream lock before invoking on_done(), because HF finalization acquires the same lock file.
    if (fs.lock) {
        fs.lock->close();
        fs.lock.reset();
    }
    if (renamed_ok && fs.task.on_done) {
        fs.task.on_done();
    }
}

} // namespace

void common_download_wait_background() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lock(g_stream_mtx);
        to_join.swap(g_bg_threads);
    }
    for (auto & t : to_join) {
        if (t.joinable()) {
            t.join();
        }
    }
}

static bool probe_gguf_data_offset(const std::string & path, size_t & out_offset) {
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ggml_ctx,
    };
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        if (ggml_ctx) {
            ggml_free(ggml_ctx);
        }
        return false;
    }
    out_offset = gguf_get_data_offset(gctx);
    gguf_free(gctx);
    if (ggml_ctx) {
        ggml_free(ggml_ctx);
    }
    return true;
}

bool common_download_prepare_stream(
        const std::vector<common_download_task> & tasks,
        const common_download_stream_opts & opts,
        std::vector<std::string> & out_paths) {
    common_download_wait_background();
    out_paths.clear();
    if (tasks.empty()) {
        return false;
    }

    auto sess = std::make_shared<stream_session>();
    sess->opts = opts;
    if (sess->opts.n_threads < 1) {
        sess->opts.n_threads = 1;
    }
    if (sess->opts.chunk_bytes < 1024 * 1024) {
        sess->opts.chunk_bytes = 1024 * 1024;
    }

    auto cleanup_on_abort = [&]() {
        for (auto & fs : sess->files) {
            if (!fs) {
                continue;
            }
            close_rw_fd(fs->fd);
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(fs->temp_path), ec);
            if (fs->lock) {
                fs->lock->close();
                fs->lock.reset();
            }
        }
    };

    for (const auto & task : tasks) {
        if (task.url.empty() || task.local_path.empty() || task.opts.offline) {
            cleanup_on_abort();
            return false;
        }
        // If already cached on disk (local blob or finalized HF snapshot), do not stream from network.
        if (std::filesystem::exists(std::filesystem::u8path(task.local_path)) ||
            (!task.alt_path.empty() && std::filesystem::exists(std::filesystem::u8path(task.alt_path)))) {
            cleanup_on_abort();
            return false;
        }
        // If another download left a partial file for resume when disk saving is requested, use regular resume.
        const std::string temp_path = task.local_path + PARTIAL_SUFFIX;
        if (std::filesystem::exists(std::filesystem::u8path(temp_path)) && opts.disk_mode != COMMON_HF_STREAM_DISK_NONE) {
            cleanup_on_abort();
            return false;
        }

        auto fs = std::make_unique<stream_file_state>();
        fs->task      = task;
        fs->temp_path = temp_path;
        fs->lock_path = task.lock_path.empty() ? (task.local_path + ".lock") : task.lock_path;
        fs->lock      = std::make_unique<common_file_lock>(fs->lock_path);

        if (!fs->lock->acquire([&]() {
                return !(task.opts.callback && task.opts.callback->is_cancelled());
            })) {
            cleanup_on_abort();
            return false;
        }
        // Re-check cache after acquiring lock
        if (std::filesystem::exists(std::filesystem::u8path(task.local_path)) ||
            (!task.alt_path.empty() && std::filesystem::exists(std::filesystem::u8path(task.alt_path)))) {
            fs->lock->close();
            cleanup_on_abort();
            return false;
        }

        if (!common_resolve_remote_file(task.url, task.opts, fs->resolved) ||
            !fs->resolved.supports_ranges ||
            fs->resolved.total_size < size_t(1024) * 1024) {
            cleanup_on_abort();
            return false;
        }

        {
            std::error_code ec;
            std::filesystem::create_directories(std::filesystem::path(fs->temp_path).parent_path(), ec);
        }

        fs->fd = open_rw_fd(fs->temp_path);
        if (fs->fd < 0 || !ftruncate_fd(fs->fd, fs->resolved.total_size)) {
            cleanup_on_abort();
            return false;
        }

        // Fetch GGUF header incrementally until gguf_init_from_file succeeds and reports data_offset
        auto [cli, url_parts] = common_http_client(fs->resolved.final_url);
        cli.set_follow_location(false);
        cli.set_keep_alive(true);
        cli.set_read_timeout(60, 0);
        const httplib::Headers hdrs = build_common_headers(task.opts, fs->resolved.same_origin);

        size_t fetched = 0;
        const size_t max_header_probe = std::min<size_t>(fs->resolved.total_size, size_t(64) * 1024 * 1024);
        size_t data_offset = 0;
        bool parsed_header = false;
        std::vector<uint8_t> hbuf;

        while (fetched < max_header_probe) {
            const size_t step = (fetched == 0) ? size_t(2) * 1024 * 1024 : size_t(4) * 1024 * 1024;
            const size_t req_end = std::min<size_t>(max_header_probe, fetched + step);
            const size_t req_len = req_end - fetched;
            hbuf.resize(req_len);
            if (!http_fetch_range_into(cli, url_parts.path, hdrs, fetched, req_len, hbuf.data(), nullptr) ||
                !pwrite_all_fd(fs->fd, hbuf.data(), req_len, fetched)) {
                cleanup_on_abort();
                return false;
            }
            fetched = req_end;
            if (probe_gguf_data_offset(fs->temp_path, data_offset)) {
                if (data_offset <= fetched) {
                    parsed_header = true;
                    break;
                }
                // Metadata knows exact data_offset; fetch remaining gap if needed
                const size_t need_end = std::min<size_t>(fs->resolved.total_size, data_offset);
                if (need_end > fetched) {
                    const size_t rem_len = need_end - fetched;
                    hbuf.resize(rem_len);
                    if (!http_fetch_range_into(cli, url_parts.path, hdrs, fetched, rem_len, hbuf.data(), nullptr) ||
                        !pwrite_all_fd(fs->fd, hbuf.data(), rem_len, fetched)) {
                        cleanup_on_abort();
                        return false;
                    }
                    fetched = need_end;
                }
                parsed_header = true;
                break;
            }
        }

        if (!parsed_header) {
            cleanup_on_abort();
            return false;
        }
        fs->header_end = fetched;
        out_paths.push_back(fs->temp_path);
        sess->files.push_back(std::move(fs));
    }

    {
        std::lock_guard<std::mutex> lock(g_stream_mtx);
        g_stream_sessions[out_paths.front()] = sess;
    }
    LOG_INF("%s: prepared %zu GGUF file(s) for direct HF->VRAM streaming (%d threads, %zu MiB chunks)\n",
            __func__, sess->files.size(), sess->opts.n_threads, sess->opts.chunk_bytes / (1024 * 1024));
    return true;
}

void * common_download_get_stream_session_ud(const std::string & first_path) {
    std::lock_guard<std::mutex> lock(g_stream_mtx);
    auto it = g_stream_sessions.find(first_path);
    if (it == g_stream_sessions.end() || it->second->consumed) {
        return nullptr;
    }
    return it->second.get();
}

bool common_download_stream_loader(
        const struct llama_model_stream_loader_ops * ops,
        void * ops_ctx,
        uint16_t n_files,
        const size_t * file_data_offsets,
        void * user_data) {
    auto * raw_sess = static_cast<stream_session *>(user_data);
    if (!ops || !raw_sess || n_files != raw_sess->files.size()) {
        return false;
    }

    std::shared_ptr<stream_session> sess;
    {
        std::lock_guard<std::mutex> lock(g_stream_mtx);
        for (auto it = g_stream_sessions.begin(); it != g_stream_sessions.end(); ++it) {
            if (it->second.get() == raw_sess) {
                sess = it->second;
                g_stream_sessions.erase(it);
                break;
            }
        }
    }
    if (!sess) {
        LOG_ERR("%s: session not found for stream user_data\n", __func__);
        return false;
    }
    sess->consumed = true;

    struct stream_chunk {
        uint16_t file_idx;
        size_t   offset;
        size_t   len;
    };

    // Determine effective buffer capacity from worker 0
    void * probe_ptr = nullptr;
    size_t buf_cap = sess->opts.chunk_bytes;
    int probe_buf_id = 0;
    if (!ops->acquire_buffer(ops_ctx, 0, &probe_buf_id, &probe_ptr, &buf_cap) || buf_cap == 0) {
        LOG_ERR("%s: acquire_buffer probe failed (cap=%zu)\n", __func__, buf_cap);
        return false;
    }
    const size_t chunk_sz = std::min<size_t>(sess->opts.chunk_bytes, buf_cap);

    std::vector<stream_chunk> chunks;
    size_t total_stream_bytes = 0;
    for (uint16_t fi = 0; fi < n_files; ++fi) {
        const auto & fs = *sess->files[fi];
        // Stream starting from min(data_offset, header_end) so any gap between data_offset and first tensor is covered
        const size_t start_off = std::min(file_data_offsets ? file_data_offsets[fi] : fs.header_end, fs.header_end);
        for (size_t off = start_off; off < fs.resolved.total_size; off += chunk_sz) {
            const size_t len = std::min(chunk_sz, fs.resolved.total_size - off);
            chunks.push_back({fi, off, len});
            total_stream_bytes += len;
        }
    }

    const int n_workers = std::min<int>(std::max(1, sess->opts.n_threads),
                                        static_cast<int>(std::min<size_t>(std::max<size_t>(1, chunks.size()), 32)));

    // Optional bounded background Disk Tee writer
    std::mutex bg_mtx;
    std::condition_variable bg_cv_nonempty;
    std::condition_variable bg_cv_space;
    std::deque<bg_write_item> bg_queue;
    size_t bg_queued_bytes = 0;
    constexpr size_t BG_MAX_QUEUE_BYTES = size_t(512) * 1024 * 1024;
    bool bg_stop = false;
    std::atomic<bool> bg_error{false};
    std::vector<std::thread> bg_writers;

    if (sess->opts.disk_mode == COMMON_HF_STREAM_DISK_BACKGROUND) {
        for (int dw = 0; dw < 2; ++dw) {
            bg_writers.emplace_back([&]() {
                while (true) {
                    bg_write_item item;
                    {
                        std::unique_lock<std::mutex> lock(bg_mtx);
                        bg_cv_nonempty.wait(lock, [&]() { return bg_stop || !bg_queue.empty(); });
                        if (bg_queue.empty() && bg_stop) {
                            break;
                        }
                        item = std::move(bg_queue.front());
                        bg_queue.pop_front();
                        bg_queued_bytes -= item.data.size();
                        bg_cv_space.notify_all();
                    }
                    if (!pwrite_all_fd(item.fd, item.data.data(), item.data.size(), item.offset)) {
                        bg_error.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
    }

    std::atomic<size_t> next_chunk{0};
    std::atomic<bool> failed{false};

    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for (int w = 0; w < n_workers; ++w) {
        workers.emplace_back([&, w]() {
            int cur_file = -1;
            std::unique_ptr<httplib::Client> cli_ptr;
            common_http_url cur_parts;
            httplib::Headers cur_hdrs;

            auto connect_for_file = [&](uint16_t fi) {
                const auto & fs = *sess->files[fi];
                auto [cli, parts] = common_http_client(fs.resolved.final_url);
                cli.set_follow_location(false);
                cli.set_keep_alive(true);
                cli.set_read_timeout(60, 0);
                cli_ptr = std::make_unique<httplib::Client>(std::move(cli));
                cur_parts = std::move(parts);
                cur_hdrs = build_common_headers(fs.task.opts, fs.resolved.same_origin);
                cur_file = static_cast<int>(fi);
            };

            while (!failed.load(std::memory_order_relaxed)) {
                const size_t cidx = next_chunk.fetch_add(1, std::memory_order_relaxed);
                if (cidx >= chunks.size()) {
                    break;
                }
                const auto & ch = chunks[cidx];
                const auto & fs = *sess->files[ch.file_idx];
                if (fs.task.opts.callback && fs.task.opts.callback->is_cancelled()) {
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                if (cur_file != static_cast<int>(ch.file_idx) || !cli_ptr) {
                    connect_for_file(ch.file_idx);
                }

                int buf_id = 0;
                void * raw_ptr = nullptr;
                size_t cap = 0;
                if (!ops->acquire_buffer(ops_ctx, w, &buf_id, &raw_ptr, &cap) || cap < ch.len) {
                    LOG_ERR("%s: worker %d acquire_buffer failed (cap=%zu, len=%zu)\n", __func__, w, cap, ch.len);
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                uint8_t * dst = static_cast<uint8_t *>(raw_ptr);

                bool fetch_ok = false;
                for (int attempt = 0; attempt < 4; ++attempt) {
                    if (attempt > 0) {
                        std::this_thread::sleep_for(std::chrono::seconds(1 << (attempt - 1)));
                        connect_for_file(ch.file_idx);
                    }
                    if (http_fetch_range_into(*cli_ptr, cur_parts.path, cur_hdrs, ch.offset, ch.len, dst, nullptr)) {
                        fetch_ok = true;
                        break;
                    }
                }
                if (!fetch_ok) {
                    LOG_ERR("%s: worker %d http_fetch_range_into failed at offset=%zu len=%zu\n", __func__, w, ch.offset, ch.len);
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }

                // 1. Schedule direct VRAM H2D DMA (and host memcpy) first
                if (!ops->submit_chunk(ops_ctx, w, buf_id, ch.file_idx, ch.offset, ch.len)) {
                    LOG_ERR("%s: worker %d submit_chunk failed at offset=%zu len=%zu\n", __func__, w, ch.offset, ch.len);
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }

                // 2. Concurrent Disk Tee while GPU PCIe DMA runs
                if (sess->opts.disk_mode == COMMON_HF_STREAM_DISK_SYNC) {
                    if (!pwrite_all_fd(fs.fd, dst, ch.len, ch.offset)) {
                        LOG_ERR("%s: worker %d pwrite_all_fd failed at offset=%zu len=%zu errno=%d\n", __func__, w, ch.offset, ch.len, errno);
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                } else if (sess->opts.disk_mode == COMMON_HF_STREAM_DISK_BACKGROUND) {
                    bg_write_item item;
                    item.fd     = fs.fd;
                    item.offset = ch.offset;
                    item.data.assign(dst, dst + ch.len);
                    {
                        std::unique_lock<std::mutex> lock(bg_mtx);
                        bg_cv_space.wait(lock, [&]() {
                            return bg_queued_bytes + item.data.size() <= BG_MAX_QUEUE_BYTES || failed.load(std::memory_order_relaxed);
                        });
                        if (failed.load(std::memory_order_relaxed)) {
                            break;
                        }
                        bg_queued_bytes += item.data.size();
                        bg_queue.push_back(std::move(item));
                    }
                    bg_cv_nonempty.notify_one();
                }
            }
        });
    }

    for (auto & t : workers) {
        t.join();
    }

    if (sess->opts.disk_mode == COMMON_HF_STREAM_DISK_BACKGROUND) {
        // Move queued background writers + session ownership into background completion thread
        // ONLY if streaming succeeded; on failure join/drain synchronously and discard.
        if (failed.load() || bg_error.load()) {
            {
                std::lock_guard<std::mutex> lock(bg_mtx);
                bg_stop = true;
            }
            bg_cv_nonempty.notify_all();
            for (auto & t : bg_writers) {
                t.join();
            }
            for (auto & fs : sess->files) {
                close_rw_fd(fs->fd);
                std::error_code ec;
                std::filesystem::remove(std::filesystem::u8path(fs->temp_path), ec);
                if (fs->lock) {
                    fs->lock->close();
                }
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(bg_mtx);
            bg_stop = true;
        }
        bg_cv_nonempty.notify_all();
        for (auto & t : bg_writers) {
            t.join();
        }
        // Finalize file renames & snapshot symlinks asynchronously after VRAM load completes
        std::thread fin_thread([sess]() {
            for (auto & fs : sess->files) {
                finalize_stream_file(*fs, COMMON_HF_STREAM_DISK_BACKGROUND);
            }
        });
        std::lock_guard<std::mutex> lock(g_stream_mtx);
        g_bg_threads.push_back(std::move(fin_thread));
        return true;
    }

    if (failed.load()) {
        for (auto & fs : sess->files) {
            close_rw_fd(fs->fd);
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(fs->temp_path), ec);
            if (fs->lock) {
                fs->lock->close();
            }
        }
        return false;
    }

    for (auto & fs : sess->files) {
        finalize_stream_file(*fs, sess->opts.disk_mode);
    }
    return true;
}
