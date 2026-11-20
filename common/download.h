#pragma once

#include "hf-cache.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

struct common_params_model;

using common_header      = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

struct common_download_progress {
    std::string url;
    size_t downloaded = 0;
    size_t total      = 0;
    bool cached       = false;
};

class common_download_callback {
public:
    virtual ~common_download_callback() = default;
    virtual void on_start(const common_download_progress & p) = 0;
    virtual void on_update(const common_download_progress & p) = 0;
    virtual void on_done(const common_download_progress & p, bool ok) = 0;
    virtual bool is_cancelled() const { return false; }
};

struct common_remote_params {
    common_header_list headers;
    long timeout  = 0;           // in seconds, 0 means no timeout
    long max_size = 0;           // unlimited if 0
};

// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);

// split HF repo with tag into <repo, tag>, for example:
// - "ggml-org/models:F16" -> <"ggml-org/models", "F16">
// tag is optional and can be empty
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);

// Result of common_list_cached_models
struct common_cached_model_info {
    std::string repo;
    std::string tag;
    std::string to_string() const {
        return repo + ":" + tag;
    }
};

// Options for common_download_file_single
struct common_download_opts {
    std::string bearer_token;
    common_header_list headers;
    bool offline = false;
    bool discover_exl3 = false;
    bool download_mmproj  = false;
    bool download_mtp     = false;
    bool download_eagle3  = false;
    bool download_dflash  = false;
    bool download_dspark  = false;
    int  n_threads        = 8;
    common_download_callback * callback = nullptr;
};

struct common_download_task {
    common_download_opts opts;
    std::string url;
    std::string local_path;
    std::function<void()> on_done;
    bool is_hf = false;
    std::string lock_path;
    std::string alt_path; // HF snapshot path, also treated as cached (the blob may be moved there)

    common_download_task() = default;
    common_download_task(hf_cache::hf_file f,
            const common_download_opts & opts,
            std::function<void()> on_done = nullptr)
        : opts(opts), url(f.url), local_path(f.local_path), on_done(on_done), is_hf(true),
          lock_path(hf_cache::get_lock_path(f.repo_id, f.oid)), alt_path(f.final_path) {}
};

void common_download_run_tasks(const std::vector<common_download_task> & tasks);

// if url is a multi-part GGUF file, returns all parts, otherwise returns the single file
std::vector<std::string> common_download_get_all_parts(const std::string & url);

// returns list of cached models
std::vector<common_cached_model_info> common_list_cached_models();

// resolve the local cached file path for a HF repo without network access (hf_file, if given, must match exactly)
// returns an empty string if the model is not present in the cache
std::string common_download_resolve_path(const std::string & hf_repo_with_tag, const std::string & hf_file = "");

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
// lock_path: lock file used to serialize downloads of the same file; when empty, "<path>.lock" is used
// alt_path: additional path treated as present in cache (see cache_hit in download.cpp)
int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts = {},
                                bool skip_etag = false,
                                const std::string & lock_path = "",
                                const std::string & alt_path = "");

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string common_docker_resolve_model(const std::string & docker);

// Remove a cached model from disk
// input format: "user/model" or "user/model:tag"
// - if tag is omitted, removes the entire repo cache directory
// - if tag is present, removes only files matching that tag (and orphaned blobs),
//   except the files another cached tag of the repo still loads
// returns true if anything was removed
bool common_download_remove(const std::string & hf_repo_with_tag);

// remove the cached sidecar files `opts` selects for "user/model:tag", as the cached
// plan resolves them, and leave the model files in place; a sidecar file another tag
// shares goes for every tag that resolves it
// returns true if anything was removed
bool common_download_remove_sidecars(const std::string & hf_repo_with_tag, const common_download_opts & opts);

struct common_download_hf_plan {
    hf_cache::hf_file primary;
    hf_cache::hf_files model_files;
    hf_cache::hf_files exl3_meta_files;
    hf_cache::hf_files exl3_tensor_files;
    bool is_exl3 = false;
    hf_cache::hf_file mmproj;
    hf_cache::hf_file mtp;
    hf_cache::hf_file eagle3;
    hf_cache::hf_file dflash;
    hf_cache::hf_file dspark;
    hf_cache::hf_file preset; // if set, only this file is downloaded
};
common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts);

// a sidecar kind a download can pull next to its model: mmproj, and the draft heads
// of --spec-type draft-<name>
struct common_download_sidecar {
    const char * name;    // the kind, as the API names it
    const char * keyword; // the part of a file name that marks a file of this kind
    bool         draft;   // a draft head, resolved at the tag even without a model

    bool              common_download_opts::*    opt;  // the option that requests it
    hf_cache::hf_file common_download_hf_plan::* file; // the plan file it resolves to
};

// every sidecar kind, the single list the resolution, the listing and the removal follow
const std::vector<common_download_sidecar> & common_download_sidecars();

// select the sidecar `kind` in `opts`; returns false for an unknown kind
bool common_download_opts_add_sidecar(common_download_opts & opts, const std::string & kind);

// the sidecar kinds `plan` resolves a file for
std::vector<std::string> common_download_plan_sidecars(const common_download_hf_plan & plan);

// a file `plan` still has to download, at the path its partial takes while it is
// incomplete; kind is the sidecar it belongs to, empty for the model files
struct common_download_partial {
    std::string kind;
    std::string path;
};

// the partials the download of `plan` writes, for the files not in the cache yet
std::vector<common_download_partial> common_download_plan_partials(const common_download_hf_plan & plan);

enum common_hf_stream_disk_mode {
    COMMON_HF_STREAM_DISK_SYNC       = 0,
    COMMON_HF_STREAM_DISK_BACKGROUND = 1,
    COMMON_HF_STREAM_DISK_NONE       = 2,
};

struct common_download_stream_opts {
    int n_threads = 8;
    size_t chunk_bytes = size_t(8) * 1024 * 1024;
    enum common_hf_stream_disk_mode disk_mode = COMMON_HF_STREAM_DISK_SYNC;
};

// Prepare GGUF model download tasks for direct HTTP Range streaming to VRAM (+ Disk Tee).
// If all tasks fit streaming preconditions (uncached, server supports byte ranges), downloads
// only the GGUF header [0, data_offset) to `.downloadInProgress`, pre-sizes the temp file,
// registers the stream session under out_paths.front(), and returns true.
// Returns false if already cached or range streaming is inapplicable (fall back to regular download).
bool common_download_prepare_stream(
    const std::vector<common_download_task> & tasks,
    const common_download_stream_opts & opts,
    std::vector<std::string> & out_paths);

// Prepare an HTTP Range streaming session for an EXL3 safetensors directory keyed by dir_path.
// Probes each safetensors header [0, 8+N) in parallel and writes .hfstream.tmp header prefixes.
bool common_download_prepare_stream_exl3(
    const std::string & dir_path,
    const std::vector<common_download_task> & tasks,
    const std::vector<std::string> & rel_paths,
    const common_download_stream_opts & opts);

// Read a small byte range [offset, offset + size) from file_idx within an active stream session.
bool common_download_stream_read_range(
    const std::string & session_key,
    size_t file_idx,
    size_t offset,
    size_t size,
    void * dst);

void * common_download_get_stream_session_ud(const std::string & first_path);

// Release a prepared-but-unused stream session (lock, fd, temp file). No-op once the loader consumed it.
void common_download_stream_abort(const std::string & first_path);

struct common_download_stream_file_info {
    std::string temp_path;
    std::string local_path;
    std::string final_path;
    std::string rel_path;
    std::string etag;
    std::string url;
    size_t      total_size = 0;
    size_t      header_end = 0;
};

// Inspect file metadata of an active (unconsumed) stream session without consuming it.
bool common_download_get_stream_file_infos(
    const std::string & first_path,
    std::vector<common_download_stream_file_info> & out_infos);

// Return all shard temp paths for an active multi-part stream session (empty if not found or single-file).
std::vector<std::string> common_download_get_stream_splits(const std::string & first_path);

// Final on-disk path of a streamed temp file ("" if unknown or disk-mode=none).
// In background mode the file appears there once common_download_wait_background() returns.
std::string common_download_stream_final_path(const std::string & temp_path);

bool common_download_stream_loader(
    const struct llama_model_stream_loader_ops * ops,
    void * ops_ctx,
    uint16_t n_files,
    const size_t * file_data_offsets,
    void * user_data);

void common_download_wait_background();
