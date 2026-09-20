#include "server-gpu-power.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#    include <dlfcn.h>
#endif

static const char * server_gpu_power_phase_name(server_gpu_power_phase phase) {
    switch (phase) {
        case server_gpu_power_phase::idle:
            return "idle";
        case server_gpu_power_phase::prefill:
            return "prefill";
        case server_gpu_power_phase::decode:
            return "decode";
    }

    return "unknown";
}

static std::string server_gpu_power_mw_to_string(uint32_t power_mw) {
    const uint32_t watts = power_mw / 1000;
    const uint32_t mw    = power_mw % 1000;

    if (mw == 0) {
        return std::to_string(watts);
    }

    std::string result = std::to_string(watts) + ".";
    if (mw < 100) {
        result += "0";
    }
    if (mw < 10) {
        result += "0";
    }
    result += std::to_string(mw);
    return result;
}

static bool server_gpu_power_w_to_mw(int32_t power_w, uint32_t & power_mw) {
    if (power_w <= 0 || static_cast<uint64_t>(power_w) > std::numeric_limits<uint32_t>::max() / 1000u) {
        return false;
    }

    power_mw = static_cast<uint32_t>(power_w) * 1000u;
    return true;
}

void server_gpu_power_phase_arbitrator::reset() {
    phase_ = server_gpu_power_phase::idle;
}

void server_gpu_power_phase_arbitrator::observe(server_gpu_power_slot_state state) {
    if (phase_ == server_gpu_power_phase::prefill) {
        return;
    }

    switch (state) {
        case server_gpu_power_slot_state::started:
        case server_gpu_power_slot_state::processing_prompt:
        case server_gpu_power_slot_state::done_prompt:
            phase_ = server_gpu_power_phase::prefill;
            break;
        case server_gpu_power_slot_state::generating:
            phase_ = server_gpu_power_phase::decode;
            break;
        case server_gpu_power_slot_state::idle:
        case server_gpu_power_slot_state::wait_other:
            break;
    }
}

server_gpu_power_phase server_gpu_power_phase_arbitrator::phase() const {
    return phase_;
}

bool server_gpu_power_config::enabled() const {
    return power_enabled() || mem_clock_enabled() || fabric_state != -1 || apu_tdp_w != -1;
}

bool server_gpu_power_config::power_enabled() const {
    return prefill_w != -1 || decode_w != -1;
}

bool server_gpu_power_config::mem_clock_enabled() const {
    return mem_clock_decode > 0 || mem_clock_prefill > 0;
}

namespace {

using nvml_return_t = unsigned int;
struct nvml_device_st;
using nvml_device_t = nvml_device_st *;

using nvml_init_t                                          = nvml_return_t (*)();
using nvml_shutdown_t                                      = nvml_return_t (*)();
using nvml_device_get_handle_by_index_t                    = nvml_return_t (*)(unsigned int, nvml_device_t *);
using nvml_device_get_name_t                               = nvml_return_t (*)(nvml_device_t, char *, unsigned int);
using nvml_device_get_power_management_limit_constraints_t = nvml_return_t (*)(nvml_device_t,
                                                                               unsigned int *,
                                                                               unsigned int *);
using nvml_device_get_power_management_limit_t             = nvml_return_t (*)(nvml_device_t, unsigned int *);
using nvml_device_set_power_management_limit_t             = nvml_return_t (*)(nvml_device_t, unsigned int);
using nvml_device_get_supported_memory_clocks_t            = nvml_return_t (*)(nvml_device_t, unsigned int *, unsigned int *);
using nvml_device_set_memory_locked_clocks_t               = nvml_return_t (*)(nvml_device_t, unsigned int, unsigned int);
using nvml_device_reset_memory_locked_clocks_t             = nvml_return_t (*)(nvml_device_t);
using nvml_device_set_mem_clk_vf_offset_t                   = nvml_return_t (*)(nvml_device_t, int);
using nvml_device_get_mem_clk_vf_offset_t                   = nvml_return_t (*)(nvml_device_t, int *);
using nvml_device_get_mem_clk_min_max_vf_offset_t           = nvml_return_t (*)(nvml_device_t, int *, int *);
using nvml_device_get_min_max_clock_of_pstate_t             = nvml_return_t (*)(nvml_device_t, unsigned int, unsigned int,
                                                                             unsigned int *, unsigned int *);
using nvml_error_string_t                                  = const char * (*) (nvml_return_t);

#if defined(_WIN32)
using server_gpu_power_library_handle = HMODULE;
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
using server_gpu_power_library_handle = void *;
#else
using server_gpu_power_library_handle = void *;
#endif

template <typename T>
static bool server_gpu_power_resolve_symbol(server_gpu_power_library_handle library, const char * name, T & target) {
    void * symbol = nullptr;

#if defined(_WIN32)
    FARPROC symbol_win = GetProcAddress(library, name);
    if (symbol_win == nullptr) {
        return false;
    }
    std::memcpy(&symbol, &symbol_win, sizeof(symbol));
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    symbol = dlsym(library, name);
    if (symbol == nullptr) {
        return false;
    }
#else
    (void) library;
    (void) name;
    (void) target;
    return false;
#endif

    static_assert(sizeof(T) == sizeof(void *), "function pointers must fit in a library symbol pointer");
    std::memcpy(&target, &symbol, sizeof(target));
    return true;
}

class server_gpu_power_nvml_backend final : public server_gpu_power_backend {
  public:
    ~server_gpu_power_nvml_backend() override { shutdown(); }

    bool init(int32_t device, server_gpu_power_device_info & info, std::string & error) override {
        shutdown();
        info = {};

        if (device < 0) {
            error = "NVML device index must be non-negative";
            return false;
        }

#if defined(_WIN32)
        const char * library_names[] = { "nvml.dll" };
#elif defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        const char * library_names[] = { "libnvidia-ml.so.1", "libnvidia-ml.so" };
#else
        error = "NVML runtime loading is not supported on this platform";
        return false;
#endif

#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        for (const char * name : library_names) {
            library_ = load_library(name);
            if (library_ != nullptr) {
                break;
            }
        }

        if (library_ == nullptr) {
            error = "could not load the NVML runtime library";
            return false;
        }

        if (!resolve_symbols(error)) {
            close_library();
            return false;
        }

        nvml_return_t result = nvml_init_();
        if (!check_result(result, "nvmlInit_v2", error)) {
            close_library();
            return false;
        }
        initialized_ = true;

        result = nvml_device_get_handle_by_index_(static_cast<unsigned int>(device), &device_);
        if (!check_result(result, "nvmlDeviceGetHandleByIndex_v2", error)) {
            shutdown();
            return false;
        }

        char name[256] = {};
        result         = nvml_device_get_name_(device_, name, sizeof(name));
        if (!check_result(result, "nvmlDeviceGetName", error)) {
            shutdown();
            return false;
        }

        unsigned int min_mw = 0;
        unsigned int max_mw = 0;
        result              = nvml_device_get_power_management_limit_constraints_(device_, &min_mw, &max_mw);
        if (!check_result(result, "nvmlDeviceGetPowerManagementLimitConstraints", error)) {
            shutdown();
            return false;
        }

        unsigned int current_mw = 0;
        result                  = nvml_device_get_power_management_limit_(device_, &current_mw);
        if (!check_result(result, "nvmlDeviceGetPowerManagementLimit", error)) {
            shutdown();
            return false;
        }

        info.name                    = name;
        info.device                  = device;
        info.original_power_limit_mw = current_mw;
        info.min_power_limit_mw      = min_mw;
        info.max_power_limit_mw      = max_mw;

        if (nvml_device_get_supported_memory_clocks_ != nullptr) {
            unsigned int count = 0;
            nvml_return_t clk_res = nvml_device_get_supported_memory_clocks_(device_, &count, nullptr);
            if (count > 0) {
                std::vector<unsigned int> clocks(count);
                clk_res = nvml_device_get_supported_memory_clocks_(device_, &count, clocks.data());
                if (clk_res == 0) {
                    clocks.resize(count);
                    info.supported_mem_clocks_mhz.assign(clocks.begin(), clocks.end());
                }
            }
        }

        unsigned int min_p2_mhz = 0;
        unsigned int max_p2_mhz = 0;
        int min_offset = 0;
        int max_offset = 0;
        // NVML_CLOCK_MEM = 2, NVML_PSTATE_2 = 2 (CUDA compute state).
        if (nvml_device_set_mem_clk_vf_offset_ != nullptr &&
            nvml_device_get_mem_clk_vf_offset_ != nullptr &&
            nvml_device_get_mem_clk_min_max_vf_offset_ != nullptr &&
            nvml_device_get_min_max_clock_of_pstate_ != nullptr &&
            nvml_device_get_mem_clk_vf_offset_(device_, &original_mem_offset_mhz_) == 0 &&
            nvml_device_get_mem_clk_min_max_vf_offset_(device_, &min_offset, &max_offset) == 0 &&
            nvml_device_get_min_max_clock_of_pstate_(device_, 2, 2, &min_p2_mhz, &max_p2_mhz) == 0 &&
            max_p2_mhz > 0 && min_offset <= max_offset) {
            info.memory_clock_p2_mhz = max_p2_mhz;
            info.min_memory_clock_offset_mhz = min_offset;
            info.max_memory_clock_offset_mhz = max_offset;
            info.memory_clock_offset_supported = true;
        }
        return true;
#else
        (void) info;
        (void) error;
        return false;
#endif
    }

    bool set_power_limit(uint32_t power_limit_mw, std::string & error) override {
#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!initialized_ || device_ == nullptr) {
            error = "NVML backend is not initialized";
            return false;
        }

        const nvml_return_t result = nvml_device_set_power_management_limit_(device_, power_limit_mw);
        return check_result(result, "nvmlDeviceSetPowerManagementLimit", error);
#else
        (void) power_limit_mw;
        error = "NVML runtime loading is not supported on this platform";
        return false;
#endif
    }

    bool set_memory_locked_clocks(uint32_t min_mhz, uint32_t max_mhz, std::string & error) override {
#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!initialized_ || device_ == nullptr || nvml_device_set_memory_locked_clocks_ == nullptr) {
            error = "NVML backend memory locked clocks not supported or not initialized";
            return false;
        }

        const nvml_return_t result = nvml_device_set_memory_locked_clocks_(device_, min_mhz, max_mhz);
        return check_result(result, "nvmlDeviceSetMemoryLockedClocks", error);
#else
        (void) min_mhz;
        (void) max_mhz;
        error = "NVML runtime loading is not supported on this platform";
        return false;
#endif
    }

    bool reset_memory_locked_clocks(std::string & error) override {
#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!initialized_ || device_ == nullptr || nvml_device_reset_memory_locked_clocks_ == nullptr) {
            error = "NVML backend memory locked clocks not supported or not initialized";
            return false;
        }

        const nvml_return_t result = nvml_device_reset_memory_locked_clocks_(device_);
        return check_result(result, "nvmlDeviceResetMemoryLockedClocks", error);
#else
        error = "NVML runtime loading is not supported on this platform";
        return false;
#endif
    }

    bool set_memory_clock_offset(int32_t offset_mhz, std::string & error) override {
#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (!initialized_ || device_ == nullptr || nvml_device_set_mem_clk_vf_offset_ == nullptr) {
            error = "NVML backend memory clock offset not supported or not initialized";
            return false;
        }

        const nvml_return_t result = nvml_device_set_mem_clk_vf_offset_(device_, static_cast<int>(offset_mhz));
        return check_result(result, "nvmlDeviceSetMemClkVfOffset", error);
#else
        (void) offset_mhz;
        error = "NVML runtime loading is not supported on this platform";
        return false;
#endif
    }

    bool reset_memory_clock_offset(std::string & error) override {
        return set_memory_clock_offset(original_mem_offset_mhz_, error);
    }

    void shutdown() override {
#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
        if (initialized_ && nvml_shutdown_ != nullptr) {
            const nvml_return_t result = nvml_shutdown_();
            if (result != 0) {
                LOG_WRN("GPU power: nvmlShutdown failed with code %u\n", result);
            }
        }

        initialized_                     = false;
        device_                          = nullptr;
        original_mem_offset_mhz_          = 0;
        nvml_device_set_mem_clk_vf_offset_ = nullptr;
        close_library();
#endif
    }

  private:
    int original_mem_offset_mhz_ = 0;

#if defined(_WIN32) || defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    server_gpu_power_library_handle load_library(const char * name) {
#    if defined(_WIN32)
        return LoadLibraryA(name);
#    else
        return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#    endif
    }

    void close_library() {
        if (library_ == nullptr) {
            return;
        }

#    if defined(_WIN32)
        FreeLibrary(library_);
#    else
        dlclose(library_);
#    endif
        library_ = nullptr;
    }

    bool resolve_symbols(std::string & error) {
        nvml_init_                                          = nullptr;
        nvml_shutdown_                                      = nullptr;
        nvml_device_get_handle_by_index_                    = nullptr;
        nvml_device_get_name_                               = nullptr;
        nvml_device_get_power_management_limit_constraints_ = nullptr;
        nvml_device_get_power_management_limit_             = nullptr;
        nvml_device_set_power_management_limit_             = nullptr;
        nvml_device_get_supported_memory_clocks_            = nullptr;
        nvml_device_set_memory_locked_clocks_               = nullptr;
        nvml_device_reset_memory_locked_clocks_             = nullptr;
        nvml_device_set_mem_clk_vf_offset_                   = nullptr;
        nvml_device_get_mem_clk_vf_offset_                   = nullptr;
        nvml_device_get_mem_clk_min_max_vf_offset_           = nullptr;
        nvml_device_get_min_max_clock_of_pstate_             = nullptr;
        nvml_error_string_                                  = nullptr;

        const bool resolved = server_gpu_power_resolve_symbol(library_, "nvmlInit_v2", nvml_init_) &&
                              server_gpu_power_resolve_symbol(library_, "nvmlShutdown", nvml_shutdown_) &&
                              server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetHandleByIndex_v2",
                                                              nvml_device_get_handle_by_index_) &&
                              server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetName", nvml_device_get_name_) &&
                              server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetPowerManagementLimitConstraints",
                                                              nvml_device_get_power_management_limit_constraints_) &&
                              server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetPowerManagementLimit",
                                                              nvml_device_get_power_management_limit_) &&
                              server_gpu_power_resolve_symbol(library_, "nvmlDeviceSetPowerManagementLimit",
                                                              nvml_device_set_power_management_limit_);

        server_gpu_power_resolve_symbol(library_, "nvmlErrorString", nvml_error_string_);

        server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetSupportedMemoryClocks",
                                        nvml_device_get_supported_memory_clocks_);
        server_gpu_power_resolve_symbol(library_, "nvmlDeviceSetMemoryLockedClocks",
                                        nvml_device_set_memory_locked_clocks_);
        server_gpu_power_resolve_symbol(library_, "nvmlDeviceResetMemoryLockedClocks",
                                        nvml_device_reset_memory_locked_clocks_);
        server_gpu_power_resolve_symbol(library_, "nvmlDeviceSetMemClkVfOffset",
                                        nvml_device_set_mem_clk_vf_offset_);
        server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetMemClkVfOffset",
                                        nvml_device_get_mem_clk_vf_offset_);
        server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetMemClkMinMaxVfOffset",
                                        nvml_device_get_mem_clk_min_max_vf_offset_);
        server_gpu_power_resolve_symbol(library_, "nvmlDeviceGetMinMaxClockOfPState",
                                        nvml_device_get_min_max_clock_of_pstate_);

        if (!resolved) {
            error = "NVML runtime is missing one of the required symbols";
            return false;
        }

        return true;
    }

    bool check_result(nvml_return_t result, const char * operation, std::string & error) const {
        if (result == 0) {
            return true;
        }

        const char * description = nvml_error_string_ != nullptr ? nvml_error_string_(result) : nullptr;
        error                    = std::string(operation) + " failed: " +
                (description != nullptr ? description : ("NVML error code " + std::to_string(result)));
        return false;
    }

    server_gpu_power_library_handle library_     = nullptr;
    nvml_device_t                   device_      = nullptr;
    bool                            initialized_ = false;

    nvml_init_t                                          nvml_init_                                          = nullptr;
    nvml_shutdown_t                                      nvml_shutdown_                                      = nullptr;
    nvml_device_get_handle_by_index_t                    nvml_device_get_handle_by_index_                    = nullptr;
    nvml_device_get_name_t                               nvml_device_get_name_                               = nullptr;
    nvml_device_get_power_management_limit_constraints_t nvml_device_get_power_management_limit_constraints_ = nullptr;
    nvml_device_get_power_management_limit_t             nvml_device_get_power_management_limit_             = nullptr;
    nvml_device_set_power_management_limit_t             nvml_device_set_power_management_limit_             = nullptr;
    nvml_device_get_supported_memory_clocks_t            nvml_device_get_supported_memory_clocks_            = nullptr;
    nvml_device_set_memory_locked_clocks_t               nvml_device_set_memory_locked_clocks_               = nullptr;
    nvml_device_reset_memory_locked_clocks_t             nvml_device_reset_memory_locked_clocks_             = nullptr;
    nvml_device_set_mem_clk_vf_offset_t                   nvml_device_set_mem_clk_vf_offset_                   = nullptr;
    nvml_device_get_mem_clk_vf_offset_t                   nvml_device_get_mem_clk_vf_offset_                   = nullptr;
    nvml_device_get_mem_clk_min_max_vf_offset_t           nvml_device_get_mem_clk_min_max_vf_offset_           = nullptr;
    nvml_device_get_min_max_clock_of_pstate_t             nvml_device_get_min_max_clock_of_pstate_             = nullptr;
    nvml_error_string_t                                  nvml_error_string_                                  = nullptr;
#endif
};

// AMDGPU sysfs backend: drives the Radeon power/clock state through the
// kernel sysfs interface (the same native mechanism hhd and other tools use),
// without depending on any external daemon.
//
//   power_dpm_force_performance_level : auto|low|high|manual|profile_*
//   pp_od_clk_voltage                 : "s 0 <min>", "s 1 <max>", "c"
//
// Legacy memory-clock options control SCLK (the graphics clock) on AMD.
// Fabric control uses the raw pp_dpm_fclk index, not MHz or a display index.

static bool server_gpu_power_write_sysfs(const std::string & path, const std::string & value) {
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f) {
        return false;
    }
    f << value;
    f.close(); // sysfs errors may only surface when the stream buffer is flushed.
    return !f.fail();
}

static bool server_gpu_power_read_sysfs(const std::string & path, std::string & value) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    value = ss.str();
    return f.good() || f.eof();
}

static std::string server_gpu_power_trim(const std::string & s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

class server_gpu_power_amdgpu_backend final : public server_gpu_power_backend {
  public:
    server_gpu_power_amdgpu_backend(std::string drm_dir,
                                   std::function<bool(const std::string &, const std::string &)> writer,
                                   const server_gpu_power_ryzenadj_api * apu_api)
        : drm_dir_(std::move(drm_dir)), writer_(writer ? std::move(writer) : server_gpu_power_write_sysfs),
          apu_api_(apu_api ? *apu_api : server_gpu_power_ryzenadj_api{}), apu_api_injected_(apu_api != nullptr) {}

    ~server_gpu_power_amdgpu_backend() override { shutdown(); }

    bool init(int32_t device, server_gpu_power_device_info & info, std::string & error) override {
        shutdown();
        info = {};

        if (device < 0) {
            error = "AMDGPU device index must be non-negative";
            return false;
        }

        // Locate the AMD GPU drm cards in /sys/class/drm (vendor 0x1002).
        std::vector<std::string> amd_cards;
        const std::string & drm_dir = drm_dir_;
        if (std::filesystem::exists(drm_dir)) {
            for (const auto & entry : std::filesystem::directory_iterator(drm_dir)) {
                const std::string name = entry.path().filename().string();
                if (name.rfind("card", 0) != 0) {
                    continue;
                }
                const std::string vendor_path = entry.path().string() + "/device/vendor";
                std::string vendor;
                if (!server_gpu_power_read_sysfs(vendor_path, vendor)) {
                    continue;
                }
                if (server_gpu_power_trim(vendor) == "0x1002") {
                    amd_cards.push_back(entry.path().string());
                }
            }
        }
        std::sort(amd_cards.begin(), amd_cards.end());

        if (amd_cards.empty()) {
            error = "no AMDGPU drm card found in " + drm_dir;
            return false;
        }
        if (static_cast<size_t>(device) >= amd_cards.size()) {
            error = "AMDGPU device index " + std::to_string(device) +
                    " is out of range (found " + std::to_string(amd_cards.size()) + " AMD card(s))";
            return false;
        }

        card_dir_ = amd_cards[device];

        const std::string level_path = card_dir_ + "/device/power_dpm_force_performance_level";
        const std::string od_path    = card_dir_ + "/device/pp_od_clk_voltage";

        if (!server_gpu_power_read_sysfs(level_path, original_level_)) {
            error = "cannot read " + level_path;
            return false;
        }
        original_level_ = server_gpu_power_trim(original_level_);

        // OD_SCLK is the current custom range; OD_RANGE is only the allowed range.
        std::string od;
        if (server_gpu_power_read_sysfs(od_path, od)) {
            bool in_sclk = false;
            for (const auto & line : split_lines(od)) {
                const std::string text = server_gpu_power_trim(line);
                if (text.rfind("OD_", 0) == 0) {
                    in_sclk = text == "OD_SCLK:";
                    continue;
                }
                std::istringstream iss(text);
                std::string label;
                uint32_t lo = 0, hi = 0;
                std::string unit;
                if (iss >> label >> lo) {
                    if (in_sclk && label == "0:") original_sclk_min_ = lo;
                    if (in_sclk && label == "1:") original_sclk_max_ = lo;
                    if (label == "SCLK:" && iss >> unit >> hi && lo > 0 && hi >= lo) {
                        od_sclk_min_ = lo;
                        od_sclk_max_ = hi;
                    }
                }
            }
        }
        std::string fabric;
        info.fabric_state_supported = original_level_ != "manual" &&
            server_gpu_power_read_sysfs(card_dir_ + "/device/pp_dpm_fclk", fabric) &&
            !server_gpu_power_trim(fabric).empty();

        std::string sclk;
        if (server_gpu_power_read_sysfs(card_dir_ + "/device/pp_dpm_sclk", sclk)) {
            for (const auto & line : split_lines(sclk)) {
                std::istringstream iss(line);
                std::string level;
                int mhz = 0;
                std::string unit;
                if (iss >> level >> mhz >> unit) {
                    if (mhz > 0) {
                        info.supported_mem_clocks_mhz.push_back(static_cast<uint32_t>(mhz));
                    }
                }
            }
            std::sort(info.supported_mem_clocks_mhz.begin(), info.supported_mem_clocks_mhz.end());
            info.supported_mem_clocks_mhz.erase(
                std::unique(info.supported_mem_clocks_mhz.begin(), info.supported_mem_clocks_mhz.end()),
                info.supported_mem_clocks_mhz.end());
        }


        info.name                    = "AMDGPU " + card_dir_ + " (sysfs)";
        info.device                  = device;
        info.original_power_limit_mw = 0;
        info.min_power_limit_mw      = 0;
        info.max_power_limit_mw      = 0;
        info.memory_clock_offset_supported = false;

        LOG_INF("AMDGPU governor: card %s, original level %s, SCLK range %u-%u MHz\n",
                card_dir_.c_str(), original_level_.c_str(), od_sclk_min_, od_sclk_max_);

        initialized_ = true;
        return true;
    }

    bool set_power_limit(uint32_t power_limit_mw, std::string & error) override {
        (void) power_limit_mw;
        error = "AMDGPU sysfs backend does not support power limit changes";
        return false;
    }

    bool set_memory_locked_clocks(uint32_t min_mhz, uint32_t max_mhz, std::string & error) override {
        if (!initialized_ || original_sclk_min_ == 0 || original_sclk_max_ < original_sclk_min_ ||
            min_mhz < od_sclk_min_ || max_mhz > od_sclk_max_ || min_mhz == 0 || min_mhz > max_mhz) {
            error = "AMDGPU SCLK range is unavailable or requested clock is outside OD_RANGE";
            return false;
        }
        if (!enter_manual(error)) {
            return rollback(error);
        }
        // Mark dirty before the first write: a later write or commit can fail.
        sclk_changed_ = true;
        if (!write("pp_od_clk_voltage", "s 0 " + std::to_string(min_mhz) + "\n", error) ||
            !write("pp_od_clk_voltage", "s 1 " + std::to_string(max_mhz) + "\n", error) ||
            !write("pp_od_clk_voltage", "c\n", error)) {
            return rollback(error);
        }
        return true;
    }

    bool reset_memory_locked_clocks(std::string & error) override {
        bool success = true;
        if (sclk_changed_) {
            // Reset and commit the driver defaults before restoring custom OD.
            level_locked_ = true;
            success = write("power_dpm_force_performance_level", "manual\n", error);
            if (success) success = write("pp_od_clk_voltage", "r\n", error);
            if (success) success = write("pp_od_clk_voltage", "c\n", error);
            if (success && (original_sclk_min_ != od_sclk_min_ || original_sclk_max_ != od_sclk_max_)) {
                success = write("pp_od_clk_voltage", "s 0 " + std::to_string(original_sclk_min_) + "\n", error) &&
                          write("pp_od_clk_voltage", "s 1 " + std::to_string(original_sclk_max_) + "\n", error) &&
                          write("pp_od_clk_voltage", "c\n", error);
            }
            if (success) sclk_changed_ = false;
        }
        if (!fabric_locked_ || !success) {
            std::string level_error;
            if (!restore_level(level_error)) {
                if (!success) error += "; restore level: " + level_error;
                else error = level_error;
                success = false;
            }
        }
        return success;
    }

    bool set_fabric_state(int32_t state, std::string & error) override {
        if (!initialized_ || state < 0 || state > 31 || original_level_ == "manual") {
            error = "AMDGPU fabric requires a raw state 0..31 and a restorable non-manual original level";
            return false;
        }
        if (!enter_manual(error)) return rollback(error);
        fabric_locked_ = true;
        if (!write("pp_dpm_fclk", std::to_string(state) + "\n", error)) return rollback(error);
        return true;
    }

    bool reset_fabric_state(std::string & error) override {
        if (!fabric_locked_) return restore_level_if_unused(error);
        // The governor resets SCLK first. Release the fabric mask and profile
        // even if that earlier OD restore failed; keep SCLK dirty for a retry.
        if (!restore_level(error)) return false;
        fabric_locked_ = false;
        return true;
    }

    bool set_memory_clock_offset(int32_t offset_mhz, std::string & error) override {
        (void) offset_mhz;
        error = "AMDGPU sysfs backend does not support clock offsets";
        return false;
    }

    bool reset_memory_clock_offset(std::string & error) override {
        (void) error;
        return true;
    }

    bool init_apu_tdp(std::string & error) override {
        if (!initialized_) {
            error = "AMDGPU backend is not initialized";
            return false;
        }
        if (apu_access_) return true;
        if (!apu_api_injected_) {
#if defined(__linux__)
            apu_library_ = dlopen("libryzenadj.so", RTLD_NOW | RTLD_LOCAL);
            if (!apu_library_) {
                error = "cannot load libryzenadj.so for --apu-tdp";
                return false;
            }
            bool resolved = server_gpu_power_resolve_symbol(apu_library_, "init_ryzenadj", apu_api_.init) &&
                            server_gpu_power_resolve_symbol(apu_library_, "cleanup_ryzenadj", apu_api_.cleanup) &&
                            server_gpu_power_resolve_symbol(apu_library_, "init_table", apu_api_.init_table) &&
                            server_gpu_power_resolve_symbol(apu_library_, "refresh_table", apu_api_.refresh_table);
            const char * names[] = {"stapm", "fast", "slow"};
            for (int i = 0; i < 3 && resolved; ++i) {
                resolved = server_gpu_power_resolve_symbol(apu_library_, ("get_" + std::string(names[i]) + "_limit").c_str(), apu_api_.get_limits[i]) &&
                           server_gpu_power_resolve_symbol(apu_library_, ("set_" + std::string(names[i]) + "_limit").c_str(), apu_api_.set_limits[i]);
            }
            if (!resolved) {
                error = "libryzenadj.so lacks required TDP symbols";
                cleanup_apu_tdp();
                return false;
            }
#else
            error = "AMDGPU RyzenAdj TDP control requires Linux";
            return false;
#endif
        }
        bool complete = apu_api_.init && apu_api_.cleanup && apu_api_.init_table && apu_api_.refresh_table;
        for (int i = 0; i < 3; ++i) complete = complete && apu_api_.get_limits[i] && apu_api_.set_limits[i];
        if (!complete) {
            error = "incomplete RyzenAdj TDP API";
            cleanup_apu_tdp();
            return false;
        }
        apu_access_ = apu_api_.init();
        if (!apu_access_ || apu_api_.init_table(apu_access_) != 0 || apu_api_.refresh_table(apu_access_) != 0) {
            error = "cannot initialize RyzenAdj access or refresh the APU power table";
            cleanup_apu_tdp();
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            const double mw = static_cast<double>(apu_api_.get_limits[i](apu_access_)) * 1000.0;
            if (!std::isfinite(mw) || mw < 1 || mw > std::numeric_limits<uint32_t>::max()) {
                error = "RyzenAdj returned an invalid original APU power limit";
                cleanup_apu_tdp();
                return false;
            }
            original_apu_limits_[i] = static_cast<uint32_t>(std::llround(mw));
        }
        LOG_INF("APU TDP: original STAPM/fast/slow limits %u/%u/%u mW\n",
                original_apu_limits_[0], original_apu_limits_[1], original_apu_limits_[2]);
        return true;
    }

    bool set_apu_tdp(uint32_t mw, std::string & error) override {
        if (!apu_access_ || mw == 0) {
            error = "APU TDP is not initialized or target is zero";
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            apu_limit_changed_[i] = true;
            const int result = apu_api_.set_limits[i](apu_access_, mw);
            if (result != 0) {
                error = "RyzenAdj set APU limit " + std::to_string(i) + " failed: " + std::to_string(result);
                std::string restore_error;
                if (!reset_apu_tdp(restore_error)) error += "; rollback: " + restore_error;
                return false;
            }
        }
        return true;
    }

    bool reset_apu_tdp(std::string & error) override {
        bool success = true;
        for (int i = 0; i < 3; ++i) {
            if (!apu_limit_changed_[i]) continue;
            const int result = apu_api_.set_limits[i](apu_access_, original_apu_limits_[i]);
            if (result != 0) {
                if (!success) error += "; ";
                else error.clear();
                error += "RyzenAdj restore APU limit " + std::to_string(i) + " failed: " + std::to_string(result);
                success = false;
            } else {
                apu_limit_changed_[i] = false;
            }
        }
        return success;
    }

    void shutdown() override {
        if (initialized_) {
            std::string error;
            if (!reset_memory_locked_clocks(error)) LOG_WRN("AMDGPU SCLK restore failed: %s\n", error.c_str());
            if (!reset_fabric_state(error)) LOG_WRN("AMDGPU fabric restore failed: %s\n", error.c_str());
            if (!reset_apu_tdp(error)) LOG_WRN("APU TDP restore failed: %s\n", error.c_str());
        }
        cleanup_apu_tdp();
        initialized_ = false;
        level_locked_ = sclk_changed_ = fabric_locked_ = false;
        od_sclk_min_ = od_sclk_max_ = original_sclk_min_ = original_sclk_max_ = 0;
        card_dir_.clear();
        original_level_.clear();
    }

  private:
    void cleanup_apu_tdp() {
        if (apu_access_) apu_api_.cleanup(apu_access_);
        apu_access_ = nullptr;
#if defined(__linux__)
        if (apu_library_) dlclose(apu_library_);
#endif
        apu_library_ = nullptr;
        if (!apu_api_injected_) apu_api_ = {};
        for (int i = 0; i < 3; ++i) {
            original_apu_limits_[i] = 0;
            apu_limit_changed_[i] = false;
        }
    }

    bool write(const char * file, const std::string & value, std::string & error) {
        const std::string path = card_dir_ + "/device/" + file;
        if (writer_(path, value)) return true;
        error = "cannot write " + path + " (" + server_gpu_power_trim(value) + ")";
        return false;
    }

    bool enter_manual(std::string & error) {
        if (level_locked_) return true;
        level_locked_ = true;
        return write("power_dpm_force_performance_level", "manual\n", error);
    }

    bool restore_level(std::string & error) {
        if (!level_locked_) return true;
        if (!write("power_dpm_force_performance_level", original_level_ + "\n", error)) return false;
        level_locked_ = false;
        return true;
    }

    bool restore_level_if_unused(std::string & error) {
        return sclk_changed_ || restore_level(error);
    }

    bool rollback(std::string & error) {
        std::string restore_error;
        if (!reset_memory_locked_clocks(restore_error)) error += "; rollback: " + restore_error;
        if (!reset_fabric_state(restore_error)) error += "; rollback: " + restore_error;
        return false;
    }

    static std::vector<std::string> split_lines(const std::string & s) {
        std::vector<std::string> out;
        std::istringstream iss(s);
        std::string line;
        while (std::getline(iss, line)) {
            out.push_back(line);
        }
        return out;
    }

    std::string drm_dir_;
    std::function<bool(const std::string &, const std::string &)> writer_;
    std::string card_dir_;
    std::string original_level_;
    uint32_t od_sclk_min_ = 0;
    uint32_t od_sclk_max_ = 0;
    uint32_t original_sclk_min_ = 0;
    uint32_t original_sclk_max_ = 0;
    bool level_locked_ = false;
    bool sclk_changed_ = false;
    bool fabric_locked_ = false;
    bool initialized_ = false;
    server_gpu_power_ryzenadj_api apu_api_;
    bool apu_api_injected_ = false;
    void * apu_library_ = nullptr;
    _ryzen_access * apu_access_ = nullptr;
    uint32_t original_apu_limits_[3] = {};
    bool apu_limit_changed_[3] = {};
};

}  // namespace

std::unique_ptr<server_gpu_power_backend> server_gpu_power_create_nvml_backend() {
    return std::make_unique<server_gpu_power_nvml_backend>();
}

std::unique_ptr<server_gpu_power_backend> server_gpu_power_create_amdgpu_backend(
    const std::string & drm_dir, std::function<bool(const std::string &, const std::string &)> write_sysfs,
    const server_gpu_power_ryzenadj_api * apu_api) {
    return std::make_unique<server_gpu_power_amdgpu_backend>(drm_dir, std::move(write_sysfs), apu_api);
}

server_gpu_power_backend_type server_gpu_power_backend_from_string(const std::string & backend) {
    if (backend == "amdgpu") {
        return server_gpu_power_backend_type::amdgpu;
    }
    if (backend == "nvml") {
        return server_gpu_power_backend_type::nvml;
    }
    return server_gpu_power_backend_type::auto_detect;
}

server_gpu_power::server_gpu_power(std::unique_ptr<server_gpu_power_backend> backend) : backend_(std::move(backend)) {}

server_gpu_power::~server_gpu_power() {
    shutdown();
}

bool server_gpu_power::init(const server_gpu_power_config & config) {
    shutdown();
    config_ = config;

    if (!config_.enabled()) {
        return true;
    }

    if (config_.fabric_state < -1 || config_.fabric_state > 31 ||
        (config_.fabric_state >= 0 && config_.backend != server_gpu_power_backend_type::amdgpu)) {
        LOG_ERR("GPU fabric state requires --gpu-power-backend amdgpu and a raw state index 0..31\n");
        return false;
    }

    if (config_.apu_tdp_w != -1 &&
        (config_.backend != server_gpu_power_backend_type::amdgpu || config_.power_enabled() ||
         !server_gpu_power_w_to_mw(config_.apu_tdp_w, apu_tdp_mw_))) {
        LOG_ERR("APU TDP requires positive watts, explicit amdgpu backend, and no GPU power-limit options\n");
        return false;
    }

    if (config_.device < 0) {
        LOG_ERR("GPU governor device index must be non-negative\n");
        return false;
    }

    if (config_.power_enabled()) {
        if (config_.prefill_w <= 0 || config_.decode_w <= 0) {
            LOG_ERR(
                "GPU power governor requires both --gpu-power-prefill and --gpu-power-decode with positive watt values\n");
            return false;
        }

        if (!server_gpu_power_w_to_mw(config_.prefill_w, prefill_power_limit_mw_) ||
            !server_gpu_power_w_to_mw(config_.decode_w, decode_power_limit_mw_)) {
            LOG_ERR("GPU power governor watt value is too large\n");
            return false;
        }
    }

    if (config_.mem_clock_enabled()) {
        decode_mem_clock_mhz_        = config_.mem_clock_decode > 0 ? static_cast<uint32_t>(config_.mem_clock_decode) : 0;
        prefill_mem_clock_mhz_       = config_.mem_clock_prefill > 0 ? static_cast<uint32_t>(config_.mem_clock_prefill) : 0;
        decode_mem_offset_mhz_       = 0;
        prefill_mem_offset_mhz_      = 0;
        last_applied_mem_offset_mhz_ = 0;
        mem_offset_applied_          = false;
    }

    if (!backend_) {
        if (config_.backend == server_gpu_power_backend_type::amdgpu) {
            backend_ = server_gpu_power_create_amdgpu_backend();
        } else if (config_.backend == server_gpu_power_backend_type::nvml) {
            backend_ = server_gpu_power_create_nvml_backend();
        } else {
            // auto uses NVML; AMDGPU sysfs requires an explicit backend.
            backend_ = server_gpu_power_create_nvml_backend();
        }
    }

    std::string error;
    if (!backend_->init(config_.device, device_info_, error)) {
        LOG_ERR("GPU governor initialization failed: %s\n", error.c_str());
        return false;
    }
    backend_initialized_ = true;

    if (config_.fabric_state >= 0 && !device_info_.fabric_state_supported) {
        LOG_ERR("GPU fabric state requires readable pp_dpm_fclk and a non-manual original performance level\n");
        backend_->shutdown();
        backend_initialized_ = false;
        return false;
    }

    if (config_.power_enabled()) {
        const auto validate_limit = [&](uint32_t power_mw, const char * profile) {
            if (power_mw < device_info_.min_power_limit_mw || power_mw > device_info_.max_power_limit_mw) {
                LOG_ERR("GPU power governor %s limit %s W is outside the allowed range %s-%s W\n", profile,
                        server_gpu_power_mw_to_string(power_mw).c_str(),
                        server_gpu_power_mw_to_string(device_info_.min_power_limit_mw).c_str(),
                        server_gpu_power_mw_to_string(device_info_.max_power_limit_mw).c_str());
                return false;
            }
            return true;
        };

        if (!validate_limit(prefill_power_limit_mw_, "prefill") || !validate_limit(decode_power_limit_mw_, "decode")) {
            backend_->shutdown();
            backend_initialized_ = false;
            return false;
        }
    }

    if (config_.mem_clock_enabled()) {
        if (device_info_.supported_mem_clocks_mhz.empty()) {
            LOG_ERR("GPU memory clock governor: supported memory clocks are unavailable\n");
            backend_->shutdown();
            backend_initialized_ = false;
            return false;
        }
        std::sort(device_info_.supported_mem_clocks_mhz.rbegin(), device_info_.supported_mem_clocks_mhz.rend());
        const uint32_t max_stock_mhz = device_info_.supported_mem_clocks_mhz.front();

        const auto setup_clock_target = [&](uint32_t & clock_mhz, int32_t & offset_mhz, const char * phase_label) {
            if (clock_mhz == 0) {
                return true;
            }

            for (uint32_t c : device_info_.supported_mem_clocks_mhz) {
                if (c == clock_mhz) {
                    offset_mhz = 0;
                    return true;
                }
            }

            if (clock_mhz > max_stock_mhz) {
                if (clock_mhz > MAX_REQUESTED_MEM_CLOCK_MHZ) {
                    LOG_WRN("GPU memory clock governor: %s target %u MHz exceeds configured ceiling (%u MHz), clamping\n",
                            phase_label, clock_mhz, MAX_REQUESTED_MEM_CLOCK_MHZ);
                    clock_mhz = MAX_REQUESTED_MEM_CLOCK_MHZ;
                }

                if (!device_info_.memory_clock_offset_supported || clock_mhz <= max_stock_mhz ||
                    device_info_.memory_clock_p2_mhz == 0 || clock_mhz <= device_info_.memory_clock_p2_mhz) {
                    LOG_ERR("GPU memory clock governor: %s overclock is unsupported or exceeds the configured ceiling\n",
                            phase_label);
                    return false;
                }
                const int64_t offset = (int64_t(clock_mhz) - device_info_.memory_clock_p2_mhz) * 2;
                if (offset < device_info_.min_memory_clock_offset_mhz || offset > device_info_.max_memory_clock_offset_mhz) {
                    LOG_ERR("GPU memory clock governor: %s offset is outside the driver limits\n", phase_label);
                    return false;
                }
                offset_mhz = static_cast<int32_t>(offset);
                LOG_INF("GPU memory clock governor: %s overclock target %u MHz -> lock %u MHz + offset %+d MHz\n",
                        phase_label, clock_mhz, max_stock_mhz, offset_mhz);
                return true;
            }

            LOG_ERR("GPU memory clock governor: %s clock %u MHz is not supported (max stock %u MHz)\n",
                    phase_label, clock_mhz, max_stock_mhz);
            return false;
        };

        if (decode_mem_clock_mhz_ > 0 && !setup_clock_target(decode_mem_clock_mhz_, decode_mem_offset_mhz_, "decode")) {
            backend_->shutdown();
            backend_initialized_ = false;
            return false;
        }

        if (prefill_mem_clock_mhz_ > 0 && !setup_clock_target(prefill_mem_clock_mhz_, prefill_mem_offset_mhz_, "prefill")) {
            backend_->shutdown();
            backend_initialized_ = false;
            return false;
        }
    }

    if (config_.apu_tdp_w != -1) {
        std::string error;
        if (!backend_->init_apu_tdp(error)) {
            LOG_ERR("APU TDP initialization failed: %s\n", error.c_str());
            backend_->shutdown();
            backend_initialized_ = false;
            return false;
        }
    }

    phase_                       = server_gpu_power_phase::idle;
    transition_count_            = 0;
    last_applied_power_limit_mw_ = device_info_.original_power_limit_mw;
    last_applied_mem_clock_mhz_  = 0;
    last_applied_mem_offset_mhz_ = 0;
    mem_clock_locked_            = false;
    mem_offset_applied_          = false;
    power_limit_changed_         = false;
    enabled_                     = true;

    LOG_INF("GPU governor enabled\n");
    LOG_INF("  device: %s\n", device_info_.name.c_str());
    LOG_INF("  device index: %d\n", device_info_.device);
    if (config_.fabric_state >= 0) LOG_INF("  fabric raw state: %d\n", config_.fabric_state);
    if (config_.apu_tdp_w != -1) LOG_INF("  active APU TDP: %d W\n", config_.apu_tdp_w);
    if (config_.power_enabled()) {
        LOG_INF("  original PL: %s W\n", server_gpu_power_mw_to_string(device_info_.original_power_limit_mw).c_str());
        LOG_INF("  allowed range: %s-%s W\n", server_gpu_power_mw_to_string(device_info_.min_power_limit_mw).c_str(),
                server_gpu_power_mw_to_string(device_info_.max_power_limit_mw).c_str());
        LOG_INF("  prefill PL: %s W\n", server_gpu_power_mw_to_string(prefill_power_limit_mw_).c_str());
        LOG_INF("  decode PL: %s W\n", server_gpu_power_mw_to_string(decode_power_limit_mw_).c_str());
    }
    if (config_.mem_clock_enabled()) {
        if (decode_mem_clock_mhz_ > 0) {
            LOG_INF("  decode memory clock: %u MHz\n", decode_mem_clock_mhz_);
        }
        if (prefill_mem_clock_mhz_ > 0) {
            LOG_INF("  prefill memory clock: %u MHz\n", prefill_mem_clock_mhz_);
        }
    }
    return true;
}

void server_gpu_power::update(server_gpu_power_phase phase) {
    if (!enabled_ || !backend_initialized_ || phase == phase_) {
        return;
    }

    const server_gpu_power_phase previous = phase_;
    phase_                                = phase;
    transition_count_++;

    if (config_.apu_tdp_w != -1 && phase != server_gpu_power_phase::idle && !apu_tdp_applied_) {
        std::string error;
        // Also retry cleanup from the governor if the backend's local rollback fails.
        apu_tdp_applied_ = true;
        if (!backend_->set_apu_tdp(apu_tdp_mw_, error)) {
            disable_after_error(error);
            return;
        }
        LOG_INF("APU TDP: active limit %d W\n", config_.apu_tdp_w);
    }

    if (config_.power_enabled()) {
        if (phase == server_gpu_power_phase::idle) {
            LOG_INF("GPU power: %s -> idle\n", server_gpu_power_phase_name(previous));
        } else {
            const uint32_t target = phase == server_gpu_power_phase::prefill ? prefill_power_limit_mw_ : decode_power_limit_mw_;
            if (target != last_applied_power_limit_mw_) {
                std::string error;
                if (!backend_->set_power_limit(target, error)) {
                    disable_after_error(error);
                    return;
                }

                last_applied_power_limit_mw_ = target;
                power_limit_changed_         = target != device_info_.original_power_limit_mw;
                LOG_INF("GPU power: %s -> %s, limit %s W\n", server_gpu_power_phase_name(previous),
                        server_gpu_power_phase_name(phase), server_gpu_power_mw_to_string(target).c_str());
            }
        }
    }

    if (config_.mem_clock_enabled()) {
        uint32_t target_mem    = 0;
        int32_t  target_offset = 0;

        if (phase == server_gpu_power_phase::decode) {
            target_mem    = decode_mem_clock_mhz_;
            target_offset = decode_mem_offset_mhz_;
        } else if (phase == server_gpu_power_phase::prefill) {
            target_mem    = prefill_mem_clock_mhz_;
            target_offset = prefill_mem_offset_mhz_;
        }

        const uint32_t max_stock_mhz = !device_info_.supported_mem_clocks_mhz.empty()
                                           ? device_info_.supported_mem_clocks_mhz.front()
                                           : 0;
        const uint32_t target_lock_mhz = target_offset != 0 ? max_stock_mhz : target_mem;

        // Restore the offset before changing the lock or returning to dynamic clocks.
        if (mem_offset_applied_ && (target_offset != last_applied_mem_offset_mhz_ ||
                                    target_lock_mhz != last_applied_mem_clock_mhz_)) {
            std::string error;
            if (!backend_->reset_memory_clock_offset(error)) {
                disable_after_error(error);
                return;
            }
            mem_offset_applied_ = false;
            last_applied_mem_offset_mhz_ = 0;
            LOG_INF("GPU memory offset: %s -> %s, restored original\n",
                    server_gpu_power_phase_name(previous), server_gpu_power_phase_name(phase));
        }

        if (target_lock_mhz != last_applied_mem_clock_mhz_) {
            std::string error;
            if (target_lock_mhz > 0) {
                if (!backend_->set_memory_locked_clocks(target_lock_mhz, target_lock_mhz, error)) {
                    disable_after_error(error);
                    return;
                }
                mem_clock_locked_ = true;
                LOG_INF("GPU memory clock: %s -> %s, locked %u MHz\n", server_gpu_power_phase_name(previous),
                        server_gpu_power_phase_name(phase), target_lock_mhz);
            } else {
                if (mem_clock_locked_) {
                    if (!backend_->reset_memory_locked_clocks(error)) {
                        disable_after_error(error);
                        return;
                    }
                    mem_clock_locked_ = false;
                    LOG_INF("GPU memory clock: %s -> %s, reset (dynamic)\n", server_gpu_power_phase_name(previous),
                            server_gpu_power_phase_name(phase));
                }
            }
            last_applied_mem_clock_mhz_ = target_lock_mhz;
        }

        if (target_offset != last_applied_mem_offset_mhz_) {
            std::string error;
            if (target_offset != 0) {
                if (!backend_->set_memory_clock_offset(target_offset, error)) {
                    disable_after_error(error);
                    return;
                }
                mem_offset_applied_ = true;
                LOG_INF("GPU memory offset: %s -> %s, offset %+d MHz (target %u MHz)\n",
                        server_gpu_power_phase_name(previous), server_gpu_power_phase_name(phase), target_offset, target_mem);
            }
            last_applied_mem_offset_mhz_ = target_offset;
        }
    }

    if (config_.fabric_state >= 0) {
        const bool active = phase != server_gpu_power_phase::idle;
        std::string error;
        if (active && !fabric_state_applied_) {
            if (!backend_->set_fabric_state(config_.fabric_state, error)) {
                disable_after_error(error);
                return;
            }
            fabric_state_applied_ = true;
        } else if (!active && fabric_state_applied_) {
            if (!backend_->reset_fabric_state(error)) {
                disable_after_error(error);
                return;
            }
            fabric_state_applied_ = false;
        }
    }
    if (phase == server_gpu_power_phase::idle && apu_tdp_applied_) {
        std::string error;
        if (!backend_->reset_apu_tdp(error)) {
            disable_after_error(error);
            return;
        }
        apu_tdp_applied_ = false;
        LOG_INF("APU TDP: restored original limits\n");
    }
}

void server_gpu_power::on_sleeping(bool sleeping) {
    if (!backend_initialized_) {
        return;
    }

    if (sleeping && !restore_original()) {
        disable_after_error("failed to restore GPU/APU state before sleep");
    }

    phase_ = server_gpu_power_phase::idle;
    if (!power_limit_changed_) {
        last_applied_power_limit_mw_ = device_info_.original_power_limit_mw;
    }
    if (!mem_clock_locked_) {
        last_applied_mem_clock_mhz_ = 0;
    }
    if (!mem_offset_applied_) {
        last_applied_mem_offset_mhz_ = 0;
    }
}

void server_gpu_power::shutdown() {
    if (backend_initialized_) {
        restore_original();
        backend_->shutdown();
        backend_initialized_ = false;
    }

    fabric_state_applied_        = false;
    apu_tdp_applied_             = false;
    enabled_                     = false;
    phase_                       = server_gpu_power_phase::idle;
    power_limit_changed_         = false;
    mem_clock_locked_            = false;
    mem_offset_applied_          = false;
    last_applied_mem_clock_mhz_  = 0;
    last_applied_mem_offset_mhz_ = 0;
}

bool server_gpu_power::enabled() const {
    return enabled_;
}

uint64_t server_gpu_power::transition_count() const {
    return transition_count_;
}

const server_gpu_power_device_info & server_gpu_power::device_info() const {
    return device_info_;
}

bool server_gpu_power::restore_original() {
    if (!backend_initialized_) {
        return true;
    }

    bool success = true;
    if (power_limit_changed_) {
        std::string error;
        if (!backend_->set_power_limit(device_info_.original_power_limit_mw, error)) {
            LOG_WRN("GPU power: failed to restore original limit %s W: %s\n",
                    server_gpu_power_mw_to_string(device_info_.original_power_limit_mw).c_str(), error.c_str());
            success = false;
        } else {
            last_applied_power_limit_mw_ = device_info_.original_power_limit_mw;
            power_limit_changed_         = false;
        }
    }

    if (mem_offset_applied_) {
        std::string error;
        if (!backend_->reset_memory_clock_offset(error)) {
            LOG_WRN("GPU memory offset: failed to reset memory clock offset: %s\n", error.c_str());
            success = false;
        } else {
            mem_offset_applied_          = false;
            last_applied_mem_offset_mhz_ = 0;
        }
    }

    if (mem_clock_locked_) {
        std::string error;
        if (!backend_->reset_memory_locked_clocks(error)) {
            LOG_WRN("GPU memory clock: failed to reset memory locked clocks: %s\n", error.c_str());
            success = false;
        } else {
            mem_clock_locked_           = false;
            last_applied_mem_clock_mhz_ = 0;
        }
    }

    if (fabric_state_applied_) {
        std::string error;
        if (!backend_->reset_fabric_state(error)) {
            LOG_WRN("GPU fabric: failed to restore original profile: %s\n", error.c_str());
            success = false;
        } else {
            fabric_state_applied_ = false;
        }
    }

    if (apu_tdp_applied_) {
        std::string error;
        if (!backend_->reset_apu_tdp(error)) {
            LOG_WRN("APU TDP: failed to restore original limits: %s\n", error.c_str());
            success = false;
        } else {
            apu_tdp_applied_ = false;
        }
    }

    return success;
}

void server_gpu_power::disable_after_error(const std::string & error) {
    LOG_WRN("GPU governor: disabling governor after error: %s\n", error.c_str());
    enabled_ = false;
    restore_original();
}
