#include "server-gpu-power.h"

#include "log.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

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
    return power_enabled() || mem_clock_enabled();
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

}  // namespace

std::unique_ptr<server_gpu_power_backend> server_gpu_power_create_nvml_backend() {
    return std::make_unique<server_gpu_power_nvml_backend>();
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
        backend_ = server_gpu_power_create_nvml_backend();
    }

    std::string error;
    if (!backend_->init(config_.device, device_info_, error)) {
        LOG_ERR("GPU governor initialization failed: %s\n", error.c_str());
        return false;
    }
    backend_initialized_ = true;

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
    LOG_INF("  NVML index: %d\n", device_info_.device);
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
}

void server_gpu_power::on_sleeping(bool sleeping) {
    if (!backend_initialized_) {
        return;
    }

    if (sleeping) {
        restore_original();
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

    return success;
}

void server_gpu_power::disable_after_error(const std::string & error) {
    LOG_WRN("GPU governor: disabling governor after error: %s\n", error.c_str());
    enabled_ = false;
    restore_original();
}
