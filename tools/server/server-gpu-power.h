#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

enum class server_gpu_power_phase {
    idle,
    prefill,
    decode,
};

enum class server_gpu_power_slot_state {
    idle,
    wait_other,
    started,
    processing_prompt,
    done_prompt,
    generating,
};

class server_gpu_power_phase_arbitrator {
  public:
    void reset();
    void observe(server_gpu_power_slot_state state);

    server_gpu_power_phase phase() const;

  private:
    server_gpu_power_phase phase_ = server_gpu_power_phase::idle;
};

enum class server_gpu_power_backend_type {
    auto_detect,
    nvml,
    amdgpu,
};

struct server_gpu_power_config {
    int32_t prefill_w         = -1;
    int32_t decode_w          = -1;
    int32_t mem_clock_decode  = -1;
    int32_t mem_clock_prefill = -1;
    int32_t device            = 0;
    server_gpu_power_backend_type backend = server_gpu_power_backend_type::auto_detect;

    int32_t fabric_state = -1;
    int32_t apu_tdp_w = -1;

    bool enabled() const;
    bool power_enabled() const;
    bool mem_clock_enabled() const;
};

struct server_gpu_power_device_info {
    std::string           name;
    int32_t               device                  = 0;
    uint32_t              original_power_limit_mw = 0;
    uint32_t              min_power_limit_mw      = 0;
    uint32_t              max_power_limit_mw      = 0;
    std::vector<uint32_t> supported_mem_clocks_mhz;
    uint32_t              memory_clock_p2_mhz          = 0;
    int32_t               min_memory_clock_offset_mhz = 0;
    int32_t               max_memory_clock_offset_mhz = 0;
    bool                  memory_clock_offset_supported = false;
    bool                  fabric_state_supported = false;
};

class server_gpu_power_backend {
  public:
    virtual ~server_gpu_power_backend() = default;

    virtual bool init(int32_t device, server_gpu_power_device_info & info, std::string & error)     = 0;
    virtual bool set_power_limit(uint32_t power_limit_mw, std::string & error)                     = 0;
    virtual bool set_memory_locked_clocks(uint32_t min_mhz, uint32_t max_mhz, std::string & error) = 0;
    virtual bool reset_memory_locked_clocks(std::string & error)                                   = 0;
    virtual bool set_memory_clock_offset(int32_t offset_mhz, std::string & error)                  = 0;
    virtual bool reset_memory_clock_offset(std::string & error)                                    = 0;
    virtual bool set_fabric_state(int32_t, std::string & error) {
        error = "fabric state control is unsupported by this backend";
        return false;
    }
    virtual bool reset_fabric_state(std::string &) { return true; }
    virtual bool init_apu_tdp(std::string & error) {
        error = "APU TDP control is unsupported by this backend";
        return false;
    }
    virtual bool set_apu_tdp(uint32_t, std::string & error) {
        error = "APU TDP control is unsupported by this backend";
        return false;
    }
    virtual bool reset_apu_tdp(std::string &) { return true; }
    virtual void shutdown()                                                                        = 0;
};

std::unique_ptr<server_gpu_power_backend> server_gpu_power_create_nvml_backend();
// Dynamically resolved RyzenAdj API; injectable for hardware-independent tests.
struct _ryzen_access;
struct server_gpu_power_ryzenadj_api {
    _ryzen_access * (*init)() = nullptr;
    void (*cleanup)(_ryzen_access *) = nullptr;
    int (*init_table)(_ryzen_access *) = nullptr;
    int (*refresh_table)(_ryzen_access *) = nullptr;
    // STAPM, fast, slow, respectively. Getters return W; setters take mW.
    float (*get_limits[3])(_ryzen_access *) = {};
    int (*set_limits[3])(_ryzen_access *, uint32_t) = {};
};

std::unique_ptr<server_gpu_power_backend> server_gpu_power_create_amdgpu_backend(
    const std::string & drm_dir = "/sys/class/drm",
    std::function<bool(const std::string &, const std::string &)> write_sysfs = {},
    const server_gpu_power_ryzenadj_api * apu_api = nullptr);

server_gpu_power_backend_type server_gpu_power_backend_from_string(const std::string & backend);

class server_gpu_power {
  public:
    // Requested overclock ceiling, not a stability guarantee for every GPU.
    static constexpr uint32_t MAX_REQUESTED_MEM_CLOCK_MHZ = 11001;

    // All methods are confined to the server_context loop thread.
    explicit server_gpu_power(std::unique_ptr<server_gpu_power_backend> backend = nullptr);
    ~server_gpu_power();

    bool init(const server_gpu_power_config & config);
    void update(server_gpu_power_phase phase);
    void on_sleeping(bool sleeping);
    void shutdown();

    bool                                 enabled() const;
    uint64_t                             transition_count() const;
    const server_gpu_power_device_info & device_info() const;

  private:
    bool restore_original();
    void disable_after_error(const std::string & error);

    std::unique_ptr<server_gpu_power_backend> backend_;
    server_gpu_power_config                   config_;
    server_gpu_power_device_info              device_info_;

    uint32_t prefill_power_limit_mw_      = 0;
    uint32_t decode_power_limit_mw_       = 0;
    uint32_t last_applied_power_limit_mw_ = 0;

    uint32_t decode_mem_clock_mhz_        = 0;
    uint32_t prefill_mem_clock_mhz_       = 0;
    int32_t  decode_mem_offset_mhz_       = 0;
    int32_t  prefill_mem_offset_mhz_      = 0;
    uint32_t last_applied_mem_clock_mhz_  = 0;
    int32_t  last_applied_mem_offset_mhz_ = 0;
    bool     mem_clock_locked_            = false;
    bool     mem_offset_applied_          = false;
    bool     fabric_state_applied_        = false;
    bool     apu_tdp_applied_             = false;
    uint32_t apu_tdp_mw_                  = 0;

    server_gpu_power_phase phase_               = server_gpu_power_phase::idle;
    uint64_t               transition_count_    = 0;
    bool                   backend_initialized_ = false;
    bool                   enabled_             = false;
    bool                   power_limit_changed_ = false;
};
