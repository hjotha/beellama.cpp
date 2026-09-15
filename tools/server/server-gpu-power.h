#pragma once

#include <cstdint>
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

struct server_gpu_power_config {
    int32_t prefill_w         = -1;
    int32_t decode_w          = -1;
    int32_t mem_clock_decode  = -1;
    int32_t mem_clock_prefill = -1;
    int32_t device            = 0;

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
    virtual void shutdown()                                                                        = 0;
};

std::unique_ptr<server_gpu_power_backend> server_gpu_power_create_nvml_backend();

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

    server_gpu_power_phase phase_               = server_gpu_power_phase::idle;
    uint64_t               transition_count_    = 0;
    bool                   backend_initialized_ = false;
    bool                   enabled_             = false;
    bool                   power_limit_changed_ = false;
};
