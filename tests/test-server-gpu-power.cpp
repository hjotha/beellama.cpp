#include "server-gpu-power.h"

#undef NDEBUG
#include <cassert>
#include <initializer_list>
#include <memory>
#include <vector>

struct fake_gpu_power_backend : server_gpu_power_backend {
    int                   init_calls         = 0;
    int                   set_calls          = 0;
    int                   set_mem_calls      = 0;
    int                   reset_mem_calls    = 0;
    int                   set_offset_calls   = 0;
    int                   reset_offset_calls = 0;
    int                   shutdown_calls     = 0;
    int                   fail_on_call       = 0;
    std::vector<uint32_t> applied_limits;
    std::vector<uint32_t> applied_mem_clocks;
    std::vector<int32_t>  applied_offsets;
    std::vector<std::string> memory_ops;
    std::vector<uint32_t> supported_mem_clocks = { 10501, 10251, 5001, 810, 405 };
    uint32_t p2_clock_mhz = 10251;
    int32_t max_offset_mhz = 6000;
    int32_t original_offset_mhz = 0;
    int32_t current_offset_mhz = 0;
    bool offset_supported = true;
    bool fail_set_offset = false;
    bool fail_reset_mem_once = false;
    bool fail_reset_offset_once = false;

    bool init(int32_t device, server_gpu_power_device_info & info, std::string &) override {
        init_calls++;
        info.name                     = "fake NVIDIA device";
        info.device                   = device;
        info.original_power_limit_mw  = 160000;
        info.min_power_limit_mw       = 100000;
        info.max_power_limit_mw       = 200000;
        info.supported_mem_clocks_mhz = supported_mem_clocks;
        info.memory_clock_p2_mhz = p2_clock_mhz;
        info.min_memory_clock_offset_mhz = -2000;
        info.max_memory_clock_offset_mhz = max_offset_mhz;
        info.memory_clock_offset_supported = offset_supported;
        current_offset_mhz = original_offset_mhz;
        return true;
    }

    bool set_power_limit(uint32_t power_limit_mw, std::string &) override {
        set_calls++;
        if (set_calls == fail_on_call) {
            return false;
        }
        applied_limits.push_back(power_limit_mw);
        return true;
    }

    bool set_memory_locked_clocks(uint32_t, uint32_t max_mhz, std::string &) override {
        set_mem_calls++;
        memory_ops.push_back("lock");
        if (set_mem_calls == fail_on_call) {
            return false;
        }
        applied_mem_clocks.push_back(max_mhz);
        return true;
    }

    bool reset_memory_locked_clocks(std::string &) override {
        reset_mem_calls++;
        memory_ops.push_back("reset-lock");
        if (fail_reset_mem_once) {
            fail_reset_mem_once = false;
            return false;
        }
        return true;
    }

    bool set_memory_clock_offset(int32_t offset_mhz, std::string &) override {
        set_offset_calls++;
        memory_ops.push_back("offset");
        if (fail_set_offset) {
            return false;
        }
        current_offset_mhz = offset_mhz;
        applied_offsets.push_back(offset_mhz);
        return true;
    }

    bool reset_memory_clock_offset(std::string &) override {
        reset_offset_calls++;
        memory_ops.push_back("reset-offset");
        if (fail_reset_offset_once) {
            fail_reset_offset_once = false;
            return false;
        }
        current_offset_mhz = original_offset_mhz;
        return true;
    }

    void shutdown() override { shutdown_calls++; }
};

static server_gpu_power_phase arbitrate(std::initializer_list<server_gpu_power_slot_state> states) {
    server_gpu_power_phase_arbitrator arbitrator;
    for (const auto state : states) {
        arbitrator.observe(state);
    }
    return arbitrator.phase();
}

int main() {
    assert(arbitrate({}) == server_gpu_power_phase::idle);
    assert(arbitrate({ server_gpu_power_slot_state::idle }) == server_gpu_power_phase::idle);
    assert(arbitrate({ server_gpu_power_slot_state::wait_other }) == server_gpu_power_phase::idle);
    assert(arbitrate({ server_gpu_power_slot_state::generating }) == server_gpu_power_phase::decode);
    assert(arbitrate({ server_gpu_power_slot_state::started }) == server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::processing_prompt }) == server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::done_prompt }) == server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::generating, server_gpu_power_slot_state::processing_prompt }) ==
           server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::processing_prompt, server_gpu_power_slot_state::generating }) ==
           server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::generating, server_gpu_power_slot_state::wait_other }) ==
           server_gpu_power_phase::decode);

    // 1. Test pure power governor
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        assert(governor.init({ 200, 165, -1, -1, 0 }));
        assert(governor.enabled());
        assert(backend_ptr->init_calls == 1);
        assert(backend_ptr->set_calls == 0);

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_calls == 1);
        assert(backend_ptr->applied_limits.back() == 200000);

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_calls == 1);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_calls == 2);
        assert(backend_ptr->applied_limits.back() == 165000);

        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->set_calls == 2);

        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->set_calls == 2);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_calls == 2);
        assert(backend_ptr->applied_limits.back() == 165000);

        governor.on_sleeping(true);
        assert(backend_ptr->set_calls == 3);
        assert(backend_ptr->applied_limits.back() == 160000);

        governor.on_sleeping(false);
        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_calls == 4);
        assert(backend_ptr->applied_limits.back() == 200000);

        governor.shutdown();
        assert(backend_ptr->set_calls == 5);
        assert(backend_ptr->applied_limits.back() == 160000);
        assert(backend_ptr->shutdown_calls == 1);
    }

    // 2. Test pure memory governor (decode only, discrete stock)
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        assert(governor.init({ -1, -1, 10501, -1, 0 }));
        assert(governor.enabled());
        assert(backend_ptr->init_calls == 1);
        assert(backend_ptr->set_mem_calls == 0);

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_mem_calls == 0);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);

        // Deduplicated
        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);

        // Transition to idle resets memory clock
        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->reset_mem_calls == 1);

        // Transition back to decode relocks memory
        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 2);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);

        governor.shutdown();
        assert(backend_ptr->reset_mem_calls == 2);
    }

    // 3. Test combined power + memory governor
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        assert(governor.init({ 200, 165, 10501, 10251, 0 }));
        assert(governor.enabled());

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_calls == 1);
        assert(backend_ptr->applied_limits.back() == 200000);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10251);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_calls == 2);
        assert(backend_ptr->applied_limits.back() == 165000);
        assert(backend_ptr->set_mem_calls == 2);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);

        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->reset_mem_calls == 1);

        governor.shutdown();
        assert(backend_ptr->set_calls == 3);
        assert(backend_ptr->applied_limits.back() == 160000);
    }

    // 4. Test unsupported memory clock rejected (e.g. non-overclock value not in supported list)
    {
        auto             backend = std::make_unique<fake_gpu_power_backend>();
        server_gpu_power governor(std::move(backend));
        // 7000 is between 5001 and 10251, not in discrete list and not an overclock > 10501
        assert(!governor.init({ -1, -1, 7000, -1, 0 }));
    }

    // 5. Test disabled backend
    {
        auto             disabled_backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           disabled_backend_ptr = disabled_backend.get();
        server_gpu_power disabled(std::move(disabled_backend));
        assert(disabled.init({ -1, -1, -1, -1, 0 }));
        assert(!disabled.enabled());
        assert(disabled_backend_ptr->init_calls == 0);
        disabled.update(server_gpu_power_phase::prefill);
        assert(disabled_backend_ptr->set_calls == 0);
    }

    // 6. Test invalid power config rejected
    {
        auto             invalid_backend = std::make_unique<fake_gpu_power_backend>();
        server_gpu_power invalid(std::move(invalid_backend));
        assert(!invalid.init({ 201, 165, -1, -1, 0 }));
    }

    // 7. Test failing power backend
    {
        auto   failing_backend        = std::make_unique<fake_gpu_power_backend>();
        auto * failing_backend_ptr    = failing_backend.get();
        failing_backend->fail_on_call = 2;
        server_gpu_power failing(std::move(failing_backend));
        assert(failing.init({ 200, 165, -1, -1, 0 }));
        failing.update(server_gpu_power_phase::prefill);
        assert(failing_backend_ptr->set_calls == 1);
        failing.update(server_gpu_power_phase::decode);
        assert(!failing.enabled());
        assert(failing_backend_ptr->set_calls == 3);
        assert(failing_backend_ptr->applied_limits.back() == 160000);
        failing.update(server_gpu_power_phase::idle);
        assert(failing_backend_ptr->set_calls == 3);
        failing.shutdown();
        assert(failing_backend_ptr->set_calls == 3);
        assert(failing_backend_ptr->applied_limits.back() == 160000);
    }

    // 8. Test memory overclock governor (target > 10501, e.g. 11001 MHz)
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        // CUDA P2 base is queried independently of the supported clock list order.
        backend_ptr->supported_mem_clocks = { 405, 5001, 10501 };
        backend_ptr->original_offset_mhz = 200;
        assert(governor.init({ -1, -1, 11001, -1, 0 }));
        assert(governor.enabled());
        assert(backend_ptr->init_calls == 1);
        assert(backend_ptr->set_mem_calls == 0);
        assert(backend_ptr->set_offset_calls == 0);

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_mem_calls == 0);
        assert(backend_ptr->set_offset_calls == 0);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);
        assert(backend_ptr->set_offset_calls == 1);
        assert(backend_ptr->applied_offsets.back() == 1500);

        // Deduplicated
        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->set_offset_calls == 1);

        // Idle resets both offset and locked clock
        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->reset_offset_calls == 1);
        assert(backend_ptr->reset_mem_calls == 1);
        assert(backend_ptr->current_offset_mhz == 200);
        assert(backend_ptr->memory_ops == std::vector<std::string>({ "lock", "offset", "reset-offset", "reset-lock" }));

        governor.shutdown();
        assert(backend_ptr->shutdown_calls == 1);
    }

    // 9. Test the requested overclock ceiling.
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        // 12501 MHz exceeds the requested ceiling (11001).
        assert(governor.init({ -1, -1, 12501, -1, 0 }));
        assert(governor.enabled());

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);
        assert(backend_ptr->set_offset_calls == 1);
        // Clamped to 11001: (11001 - 10251) * 2 = 1500
        assert(backend_ptr->applied_offsets.back() == 1500);

        governor.shutdown();
        assert(backend_ptr->reset_offset_calls == 1);
    }

    // Offset failures must immediately undo a lock that was already applied.
    {
        auto backend = std::make_unique<fake_gpu_power_backend>();
        auto * ptr = backend.get();
        ptr->fail_set_offset = true;
        server_gpu_power governor(std::move(backend));
        assert(governor.init({ 200, 165, 11001, -1, 0 }));
        governor.update(server_gpu_power_phase::decode);
        assert(!governor.enabled());
        assert(ptr->reset_mem_calls == 1);
        assert(ptr->applied_limits.back() == 160000);
        assert(ptr->current_offset_mhz == 0);
    }

    // Both reset failure paths restore immediately and preserve the original offset.
    for (bool fail_offset_reset : { false, true }) {
        auto backend = std::make_unique<fake_gpu_power_backend>();
        auto * ptr = backend.get();
        ptr->original_offset_mhz = 200;
        server_gpu_power governor(std::move(backend));
        assert(governor.init({ -1, -1, 11001, -1, 0 }));
        governor.update(server_gpu_power_phase::decode);
        ptr->fail_reset_mem_once = !fail_offset_reset;
        ptr->fail_reset_offset_once = fail_offset_reset;
        governor.update(server_gpu_power_phase::idle);
        assert(!governor.enabled());
        assert(ptr->current_offset_mhz == 200);
        assert(ptr->reset_offset_calls == (fail_offset_reset ? 2 : 1));
        assert(ptr->reset_mem_calls == (fail_offset_reset ? 1 : 2));
        assert(ptr->memory_ops[2] == "reset-offset");
    }

    // Reject missing capabilities, driver limits and clocks above the ceiling before any write.
    for (int scenario = 0; scenario < 5; ++scenario) {
        auto backend = std::make_unique<fake_gpu_power_backend>();
        auto * ptr = backend.get();
        if (scenario == 0) ptr->offset_supported = false;
        if (scenario == 1) ptr->max_offset_mhz = 1000;
        if (scenario == 2) ptr->supported_mem_clocks.clear();
        if (scenario == 3) ptr->p2_clock_mhz = 0;
        if (scenario == 4) {
            ptr->supported_mem_clocks = { 12000, 11750 };
            ptr->p2_clock_mhz = 11750;
        }
        server_gpu_power governor(std::move(backend));
        assert(!governor.init({ -1, -1, scenario == 4 ? 12501 : 11001, -1, 0 }));
        assert(ptr->memory_ops.empty());
        assert(ptr->shutdown_calls == 1);
    }

    return 0;
}
