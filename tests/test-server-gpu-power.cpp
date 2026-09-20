#include "server-gpu-power.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <initializer_list>
#include <limits>
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

struct fake_amdgpu_sysfs {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("test-amdgpu-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path device = root / "card0/device";
    std::vector<std::string> writes;
    std::string fail_once;

    fake_amdgpu_sysfs(const std::string & level = "auto", bool custom = false) {
        std::filesystem::create_directories(device);
        put("vendor", "0x1002\n");
        put("power_dpm_force_performance_level", level + "\n");
        put("pp_od_clk_voltage", custom ?
            "OD_SCLK:\n0: 900Mhz\n1: 2400Mhz\nOD_RANGE:\nSCLK: 800Mhz 2700Mhz\n" :
            "OD_SCLK:\n0: 800Mhz\n1: 2700Mhz\nOD_RANGE:\nSCLK: 800Mhz 2700Mhz\n");
        put("pp_dpm_sclk", "0: 800Mhz\n1: 2700Mhz *\n");
        // Display order is deliberately reversed relative to firmware indices.
        put("pp_dpm_fclk", "0: 400Mhz\n1: 933Mhz\n2: 1467Mhz\n3: 1875Mhz *\n");
    }
    ~fake_amdgpu_sysfs() { std::filesystem::remove_all(root); }
    void put(const std::string & name, const std::string & text) {
        std::ofstream f(device / name);
        f << text;
        f.close();
        assert(!f.fail());
    }
    std::unique_ptr<server_gpu_power_backend> backend(const server_gpu_power_ryzenadj_api * api = nullptr) {
        return server_gpu_power_create_amdgpu_backend(root.string(),
            [this](const std::string & path, const std::string & value) {
                const auto op = std::filesystem::path(path).filename().string() + ":" + value;
                writes.push_back(op);
                if (op == fail_once) {
                    fail_once.clear();
                    return false;
                }
                return true;
            }, api);
    }
};

struct fake_ryzenadj {
    static fake_ryzenadj * current;
    float limits[3] = {8, 20, 15};
    int init_calls = 0;
    int cleanup_calls = 0;
    int init_table_calls = 0;
    int refresh_calls = 0;
    bool fail_init = false;
    bool fail_table = false;
    bool fail_refresh = false;
    std::vector<std::pair<int, uint32_t>> writes;
    std::vector<std::pair<int, uint32_t>> failures;
    fake_ryzenadj() { current = this; }
    static fake_ryzenadj & from(_ryzen_access * access) {
        return *reinterpret_cast<fake_ryzenadj *>(access);
    }
    template<int I> static float get(_ryzen_access * access) { return from(access).limits[I]; }
    template<int I> static int set(_ryzen_access * access, uint32_t mw) {
        auto & fake = from(access);
        const auto op = std::make_pair(I, mw);
        fake.writes.push_back(op);
        fake.limits[I] = mw / 1000.0f;
        // Even a failing setter may have partially applied its hardware write.
        if (!fake.failures.empty() && fake.failures.front() == op) {
            fake.failures.erase(fake.failures.begin());
            return -4;
        }
        return 0;
    }
    server_gpu_power_ryzenadj_api api() {
        return {
            []() -> _ryzen_access * {
                ++current->init_calls;
                return current->fail_init ? nullptr : reinterpret_cast<_ryzen_access *>(current);
            },
            [](_ryzen_access * access) { ++from(access).cleanup_calls; },
            [](_ryzen_access * access) { ++from(access).init_table_calls; return from(access).fail_table ? -5 : 0; },
            [](_ryzen_access * access) { ++from(access).refresh_calls; return from(access).fail_refresh ? -5 : 0; },
            {get<0>, get<1>, get<2>}, {set<0>, set<1>, set<2>}
        };
    }
    void assert_original() const {
        assert(limits[0] == 8 && limits[1] == 20 && limits[2] == 15);
    }
};
fake_ryzenadj * fake_ryzenadj::current = nullptr;

static void test_apu_tdp() {
    const auto amd = server_gpu_power_backend_type::amdgpu;
    // Real AMD backend and governor, fake RyzenAdj: no writes until inference.
    {
        fake_amdgpu_sysfs sys;
        fake_ryzenadj ryzen;
        auto api = ryzen.api();
        server_gpu_power governor(sys.backend(&api));
        assert(governor.init({-1, -1, -1, -1, 0, amd, -1, 20}));
        assert(ryzen.init_calls == 1 && ryzen.refresh_calls == 1 && ryzen.writes.empty());
        governor.update(server_gpu_power_phase::prefill);
        assert((ryzen.writes == std::vector<std::pair<int, uint32_t>>{{0, 20000}, {1, 20000}, {2, 20000}}));
        governor.update(server_gpu_power_phase::decode);
        assert(ryzen.writes.size() == 3);
        governor.update(server_gpu_power_phase::idle);
        ryzen.assert_original();
        assert((ryzen.writes == std::vector<std::pair<int, uint32_t>>{
            {0, 20000}, {1, 20000}, {2, 20000}, {0, 8000}, {1, 20000}, {2, 15000}}));
        governor.update(server_gpu_power_phase::decode);
        governor.on_sleeping(true);
        ryzen.assert_original();
        governor.on_sleeping(false);
        governor.update(server_gpu_power_phase::prefill);
        governor.shutdown();
        ryzen.assert_original();
        assert(ryzen.cleanup_calls == 1);
        assert(sys.writes.empty()); // TDP does not touch DPM, clocks, thermal limits or time constants.
    }
    // Existing GFX/fabric options do not initialize RyzenAdj at all.
    {
        fake_amdgpu_sysfs sys;
        fake_ryzenadj ryzen;
        auto api = ryzen.api();
        server_gpu_power governor(sys.backend(&api));
        assert(governor.init({-1, -1, 2700, 2700, 0, amd, 0}));
        governor.update(server_gpu_power_phase::prefill);
        governor.update(server_gpu_power_phase::idle);
        assert(ryzen.init_calls == 0 && ryzen.writes.empty());
    }
    // Each partial apply failure is undone; a failed local rollback is retried by the governor.
    for (int failed_limit = 0; failed_limit < 3; ++failed_limit) {
        for (bool fail_rollback : {false, true}) {
            fake_amdgpu_sysfs sys;
            fake_ryzenadj ryzen;
            auto api = ryzen.api();
            server_gpu_power governor(sys.backend(&api));
            assert(governor.init({-1, -1, -1, -1, 0, amd, -1, 20}));
            ryzen.failures = {{failed_limit, 20000}};
            if (fail_rollback) ryzen.failures.push_back({0, 8000});
            governor.update(server_gpu_power_phase::prefill);
            assert(!governor.enabled());
            assert(ryzen.failures.empty());
            ryzen.assert_original();
        }
    }
    // Idle/sleep restore continues past an error and retries only dirty limits.
    for (bool sleeping : {false, true}) {
        fake_amdgpu_sysfs sys;
        fake_ryzenadj ryzen;
        auto api = ryzen.api();
        server_gpu_power governor(sys.backend(&api));
        assert(governor.init({-1, -1, -1, -1, 0, amd, -1, 20}));
        governor.update(server_gpu_power_phase::decode);
        ryzen.failures = {{0, 8000}};
        if (sleeping) governor.on_sleeping(true);
        else governor.update(server_gpu_power_phase::idle);
        assert(!governor.enabled());
        ryzen.assert_original();
        assert((ryzen.writes == std::vector<std::pair<int, uint32_t>>{
            {0, 20000}, {1, 20000}, {2, 20000}, {0, 8000}, {1, 20000}, {2, 15000}, {0, 8000}}));
    }
    // A clock failure after TDP application must also return the original TDP limits.
    {
        fake_amdgpu_sysfs sys;
        fake_ryzenadj ryzen;
        auto api = ryzen.api();
        server_gpu_power governor(sys.backend(&api));
        assert(governor.init({-1, -1, 2700, 2700, 0, amd, 0, 20}));
        sys.fail_once = "pp_od_clk_voltage:s 0 2700\n";
        governor.update(server_gpu_power_phase::prefill);
        assert(!governor.enabled());
        ryzen.assert_original();
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
    }
    // API, access, table and snapshot failures clean up without writing any limit.
    for (int scenario = 0; scenario < 8; ++scenario) {
        fake_amdgpu_sysfs sys;
        fake_ryzenadj ryzen;
        auto api = ryzen.api();
        if (scenario == 0) api.set_limits[2] = nullptr;
        if (scenario == 1) ryzen.fail_init = true;
        if (scenario == 2) ryzen.fail_table = true;
        if (scenario == 3) ryzen.fail_refresh = true;
        if (scenario == 4) ryzen.limits[0] = std::numeric_limits<float>::quiet_NaN();
        if (scenario == 5) ryzen.limits[1] = std::numeric_limits<float>::infinity();
        if (scenario == 6) ryzen.limits[2] = 0;
        if (scenario == 7) ryzen.limits[0] = std::numeric_limits<float>::max();
        server_gpu_power governor(sys.backend(&api));
        assert(!governor.init({-1, -1, -1, -1, 0, amd, -1, 20}));
        assert(ryzen.writes.empty());
        assert(ryzen.cleanup_calls == (scenario >= 2 ? 1 : 0));
    }
    for (int watts : {-2, 0, std::numeric_limits<int32_t>::max()}) {
        fake_amdgpu_sysfs sys;
        server_gpu_power governor(sys.backend());
        assert(!governor.init({-1, -1, -1, -1, 0, amd, -1, watts}));
        assert(sys.writes.empty());
    }
    for (auto backend : {server_gpu_power_backend_type::auto_detect, server_gpu_power_backend_type::nvml}) {
        fake_amdgpu_sysfs sys;
        server_gpu_power governor(sys.backend());
        assert(!governor.init({-1, -1, -1, -1, 0, backend, -1, 20}));
    }
    {
        fake_amdgpu_sysfs sys;
        server_gpu_power governor(sys.backend());
        assert(!governor.init({20, 20, -1, -1, 0, amd, -1, 20}));
    }
}

static void test_amdgpu() {
    const auto amd = server_gpu_power_backend_type::amdgpu;
    // Real AMD backend over fake sysfs: exact restore sequence, including custom OD.
    for (bool custom : {false, true}) {
        fake_amdgpu_sysfs sys("auto", custom);
        auto backend = sys.backend();
        server_gpu_power_device_info info;
        std::string error;
        assert(backend->init(0, info, error));
        assert(backend->set_memory_locked_clocks(2700, 2700, error));
        assert(backend->reset_memory_locked_clocks(error));
        std::vector<std::string> expected = {
            "power_dpm_force_performance_level:manual\n", "pp_od_clk_voltage:s 0 2700\n",
            "pp_od_clk_voltage:s 1 2700\n", "pp_od_clk_voltage:c\n",
            "power_dpm_force_performance_level:manual\n", "pp_od_clk_voltage:r\n", "pp_od_clk_voltage:c\n"};
        if (custom) {
            expected.push_back("pp_od_clk_voltage:s 0 900\n");
            expected.push_back("pp_od_clk_voltage:s 1 2400\n");
            expected.push_back("pp_od_clk_voltage:c\n");
        }
        expected.push_back("power_dpm_force_performance_level:auto\n");
        assert(sys.writes == expected);
        assert(backend->reset_memory_locked_clocks(error));
        assert(sys.writes == expected);
    }
    // Every partial setter failure rolls back immediately, before governor bookkeeping.
    for (const auto & fail : {"power_dpm_force_performance_level:manual\n",
                              "pp_od_clk_voltage:s 0 2700\n", "pp_od_clk_voltage:s 1 2700\n",
                              "pp_od_clk_voltage:c\n"}) {
        fake_amdgpu_sysfs sys;
        auto backend = sys.backend();
        server_gpu_power_device_info info;
        std::string error;
        assert(backend->init(0, info, error));
        sys.fail_once = fail;
        assert(!backend->set_memory_locked_clocks(2700, 2700, error));
        assert(!error.empty());
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
    }
    // Reset failures propagate and retain dirty state so the next cleanup retries.
    for (const auto & fail : {"pp_od_clk_voltage:r\n", "pp_od_clk_voltage:c\n",
                              "power_dpm_force_performance_level:auto\n"}) {
        fake_amdgpu_sysfs sys;
        auto backend = sys.backend();
        server_gpu_power_device_info info;
        std::string error;
        assert(backend->init(0, info, error));
        assert(backend->set_memory_locked_clocks(2700, 2700, error));
        sys.fail_once = fail;
        assert(!backend->reset_memory_locked_clocks(error));
        assert(backend->reset_memory_locked_clocks(error));
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
    }
    // A failed OD restore must still release manual mode while fabric is locked.
    for (const auto & fail : {"pp_od_clk_voltage:r\n", "pp_od_clk_voltage:c\n"}) {
        fake_amdgpu_sysfs sys;
        auto backend = sys.backend();
        server_gpu_power_device_info info;
        std::string error;
        assert(backend->init(0, info, error));
        assert(backend->set_fabric_state(0, error));
        assert(backend->set_memory_locked_clocks(2700, 2700, error));
        sys.fail_once = fail;
        assert(!backend->reset_memory_locked_clocks(error));
        assert(error.find("pp_od_clk_voltage") != std::string::npos);
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
        assert(backend->reset_memory_locked_clocks(error));
        assert(backend->reset_fabric_state(error));
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
    }
    // Fabric alone and combined decode-only GFX: active in BOTH phases, raw index unchanged.
    for (bool gfx : {false, true}) {
        fake_amdgpu_sysfs sys;
        server_gpu_power governor(sys.backend());
        assert(governor.init({-1, -1, gfx ? 2700 : -1, -1, 0, amd, 0}));
        assert(governor.enabled());
        governor.update(server_gpu_power_phase::prefill);
        assert(sys.writes == std::vector<std::string>({"power_dpm_force_performance_level:manual\n", "pp_dpm_fclk:0\n"}));
        governor.update(server_gpu_power_phase::decode);
        governor.update(server_gpu_power_phase::prefill);
        assert(sys.writes.back() != "power_dpm_force_performance_level:auto\n");
        governor.update(server_gpu_power_phase::idle);
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
        governor.update(server_gpu_power_phase::decode);
        governor.on_sleeping(true);
        assert(sys.writes.back() == "power_dpm_force_performance_level:auto\n");
    }
    // Kernel rejection restores the prior performance profile and disables the governor.
    {
        fake_amdgpu_sysfs sys("high");
        server_gpu_power governor(sys.backend());
        assert(governor.init({-1, -1, 2700, 2700, 0, amd, 31}));
        sys.fail_once = "pp_dpm_fclk:31\n";
        governor.update(server_gpu_power_phase::prefill);
        assert(!governor.enabled());
        assert(sys.writes.back() == "power_dpm_force_performance_level:high\n");
    }
    // Original manual masks cannot be read back faithfully; fail without touching hardware.
    {
        fake_amdgpu_sysfs sys("manual");
        server_gpu_power governor(sys.backend());
        assert(!governor.init({-1, -1, -1, -1, 0, amd, 0}));
        assert(sys.writes.empty());
    }
    for (int state : {-2, 32}) {
        fake_amdgpu_sysfs sys;
        server_gpu_power governor(sys.backend());
        assert(!governor.init({-1, -1, -1, -1, 0, amd, state}));
        assert(sys.writes.empty());
    }
    for (auto backend : {server_gpu_power_backend_type::auto_detect, server_gpu_power_backend_type::nvml}) {
        fake_amdgpu_sysfs sys;
        server_gpu_power governor(sys.backend());
        assert(!governor.init({-1, -1, -1, -1, 0, backend, 0}));
        assert(sys.writes.empty());
    }
#if defined(__linux__)
    // A buffered ofstream write to /dev/full succeeds until flush/close: detect it.
    {
        fake_amdgpu_sysfs sys;
        auto backend = server_gpu_power_create_amdgpu_backend(sys.root.string());
        server_gpu_power_device_info info;
        std::string error;
        assert(backend->init(0, info, error));
        std::filesystem::remove(sys.device / "pp_od_clk_voltage");
        std::filesystem::create_symlink("/dev/full", sys.device / "pp_od_clk_voltage");
        assert(!backend->set_memory_locked_clocks(2700, 2700, error));
        std::ifstream level(sys.device / "power_dpm_force_performance_level");
        std::string current;
        level >> current;
        assert(current == "auto");
    }
#endif
}

int main() {
    test_amdgpu();
    test_apu_tdp();
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
