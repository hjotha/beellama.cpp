#undef NDEBUG
#include "ggml-remote-attn.h"

#include <cassert>
#include <cstdio>
#include <thread>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

int main() {
#ifdef _WIN32
    std::puts("SKIP: remote attention backend unavailable on Windows");
#else
    auto * first = ggml_backend_remote_attn_init("127.0.0.1", 1);
    auto * second = ggml_backend_remote_attn_init("127.0.0.1", 2);
    assert(first && second);
    assert(ggml_backend_remote_attn_set_upload_format(first, "kvarn4"));
    assert(ggml_backend_remote_attn_set_upload_format(second, "saw_int4"));
    assert(ggml_remote_attn_try_activate(first));
    assert(ggml_remote_attn_try_activate(first));
    assert(!ggml_remote_attn_try_activate(second));
    assert(!ggml_remote_attn_reserve_stage(second));
    ggml_remote_attn_deactivate(second);
    assert(!ggml_remote_attn_try_activate(second));
    ggml_remote_attn_deactivate(first);
    assert(ggml_remote_attn_reserve_stage(first));
    assert(!ggml_remote_attn_reserve_stage(second));
    assert(!ggml_remote_attn_try_activate(second));
    ggml_remote_attn_release_stage(second);
    assert(!ggml_remote_attn_try_activate(second));
    assert(ggml_remote_attn_try_activate(first));
    ggml_remote_attn_release_stage(first);
    assert(!ggml_remote_attn_try_activate(second));
    ggml_remote_attn_deactivate(first);
    assert(ggml_remote_attn_try_activate(second));
    ggml_remote_attn_deactivate(first);
    assert(!ggml_remote_attn_try_activate(first));
    ggml_remote_attn_deactivate(second);
    ggml_remote_attn_set_active(first);
    assert(!ggml_remote_attn_try_activate(first));
    assert(!ggml_remote_attn_try_activate(second));
    ggml_remote_attn_set_active(nullptr);
    assert(ggml_remote_attn_try_activate(second));
    ggml_remote_attn_set_active(nullptr);
    assert(!ggml_remote_attn_try_activate(first));
    ggml_remote_attn_deactivate(second);
    ggml_remote_attn_set_active(first);
    ggml_remote_attn_deactivate(first);
    assert(ggml_remote_attn_try_activate(second));
    ggml_remote_attn_set_active(nullptr);
    assert(!ggml_remote_attn_try_activate(first));
    ggml_remote_attn_deactivate(second);

    // A refused legacy caller has no identity in exec; each conflict must
    // permanently retire dispatch even after both apparent owners release.
    for (int scenario = 0; scenario < 3; ++scenario) {
        const pid_t child = fork();
        assert(child >= 0);
        if (child == 0) {
            if (scenario == 0) assert(ggml_remote_attn_try_activate(first));
            if (scenario == 1) ggml_remote_attn_set_active(first);
            if (scenario == 2) assert(ggml_remote_attn_reserve_stage(first));
            ggml_remote_attn_set_active(second);
            ggml_remote_attn_set_active(nullptr);
            ggml_remote_attn_deactivate(first);
            ggml_remote_attn_release_stage(first);
            assert(!ggml_remote_attn_exec(nullptr));
            assert(!ggml_remote_attn_try_activate(second));
            assert(!ggml_remote_attn_reserve_stage(second));
            _exit(0);
        }
        int status = 0;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    bool first_won = false, second_won = false;
    std::thread a([&] { first_won = ggml_remote_attn_try_activate(first); });
    std::thread b([&] { second_won = ggml_remote_attn_try_activate(second); });
    a.join(); b.join();
    assert(first_won != second_won);
    ggml_remote_attn_deactivate(first);
    ggml_remote_attn_deactivate(second);
    ggml_backend_free(first);
    ggml_backend_free(second);
    std::puts("PASS: remote attention ownership is exclusive and owner-bound");
#endif
}
