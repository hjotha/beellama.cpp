#include "server-task.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>

int main() {
    server_prompt_cache cache(1,0);
    cache.states.emplace_back(); cache.states.back().data.main.assign(16000,0x15);
    cache.states.emplace_back(); cache.states.back().data.main.assign(24000,0x27);
    auto checkpoint=std::make_shared<common_prompt_checkpoint>();
    checkpoint->data_tgt.assign(4096,0x39);
    cache.states.front().prompt.checkpoints.push_back(checkpoint);
    cache.states.back().prompt.checkpoints.push_back(checkpoint);
    auto * first=&cache.states.front();auto * last=&cache.states.back();
    const size_t base=cache.size();cache.limit_size=base+1000;
    assert(cache.reserve_transient_without_eviction(600));
    assert(cache.size()==base+600);
    assert(cache.reserve_transient_without_eviction(400));
    assert(cache.size()==cache.limit_size);
    assert(!cache.reserve_transient_without_eviction(1));
    assert(cache.size()==base+1000 && cache.evictions==0 && cache.states.size()==2);
    assert(&cache.states.front()==first && &cache.states.back()==last);
    assert(cache.states.front().data.main.front()==0x15 && cache.states.back().data.main.front()==0x27);
    assert(cache.states.front().prompt.checkpoints.front()==cache.states.back().prompt.checkpoints.front());
    cache.release_transient(1000);assert(cache.size()==base);
    assert(!cache.reserve_transient_without_eviction(1001));
    assert(cache.states.size()==2 && cache.evictions==0 && cache.size()==base);
    cache.limit_size=base-1;
    assert(!cache.reserve_transient_without_eviction(0));
    assert(cache.states.size()==2 && cache.evictions==0);
    cache.limit_size=0; // unlimited still guards size_t arithmetic
    assert(!cache.reserve_transient_without_eviction(std::numeric_limits<size_t>::max()));
    assert(cache.states.size()==2 && cache.size()==base);
    assert(cache.reserve_transient_without_eviction(1));
    assert(!cache.reserve_transient_without_eviction(std::numeric_limits<size_t>::max()));
    assert(cache.states.size()==2 && cache.evictions==0 && cache.size()==base+1);
    cache.release_transient(1);

    // Existing admission retains its eviction semantics for explicit requests.
    server_prompt_cache explicit_cache(1,0);
    explicit_cache.states.emplace_back();explicit_cache.states.back().data.main.assign(4096,0x44);
    const size_t occupied=explicit_cache.size();explicit_cache.limit_size=occupied+1;
    assert(explicit_cache.reserve_transient(2));
    assert(explicit_cache.states.empty() && explicit_cache.evictions==1);
    explicit_cache.release_transient(2);
    std::puts("prompt cache non-evicting cumulative reservation: PASS");
}
