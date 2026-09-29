#include "dflash-pipeline-candidate.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>

int main() {
    const int32_t positions[] = {20, 21, 22, 23};
    assert(dflash_pipeline_prefix_offset(positions, 4, 20, 0) == 0);
    assert(dflash_pipeline_prefix_offset(positions, 4, 20, 3) == 0);
    assert(dflash_pipeline_prefix_offset(positions, 4, 21, 2) == 1);
    assert(dflash_pipeline_prefix_offset(positions, 4, 19, 1) == -1); // missing anchor
    assert(dflash_pipeline_prefix_offset(positions, 4, 22, 2) == -1); // missing tail
    const int32_t gap[] = {20, 22, 23};
    assert(dflash_pipeline_prefix_offset(gap, 3, 20, 1) == -1);
    assert(dflash_pipeline_prefix_offset(nullptr, 4, 20, 1) == -1);
    assert(dflash_pipeline_prefix_offset(positions, 4, 20, -1) == -1);
    assert(dflash_pipeline_prefix_offset(positions, 4, 20, std::numeric_limits<int32_t>::max()) == -1);

    double prefix = 1.0;
    dflash_pipeline_candidate best;
    const float ids0[] = {11, 12, 13};
    const float scores0[] = {2.0f, 1.0f, 0.0f};
    best = dflash_pipeline_rank2_step(ids0, scores0, 3, 0, 0, prefix, best);
    assert(best.accepted == 0 && best.token == 12 && best.token != 11);
    const float ids1[] = {21, 22};
    const float scores1[] = {0.0f, -0.1f};
    best = dflash_pipeline_rank2_step(ids1, scores1, 2, 0, 1, prefix, best);
    assert(best.accepted == 1 && best.token == 22 && best.score > 0.25f);

    prefix = 1.0;
    best = {};
    const float duplicate_ids[] = {31, 31};
    best = dflash_pipeline_rank2_step(duplicate_ids, scores1, 2, 0, 0, prefix, best);
    assert(best.accepted == -1 && best.token == -1);
    best = dflash_pipeline_rank2_step(ids1, scores1, 1, 0, 0, prefix, best);
    assert(best.accepted == -1 && best.token == -1);

    prefix = 1.0;
    const float invalid_ids[] = {std::numeric_limits<float>::quiet_NaN(), 42};
    best = dflash_pipeline_rank2_step(invalid_ids, scores1, 2, 0, 0, prefix, best);
    assert(best.accepted == -1 && prefix == 0.0);
    prefix = 1.0;
    best = dflash_pipeline_rank2_step(ids1, scores1, 2, 2, 0, prefix, best);
    assert(best.accepted == -1 && prefix == 0.0);
}
