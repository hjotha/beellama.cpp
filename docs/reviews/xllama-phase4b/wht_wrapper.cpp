#include "/home/hjotha/xllama-qwen35-xbox/uwp/rkva/rkva-kvarn-range.h"
extern "C" void wht_cpu(const float * src, float * dst) {
    rkva_kvarn_range::butterfly_wht256(src, dst);
}
