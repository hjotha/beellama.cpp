// Included only by RKVA_DECODE_ROWLANE in CSRangeAttnKvarn4Decode.
// Wave64: lane = row during QK, lane = four dimensions during PV.
// Every WaveReadLaneAt index is a wave-UNIFORM loop induction variable.
// Never replace these broadcasts with lane-dependent indices on Xbox DXIL.
// Q*scale stays in dimension-owned registers; row weights stay in row-owned
// registers. Uniform broadcasts avoid LDS addressing and all record barriers.
[loop] for (uint t0 = fast_lo; t0 < fast_hi; t0 += 128) {
    uint rgroup = t0 / 128;
    uint bits = rkva_kvarn_bits();
    uint pay_bytes = bits * 2048u;
    uint rec_bytes = pay_bytes + 768u;
    uint base = (rgroup * n_head_kv * 2 + kv_head * 2) * rec_bytes;
    // Zero correction per slice; one reduction per head/slice/group, not row.
    uint dim = (u & 31) * 4;
    uint rb_dim = base + (u >> 5) * rec_bytes;
    float4 kz = halves4(keys.Load2(rb_dim + pay_bytes + 256u + dim * 2));
    float4 ks = halves4(keys.Load2(rb_dim + pay_bytes + dim * 2));
    float4 qs[RKVA_GQA];
    float z0[RKVA_GQA], z1[RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        qs[g] = q[g] * ks;
        float z = dot(q[g], kz);
        z0[g] = WaveActiveSum(u < 32 ? z : 0);
        z1[g] = WaveActiveSum(u >= 32 ? z : 0);
    }

    uint pos = wave * 64 + lane;
    uint pos_bit = pos * bits;
    uint pos_word = (pos_bit >> 5) * 4u;
    uint shift = pos_bit & 31u;
    uint mask = (1u << bits) - 1u;
    uint row_stride = 16u * bits;
    uint du_stride = 64u * bits;
    bool cross_dword = (shift + bits) > 32u;
    float score[RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) score[g] = 0;
    [unroll] for (uint slice = 0; slice < 2; ++slice) {
        uint rb = base + slice * rec_bytes;
        float acc[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) acc[g] = 0;
        // Four dimensions per iteration. src_lane is uniform in the wave;
        // the row lane only affects the coalesced K address and bit shift.
        [loop] for (uint du = 0; du < 32; ++du) {
            uint kb = rb + du * du_stride + pos_word;
            float4 nib;
            [branch] if (!cross_dword) {
                nib = float4((keys.Load(kb) >> shift) & mask,
                             (keys.Load(kb + row_stride) >> shift) & mask,
                             (keys.Load(kb + 2u * row_stride) >> shift) & mask,
                             (keys.Load(kb + 3u * row_stride) >> shift) & mask);
            } else {
                uint rshift = 32u - shift;
                uint2 w0 = keys.Load2(kb);
                uint2 w1 = keys.Load2(kb + row_stride);
                uint2 w2 = keys.Load2(kb + 2u * row_stride);
                uint2 w3 = keys.Load2(kb + 3u * row_stride);
                nib = float4(((w0.x >> shift) | (w0.y << rshift)) & mask,
                             ((w1.x >> shift) | (w1.y << rshift)) & mask,
                             ((w2.x >> shift) | (w2.y << rshift)) & mask,
                             ((w3.x >> shift) | (w3.y << rshift)) & mask);
            }
            uint src_lane = slice * 32 + du;
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                acc[g] = mad(WaveReadLaneAt(qs[g].x, src_lane), nib.x, acc[g]);
                acc[g] = mad(WaveReadLaneAt(qs[g].y, src_lane), nib.y, acc[g]);
                acc[g] = mad(WaveReadLaneAt(qs[g].z, src_lane), nib.z, acc[g]);
                acc[g] = mad(WaveReadLaneAt(qs[g].w, src_lane), nib.w, acc[g]);
            }
        }
        float other = half_at(keys, rb + pay_bytes + 512u + pos * 2);
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
            // other is independent for the two physical records/slices.
            score[g] += (acc[g] + (slice == 0 ? z0[g] : z1[g])) * other;
        }
    }
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        score[g] *= kq_scale;
        float next_max = max(maximum[g], WaveActiveMax(score[g]));
        float old_weight = denom[g] == 0 ? 0 : exp(maximum[g] - next_max);
        float w = exp(score[g] - next_max);
        score[g] = w;
        body[g] *= old_weight;
        denom[g] = denom[g] * old_weight + WaveActiveSum(w);
        maximum[g] = next_max;
    }

    float4 vo = halves4(values.Load2(rb_dim + pay_bytes + 512u + dim * 2));
    [loop] for (uint t = 0; t < 64; t += 8) {
        uint vp = wave * 64 + t;
        uint4 vs = values.Load4(rb_dim + pay_bytes + vp * 2);
        uint4 vz = values.Load4(rb_dim + pay_bytes + 256u + vp * 2);
        [unroll] for (uint j = 0; j < 8; ++j) {
            uint sh = (j & 1) * 16;
            float4 vv = tile_value(rb_dim, vp + j, u,
                                   f16tof32((vs[j / 2] >> sh) & 65535u),
                                   f16tof32((vz[j / 2] >> sh) & 65535u), vo);
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                body[g] += WaveReadLaneAt(score[g], t + j) * vv;
            }
        }
    }
}
