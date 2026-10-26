// Included only by RKVA_DECODE_ROWLANE in CSRangeAttnKvarn4Decode.
// Wave64: lane = row during QK, lane = four dimensions during PV.
// Every WaveReadLaneAt index is a wave-UNIFORM loop induction variable.
// Never replace these broadcasts with lane-dependent indices on Xbox DXIL.
// Q*scale stays in dimension-owned registers; row weights stay in row-owned
// registers. Uniform broadcasts avoid LDS addressing and all record barriers.
[loop] for (uint t0 = fast_lo; t0 < fast_hi; t0 += 128) {
    uint rgroup = t0 / 128;
    uint base = (rgroup * n_head_kv * 2 + kv_head * 2) * RKVA_RECORD;
    // Zero correction per slice; one reduction per head/slice/group, not row.
    uint dim = (u & 31) * 4;
    uint rb_dim = base + (u >> 5) * RKVA_RECORD;
    float4 kz = halves4(keys.Load2(rb_dim + 8448 + dim * 2));
    float4 ks = halves4(keys.Load2(rb_dim + 8192 + dim * 2));
    float4 qs[RKVA_GQA];
    float z0[RKVA_GQA], z1[RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
        qs[g] = q[g] * ks;
        float z = dot(q[g], kz);
        z0[g] = WaveActiveSum(u < 32 ? z : 0);
        z1[g] = WaveActiveSum(u >= 32 ? z : 0);
    }

    uint pos = wave * 64 + lane;
    uint shift = (pos & 7) * 4;
    float score[RKVA_GQA];
    [unroll] for (uint g = 0; g < RKVA_GQA; ++g) score[g] = 0;
    [unroll] for (uint slice = 0; slice < 2; ++slice) {
        uint rb = base + slice * RKVA_RECORD;
        float acc[RKVA_GQA];
        [unroll] for (uint g = 0; g < RKVA_GQA; ++g) acc[g] = 0;
        // Four dimensions per iteration. src_lane is uniform in the wave;
        // the row lane only affects the coalesced K address and nibble shift.
        [loop] for (uint du = 0; du < 32; ++du) {
            uint kb = rb + du * 256 + (pos >> 3) * 4;
            float4 nib = float4((keys.Load(kb) >> shift) & 15u,
                                (keys.Load(kb + 64) >> shift) & 15u,
                                (keys.Load(kb + 128) >> shift) & 15u,
                                (keys.Load(kb + 192) >> shift) & 15u);
            uint src_lane = slice * 32 + du;
            [unroll] for (uint g = 0; g < RKVA_GQA; ++g) {
                acc[g] = mad(WaveReadLaneAt(qs[g].x, src_lane), nib.x, acc[g]);
                acc[g] = mad(WaveReadLaneAt(qs[g].y, src_lane), nib.y, acc[g]);
                acc[g] = mad(WaveReadLaneAt(qs[g].z, src_lane), nib.z, acc[g]);
                acc[g] = mad(WaveReadLaneAt(qs[g].w, src_lane), nib.w, acc[g]);
            }
        }
        float other = half_at(keys, rb + 8704 + pos * 2);
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

    float4 vo = halves4(values.Load2(rb_dim + 8704 + dim * 2));
    [loop] for (uint t = 0; t < 64; t += 8) {
        uint vp = wave * 64 + t;
        uint4 vs = values.Load4(rb_dim + 8192 + vp * 2);
        uint4 vz = values.Load4(rb_dim + 8448 + vp * 2);
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
