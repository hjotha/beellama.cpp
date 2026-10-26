// GPU ceiling probe: STREAM read / write / copy.
//
// One dispatch moves one whole buffer with 16-byte vectors and a grid-stride
// loop, so the grid size is independent of the buffer size and the same PSO
// can be swept over occupancy. Every entry point is a separate PSO compiled
// from this file by tools/xbox/scripts/compile-gpupeak-shaders.py.
//
// Register contract (see gpupeak.cpp):
//   b0 CBV  Params
//   t0 SRV  Src   (raw, StructureByteStride = 0)
//   u0 UAV  Sink  (structured uint, stride 4)
//   u1 UAV  Dst   (raw, StructureByteStride = 0)
// Unused registers stay bound but are not declared per entry point.

cbuffer Params : register(b0) {
    uint n_vec;      // 16-byte vectors in the buffer
    uint groups_x;   // Dispatch X, for the flat group index
    uint groups_y;   // Dispatch Y
    uint mode;       // reserved (0 read, 1 write, 2 copy) — documentation only
};

ByteAddressBuffer Src : register(t0);
RWStructuredBuffer<uint> Sink : register(u0);
RWByteAddressBuffer Dst : register(u1);

#define GPUPK_THREADS 256u
#define GPUPK_VEC_BYTES 16u

uint gpupk_flat_thread(uint3 dtid) {
    // Grid-stride flattening: the dispatch is (x, y) so 1 GiB (67 M vectors)
    // stays inside the per-axis limit.
    return dtid.x + dtid.y * (groups_x * GPUPK_THREADS);
}

uint gpupk_flat_group(uint3 gid) {
    return gid.y * groups_x + gid.x;
}

// Pure streaming read. In timing mode the conditional store depends on every
// loaded word, so no load can be eliminated and the store is never taken for
// the probe's own pattern. mode == 1 forces the reduction to be written so the
// host can XOR-fold it and prove the whole buffer was read (validation pass).
//
// The accumulator goes to its own flat-thread slot. Storing it in
// Sink[flat_group] would make all 256 threads of a group race on one plain
// (non-atomic) store: the slot would keep an arbitrary thread's accumulator,
// not the group XOR. The host folds groups_x * groups_y * 256 slots, and the
// Sink UAV has exactly that many elements for the 2048-group stream plan.
[numthreads(GPUPK_THREADS, 1, 1)]
void CSRead(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const uint stride = groups_x * groups_y * GPUPK_THREADS;
    uint acc = 0u;
    for (uint i = gpupk_flat_thread(dtid); i < n_vec; i += stride) {
        const uint4 v = Src.Load4(i * GPUPK_VEC_BYTES);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (mode == 1u || acc == 0x9e3779b9u) {
        Sink[gpupk_flat_thread(dtid)] = acc;
    }
}

// Pure streaming write: no reads, so the rate is the write bandwidth.
[numthreads(GPUPK_THREADS, 1, 1)]
void CSWrite(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const uint stride = groups_x * groups_y * GPUPK_THREADS;
    const uint4 pattern = uint4(dtid.x + 1u, gid.x + 1u, n_vec, mode);
    for (uint i = gpupk_flat_thread(dtid); i < n_vec; i += stride) {
        Dst.Store4(i * GPUPK_VEC_BYTES, pattern);
    }
    if (dtid.x == 0u && dtid.y == 0u && dtid.z == 0u) {
        Sink[0] = 1u;
    }
}

// Streaming copy: reads Src and writes Dst, 16 bytes per thread per step.
[numthreads(GPUPK_THREADS, 1, 1)]
void CSCopy(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    const uint stride = groups_x * groups_y * GPUPK_THREADS;
    uint acc = 0u;
    for (uint i = gpupk_flat_thread(dtid); i < n_vec; i += stride) {
        const uint4 v = Src.Load4(i * GPUPK_VEC_BYTES);
        Dst.Store4(i * GPUPK_VEC_BYTES, v);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) {
        Sink[gpupk_flat_thread(dtid)] = acc;
    }
}
