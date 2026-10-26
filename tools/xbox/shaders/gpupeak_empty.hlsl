// GPU ceiling probe: near-empty dispatch.
//
// The only work is one store from thread 0 of group 0. Everything else is the
// fixed D3D12 cost of a dispatch: command-list recording, submit, queue
// latency, fence completion and (on the host side) the readback Map. The
// timestamp pair around the dispatch in the same command list is what the
// engine reports as gpu_total for a real call, minus the real work.

cbuffer Params : register(b0) {
    uint marker;    // written so the compiler keeps a live UAV access
    uint groups_x;
    uint groups_y;
    uint reserved;
};

RWStructuredBuffer<uint> Sink : register(u0);

[numthreads(64, 1, 1)]
void CSEmpty(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x == 0u && dtid.y == 0u && dtid.z == 0u) {
        Sink[0] = marker;
    }
}
