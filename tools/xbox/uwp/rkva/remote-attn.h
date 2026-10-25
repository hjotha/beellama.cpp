#pragma once

#include <string>

// RKVA — remote KVarN attention service (Xbox Series X side).
//
// A dedicated persistent TCP listener (default port 50054) that owns the
// target model's full-attention KV cache and runs the attention core on the
// console's D3D12 APU. It is separate from the DFlash DFL2 service (50053) so
// the validated drafter path is never disturbed. One connection owns one host
// context; calls are strictly serialized (the 16 full-attention layers are
// sequential on the host).
//
// Control plane (HELLO/CREATE_SESSION/DESTROY_SESSION/RESET/TRIM/REWIND/PING/
// GET_STATS) is fully implemented here. The attention data plane
// (ATTN_DECODE/ATTN_PREFILL) is served by RemoteAttnEngine's D3D12 KVarN
// kernels; until those land it returns RKVA_ERR_UNSUPPORTED_OP rather than
// silently producing wrong results.
void serve_remote_attn(const std::string& local_folder);
