#pragma once

#include "ggml.h"

// CPU registry proc "ggml_backend_cpu_kvarn_store_direct_v1". Executes the
// canonical single-thread CPU KVarN store inside an existing CPU graph worker;
// it never starts a nested graph or thread pool. The caller owns all tensors.
typedef bool (*ggml_backend_cpu_kvarn_store_direct_v1_t)(struct ggml_tensor * store);
