#!/usr/bin/env python3
"""CPU-only guard/structure checks; not a CUDA execution qualification."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
start = source.index('static bool ggml_cuda_remote_singleton_eligible(')
end = source.index('\n}\n', start) + 3
predicate = source[start:end]
# Compile the actual production predicate against a deliberately small metadata
# fixture. CUDA availability/placement is an explicit stub, never simulated GPU
# execution. No CUDA compiler, driver, model or remote peer is needed.
fixture = r'''
#include <cassert>
#include <cstdint>
enum { GGML_OP_REMOTE_ATTN=1, GGML_TYPE_F32=2, GGML_TENSOR_FLAG_COMPUTE=4,
       GGML_REMOTE_ATTN_PARAM_RANGE=0, GGML_REMOTE_ATTN_PARAM_WAIT=1,
       GGML_REMOTE_ATTN_PARAM_ASYNC=2 };
struct ggml_tensor {
    int op=GGML_OP_REMOTE_ATTN, flags=GGML_TENSOR_FLAG_COMPUTE, type=GGML_TYPE_F32;
    ggml_tensor *src[1]={}; int64_t ne[2]={256,1}; int params[3]={1,0,0};
    bool contiguous=true, placement=true;
};
struct ggml_cgraph { uint64_t uid=1; int n_nodes=1; ggml_tensor *nodes[2]={}; };
bool ggml_is_contiguous_rows(const ggml_tensor *n) { return n->contiguous; }
int ggml_get_op_params_i32(const ggml_tensor *n, int i) { return n->params[i]; }
bool ggml_backend_remote_attn_cuda_enabled_for_node(const ggml_tensor *n) { return n->placement; }
'''
checks = r'''
int main() {
    ggml_tensor q, n; n.src[0]=&q; ggml_cgraph g; g.nodes[0]=&n;
    assert(!ggml_cuda_remote_singleton_eligible(nullptr));
    for (int nq=1; nq<=9; ++nq) { n.ne[1]=nq; assert(ggml_cuda_remote_singleton_eligible(&g)); }
    for (int nq : {0, 10, 17, 256}) { n.ne[1]=nq; assert(!ggml_cuda_remote_singleton_eligible(&g)); }
    n.ne[1]=1;
    g.uid=0; assert(!ggml_cuda_remote_singleton_eligible(&g)); g.uid=1;
    g.n_nodes=0; assert(!ggml_cuda_remote_singleton_eligible(&g));
    g.n_nodes=2; assert(!ggml_cuda_remote_singleton_eligible(&g)); g.n_nodes=1;
    g.nodes[0]=nullptr; assert(!ggml_cuda_remote_singleton_eligible(&g)); g.nodes[0]=&n;
    n.op=0; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.op=GGML_OP_REMOTE_ATTN;
    n.flags=0; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.flags=GGML_TENSOR_FLAG_COMPUTE;
    n.type=0; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.type=GGML_TYPE_F32;
    n.src[0]=nullptr; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.src[0]=&q;
    n.params[0]=0; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.params[0]=1;
    for (int slot : {1,2}) { n.params[slot]=1; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.params[slot]=0; }
    n.contiguous=false; assert(!ggml_cuda_remote_singleton_eligible(&g)); n.contiguous=true;
    n.placement=false; assert(!ggml_cuda_remote_singleton_eligible(&g));
}
'''
with tempfile.TemporaryDirectory(prefix='rkva-singleton-') as tmp:
    cpp = Path(tmp) / 'predicate.cpp'
    cpp.write_text('#include <initializer_list>\n' + fixture + predicate + checks)
    binary = Path(tmp) / 'predicate'
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++11', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)

compute = source[source.index('static enum ggml_status ggml_backend_cuda_graph_compute('):source.index('static void ggml_backend_cuda_event_record(')]
assert 'return option && strcmp(option, "1") == 0;' in compute
assert '!cuda_ctx->disable_cuda_graphs_due_to_memory_pressure' in compute
assert 'cuda_ctx->curr_stream_no == 0' in compute
assert 'cuda_ctx->stream_context().concurrent_events.empty()' in compute
assert 'if (!remote_singleton_fastpath) {' in compute
assert 'if (graph && graph->is_enabled())' in compute
assert 'if (bookkeeping.enabled && graph)' in compute
assert compute.count('check_memory_pressure_before_graph()') == 1
assert compute.index('check_memory_pressure_before_graph()') < compute.index('ggml_cuda_graph_evaluate_and_capture(')
assert 'return GGML_STATUS_FAILED;' in compute
assert 'ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required,' in compute
assert 'return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_ABORTED;' in compute
assert 'ggml_remote_attn_exec_cuda(' not in compute  # never bypass the evaluator
print('RKVA singleton predicate matrix and dispatch structure checks passed (CPU-only)')
