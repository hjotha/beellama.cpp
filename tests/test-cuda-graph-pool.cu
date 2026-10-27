// White-box hardware regression: compile with ggml-cuda.cu's exact build flags.
// Deliberately includes the implementation to exercise its private pool/cache.
#include "../ggml/src/ggml-cuda/ggml-cuda.cu"

static __global__ void fixture_write(float * scratch, float * output, int value) {
    const int i = threadIdx.x;
    scratch[i] = float(value + i);
    output[i] = scratch[i];
}
static void check_output(float * output, int value) {
    float host[256];
    CUDA_CHECK(cudaMemcpy(host, output, sizeof(host), cudaMemcpyDeviceToHost));
    for (int i=0; i<256; ++i) GGML_ASSERT(host[i] == float(value+i));
}
static void capture_fixture(ggml_backend_cuda_context & ctx, ggml_cgraph & gf,
                            float * output, int value, bool overflow) {
    auto & pool = static_cast<ggml_cuda_pool_leg &>(ctx.pool());
    auto * graph = ctx.cuda_graph(gf.nodes[0]);
    ggml_cuda_graph_update_required(&ctx, &gf);
    size_t actual;
    float * scratch = (float *)pool.alloc(1024*1024, &actual);
    const auto before = ctx.pool_epoch;
    ctx.pool_capture_active = true;
    CUDA_CHECK(cudaStreamBeginCapture(ctx.stream(),cudaStreamCaptureModeRelaxed));
    fixture_write<<<1,256,0,ctx.stream()>>>(scratch,output,value);
    pool.free(scratch,actual);
    if (overflow) {
        GGML_ASSERT(pool.has_deferred_frees());
        GGML_ASSERT(ctx.pool_deferred_frees);
        GGML_ASSERT(ctx.pool_epoch == before);
    }
    CUDA_CHECK(cudaStreamEndCapture(ctx.stream(),&graph->graph));
    ctx.pool_capture_active = false;
    graph->pool_epoch = ctx.pool_epoch;
    GGML_ASSERT(ggml_cuda_graph_instantiate(&ctx,gf.nodes[0]));
    graph->warmup_complete = true;
    CUDA_CHECK(cudaGraphLaunch(graph->instance,ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    check_output(output,value);
}
int main() {
    ggml_cuda_set_device(0);
    float * output;
    CUDA_CHECK(cudaMalloc(&output,256*sizeof(float)));
    {
        ggml_backend_cuda_context ctx(0);
        ggml_tensor node = {};
        snprintf(node.name,sizeof(node.name),"pool-regression");
        node.data = output;
        node.type = GGML_TYPE_F32;
        for (int i=0;i<GGML_MAX_DIMS;++i) {node.ne[i]=1;node.nb[i]=4;}
        ggml_tensor * nodes[] = {&node};
        ggml_cgraph gf = {};gf.nodes=nodes;gf.n_nodes=1;gf.uid=1;
        auto & pool = static_cast<ggml_cuda_pool_leg &>(ctx.pool());
        auto * graph = ctx.cuda_graph(&node);
        capture_fixture(ctx,gf,output,7,false);
        const auto captured=graph->pool_epoch;
        pool.clear_pool(); // Same release path used by a real allocation-OOM retry.
        GGML_ASSERT(ctx.pool_epoch != captured);
        ggml_cuda_graph_update_required(&ctx,&gf);
        GGML_ASSERT(!graph->instance && !graph->graph && !graph->warmup_complete);
        capture_fixture(ctx,gf,output,13,false);
        CUDA_CHECK(cudaGraphLaunch(graph->instance,ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));check_output(output,13);
        puts("PASS: post-capture pool flush invalidates stale replay; recapture numeric output exact");
        pool.clear_pool();ggml_cuda_graph_update_required(&ctx,&gf);
        // Fill every legacy cache entry while keeping the captured scratch too
        // large to reuse any of those entries.
        std::vector<std::pair<void *,size_t>> blocks;
        for (int i=0;i<ggml_cuda_pool_leg::MAX_BUFFERS;++i) {
            size_t size;void * ptr=pool.alloc(4096+i*256,&size);blocks.push_back({ptr,size});
        }
        for (auto b:blocks)pool.free(b.first,b.second);
        capture_fixture(ctx,gf,output,29,true);
        const auto spill_capture=graph->pool_epoch;
        ggml_cuda_release_deferred_pool_frees(&ctx);
        GGML_ASSERT(!ctx.pool_deferred_frees && !pool.has_deferred_frees());
        GGML_ASSERT(ctx.pool_epoch != spill_capture);
        ggml_cuda_graph_update_required(&ctx,&gf);
        GGML_ASSERT(!graph->instance && !graph->graph && !graph->warmup_complete);
        pool.clear_pool();
        capture_fixture(ctx,gf,output,41,false);
        CUDA_CHECK(cudaGraphLaunch(graph->instance,ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));check_output(output,41);
        puts("PASS: cache-full free deferred during capture; first launch exact; release invalidates replay; recapture exact");
    }
    CUDA_CHECK(cudaFree(output));
    puts("PASS: legacy CUDA graph pool regression");
}
