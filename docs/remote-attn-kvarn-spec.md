# KVarN -> HLSL/D3D12 port spec (extracted reference)

> Fonte: sub-agente Explore sobre kvarn_store.comp / kvarn.cu / llama-kvarn.cpp /
> fattn-mma-kvarn*.cuh (2026-09-30). Cobre secoes 1-2a (constantes, layout de
> records/tiles/stage/indices, estrutura do dispatch de store). TRUNCADO no meio
> da secao 2 (MODE_COMMIT); as secoes 3 (WHT), 4 (flash-attn), 5 (materialize),
> 6 (dominios), 7 (barreiras), 8 (riscos de paridade) devem ser relidas direto
> das fontes canonicas listadas abaixo durante a fase KVarN.
>
> Uso: referencia para portar os kernels KVarN ao Xbox (RKVA cache_bits 2..8),
> que hoje retornam UNSUPPORTED. O caminho F16 (cache_bits 0/16) ja esta validado
> (cos=1.0) e usa rkva_flash_attn.hlsl + rkva_store_f16.hlsl.

---

## Constants and Geometry (qwen35 / kvarn4)
- KVAR_N_GROUP = 128 tokens/group
- head_dim = 256 → head_slices = 2, record_dim = 128
- Physical KV heads n_heads = n_head_kv * head_slices = 4*2 = 8; logical KV heads 4; Q heads 24; gqa = 6
- bits K = 4, V = 4 → qmax = 15
- sinkhorn_iters = 16 (default), clamped iterations
- sink_tokens = 128 (group 0 never quantized)
- kq_scale = 1/sqrt(256) = exactly 0.0625
- WHT scale 128: 0.08838834764831845 = 1/sqrt(128); slice mix 0.7071067811865475 = 1/sqrt(2); combined 1/sqrt(256) = 0.0625
- tail_groups = 2 (RKVA hello), stage_groups = tail_groups+1 = 3 (non-SWA constraint tail_groups < stage_groups; group 0 → slot 0; groups g>0 → slot 1 + ((g-1) % tail_groups))

## Record Layout (GPU, per physical head / per group)
- record_words (in u32 units) = bits*512 + 192 (Vulkan flash attn); record_bytes = bits*2048 + 768; for bits=4 = 8192+768 = 8960 B
- K record: row=dim(128), col=token(128): payload index = row*128+col, LSB-first packed 4-bit little-endian bitstream
- Axis0 (scale) at payload_bytes position: 128 u16 = f16(best_row[row]*scale[row]) — for K row=dim
- Axis1 (zp): +128 u16 = f16(best_row[row]*lo[row])
- Axis2 (other): +256 u16 = f16(best_col[col]) — for K col=token
- V record: row=token, col=dim: axis0=f16(best_row[token]*scale), axis1=f16(best_row[token]*lo), axis2=f16(best_col[dim])
- Dequantization: value = (float(q) * f16(scale[row]) + f16(zp[row])) * f16(other[col]) — operations in F32 after f16→f32 conversion
- Record base offset = ((stream*groups_per_stream + group)*n_heads + head)*record_bytes; head = logical_kv_head*head_slices + slice
- K and V are separate buffers.
- CPU/state tile layout (llama_kvarn_make_layout): K payload, then k_s_col(head_dim u16), k_zp(head_dim u16), k_s_row(group u16), V payload, v_s_col(head_dim u16), v_s_row(group u16), v_zp(group u16), tile_bytes aligned to 8. Note: for head_dim 256 the state file uses per-slice 128 records (the conversion layout uses KVAR_N_GROUP axis = 128 entries). Record layout (make_record_layout): payload_bytes = rows*cols*bits/8; scale_off=payload, zp_off=+rows*2, other_off=+rows*2*2, record_bytes=other+cols*2. K has rows=record_dim, cols=128; V has rows=128, cols=record_dim.

## Stage Layout
- stage: F16, index (stage_pos * n_heads + head) * 128 + dim; stage_pos = stream*128*stage_groups + stage_slot*128 + pos; slot: group0→0, group g≥1→1+((g-1)%tail_groups) (or host-assigned slot via upper word of index; SWA→group%stage_groups)
- Content: for both K and V, rotation domain (post full 256-WHT), F16 rounded.

## Index Encoding
- indices: I64 per token; normally lower 32 = absolute cell (position); upper 32 = slot+1 (0 = stateless fallback); encoded < -1 → explicitly staged: payload = -(encoded+2). -1 = padding/invalid. SWA: lower=absolute position, upper=stream.

## WHT (kvarn_wht.comp / parallel / CUDA)
- Per row of head_width (256 for qwen35): per-128-slice butterfly (7 stages: within subgroup strides 1..64 via shuffle xor, cross via shared memory), scale 1/sqrt(128), then cross-slice Hadamard on slice values: stride loop (a+b, a-b), scale 1/sqrt(slices). Equivalent to normalized 256-point WHT (self-inverse, symmetric). data_type: 0=f32, 1=f16, 2=bf16 (bf16 conversion uses rounding bias 0x7fff + lsb — round to nearest even). In-place? Not possible — src/dst are separate buffers.
- Butterfly semantics: value = (lane & stride)==0 ? value+other : other-value. This is XOR butterfly producing H·x with normalization.
- For head_width=256 the parallel version uses local_size 256; sequential version uses 128 threads iterating slices.

## Store (Sinkhorn + Quantization)
Two implementations (CUDA monolithic head-wide / per-head; Vulkan workspace mode), mathematically identical, both 128 threads:

Per-token WHT (monolithic): load F32 current[(token*n_heads+head)*128+lane] per slice, wht_128, cross-slice mix, round to F16 → stage.

quantize_tile (128 threads, lane = row index of transposed stage matrix):
1. Build tile (128x128 F32) from stage: K: tile[row=dim][col=token]; V: tile[row=token][col=dim]. CUDA's kvarn_quantize_stage loads tile[i] with row=i/128, col=i%128, token = value?row:col, dim=value?col:row.
2. Initialize log_s_col=0, log_s_row=0, s_col=s_row=1, best_col=best_row=1; initial col_std/row_std at identity transform; best_imbalance=FLT_MAX (CUDA) → update_best with candidate 1 (so initial imbalance becomes baseline and best=1).
   Vulkan: computes std, lane0 computes baseline imbalance directly into sh_best_imbalance (best remains 1). Semantics are equivalent.
3. iterations=16 loop:
   a. col_std_i = clamp(std_col(tile, s_col, s_row, i), 1e-3, 1e3); log_s_col_i = clamp(log_s_col_i + log(col_std_i), -0.3, 10); s_col = exp(log_s_col)
      std_col(col): scaled = tile[r][col]/(s_col[col]*s_row[r]) accumulated sequentially over r=0..127; sum, sum_sq; mean=sum/128; var=max((sum_sq - 128*mean*mean)/127, 0); std=sqrt(var). (F32 accumulation; CPU reference uses double — GPU is canonical.)
   b. row_std recomputed with new s_col; row_std_i=clamp(...); log_s_row_i=clamp(log_s_row_i+log(row_std_i), -0.3, 10); s_row=exp
      Note: CUDA updates s_col, then before row update recomputes row_std with (new s_col, old s_row) — Vulkan computes row_std at start of iteration with (old s_col, old s_row)?? Wait — check Vulkan:
      
      Vulkan loop:
      ```
      const float col_std = clamp(std_col(...lane), 1e-3, 1e3);   // under current log_s
      sh_log_s_col[lane] = clamp(sh_log_s_col[lane] + log(col_std), -0.3, 10.0);
      barrier();
      const float row_std = clamp(std_row(...lane), 1e-3, 1e3);   // std_row uses exp(sh_log_s_col) — after col update! Yes: std_row divides by (exp(sh_log_s_col[col]) * sr) reading updated log_s_col
      sh_log_s_row[lane] = clamp(... + log(row_std), ...);
      barrier();
      update_best(...);   // recompute both std with new scales, lane0 compares imbalance
      ```
      So Vulkan's row_std is computed after col update (new s_col, old s_row) — matches CUDA order. CUDA computes std_col for next iteration's col update at end of previous iteration (update path) — values equivalent. One subtle difference: CUDA first iteration col_std comes from init (identity) — same as Vulkan's new computation at identity. OK: order is identical.
   c. update_best: recompute col_std and row_std under current (s_col, s_row); reduce min/max over all 128 (CUDA: warp reduce + 4-warp combine; Vulkan: lane0 sequential); imbalance = col_max/max(col_min,1e-8) + row_max/max(row_min,1e-8); if imbalance <= best_imbalance then best_imbalance=imbalance, best_col=exp? — CUDA's candidate is s_col itself (not log) → best_col[i]=s_col[i]; Vulkan: best_col=exp(log_s_col). Same.
4. Per row (lane=row): lo=min, hi=max of x=tile[row][col]/(best_col[col]*best_row[row]) over col; scale=max((hi-lo)/15, 1e-10); q[col]=clamp(round((x-lo)/scale),0,15) — round is GLSL round() = round to nearest even?? GLSL round: "rounds to nearest even integer"? Actually the GLSL spec has round() implementation-defined for .5 (round to nearest even or nearest integer?)! GLSL round: "The fraction 0.5 will round in a direction chosen by the implementation" — hmm. CUDA's roundf = round to nearest, ties away from zero. Vulkan's GLSL round — on most drivers... this is a parity risk to flag. HLSL's round() = round to nearest even. Divergence possible when (x-lo)/scale lands exactly on .5 — measure-zero but flag.
5. Payload packing: bit_offset = col*bits, LSB-first little-endian; CUDA byte-wise; Vulkan word-wise: data_records[row_base + (bit_offset>>5)] |= q << (bit_offset&31); carryover to next word. Equivalent to flat LSB-first bitstream. Row words = bits*4 = 16 u32 per row (bits=4).
6. Axis: f16 round to nearest even (CUDA uses __float2half_rn; Vulkan uses packHalf2x16 = RNE). scale_axis[row]=f16(best_row[row]*scale), zp_axis[row]=f16(best_row[row]*lo), other_axis[col]=f16(best_col[col]).
   Vulkan packs pairs: axis_word<64: word payload_words+axis_word = (best_row[r0]*col_std[r0], best_row[r1]*col_std[r1]) — r0=2w, r1=2w+1 (col_std repurposed as scale holder); +64: zp; +128: best_col pair.

Store variants:
- Delayed (eager_records=0): when token arrives at pos==0 and group>tail_groups (non-SWA), flush group-tail_groups: quantize from stage slot (1+((flush_group-1)%tail_groups)) into record (stream*groups_per_stream+flush_group). Group 0 (sink) and the most recent tail_groups groups stay F16 in stage.
- Eager (eager_records=1): when pos==127 of a group (group>0), quantize immediately from the current stage slot.
- Vulkan workspace path (parallel prefill): MODE_VALIDATE checks that each active stream has contiguous dense indices (first_idx+t) and high word zero (no explicit slots); MODE_STAGE does per-token WHT into workspace F16 (indexed [token][head][128] packed as f16x2); MODE_FLUSH: per (head, stream×flush_candidates): quantize the group matrix reading from workspace+stage (workspace_matrix_value falls back to stage via sh_workspace_stage_slot); MODE_COMMIT copies workspace F16 into stage slot (pick latest token per (slot,pos)); MODE_FALLBACK is monolithic path if invalid (all groups re-execute — idempotent? fallback re-executes whole monolithic store; workspace writes are identical so safe... note both flush+monolithic write same records).

CUDA store kernel mirrors: monolithic per-head block (n_heads/head_slices blocks? headwide: blockIdx.x = logical head, 128 threads); workspace kernels: stage (grid n_heads×chunks), flush (grid head × streams*flush_candidates), commit; direct_flush/direct_stage for decode; d64 variants (record_dim 64, 32 threads).

## Materialize
- MODE_PREPARE_LIVE: per stream compute (live_group, live_pos) = max over indices (group, then pos) → data_live[2*stream+{0,1}].
- Main: flat block over (logical_head, token, out_stream): logical_heads = n_heads/head_slices; flat_block → logical_head = flat % logical_heads; token_stream = flat / logical_heads; token = token_stream % n_kv; out_stream = token_stream / n_kv.
- load_materialized: resolve absolute position (via index if indirect or read_indirect); determine from_stage/from_record with the same rules as attention (eager/delayed/swa); stage slot fallback formula; record dequantization: row = value? pos : lane; col = value? lane : pos; value=(q*scale[row]+zp[row])*other[col].
- If emit_rotated==0: apply wht_128 (shared memory butterfly) per slice, then cross-slice mix → emit original domain F16; otherwise emit rotated values directly. Output packed as f16x2: dst[((out_stream*n_kv+token)*n_heads+head)*64 + lane] = pack(f16(v[2*lane]), f16(v[2*lane+1])).
- dst shape: [128 dim][n_heads][n_kv][n_stream] F16 (ne0=record_dim, ne1=n_heads(slices), ne2=n_kv, ne3=n_stream), for downstream generic fattn.

## Flash Attention (Vulkan shader — the canonical version for RDNA port)
Grid: x = n_query * split_k, y = n_kv_heads * gqa_groups (=4*2=8), z = n_stream. 128 threads.
- split_k from root policy: target_workgroups = cores*2; split_k = clamp(ceil(target/workgroups), 1, record_groups=ceil(n_kv/128)).
- gqa_group_size = min(gqa,4) = 4 for qwen35 (gqa=6) → gqa_groups=2; group0: 4 heads, group1: 2 heads. query_head_first = logical_head*gqa + gqa_group*gqa_group_size.
- Per workgroup: query = wg.x / split_k; split_index = wg.x % split_k; out_stream = wg.z; logical_head = wg.y / gqa_groups; gqa_group = wg.y % gqa_groups.
- Tail activation: lane0 searches query_order for query_id = out_stream*n_query + query; sh_active = packed/query_order_ne0 (row); if tail present and not found → return.
- compute_live of K and V (shared indices flag).
- accumulators per (gi × slice): register [GQA_GROUP_MAX*4=16] floats. maximum/denominator per gi in shared memory, processed serially by lane0 (not per-lane!) — scores are computed by subgroup reduce then lane0 does online softmax update in shared memory, broadcast old_scale/weight.
- Score: per-lane partial = Σ_slices q[base+dim]*k[slice]; subgroupAdd; sh_reduce[subgroup]; lane0 sums gl_NumSubgroups entries sequentially. Order: subgroup partial sums then sequential — note this differs from CUDA portable version (block tree reduction over 128 lanes: reduction[tid] then tree halving with THREADS/2=64→1). Reduction order differs → floating-point non-associativity → scores slightly different. Flag as parity risk (tolerable? need to pick one for bit-parity; RKVA parity harness probably compares to CUDA with tolerance).
  Actually with wave32 and 128 threads = 4 subgroups: subgroupAdd within wave32 then 4 sequential adds — same shape as CUDA wave kernel? CUDA portable: tree: stride 64,32,16,8,4,2,1. Different order. Flag.
- mask: slopes[gi] * load_mask(token, query, out_stream); mask F16 buffer indexed by half_index = token + query*n_kv + (stream%mask_ne3)*n_kv*n_query. (For causal, mask is -inf/0 in F16.)
- score = total*scale; softcap: score = cap*tanh(score) (if scale is pre-divided by cap); score += mask; if mask==-inf: old_scale=1, weight=0; else: next_max=max(m,score); old_scale = (m < -3e38 ? 0 : exp(m-next_max)); weight=exp(score-next_max); m=next_max; denom=denom*old_scale+weight.
- acc[gi*4+slice] = acc*old_scale + v*weight (per-lane register).
- Tail (exact precision tail): only in the final split (split_index+1==split_k): run_desc row = sh_active*run_desc_ne0; n_tail = max(desc[4],0); per tail token: slot=desc[6+t]; K/V from k_tail/v_tail buffers: half_index = slot*nb1 + head*nb2 + dim (in u16 elements); from_current if FLAG_TAIL_CURRENT and slot >= history_slots (read via device address k_tail_current_addr with slot-history_slots). bf16 flag → bits<<16. tail mask: half_index = t + query*tail_stride + stream*tail_stride*n_query; F16 always (tail mask). Same softmax merge.
- Body tokens: token range [body_tokens*split/split_k, body_tokens*(split+1)/split_k), body_tokens = bodyless? 0 : n_kv. (Vulkan has no body_packed; CUDA portable supports packed body list desc[6+tail_mask_ne0+packed] with n_body=desc[5] — flag as divergence/extension.)
- load_kvarn: resolve per-token group/pos (indirect index); from_stage/from_record determination (eager vs delayed vs swa); stage: F16 load at (stage_pos*n_heads+head)*64 + dim>>1 unpacked; record: record_words = bits*512+192; record_base = ((stream*groups_per_stream+ring)*n_heads+head)*record_words; payload index = row*128+col, row=value?pos:dim(lane), col=value?dim:pos; scale=axis0[row], zp=axis1[row], other=axis2[col]; out=(q*scale+zp)*other. head = logical_head*head_slices+slice.
- Q layout: q_bases[gi] = query*q_nb1 + query_head*q_nb2 + out_stream*head_dim*n_query*n_query_heads (u32 elements, F32). q_nb1/q_nb2 are row/head strides in element units; stream stride is contiguous head_dim*n_query*n_query_heads.
- Output (split_k==1): lane0 merges sinks if FLAG_SINKS (not for qwen35: has_sinks=0): sh_old_scale, weight = denom>0 ? 1/denom : 0; dst_base = query_head*head_dim + query*head_dim*n_query_heads + out_stream*head_dim*n_query_heads*n_query; data_dst[dst_base+dim] = acc*old_scale*weight (F32). Layout: [head_dim][n_query_heads][n_query][n_stream] F32 (dst ne0=head_dim... actually ggml dst is [D, n_head, n_query, n_stream]).
- split_k>1: write partial matrix + (denominator, maximum) metadata into split-K workspace; then generic flash_attn_split_k_reduce: per-row (query_head=n? mapping: workgroup.x=n → row index over ne1=heads? In reduce shader: n=wg.x indexes head dim? Actually reduce_pc = {q->ne[0]=D, q->ne[2]=heads, q->ne[1]=n_query, q->ne[3]=streams, split_k, sinks}. Grid (heads, D/32, n_query*streams). In shader: p.ne1=heads... wait pc.D=q->ne[0], ne1=q->ne[2] (heads), ne2=q->ne[1] (n_query), ne3=q->ne[3]. wg.x = n → row over ne1 (heads); i2=wg.z%ne2 (query), i3=wg.z/ne2 (stream). o_offset = D*ne1*(k + k_num*(i2+ne2*i3)) + D*n + d. So partial layout: [split][stream][query][head][D]. lm region is matrix_values = D*n_query*heads*streams*split_k, lm_base = matrix_values + heads*2*(split + split_k*(query + n_query*stream)); data_dst[lm_base + head] = denominator; [+heads + head] = maximum. Matches shader write: matrix_base = head_dim*n_query_heads*(split_index + split_k*(query + n_query*out_stream)) + head_dim*query_head.
- Reduce formula: m_max = max over splits of m (regardless of denom); L = Σ exp(m_s - m_max)*l_s; sink merge (if any); O = Σ_s exp(m_s-m_max)*partial_s; if sink > m_max, O *= ms; O *= 1/L; clamp to ±FLT_MAX. Note: differs from CUDA portable combine which skips splits with denom<=0 when computing max (meta.y>0). Does Vulkan include m from empty splits (m=-FLT_MAX→ exp(-inf)=0, harmless; but if split is empty then m=-3.4e38 with l=0 → weight=exp(m-m_max)*0=0 no problem). Minor: CUDA combine m only over denom>0 splits; Vulkan over all. If a split has m=-FLT_MAX and denom=0, both give weight 0. Except in the case where all are empty and m_max=-FLT_MAX (then L=0 → output 0). OK.

Decode vs prefill (CUDA): decode has dedicated routes (fattn-mma-kvarn-decode / vec kernel, specialized up to Q≤16); portable kernel n_splits=1 for D≥128 (only D=64 is split with 256-token granularity). Vulkan: single shader, split_k from occupancy policy. In RKVA server: decode is n_tokens=1: n_query=1 → split_k = min(ceil(cores*2/(1*8*1)), record_groups) — on Series X (52 CU) target=104 → split_k = 13 clamped by record_groups.

## Domain Semantics
- ROTATED (1): Q rotated by WHT (host or server), K/V stored rotated, attention entirely in rotated domain, output rotated; graph applies WHT to output again (WHT self-inverse: H symmetric, H·H=I). Since dot products are invariant under orthogonal transform, scores equal original domain.
- ORIGINAL (2): kernel reconstructs K and V in original domain before attention (inverse WHT on dequantized rows, MMA loader or portable v_original_domain path — portable only V; MMA loader inverse-transforms K or V depending on type_K/type_V ORIGINAL_TYPE); Q original; output original.
- ROTATED_K_ORIGINAL_V (3): K is rotated (Q rotated), V inverse-transformed to original inside kernel → output directly original domain (V side only: out = W^{-1}(Σ p W v)?? no wait: scores in rotated domain: q_rot·k_rot = q·k. Output accumulator is Σ p * v_original → already original domain. But wait: kernel loads v from record (rotation storage), inverse WHT per row → v_orig. Then output original. Q rotated. So no output WHT needed in graph. This is used for prefill when native_original_v is supported.
- AUTO (0): backend chooses; CUDA mapping: decode (n_query==1) → rotated; prefill (n_query>1) → original (both K and V original, Q not rotated; kernel dequantizes and inverse-rotates).
- qwen35 local plan: decode (n_query_tokens ≤ native_rotated_max_query_tokens = 16 in CUDA) → ROTATED native; larger prefill → if backend supports native_original_v → ROTATED_K_ORIGINAL_V, else materialize fallback (generic fattn over materialized F16 with domain ROTATED for Q/output rotation... actually when !native_attention the plan returns {false, ROTATED}: graph rotates Q and output, materializes K/V to emit ROTATED F16).
- RKVA server with domain=AUTO: "server mirrors KVarN rotated domain plan" — i.e. server: rotates Q, K, V (WHT), stores rotated, computes attention in rotated domain, inverse WHTs output (one WHT) before returning. Output is same as local rotated path.

For RKVA: host sends Q/K/V F32 post-MRoPE pre-WHT + absolute position i32. Server does: WHT(Q) (needed for rotated attention), WHT+quantize+stage K/V, attention with causal mask (server-generated: mask value = 0 if kv_pos<=q_pos, else -inf), no sinks, scale=kq_scale=0.0625, max_bias=0, softcap=0, exact tail merge if tail set up, output WHT → original domain, F32 out.

Causal mask detail: local path uses pre-built kq_mask F16; server constructs equivalently from position: for query with absolute position p_q and kv token position p_k, mask = (p_k <= p_q) ? 0 : -INF.

## Precision Tail (server)
On local host path, exact tail = k_tail/v_tail F16 tensors with most recent tail_tokens rows (original domain, then WHT-rotated by graph if rotated domain), tail mask (F16, causal within tail), query_order/run_desc packing. Server needs to replicate: keep exact F16 (or BF16 per hello.tail_type) of last tail_tokens KV rows in rotated domain (graph applies wht to k_tail always; to v_tail only for rotated domain), merge after body loop with per-query causal mask; process only in final split. run_desc fields: [4]=n_tail, [5]=n_body, [6..6+n_tail)=tail slot. tail mask: [n_tail][n_query][stream] F16, stride tail_stride=ne0=n_tail.

Also interaction with stage: group-0 sink stays in stage; last tail_groups=2 groups in stage slots {1,2} cyclically; delayed flush quantizes group g-tail_groups when group g opens. Eager mode quantizes on group completion.

## Numerical Parity Risks on RDNA2 wave32
- No FP64: CPU reference uses double for std (llama_kvarn_sample_std), GPU uses F32 — server needs to match GPU's F32 (canonical for parity with local GPU path). DFlash precedent: mad() instead of FP64 emulation.
- exp/log: ggml shaders use GLSL exp/log (mapping to hardware ex2/lg2 approx?). Actually GLSL exp is precise, not ex2 approximation — driver dependent. HLSL's exp()/log() compile to ex2/lg2-based sequences with similar range reduction. CUDA uses expf/logf (SFU-based __expf? no — expf is ~2 ulp software implementation). Slight ulp divergence unavoidable; softmax exp differences are tolerable, but Sinkhorn best-imbalance comparisons (<=) can flip on ulp differences → different best scales → different 4-bit payload → visible diff in records. This is the main bit-exactness risk: quantization decisions are discrete. Mitigation: replicate exact operation order (per-lane sequential sum over 128 in std, same clamp/log/exp sequence). exp(log(x)) round-trip, clamp constants -0.3/10.0, 1e-3/1e3, 1e-8, 1e-10.
- round(): CUDA roundf is ties-away-from-zero; GLSL round is implementation-defined (AMD RADV rounds to even? Actually most implementations round-half-away like roundf? SPIR-V GLSL.std.450 Round: "round to nearest even, ties..."? RoundEven vs Round: GLSL round() maps to SPIR-V Round, which rounds half away from zero? No — SPIR-V Round: "round to nearest integer, ties away from zero"? Let me recall: SPIR-V Round = round to nearest, ties away from zero; RoundEven = ties to even. GLSL round() → Round (ties away). HLSL round() → ties to even! Divergence: HLSL port must not use round() — use floor(x+0.5) to match ties-away (for positive x; here (x-lo)/scale ≥ 0 always since lo=min so argument ≥0; floor(x+0.5) matches round-half-away-from-zero for non-negative). Important pitfall — include.
- packHalf2x16 / unpackHalf2x16: HLSL SM6.0 has f32tof16/f16tof32 — RNE, matches.
- F16 rounding on stage write: __float2half_rn = RNE = f32tof16.
- bf16 conversion: rounding bias 0x7fff + ((bits>>16)&1) then >>16 (RNE). No native bf16 in HLSL on SM6.0? SM6.6 has pack/unpack bf16; emulate with bit ops.
- subgroupAdd → WaveActiveSum on wave32; reduction order across 4 waves is sequential in lane0.
- tanh softcap: qwen35 doesn't use (logit_softcap=0), skip.
- exp for softmax: CUDA expf vs GLSL exp vs HLSL exp — ulp-level differences only affect F32 output, tolerable unless strict bit-parity required (impossible cross-vendor anyway).
- std Col/row: sum order sequential r=0..127 — preserve exactly.
- imbalance compare <= (ties adopt newer).
- Sinkhorn uses clamp(log_s, -0.3, 10.0) — note the lower bound -0.3 (odd but canonical).
- quantize lo/hi scan over cols 0..127 sequential min/max — deterministic.
- Scale computation: max((hi-lo)/qmax, 1e-10), qmax=15 float.
- Dequant in attention: (q*scale+zp)*other — F32 ops, order matters: multiply-add then multiply by other.
- Q·K dot: per-lane partial over slices: slice0's dim lane then slice1's dim lane: partial += q*k sequentially per slice (Vulkan: for slice 0..1 partial += q[base+dim]*k). Same order on wave32.
- No atomics in these kernels other than atomicAnd in validate (uint).

## Dependencies / Barriers (D3D12)
Per-layer server-side order:
1. WHT kernel? In RKVA design: the store shader does WHT internally (monolithic/workspace stage mode). WHT of Q is a separate dispatch (kvarn_wht over Q rows, head_width=256, f32).
2. Store K (dispatch), Store V — write stage+records. RAW hazards: attention reads stage/records → UAV barrier (D3D12: barrier with UAV access before compute, or use aliased barriers; in Vulkan terms memory barrier with storage write→read).
3. Materialize's live prepare → materialize (only for fallback/parity harness).
4. Attention reads Q (post WHT), records, stage, indices, mask, tail → writes dst.
5. If domain=rotated, output WHT over dst rows.
6. If split_k>1, split-K reduce.
In Vulkan backend ggml_vk_sync_buffers (memory barrier) between modes 1→2→3→4→5. In D3D12: UAV barriers between store dispatches; resource transitions are trivial since all buffers are UAV state.

Decode vs prefill differences:
- Decode (n_tokens=1): CUDA uses direct stage/flush kernels (parallel per head×token for stage; flush only for group boundary tokens), or specialized decode attention with split. Vulkan uses monolithic or workspace with active_streams... store monolithic loops n_tokens serially per head-block. For decode RKVA_ATTN_DECODE: n_tokens=1 → monolithic sufficient; attention: n_query=1 → large split_k.
- Prefill: workspace parallel store; attention processes n_query rows per stream, GQA batching 4+2.

Record arena sizing: groups_per_stream = ceil(capacity/128); records buffer bytes = n_stream*groups_per_stream*n_heads*record_bytes. For qwen35 on Xbox: n_heads=8 (4 KV × 2 slices), record_bytes=8960 → per layer per 128-token group: 8*8960 = 71680 B for K, same for V. 16 layers × ctx groups.

OK, let me write the final report. It must be exhaustive and structured per the 8 questions. Include absolute file paths. Flag CUDA/Vulkan divergences:
1. Reduction order of score (Vulkan subgroup + sequential vs CUDA tree)
2. CUDA portable supports packed body list (run_desc[5], body_packed) but Vulkan body always n_kv dense
3. CUDA MMA original-domain loader (prefill) vs Vulkan always rotated (Vulkan shader has no v_original_domain — verify: Vulkan flash attn always loads rotated; graph emits rotated and rotates Q and output; i.e. Vulkan only supports ROTATED domain natively — domain op param? Vulkan pipeline doesn't take domain. Right: Vulkan native = rotated only).
4. Round ties (CUDA roundf vs GLSL round vs HLSL round).
5. Double vs float in CPU reference.
6. Eager vs delayed records.
7. CUDA's decode-specialized kernel uses per-head desc array; portable kernel desc resolution logic identical.

Sinkhorn iteration count: 16 (from llama_kvarn_default_params; RKVA hello.sinkhorn_iters).

Let me also mention the KVAR_N_OP_PARAM index in ggml-cuda for store op params (not important for HLSL).

Stage slot assignment: host can pack explicit slot in upper 32 bits (slot+1). Server can implement stateless fallback: slot = group==0 ? 0 : 1+((group-1)%tail_groups).

Also note index high-word encoding: read_stage_slot: encoded==0 → fallback; else slot=encoded-1.

Tail geometry from hello: tail_tokens (cparams.kv_tail_tokens; 0 = server policy), tail_groups=2, tail_type f16/bf16.

Now let me structure the report. Long; structure into the 8 requested sections + a constants table + sources. Use Portuguese (user wrote in Portuguese).Li todas as fontes canônicas (shaders Vulkan completos, `kvarn.cu` (núcleos de store/quantize/WHT), `kvarn-wht.cu`, `llama-kvarn.cpp/.h`, `fattn-mma-kvarn.cuh`, `fattn-mma-kvarn-load.cuh`, `fattn-mma-kvarn-case-decl.cuh`, `fattn-mma-kvarn-impl.cuh`, `fattn-kvarn-portable.cuh`, `fattn-kvarn-route-policy.h`, `flash_attn_split_k_reduce.comp`, dispatch host em `ggml-vulkan.cpp`, `llama-graph.cpp`, `llama-context.cpp`, `models/qwen35.cpp`, `remote-attn-protocol.h`, `ggml-remote-attn.h`). Segue a especificação completa.

# Especificação de portabilidade KVarN → HLSL/D3D12 (SM 6.0, RDNA2 wave32)

## Fontes (caminhos absolutos)

- `/home/hjotha/beellama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/kvarn_store.comp` (892 linhas)
- `/home/hjotha/beellama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/kvarn_flash_attn.comp`
- `/home/hjotha/beellama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/kvarn_wht.comp`, `kvarn_wht_parallel.comp`, `kvarn_materialize.comp`, `flash_attn_split_k_reduce.comp`
- `/home/hjotha/beellama.cpp/ggml/src/ggml-cuda/kvarn.cu` (3356 linhas; `kvarn_quantize_tile` na linha 662 é a referência bit-exata do store)
- `/home/hjotha/beellama.cpp/ggml/src/ggml-cuda/kvarn-wht.cu`
- `/home/hjotha/beellama.cpp/ggml/src/ggml-cuda/fattn-mma-kvarn.cuh`, `fattn-mma-kvarn-load.cuh`, `fattn-mma-kvarn-case-decl.cuh`, `fattn-mma-kvarn-impl.cuh`, `fattn-kvarn-portable.cuh`
- `/home/hjotha/beellama.cpp/src/llama-kvarn.cpp` / `.h` (layouts, referência CPU, params)
- `/home/hjotha/beellama.cpp/src/llama-kv-cache-kvarn.cpp` (linhas 3463–3483: layouts de conversão por fatia; 3520–3560: quantize CPU por fatia)
- `/home/hjotha/beellama.cpp/src/llama-graph.cpp` (linhas 3944–3960, 4040–4180: aplicação de WHT por domínio)
- `/home/hjotha/beellama.cpp/src/llama-context.cpp` (linhas 1064–1131: geometria RKVA enviada ao Xbox)
- `/home/hjotha/beellama.cpp/ggml/src/ggml-remote-attn/remote-attn-protocol.h` (contrato RKVA v1)
- `/home/hjotha/beellama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` (linhas 10772–11200: dispatch WHT/store/materialize; 12280–12630: dispatch flash attn; 1600–1630: push constants)
- `/home/hjotha/beellama.cpp/ggml/src/ggml-vulkan/fattn-kvarn-route-policy.h` (política de split_k)

## Constantes globais (qwen35 + kvarn4, valores confirmados no código)

| Constante | Valor | Origem |
|---|---|---|
| `KVAR_N_GROUP` | 128 tokens | `llama-kvarn.h` |
| `record_dim` | 128 | `llama_kvarn_geometry_for(256)` → `{128, 128, 256, 2}` |
| `head_slices` | 2 (256 = 2×128) | idem |
| `key_bits`/`value_bits` | 4/4 → `qmax = (1<<4)-1 = 15` | tipo `kvarn_k4v4_g128` |
| `sinkhorn_iters` | **16** | `llama_kvarn_default_params()` (`llama-kvarn.cpp:160`); vai no HELLO RKVA |
| `sink_tokens` | 128 (grupo 0 nunca é quantizado) | validação exige exatamente 128 |
| WHT scale 128 | `0.08838834764831845f` (= 1/√128) | todos os shaders/kernels |
| mix cross-slice (2 slices) | `0.7071067811865475f` (= 1/√2) | `kvarn_wht_cross_slices` |
| WHT efetivo 256 | 1/√256 = **0.0625** exato (produto dos dois) | — |
| `kq_scale` | `1/sqrtf(256)` = 0.0625 (f_attention_scale==0 no qwen35) | `llama-context.cpp:1108`, `qwen35.cpp:397` |
| `max_bias`, `logit_softcap` | 0, 0 (qwen35) | — |
| `stage_groups` / `tail_groups` | 3 / 2 (RKVA hello fixa `tail_groups=2`; restrição `tail_groups < stage_groups` não-SWA) | `llama-context.cpp:1104`, `llama-kvarn.h` |
| n_heads físico KV | `n_head_kv * head_slices = 4*2 = 8` | — |
| GQA | 24 Q / 4 KV → gqa=6 | — |
| Clamps Sinkhorn | std ∈ [1e-3, 1e3]; log_s ∈ [-0.3, 10.0]; min denominador 1e-8; scale mín 1e-10 | `kvarn.cu:662+`, `kvarn_store.comp` |
| Sentinelas | `FLT_MAX = 3.402823466e+38`, "vazio" = `-3.402823466e38` (Vulkan) / `-FLT_MAX` (CUDA), teste `< -3.0e38` | — |

---

## 1. Record/tile layout

### 1a. Record GPU (o que o servidor Xbox precisa) — por (stream, grupo, cabeça física)

Definido implicitamente por `record_words = bits*512 + 192` (u32) em `kvarn_flash_attn.comp:load_kvarn` e `kvarn.cu` (`record_bytes = KVAR_N_TILE_VALUES*bits/8 + 3*128*sizeof(half)`). Para bits=4:

```
record_bytes = 128*128*4/8 + 3*128*2 = 8192 + 768 = 8960 bytes   (K e V iguais)
record_words (u32) = 4*512 + 192 = 2240
```

Layout interno de UM record (bytes, little-endian):

| Offset | Tamanho | Campo | Conteúdo |
|---|---|---|---|
| 0 | `bits*2048` = 8192 B | payload | 16384 valores de 4 bits, índice `row*128 + col`, bitstream LSB-first: valor `i` ocupa bits `[i*4, i*4+4)` do fluxo; em u32: `word = (i*4)>>5`, `bit = (i*4)&31` |
| 8192 | 256 B (128×u16) | eixo 0 "scale" | `f16(best_row[row] * scale[row])` (RNE) |
| 8448 | 256 B (128×u16) | eixo 1 "zp" | `f16(best_row[row] * lo[row])` |
| 8704 | 256 B (128×u16) | eixo 2 "other" | `f16(best_col[col])` |

Orientação (crítico):
- **K**: `row = dim` (0..127, dim dentro da fatia), `col = token` (posição dentro do grupo). Eixos scale/zp são por **dim**; "other" é por **token**. (Em `load_kvarn`: `row = value ? pos : dim; col = value ? dim : pos`.)
- **V**: `row = token`, `col = dim`. scale/zp por **token**; "other" por **dim**.

Descompressão (idêntica em Vulkan/CUDA, toda em F32 após conversão das f16):

```
value = (float(q) * f16_to_f32(scale[row]) + f16_to_f32(zp[row])) * f16_to_f32(other[col])
```

Ordem das operações é `fma-na-mão`: primeiro `q*scale + zp` (multiplicação e soma F32 distintas, sem fused), depois `* other`. Preserve exatamente.

Indexação do buffer de records (K e V são **buffers separados**):

```
record_base_bytes = ((stream * groups_per_stream + group) * n_heads + head) * record_bytes
head = logical_kv_head * head_slices + slice      // n_heads = 8 (físico), logical_kv_head ∈ [0,4), slice ∈ {0,1}
```

`groups_per_stream = ceil(capacity/128)`; `group = absolute_pos / 128`; o grupo 0 (sink, primeiros 128 tokens) **não tem record** — vive sempre no stage F16 (ver §2). `layer` não entra no offset: cada camada tem seus próprios buffers (records/stage/indices), como no host.

### 1b. `llama_kvarn_make_record_layout(record_dim, bits, value)` (llama-kvarn.cpp:~660)

```
rows = value ? 128 : record_dim;  cols = value ? record_dim : 128
payload_bytes = rows*cols*bits/8
scale_off = payload_bytes;  zp_off = scale_off + rows*2;  other_off = zp_off + rows*2
record_bytes = other_off + cols*2      // = bits*2048+768 para record_dim=128, bits=4
```

Confirma o layout da §1a (scale→zp→other, nessa ordem fixa).

### 1c. `llama_kvarn_make_layout(head_dim, group, key_bits, value_bits)` — tile combinado K+V (formato de estado/CPU, NÃO é o formato GPU)

```
off=0: k_payload (packed_bytes(head_dim*group, kb))
       k_s_col   (head_dim × u16)   ← eixo scale do K (por dim, absorve best_row)
       k_zp      (head_dim × u16)
       k_s_row   (group × u16)      ← eixo "other" do K (por token)
       v_payload (packed_bytes(group*head_dim, vb))
       v_s_col   (head_dim × u16)   ← "other" do V (por dim)
       v_s_row   (group × u16)      ← scale do V (por token)
       v_zp      (group × u16)
tile_bytes = align_up(off, 8)
```

Para head_dim=256, bits 4/4: k_payload=v_payload=16384 B; eixos 512+512+256 (K) e 512+256+256 (V); total 35072 B alinhado a 8 = 35072. **Atenção**: o path de conversão/estado (`llama-kv-cache-kvarn.cpp:3463`) usa na prática layouts por fatia de 128 (`kvarn_convert_k_layout`: payload 8192, scale@8192, zp@8448, other@8704, tile_bytes=8960 — idêntico ao record GPU) e chama `quantize_k_tile`/`quantize_v_tile` por fatia com tile transposto: K `tile[dim*128 + token]`, V `tile[token*128 + dim]`. `head_slices=2` não muda o record: cada fatia vira uma "cabeça física" independente com record próprio; a única interação entre fatias é o mix Hadamard cross-slice no WHT antes do stage (§3).

### 1d. Stage F16 (grupo incompleto + sinks + cauda exata)

```
stage_index(stage_pos, head, dim) = (stage_pos * n_heads + head) * 128 + dim     // u16/F16
stage_pos = stream * 128 * stage_groups + stage_slot * 128 + pos
```

Slot (sem atribuição explícita do host; fórmula stateless canônica, não-SWA):

```
stage_slot = (group == 0) ? 0 : 1 + ((group - 1) % tail_groups)      // tail_groups=2 → slots {1,2}
```

(SWA seria `group % stage_groups`; não se aplica ao qwen35.) O conteúdo do stage é **domínio rotacionado para K e V** (comentário literal em `kvarn.cu:1313` e `kvarn_store.comp:monolithic_store`: "Stage and records are rotated-domain for both K and V"), arredondado RNE para F16.

### 1e. Índices (células)

Buffer `indices` de i64 por token (no servidor: você o constrói a partir das posições absolutas recebidas):
- Célula normal: low32 = posição absoluta; high32 = 0 (fallback stateless) ou `slot+1` (slot explícito, 1-based para distinguir de 0).
- `encoded < -1`: "explicitamente staged" — payload = `-(encoded+2)` (nunca vem de record).
- `-1`: padding/inválido → contribuição 0.
- Decode no kernel (`read_cell` em `kvarn_flash_attn.comp`/`fattn-mma-kvarn.cuh:ggml_cuda_fattn_kvarn_read_cell`): `explicitly_staged = encoded < -1; payload = explicitly_staged ? -(encoded+2) : encoded; assigned_slot = (payload>>32)==0 ? -1 : (payload>>32)-1; cell = int32(payload)`.

Para o servidor RKVA com posições contíguas por sessão, `cell = pos` e `group = pos/128`, `pos_in_group = pos%128` bastam; o caminho `read_indirect` só é necessário se houver buracos/compactação (`llama_kvarn_compact_read_plan`).

---

## 2. KVARN_STORE / quantize (Sinkhorn + empacotamento)

### 2a. Estrutura do dispatch (Vulkan, `kvarn_store.comp`, 128 threads/workgroup)

Push constants: `n_heads` (físico, =8), `n_tokens`, `n_stream`, `groups_per_stream`, `record_words` (=2240), `bits`(=4), `iterations`(=16), `value`(0=K,1=V), `head_slices`(=2), `swa`(=0), `stage_groups`(=3), `tail_groups`(=2), `eager_records`, `tokens_per_stream`, `active_streams`, `flush_candidates`, `mode`.

Buffers: 0=current (F32 `[token][head][128]` pré-WHT), 1=indices (u32×2 por token = i64), 2=stage (u32 = f16x2), 3=records (u32), 4=workspace (u32 = f16x2), 5=workspace_valid (u32).

**Modo monolítico (MODE=0/5)** — grid `(n_heads/head_slices, 1, 1)` = (4,1,1); cada workgroup = 1 cabeça lógica (2 fatias), loop serial sobre tokens:

Para cada token t:
1. Ler célula → `group_global, pos, stream, group` (`stream = group_global / groups_per_stream`, `group = group_global - stream*groups_per_stream`; abortar se `stream >= n_stream || group >= groups_per_stream`).
2. **Flush tardio** (`eager_records==0`): se `pos == 0 && group > tail_groups` → quantizar grupo `group - tail_groups` a partir do stage slot `1 + ((flush_group-1) % tail_groups)` para `record_base = ((stream*groups_per_stream + flush_group)*n_heads + head)*record_words` (por fatia).
3. WHT do token corrente: por fatia, `values[slice] = wht_stage_value(current[(t*n_heads+head)*128 + lane], lane)` (butterfly §3, escala 1/√128); depois mix cross-slice: `a=values[0], b=values[1]; values[0]=(a+b)*0.7071067811865475; values[1]=(a-b)*0.7071067811865475`.
4. Escrever no stage: `stage_pos = stream*128*stage_groups + stage_slot*128 + pos`; lane<64: `data_stage[(stage_pos*n_heads+head)*64 + lane] = packHalf2x16(v[2*lane], v[2*lane+1])` (RNE).
5. **Eager** (`eager_records!=0`): se `pos == 127 && group > 0` → quantizar imediatamente o grupo corrente do stage.

**Modo workspace (paralelo, prefill denso)** — sequência de 5 dispatches com barreira de memória entre cada um (em D3D12: UAV barrier entre todos):

1. `MODE_VALIDATE` (grid 1,1,1): lane0 + atomicAnd validam, por stream ativa, que os índices são densos/contíguos (`idx[t] == first_idx + t`), high word zero (sem slots explícitos), e cabem no stream/grupo; resultado em `data_workspace_valid[0]`.
2. `MODE_STAGE` (grid `(n_heads/head_slices, n_tokens, 1)`): WHT (passos 3 acima) de cada token para o **workspace** F16 `[token][head][64 palavras]`.
3. `MODE_FLUSH` (grid `(n_heads, active_streams*flush_candidates, 1)`; `flush_candidates = ceil(tokens_per_stream/128) + stage_groups`): cada candidato enumera um `record_group`; **eager**: grupos que COMPLETAM dentro do store, ancorados em `start_local/128 + candidate`, selando só se `(rg+1)*128 <= end_local`; **tardio**: `ceil((start_local+127)/128) + candidate - tail_groups`, selando só se `(rg+tail_groups)*128 < end_local`. Lê a matriz do grupo via `workspace_matrix_value` (posição local ∈ [start_local,end_local) → workspace; senão → stage no slot `sh_workspace_stage_slot`, preferindo slot explícito do token de fronteira). Depois `quantize_workspace` (mesmo algoritmo de `quantize_stage`).
4. `MODE_COMMIT` (grid `(n_heads, active_streams*128*stage_groups, 1)`): para cada (slot, pos) procur