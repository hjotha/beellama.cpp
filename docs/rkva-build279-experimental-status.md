# Build 279 experimental RKVA work: qualification status

This feature branch saves the real serving wavefront, optional mapped CUDA codec, and independent-layer RPC cohorts. These experiments are not approved for production integration. The qualified Build 278 policy remains GPU FP8 for prefill and CPU FP8 for small decode queries.

## Matched measurements

Five alternating A/B pairs followed a complete excluded A/B warmup. Both arms used Xbox test package 279, identical model/prompt/snapshot/seed, P=96256, C=131072, b=ub=256, FP8 wire and compact stage. Profiling, host tracing and CUPTI were off. Each completion generated 64 tokens; decode covers 63 inter-emission intervals and excludes the first token.

| Context | Reference late prefill t/s | Mapped late prefill t/s | Reference decode t/s | Mapped decode t/s | Median paired decode change |
| ------- | -------------------------: | ----------------------: | -------------------: | ----------------: | --------------------------: |
| 102400  |                   139.6364 |                139.6364 |              19.4378 |           17.6420 |                   -10.5770% |
| 112640  |                   108.9877 |                108.4746 |              18.2016 |           15.2852 |                   -16.7688% |

Changes are medians of paired percentages, not ratios of arm medians. The 102400 mapped observations ranged 16.8031–19.3540 t/s; its last two pairs nearly tied the reference. The 112640 mapped arm lost every pair. n=5 bootstrap intervals are coarse and temporal drift remains possible. A UI pipe interruption was preserved as a failed attempt; complete pairs 1/2 were reused and whole pairs 3–5 resumed under new IDs, without source/control changes.

## Implementation and open gates

- Qwen3.5 serving now has an opt-in tile DAG for 64/128 tokens, explicit recurrent/KV dependencies and graph reuse. Eligibility is deliberately restricted to a single contiguous sequence wholly in overflow with n_rs_seq=0. Default behavior is unchanged.
- The mapped small-query codec uses two GPU kernels, pinned buffers and explicit retirement. Across all ten measured candidate points its counters record 1008 small packs and 1008 small unpacks. Two complete returned-logprob distributions of 248320 IDs per context matched the Build 278reference exactly; this is not a full-model/MTP quality proof.
- The WF128 full-model exact-logprob gate failed at both contexts in R5c. R5e fixed mapping/lifetime/error handling and completed one-off screens, but did not close that quality gate. No approximate tolerance was introduced after seeing the differences.
- Op23/cohort capability bit 14 combines up to four already-ready independent layer entries. The server pre-admits all entries, keeps the existing per-entry fence and poisons uncertain execution failures. It cannot batch dependent future Q values from the same tile.
- Per 256-token parent and 16 remote stages: baseline 16 tasks / 16 envelopes; WF128/B1=32/32; WF128/B4=32/17; WF64/B4=64/19. The requested literal 16-to-4 transactions was not achieved.
- WF128/B1late-prefill screens were 175.0760/129.7297 t/s; WF64/B4was 101.1414/84.0263 t/s at102400/112640. These are single experimental runs with an open quality gate, not qualified gains.

## Verification and evidence

The measured R5e serving sources passed 18 CTest cases, helper/range ComputeSanitizer with zero errors, rebuilt range ASan/UBSan and TSan, and strict source/bundle identity gates. Xbox test package 279was built on HPBEATS and installed in place without uninstalling LocalState. Legacy and cohort physical protocol/numeric checks passed. MTP, full recurrent rollback and physical fatal-device failure are not qualified. Four pre-existing coherence evidence-link groups still fail in both baseline and candidate.

A full 63-forward diagnostic with the original CUPTI collector passed attribution/timestamp/provider checks for both arms. It observed 1008 pack and 1008 unpack kernels, plus 2016 extra host CUDA subgraph evaluations. Its observer also recorded memory-pressure decisions, so these timings do not establish the complete cause of the uninstrumented regression. The earlier reduced-buffer collector was rejected for invalid timestamps even with zero reported drops.

Operator artifacts are archived outside Git at `/home/hjotha/build/rkva-next-20261021/`. They include `reports/mapped-r5e-vs-278-final-paired-resumed.json`, `reports/delivery-r5e.md`, `reports/hardware-matrix-r5e/actual-r1/`, `decode63-a16-diagnostic-r1/post-data-comparison-r1.json`, and the frozen source/package manifests. These are local artifact paths, not portable repository evidence links.

Frozen measured host manifest SHA256: `69b024dbe4bd014662e09436d66d53cc126f439b8d5d8caf7c07c7b540ec0f21`. Measured dirty patch SHA256: `0164c993de679f02b5d611177545e97e61480a300a682d414551208b2602e36b`. Xbox test MSIX SHA256: `50e92a74fe767607c208561070204397a2e07fc9179a2d5989bcc3d4bab63684`.
