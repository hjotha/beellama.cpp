# T4 task-668 evidence assessment — 2026-11-17

Located the owner's original reported summary in ChatGPT, not the originating
benchmark task or raw logs. The adjacent assistant messages are unavailable via
the history reader. Saved the six reported points separately as
`t4-task-668-reported.json`; deliberately NOT rows.json or measured calibration.

Search coverage: saved ChatGPT searches for task-668 and 13.46/T4; filename
searches in /home/hjotha (depth 4, then excluding caches) and /tmp (depth 3);
text inspection of root bench artifacts and benchmark-results/benchmarks/tier-bench;
exact task-668 search in beellama-agents, repo .agents/docs,
mtp-threshold-sweep-20260923 and beellama-benchmark-results. No matching raw task
artifact was located. This is a scoped negative result, not proof that logs do
not exist on Colab or another executor. Unrelated numerical substring hits are
not evidence for this sweep. No native chat database/session contents read.

Exploratory utility: the reported sweep shows N2 at 13.46 and N12 at 10.34 TPS,
a descriptive 23.18% decrease relative to N2. This motivates testing smaller N
and including N0; it does not isolate the effect of N because context and other
unrecorded controls may differ. It cannot select 2 tiers, establish an N2/N0
crossover, validate a GPU-transfer model or infer 4070 limits. Acceptance counts
are token aggregates, not cycle/position distributions. VRAM is a constraint,
not the performance objective, and snapshots do not establish safe headroom.

Use as a qualitative plausibility check and to prioritize controlled probes.
Only after recovering exact per-point metadata and repeated comparable traces
could it serve as an external quantitative validation set. Never silently train
on it and report the same points as held-out validation.

No GPU workload, service changes or production configuration changes performed.
