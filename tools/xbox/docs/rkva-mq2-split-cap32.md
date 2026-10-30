# Experimental KVarN4 MQ2 split cap

The default remains the existing MQ2/fine policy: target 256 remote rows and
at most 64 splits. The presence of `rkva-mq2-split-cap32.flag` in the app's
LocalState opts into the same target with at most 32 splits. The file contents
are ignored. It is read at HELLO, so changing it does not affect an existing
engine. The policy applies only when the F32 KVarN4 revision-3 MQ2/fine PSO is
actually selected (GQA6, tail128, nq 2..9). MQ4, decode, other formats, missing
MQ2 PSO, and a selected INT8 PSO retain their existing split policies. Enabling
INT8 but falling back to F32 due to a missing INT8 PSO can still select cap32.

The unchanged split buffer supports 256 splits and the reducer supports 256;
this experiment only reduces the limit. Split rows remain 8-aligned. Using
KVarN4 and F32 does not imply bit identity: changing the partition changes the
order of floating-point accumulation and reduction.

## Hardware harness procedure

Reserve the console and use the existing controlled harness; a new connection
and HELLO can reset an active engine. Before any change, capture the presence
and exact bytes of `rkva-mq2-split-cap32.flag` and `rkva-mq-i8.flag`, and the
other runtime flags, package/build hashes, model, prompt and batch settings.
Keep INT8 off in both arms. Run the baseline with the cap32 file absent and the
candidate with it present; establish a fresh HELLO after each change. In a
finally/cleanup path restore the prior file presence and exact bytes, including
on failure or interruption, and verify the restored state. Do not assume an
empty prior file or overwrite an existing flag permanently.

Use mode `fine` in both arms, with identical inputs and segment boundaries.
Collect the range profile: `kvarn4_multi_query.mq2.cap32` reports the requested
configuration, `mq2.available` and `mq2.calls` establish availability and
actual MQ2 selection, and `mq2_i8.calls` must remain zero. Inspect split counts
and query-tile samples, including bulk segments; configuration alone does not
prove execution. Counts depend on active rows and rounding: at 16384 rows the
default is 64 splits of 256 and cap32 is 32 splits of 512. At 34816 rows they
are 64 splits of 544 and 32 splits of 1088. Small spans may be identical.

Compare O/LSE, full logits and recurrent state with matching prefill segmentation
before timing; exercise seams at 128 rows and odd nq. Then measure request and
pass1 timings using the campaign's paired ABBA protocol, including roughly 1k,
8k, 28k and 56k remote rows. CPU oracle results are not console shader, quality,
or performance qualification.
