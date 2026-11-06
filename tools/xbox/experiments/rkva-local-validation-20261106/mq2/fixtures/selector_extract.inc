    uint split_begin = split * split_rows;
    uint split_end = split_begin + split_rows;
    uint record_lo = origin == 0 ? 128 : 0;
    uint window[RKVA_MQ_QT], end_q[RKVA_MQ_QT], body_end_q[RKVA_MQ_QT], tail_begin[RKVA_MQ_QT];
    uint head0[RKVA_MQ_QT];
    bool act[RKVA_MQ_QT];
    uint shared_hi = 0xffffffffu;
    uint body_end_max = split_begin;
    bool any_active = false;
    for (uint j = 0; j < RKVA_MQ_QT; ++j) {
        uint qi = q_first + j;
        uint qc = min(qi, n_query - 1); // phantom queries of a partial tile stay in bounds
        int query_pos = asint(positions.Load(qc * 4));
        uint causal_count = query_pos < int(origin) ? 0 : uint(query_pos) - origin + 1;
        uint limit = qi < n_query ? min(active_count, causal_count) : 0;
        window[j] = limit;
        tail_begin[j] = limit - min(limit, precision_tail_tokens);
        uint end = min(limit, split_end);
        act[j] = end > split_begin;
        end_q[j] = act[j] ? end : 0;
        body_end_q[j] = min(end_q[j], tail_begin[j]);
        head0[j] = qc * n_head + kv_head * RKVA_GQA RKVA_MQ_HEAD0_OFFSET;
        if (act[j]) {
            any_active = true;
            body_end_max = max(body_end_max, body_end_q[j]);
            shared_hi = min(shared_hi, min(min(end, limit & ~127u), tail_begin[j]));
        }
    }
    uint fast_lo = (max(split_begin, record_lo) + RKVA_TILE - 1) & ~(RKVA_TILE - 1);
    uint fast_hi = any_active ? shared_hi & ~(RKVA_TILE - 1) : 0;
    if (fast_hi <= fast_lo) {
        fast_lo = split_begin;
        fast_hi = split_begin;
    }

