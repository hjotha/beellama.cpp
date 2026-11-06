uint row_kind(uint row, uint end, uint window, uint tail_begin) {
    if (row >= end) {
        return RKVA_KIND_NONE;
    }
    if (row >= tail_begin) {
        return RKVA_KIND_TAIL;
    }
    uint group = row / 128;
    uint live_group = window ? (window - 1) / 128 : 0;
    bool staged = origin / 128 + group == 0 || (group == live_group && (window & 127u) != 0);
    return staged ? RKVA_KIND_STAGE : RKVA_KIND_RECORD;
}

