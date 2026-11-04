#pragma once

// SET_ROWS op_params slot 0: checked SAW-INT4 stores used by the local
// exact-tail path. A value of 1 requires transactional validation and an
// error status instead of publishing rows from an invalid input batch.
enum { GGML_SET_ROWS_OP_PARAM_SAW_CHECKED = 0 };
