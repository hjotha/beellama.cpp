"""Static ordering contracts; actual GPU behavior still requires console validation."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
ENGINE = (ROOT / 'tools/xbox/uwp/rkva/remote-attn-engine.cpp').read_text()
COMPACT = (ROOT / 'tools/xbox/uwp/rkva/rkva-kvarn-bulk-f16.inc').read_text()
GRAPH = (ROOT / 'src/llama-graph.cpp').read_text()


class CaptureSource(unittest.TestCase):
    def test_readback_order_and_no_extra_wait(self):
        body = ENGINE[ENGINE.index('        bool capture = false;'):]
        before, after = body.split('cmd_->Dispatch(dispatch_x, query_groups, split_k);', 1)
        self.assertIn('if (capture_armed_ && prefix', before)
        self.assertIn('CopyBufferRegion(capture_readback_', before)
        self.assertLess(after.index('uav_barrier(range_opt_partials_)'),
                        after.index('capture_extents_.partial'))
        self.assertLess(after.index('capture_extents_.partial'),
                        after.index('cmd_->Dispatch(geo_.n_head, n_tokens, 1)'))
        self.assertLess(after.index('uav_barrier(range_output_)'),
                        after.index('capture_extents_.output'))
        self.assertNotIn('submit(', body[:body.index('return finish(')])
        self.assertNotIn('WaitFor', body[:body.index('return finish(')])
        submit = ENGINE[ENGINE.index('bool RemoteAttnEngine::submit('):
                        ENGINE.index('bool RemoteAttnEngine::create_device(')]
        self.assertLess(submit.index('unconfirmed_submission_ = false;'),
                        submit.index('persist_mq_capture();'))
        self.assertIn('~CaptureSubmitGuard() { if (!complete) pending = false; }', submit)
        self.assertLess(submit.index('if (completed < value)'),
                        submit.index('capture_guard.complete = true;'))
        self.assertLess(submit.index('if (FAILED(hr)) {\n        terminal_failure_ = true;\n'
                                     '        return fail(error, "D3D12 device removed", hr);'),
                        submit.index('persist_mq_capture();'))
        self.assertIn('Buffer capture_readback_;',
                      (ROOT / 'tools/xbox/uwp/rkva/remote-attn-engine.h').read_text())
        self.assertIn('if (!confirm_execution_retired())', ENGINE)
        self.assertIn('capture_pending_ = false;', COMPACT)
        bulk = ENGINE[ENGINE.index('bool RemoteAttnEngine::attend_kvarn_bulk('):
                      ENGINE.index('bool RemoteAttnEngine::attend_native_rows(')]
        self.assertIn('capture_pending_ = false;', bulk)

    def test_gpu_unpack_and_pack_bracket_attention(self):
        self.assertLess(COMPACT.index('cmd_->Dispatch(unpack_groups, 1, 1)'),
                        COMPACT.index('ok = attend_range('))
        self.assertLess(COMPACT.index('ok = attend_range('),
                        COMPACT.index('cmd_->Dispatch(pack_groups, 1, 1)'))

    def test_remote_layer_is_model_full_attention_ordinal(self):
        self.assertIn('if (hparams.has_kv(l) && !hparams.is_recr(l)) ++remote_layer;', GRAPH)
        self.assertIn('inp->self_overflow_cell_idxs, positions, remote_layer,', GRAPH)


if __name__ == '__main__':
    unittest.main()
