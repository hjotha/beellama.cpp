"""Source contract for Windows-only profile plumbing; CPU bucket tests are C++.

In particular, dispatch selection alone must not pass if finish_bulk_profile
loses the rowlane identity when it constructs its separate attention Sample.
This check does not qualify MSVC or console execution.
"""
from pathlib import Path
import unittest


class RowlaneProfilePlumbing(unittest.TestCase):
    def test_bulk_sample_preserves_kernel_identity(self):
        src = (Path(__file__).resolve().parents[1] /
               'uwp/rkva/remote-attn-engine.cpp').read_text()
        finish = src.split('void RemoteAttnEngine::finish_bulk_profile(', 1)[1]
        finish = finish.split('\n}\n', 1)[0]
        self.assertIn('attn_sample.decode_rowlane = seg.decode_rowlane;', finish)
        self.assertIn('if (seg.decode_rowlane) ++c.kvarn4_decode_rowlane_calls;', finish)
        self.assertIn('seg.decode_rowlane = uint8_t(rowlane);', src)
        self.assertIn('sample.decode_rowlane = uint8_t(rowlane);', src)


if __name__ == '__main__':
    unittest.main()
