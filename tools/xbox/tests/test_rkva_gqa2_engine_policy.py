"""Offline source contracts for the Windows-only GQA2 engine policy.

The executable policy is covered by rkva_range_profile_cpu.cpp. This file keeps
the D3D12 engine on the same tested plan: PSO selection, DispatchX, group
reporting and the HELLO arm/conflict path must not drift back to ad-hoc code.
It does not compile or qualify UWP/MSVC.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / 'tools/xbox/uwp/rkva/remote-attn-engine.cpp'
REMOTE = ROOT / 'tools/xbox/uwp/rkva/remote-attn.cpp'


class Gqa2EnginePolicySourceTest(unittest.TestCase):
    def test_engine_uses_the_tested_plan_for_dispatch_and_groups(self):
        src = ENGINE.read_text()
        self.assertIn('rp::kvarn4_gqa2_plan(', src)
        self.assertIn('s->range_descriptor.layout_version', src)
        self.assertIn('rp::kvarn4_gqa2_groups(gqa2_plan, query_groups, split_k)', src)
        self.assertIn('cmd_->Dispatch(dispatch_x, query_groups, split_k);', src)
        self.assertIn('rp::count_attention(c, attn_sample);', src)

    def test_hello_reads_the_flag_and_applies_the_conflict_arm(self):
        src = REMOTE.read_text()
        self.assertIn('rkva-mq-gqa2.flag', src)
        self.assertIn('rkva_rprof::mq_gqa2_arm(', src)
        self.assertIn('set_mq_gqa2(', src)
        self.assertIn('MqGqa2Arm::kOn', src)
        self.assertIn('MqGqa2Arm::kConflict', src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
