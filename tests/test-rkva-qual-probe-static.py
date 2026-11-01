#!/usr/bin/env python3
"""Source-level guard checks for the test-only qualification probe.

These are not a substitute for the MSVC/UWP compile or a console run. They
pin the properties a default build must keep:

  * the autopilot op is reachable only under XLLAMA_QUAL_PROBE;
  * the UWP dispatcher branch and listener provider are inside the guard;
  * HELLO/supported_ops never advertise the probe or a managed capability;
  * the MSBuild property defaults to false.
"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
UWP = ROOT / 'tools/xbox/uwp'
PROBE = UWP / 'rkva/rkva-qual-probe.h'
REMOTE = UWP / 'rkva/remote-attn.cpp'
ENGINE = UWP / 'rkva/remote-attn-engine.cpp'
MAIN = UWP / 'MainPage.cpp'
AUTOPILOT = ROOT / 'tools/xbox/src/bridge/autopilot.cpp'
VCX = UWP / 'xllama.vcxproj'


def guarded_block(text, marker, start_hint):
    """Return the text between the #ifdef that precedes marker and its #endif."""
    position = text.find(marker, start_hint)
    if position < 0:
        return None
    before = max(text.rfind('#ifdef XLLAMA_QUAL_PROBE', 0, position),
                 text.rfind('#if defined(XLLAMA_QUAL_PROBE)', 0, position))
    if before < 0:
        return None
    after = text.find('#endif', position)
    if after < 0:
        return None
    # Reject a guard that closed before the marker.
    if text.find('#endif', before) < position:
        return None
    return text[position:after]


class GuardPlacementTests(unittest.TestCase):
    def test_probe_action_is_guarded(self):
        text = MAIN.read_text()
        block = guarded_block(text, 'else if (a.op == "rkva_qual_probe")', 0)
        self.assertIsNotNone(block, 'rkva_qual_probe branch is not inside its guard')
        self.assertIn('query_resident', block)
        self.assertIn('write_local_bytes', block)

    def test_include_and_helpers_are_guarded(self):
        text = MAIN.read_text()
        self.assertIsNotNone(guarded_block(text, '#include "rkva/rkva-qual-probe.h"', 0))
        self.assertIsNotNone(guarded_block(text, 'static rkva_qual_probe::build_info', 0))

    def test_listener_provider_is_guarded(self):
        text = REMOTE.read_text()
        self.assertIsNotNone(guarded_block(text, 'bool qual_provider(', 0))
        self.assertIsNotNone(
            guarded_block(text, 'rkva_qual_probe::set_provider(&qual_provider', 0))
        self.assertIsNotNone(
            guarded_block(text, 'rkva_qual_probe::set_provider(nullptr', 0))

    def test_engine_accessor_is_guarded(self):
        self.assertIsNotNone(
            guarded_block(ENGINE.read_text(), 'RemoteAttnEngine::qual_resident_probe', 0))

    def test_header_is_fully_guarded(self):
        text = PROBE.read_text()
        self.assertIn('#if defined(XLLAMA_QUAL_PROBE)', text)
        default_branch = text.split('#else // !XLLAMA_QUAL_PROBE', 1)
        self.assertEqual(len(default_branch), 2, 'missing default-off branch')
        self.assertIn('compiled = false', default_branch[1])
        self.assertNotIn('make_response', default_branch[1])


class NoAdvertisementTests(unittest.TestCase):
    def test_hello_mask_has_no_probe_or_managed_caps(self):
        text = REMOTE.read_text()
        start = text.index('constexpr uint32_t supported_ops_mask()')
        end = text.index('\n}\n', start)
        mask = text[start:end]
        for forbidden in ('QUAL', 'qual', 'rkva_qual_probe', 'REGISTRY_V1', 'AUTH_V2',
                          'FRAMING_V3', 'RKVA_MANAGED_AUTH_HELLO',
                          'RKVA_MANAGED_ATTEST'):
            self.assertNotIn(forbidden, mask)

    def test_autopilot_op_list_is_conditional(self):
        text = AUTOPILOT.read_text()
        self.assertNotIn('"rkva_qual_probe"', text.split('#ifdef XLLAMA_QUAL_PROBE')[0])
        self.assertIn('#ifdef XLLAMA_QUAL_PROBE', text)
        self.assertIn('if (op == "rkva_qual_probe")', text)

    def test_msbuild_default_is_off(self):
        text = VCX.read_text()
        self.assertIn("""<XllamaQualProbe Condition="'$(XllamaQualProbe)' == ''">false</XllamaQualProbe>""",
                      text)
        self.assertIn("""Condition="'$(XllamaQualProbe)' == 'true'">XLLAMA_QUAL_PROBE;""",
                      text)
        self.assertIn("""Condition="'$(XllamaQualProbe)' == 'true'">RKVA_MANAGED_SNAPSHOT_PHASE3;""",
                      text)


class ContractTests(unittest.TestCase):
    def test_probe_never_writes_a_staging_object(self):
        text = PROBE.read_text()
        handle = text[text.index('inline std::string make_fault_response'):]
        for forbidden in ('ofstream', 'fopen', 'remove(', 'rename(', 'create_directory'):
            self.assertNotIn(forbidden, handle)

    def test_response_fields_are_nullable_not_sentinels(self):
        text = PROBE.read_text()
        self.assertIn('set_null("session_id")', text)
        self.assertIn('set_null("host_key_sha256")', text)
        self.assertIn('set_null("generation")', text)
        self.assertIn('unavailable_fields', text)

    def test_no_generated_fixture_literals(self):
        for path in (PROBE,):
            text = path.read_text()
            self.assertNotIn('4443ba6', text)
            self.assertNotIn('"0000000000000000', text)


if __name__ == '__main__':
    unittest.main()
