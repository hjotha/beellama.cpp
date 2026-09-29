import json
from pathlib import Path
import tempfile
import unittest

from analyze_active import audit_group


class ActiveAudit(unittest.TestCase):
    def fixture(self, directory, mutation=None):
        def trace(event, **fields):
            return 'SHADOWTRACE ' + json.dumps(dict(event=event, request=0, slot=0, **fields))
        lines = [
            trace('begin', temperature=0),
            trace('token', pos=100, token=10, done_us=1),
            trace('draft', pos0=100, anchor=10, proposed=[11], start_us=2, done_us=3),
            trace('commit', pos0=100, anchor=10, confirmed=[11, 12], done_us=4),
            'SHADOWv2 job=1 epoch=3 request=0 pos0=100 anchor=10 c=2 d=7 ready=1 prefix_ok=1 position_ok=1 usable=1 usable_tokens=4 confirmed=[11,12] proposed=[11,12,13,14,15,16,17]',
            'SHADOWACTIVE select job=1 epoch=3 request=0 pos0=102 anchor=12 n=4',
            trace('draft', pos0=102, anchor=12, proposed=[13, 14, 15, 16], start_us=5, done_us=6),
            'SHADOWACTIVE accept job=1 request=0 pos0=102 proposed=4 accepted=1 confirmed=2',
            trace('commit', pos0=102, anchor=12, confirmed=[13, 99], done_us=7),
            trace('draft', pos0=104, anchor=99, proposed=[], start_us=8, done_us=9),
            trace('token', pos=105, token=20, done_us=10),
            trace('end', generated=6, truncated=False),
            'shadow active: selected=1 proposed=4 verified=1 accepted=1 cancelled=0 pending=0',
            'shadow auxiliary: errors=0',
        ]
        text = '\n'.join(lines) + '\n'
        if mutation:
            text = mutation(text)
        (directory / 'server-output.txt').write_text(text)
        (directory / 'plan.json').write_text(json.dumps(dict(mode='active', group='test', requests=[{}])))
        (directory / 'test.tokens.json').write_text(json.dumps([10, 11, 12, 13, 99, 20]))
        (directory / 'test.result.json').write_text(json.dumps(dict(request_index=0, scenario='test', case='curto192',
            repeat=0, warmup=False, token_count=6, config=dict(verifier_max=4, mtp_max=2, min_suffix=1))))

    def test_partial_acceptance_and_primary_resume(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            self.fixture(path)
            result = audit_group(path)[0]
            self.assertEqual(result['selected'], 1)
            self.assertEqual(result['accepted'], 1)
            self.assertEqual(result['partial'], 1)
            self.assertEqual(result['resumed_primary'], 1)

    def test_bad_handoffs_are_rejected(self):
        for old, new in [
            ('select job=1 epoch=3', 'select job=1 epoch=4'),
            ('ready=1', 'ready=0'),
            ('accepted=1 confirmed=2', 'accepted=2 confirmed=2'),
            ('pos0=102 anchor=12 n=4', 'pos0=102 anchor=90 n=4'),
            ('SHADOWACTIVE accept', 'MISSING accept'),
        ]:
            with self.subTest(old=old), tempfile.TemporaryDirectory() as name:
                path = Path(name)
                self.fixture(path, lambda text: text.replace(old, new))
                with self.assertRaises(AssertionError):
                    audit_group(path)


if __name__ == '__main__':
    unittest.main()
