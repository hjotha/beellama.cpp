"""Re-run six exact branch tests against reconstructed HLSL and current main base.

No pinned historical Git object and no generated GQA2 header is present locally.
Therefore this deliberately runs only Gqa2CompileContractTest, overrides its
base_source with the preserved main HLSL, and does NOT run GeneratedDxilTest.
"""
from pathlib import Path
from output_paths import RESULTS
import importlib.util,json,unittest
root=Path(__file__).resolve().parent
path=root/'.build/branch/tools/xbox/tests/test_rkva_mq_gqa2_layout.py'
spec=importlib.util.spec_from_file_location('branch_test',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.base_source=lambda:(root/'fixtures/main_mq.hlsl').read_text()
suite=unittest.defaultTestLoader.loadTestsFromTestCase(m.Gqa2CompileContractTest)
r=unittest.TextTestRunner(verbosity=2).run(suite)
(RESULTS/'gqa2_layout_results.json').write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'success':r.wasSuccessful(),'base':'preserved main snapshot, not pinned historical Git revision','not_tested':['GeneratedDxilTest','DXC compilation','engine integration','GPU']},indent=2)+'\n')
raise SystemExit(not r.wasSuccessful())
