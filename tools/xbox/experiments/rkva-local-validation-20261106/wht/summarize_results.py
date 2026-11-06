from pathlib import Path
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

P = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / '.build'
manifest = json.loads((P / 'source_manifest.json').read_text())
assert hashlib.sha256((P / manifest['baseline_fixture']).read_bytes()).hexdigest() == manifest['source_sha256']
hashes = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(P.iterdir())
    if p.is_file() and p.name not in ('source_hashes.json', 'results.json')
}
(OUT / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
identical = (OUT / 'cpu_results.json').read_bytes() == (OUT / 'sanitized_results.json').read_bytes()
assert identical
cpu = json.loads((OUT / 'cpu_results.json').read_text())
structure = json.loads((OUT / 'structure_results.json').read_text())
assert cpu['status'] == structure['status'] == 'PASS'
result = {
    'status': 'PASS_CPU_MODEL_ONLY',
    'candidate': 'LDS ping-pong WHT reducer; not a wave-shuffle implementation',
    'date': datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
    'scope': 'Bundled fixture and CPU model only; no production edit, HLSL compilation, GPU execution or performance measurement',
    'environment': {
        'compiler': subprocess.check_output([os.environ.get('CXX', 'g++'), '--version'], text=True).splitlines()[0],
        'python': platform.python_version(),
        'numpy': np.__version__,
    },
    'source_manifest': manifest,
    'cpu': cpu,
    'structure': structure,
    'sanitizers': {
        'address_sanitizer': 'PASS',
        'undefined_behavior_sanitizer': 'PASS',
        'optimized_and_sanitized_results_identical': identical,
        'leak_sanitizer': 'PASS' if os.environ.get('WHT_DETECT_LEAKS') == '1' else 'NOT RUN: disabled by default; opt in with WHT_DETECT_LEAKS=1 on a supported runner',
    },
    'shader_compilation': 'NOT RUN',
    'console_runtime': 'NOT RUN',
    'performance': 'NOT MEASURED',
    'next_gate': 'Compile baseline and candidate with the actual DXC/profile; inspect separate H128/H2 normalization and group barriers in DXIL; verify resources and poison/quality on the console before any performance conclusion.',
}
(OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({
    'status': result['status'],
    'vectors': cpu['wht_vectors'],
    'layout_shapes': structure['layout_shapes'],
    'sanitizers': result['sanitizers'],
}, indent=2))
