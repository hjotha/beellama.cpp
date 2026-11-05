#!/usr/bin/env python3
"""Build a private VMM A/B from one existing CMake CUDA build; never deploy.

--check reads configuration only. --build compiles and links; run it only when
GPU/CPU resources are available and the input build/source are not changing.
The input build is not the production release; pair results are experimental.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

BUILD = Path('/tmp/beellama-upstream-merge-cuda-20261105')
OBJDIR = BUILD / 'ggml/src/ggml-cuda'
CM = OBJDIR / 'CMakeFiles/ggml-cuda.dir'
SOURCE = Path('/home/hjotha/beellama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu')
STUBS = Path('/opt/cuda/targets/x86_64-linux/lib/stubs')
OUT = Path('/home/hjotha/.codex/artifacts/vmm-single-object-20261105/pair')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args, **kw):
    print('+', shlex.join(map(str, args)), flush=True)
    subprocess.run(args, check=True, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', action='store_true', help='compile one .cu and create private pair')
    args = p.parse_args()
    flags = (CM / 'flags.make').read_text()
    values = {}
    for line in flags.splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            values[key] = shlex.split(value)
    defs = values['CUDA_DEFINES']
    assert defs.count('-DGGML_CUDA_NO_VMM') == 1
    assert (STUBS / 'libcuda.so').exists()
    objects = shlex.split((CM / 'objects1.rsp').read_text())
    old_obj = 'CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o'
    assert objects.count(old_obj) == 1
    assert all((OBJDIR / o).is_file() for o in objects)
    link = shlex.split((CM / 'link.txt').read_text())
    assert link.count('@CMakeFiles/ggml-cuda.dir/objects1.rsp') == 1
    assert link.count('-o') == 1
    baseline = BUILD / 'bin/libggml-cuda.so.0.27.0'
    depfile = CM / 'ggml-cuda.cu.o.d'
    deps = [Path(x) if Path(x).is_absolute() else OBJDIR / x for x in
            shlex.split(depfile.read_text().replace('\\\n', ' '))[1:]]
    deps = [p.resolve() for p in deps]
    assert all(p.is_file() for p in deps)
    old_time = (OBJDIR / old_obj).stat().st_mtime_ns
    assert all(p.stat().st_mtime_ns <= old_time for p in deps), 'baseline object is stale'
    assert all((OBJDIR / o).stat().st_mtime_ns <= baseline.stat().st_mtime_ns for o in objects), \
        'baseline library is stale'
    production = Path('/home/hjotha/beellama-releases/upstream-v049-4260165-20261105/bin/libggml-cuda.so.0.27.0')
    print(json.dumps({'source': str(SOURCE), 'object_count': len(objects),
                      'build_cuda_sha256': digest(baseline), 'production_cuda_sha256': digest(production),
                      'note': 'Pair uses build/bin, not production; no GPU check performed.'}, indent=2))
    if not args.build:
        return
    OUT.mkdir(parents=False, exist_ok=False)
    tracked = list(dict.fromkeys([OBJDIR / o for o in objects] + deps + [SOURCE, depfile,
              CM / 'flags.make', CM / 'includes_CUDA.rsp', CM / 'link.txt',
              CM / 'objects1.rsp', CM / 'linkLibs.rsp', baseline]))
    before = {str(p): digest(p) for p in tracked}
    (OUT / 'inputs.json').write_text(json.dumps(before, indent=2) + '\n')
    # --reflink=auto keeps a private snapshot without hardlinks into the build.
    run(['cp', '-a', '--reflink=auto', str(BUILD / 'bin'), str(OUT / 'no-vmm')])
    run(['cp', '-a', '--reflink=auto', str(OUT / 'no-vmm'), str(OUT / 'vmm')])
    # Prove the exact objects and link command reproduce the control library.
    control = OUT / 'control-relinked.so'
    control_link = link.copy()
    control_link[control_link.index('-o') + 1] = str(control)
    run(control_link, cwd=OBJDIR)
    assert digest(control) == before[str(baseline)], 'objects cannot reproduce baseline; discard pair'
    obj = OUT / 'ggml-cuda-vmm.cu.o'
    nvcc = ['ccache', '/opt/cuda/bin/nvcc', '-forward-unknown-to-host-compiler']
    nvcc += [x for x in defs if x != '-DGGML_CUDA_NO_VMM']
    nvcc += values['CUDA_INCLUDES'] + values['CUDA_FLAGS']
    nvcc += ['-x', 'cu', '-c', str(SOURCE), '-o', str(obj)]
    run(nvcc, cwd=OBJDIR)
    response = OUT / 'objects-vmm.rsp'
    response.write_text(' '.join(shlex.quote(str(obj) if o == old_obj else o) for o in objects) + '\n')
    output = OUT / 'vmm/libggml-cuda.so.0.27.0'
    link[link.index('-o') + 1] = str(output)
    link[link.index('@CMakeFiles/ggml-cuda.dir/objects1.rsp')] = '@' + str(response)
    link += ['-L' + str(STUBS), '-lcuda']
    run(link, cwd=OBJDIR)
    dynamic = subprocess.check_output(['readelf', '-d', str(output)], text=True)
    assert 'libcuda.so.1' in dynamic, 'driver dependency missing'
    assert {str(p): digest(p) for p in tracked} == before, 'input changed during build; discard pair'
    no_vmm = OUT / 'no-vmm/libggml-cuda.so.0.27.0'
    assert digest(no_vmm) == before[str(baseline)]
    result = {'before_sha256': digest(no_vmm), 'after_sha256': digest(output),
              'source_sha256': digest(SOURCE), 'runtime_warning':
              'Do not add CUDA stubs to LD_LIBRARY_PATH; use the real NVIDIA driver. '
              'This pair is not the production release. No tests or deployment executed.'}
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
