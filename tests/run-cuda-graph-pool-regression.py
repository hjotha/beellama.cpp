#!/usr/bin/env python3
"""Build a white-box legacy CUDA pool regression; GPU execution needs --execute.

Requires a Release CUDA build with CMAKE_EXPORT_COMPILE_COMMANDS=ON and
GGML_CUDA_NO_VMM=ON. Uses the real CUDA implementation without a public test API.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve(strict=True)
    cache = (build / 'CMakeCache.txt').read_text()
    if 'GGML_CUDA_NO_VMM:BOOL=ON' not in cache:
        parser.error('requires GGML_CUDA_NO_VMM=ON; this test exercises the legacy pool')
    if 'GGML_CUDA_GRAPHS:BOOL=OFF' in cache:
        parser.error('requires CUDA graphs enabled')
    entries = json.loads((build / 'compile_commands.json').read_text())
    entry = next(x for x in entries if x['file'].endswith('/ggml-cuda.cu'))
    command = entry.get('arguments') or shlex.split(entry['command'])
    command = list(command)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    obj, exe = output / 'test-cuda-graph-pool.o', output / 'test-cuda-graph-pool'
    command[command.index('-o') + 1] = str(obj)
    command[command.index('-c') + 1] = str(root / 'tests/test-cuda-graph-pool.cu')
    subprocess.run(command, cwd=entry['directory'], check=True)
    libraries = ['-lggml-cuda', '-lggml', '-lggml-base', '-lcublas', '-lcublasLt', '-lcudart', '-lcuda']
    if any(x.startswith('-DGGML_USE_NCCL') for x in command):
        libraries.append('-lnccl')
    subprocess.run([command[0], str(obj), '-o', str(exe), '-L' + str(build / 'bin'),
                    *libraries, '-Xlinker', '-rpath', '-Xlinker', str(build / 'bin')], check=True)
    if args.execute:
        subprocess.run([str(exe)], check=True)
        if args.memcheck:
            sanitizer = str(Path(command[0]).parent / 'compute-sanitizer')
            subprocess.run([sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', str(exe)], check=True)
    elif args.memcheck:
        parser.error('--memcheck requires --execute')


if __name__ == '__main__':
    main()
