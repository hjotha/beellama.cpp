"""Offline numeric model of the experimental Q8 paired reducer.

The model does not execute HLSL; Xbox replay and output validation remain gates.
"""
from pathlib import Path
import shutil
import subprocess
import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which('g++') is None, reason='g++ required')
def test_paired_reducer_cpu_parity_and_invalid_propagation(tmp_path):
    binary = tmp_path / 'paired'
    subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-Wall', '-Wextra',
                    '-Werror', str(HERE / 'rkva_q8_paired_reduce_cpu.cpp'), '-o', str(binary)], check=True)
    out = subprocess.check_output([str(binary)], text=True)
    assert 'PASS: 192 CPU parity and invalid-propagation cases' in out
