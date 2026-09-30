"""Privileged, stdin-controlled CPU policy and APERF/MPERF helper.

Derived from mtp-dflash2-suffix-dense-20260929/cpu-controller.py. The runner
starts this helper through sudo; EOF, close and termination restore policies.
"""

import json
import os
from pathlib import Path
import signal
import struct
import sys
import time


KEYS = ('scaling_governor', 'energy_performance_preference',
        'scaling_min_freq', 'scaling_max_freq')
BENCHMARK = dict(zip(KEYS, ('powersave', 'power', '419175', '3301000')))


def snapshot(policies):
    return {str(p): {key: (p / key).read_text().strip() for key in KEYS}
            for p in policies}


def apply_settings(directory, settings):
    # Widen the current interval before moving either boundary across it.
    current_max = int((directory / 'scaling_max_freq').read_text())
    bounds = ('scaling_max_freq', 'scaling_min_freq') if (
        int(settings['scaling_min_freq']) > current_max
    ) else ('scaling_min_freq', 'scaling_max_freq')
    for key in ('scaling_governor', 'energy_performance_preference') + bounds:
        path = directory / key
        if path.read_text().strip() != settings[key]:
            path.write_text(settings[key])


def restore(original, policies):
    errors = []
    for directory, settings in original.items():
        try:
            apply_settings(Path(directory), settings)
        except Exception as exc:
            errors.append(f'{directory}: {exc}')
    actual = snapshot(policies)
    if actual != original:
        errors.append('CPU restore readback mismatch')
    if errors:
        raise RuntimeError('; '.join(errors))
    return actual


def emit(value):
    print(json.dumps(value), flush=True)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    policies = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*'),
                      key=lambda p: int(p.name[6:]))
    if not policies:
        raise RuntimeError('No cpufreq policies available')
    original = snapshot(policies)
    fds = {}
    nominal_mhz = {}
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for policy in policies:
            for cpu_text in (policy / 'affected_cpus').read_text().split():
                cpu = int(cpu_text)
                if cpu in fds:
                    continue
                fds[cpu] = os.open(f'/dev/cpu/{cpu}/msr', os.O_RDONLY)
                nominal_mhz[cpu] = int(Path(
                    f'/sys/devices/system/cpu/cpu{cpu}/acpi_cppc/nominal_freq'
                ).read_text())
        if not fds or any(value <= 0 for value in nominal_mhz.values()):
            raise RuntimeError('No valid APERF/MPERF reference frequencies')
        emit({'ready': True, 'original': original, 'nominal_mhz': nominal_mhz})
        for line in sys.stdin:
            try:
                action = json.loads(line)['action']
                if action == 'sample':
                    counters = {
                        cpu: [struct.unpack('<Q', os.pread(fd, 8, 0xE8))[0],
                              struct.unpack('<Q', os.pread(fd, 8, 0xE7))[0]]
                        for cpu, fd in fds.items()
                    }
                    emit({'time': time.monotonic(), 'counters': counters})
                elif action == 'benchmark':
                    for policy in policies:
                        apply_settings(policy, BENCHMARK)
                    actual = snapshot(policies)
                    if any(value != BENCHMARK for value in actual.values()):
                        raise RuntimeError('CPU benchmark policy readback mismatch')
                    emit({'policy': actual})
                elif action == 'original':
                    emit({'policy': restore(original, policies)})
                elif action == 'close':
                    emit({'restored': restore(original, policies)})
                    break
                else:
                    raise RuntimeError('Unknown action: ' + str(action))
            except Exception as exc:
                emit({'error': str(exc)})
    finally:
        try:
            restore(original, policies)
        finally:
            for fd in fds.values():
                os.close(fd)


if __name__ == '__main__':
    main()
