import json
import os
from pathlib import Path
import struct
import sys
import time

policies = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*'),key=lambda p:int(p.name[6:]))
keys = ['scaling_governor','energy_performance_preference','scaling_min_freq','scaling_max_freq']
original = {str(p):{k:(p/k).read_text().strip() for k in keys} for p in policies}
fds = {}
nominal_mhz = {}

def snapshot():
    return {str(p):{k:(p/k).read_text().strip() for k in keys} for p in policies}

def restore():
    errors=[]
    for directory, settings in original.items():
        for key in keys:
            try:
                path=Path(directory)/key
                if path.read_text().strip()!=settings[key]:path.write_text(settings[key])
            except Exception as exc:errors.append(str(path)+': '+str(exc))
    if errors:raise RuntimeError('; '.join(errors))

def emit(value):
    print(json.dumps(value),flush=True)

try:
    for p in policies:
        cpu=int(p.name[6:])
        fds[cpu]=os.open(f'/dev/cpu/{cpu}/msr',os.O_RDONLY)
        nominal_mhz[cpu]=int(Path(f'/sys/devices/system/cpu/cpu{cpu}/acpi_cppc/nominal_freq').read_text())
    emit({'ready':True,'original':original,'nominal_mhz':nominal_mhz})
    for line in sys.stdin:
        try:
            req=json.loads(line)
            action=req['action']
            if action=='sample':
                counters={}
                for cpu,fd in fds.items():
                    aperf=struct.unpack('<Q',os.pread(fd,8,0xE8))[0]
                    mperf=struct.unpack('<Q',os.pread(fd,8,0xE7))[0]
                    counters[cpu]=[aperf,mperf]
                emit({'time':time.monotonic(),'counters':counters})
            elif action=='performance':
                for p in policies:
                    (p/'scaling_governor').write_text('performance')
                    (p/'energy_performance_preference').write_text('performance')
                state=snapshot()
                if any(v['scaling_governor']!='performance' or v['energy_performance_preference']!='performance' for v in state.values()):
                    raise RuntimeError('CPU performance policy readback mismatch')
                emit({'policy':state})
            elif action=='original':
                restore();emit({'policy':snapshot()})
            elif action=='close':
                restore();emit({'restored':snapshot()});break
            else:raise RuntimeError('Unknown action')
        except Exception as exc:
            emit({'error':str(exc)})
finally:
    restore()
    for fd in fds.values():os.close(fd)
