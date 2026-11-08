#!/usr/bin/env python3
"""Fault-inject the actual CUDA producer's host orchestration with fake APIs.

No CUDA compiler, driver, GPU, or real CUDA runtime execution is involved.
Requires Python 3 and a GCC-compatible C++17 compiler (CXX, default g++).
ASan/UBSan are enabled by default; --no-sanitize is an explicitly weaker run.
LeakSanitizer is disabled: the CUDA-style process-lifetime scratch is retained,
and LSan cannot operate under some sandbox ptrace environments.

Usage: python tests/test-rkva-low4-cuda-lifecycle.py [--output-dir PATH]
Default artifacts live in a temporary directory and are removed after the run.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def extract_orchestration():
    source = (ROOT / "ggml/src/ggml-cuda/native-low4-producer.cu").read_text()
    source = re.sub(r"^\s*#\s*include .*$", "", source, flags=re.M)
    marker = "static __global__ void low4_native_prepare_kernel"
    if source.count(marker) != 1:
        raise RuntimeError("Expected exactly one CUDA device kernel definition")
    start = source.index(marker)
    end = source.index("\n#    endif", start)
    source = source[:start] + source[end:]
    source, count = re.subn(r"low4_native_prepare_kernel<<<.*?>>>\(.*?\);",
                            "fake_launch();", source, flags=re.S)
    if count != 2:
        raise RuntimeError(f"Expected two kernel launches, found {count}")
    # Integer fake streams make stale-handle reuse deterministic and auditable.
    marker = "cudaStream_t stream = nullptr"
    if source.count(marker) != 1:
        raise RuntimeError("Expected exactly one stream initializer")
    return source.replace(marker, "cudaStream_t stream = 0")


STUB = r'''
#undef NDEBUG
#include "ggml.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
using cudaError_t=int; using cudaStream_t=int; using cudaMemcpyKind=int;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3,cudaMemoryTypeDevice=1,cudaStreamNonBlocking=1;
struct cudaPointerAttributes { int type=0,device=0; };
int current=0, fail_set=-1, next_stream=1, last_error=0, launches=0,fail_launch=0,downloads=0,syncs=0,fail_free=0,fail_destroy=0;
std::set<void*> live; std::set<int> streams;
namespace ggml_cuda_graph_guard { void note_device_reservation(){} }
int cudaGetDevice(int* d){*d=current;return 0;}
int cudaSetDevice(int d){if(d==fail_set){fail_set=-1;return 1;}current=d;return 0;}
int cudaMalloc(void**p,size_t n){*p=std::malloc(n);live.insert(*p);return 0;}
int cudaFree(void*p){assert(live.erase(p)==1);std::free(p);int e=fail_free;fail_free=0;return e;}
int cudaStreamDestroy(int s){assert(streams.erase(s)==1);int e=fail_destroy;fail_destroy=0;return e;}
int cudaStreamCreateWithFlags(int*s,int){*s=next_stream++;streams.insert(*s);return 0;}
int cudaStreamSynchronize(int s){assert(streams.count(s));++syncs;return 0;}
int cudaPointerGetAttributes(cudaPointerAttributes*,const void*){last_error=1;return 1;}
int cudaGetLastError(){int e=last_error;last_error=0;return e;}
int cudaMemcpyAsync(void*d,const void*s,size_t n,int kind,int stream){assert(streams.count(stream));if(kind==2)++downloads;std::memcpy(d,s,n);return 0;}
int cudaMemsetAsync(void*d,int v,size_t n,int stream){assert(streams.count(stream));std::memset(d,v,n);return 0;}
void fake_launch(){++launches;if(launches==fail_launch)last_error=1;}
'''
MAIN = r'''
int main(){
 float k[512]={},v[512]={}; unsigned char pk[320]={},pv[320]={};
 auto call=[&]{return ggml_cuda_low4_native_prepare(GGML_TYPE_Q4_1,0,k,v,1,1,pk,pv);};
 assert(call()); assert(streams.size()==1&&live.size()==1);
 current=1;fail_set=1;assert(!call());assert(streams.empty()&&live.empty());
 // Restoration failed while current device is the old device: next call must
 // create fresh state, never reuse/double-free old stream/allocation handles.
 assert(current==0);assert(call());assert(streams.size()==1&&live.size()==1);
 current=1;fail_set=0;assert(!call());assert(streams.size()==1&&live.size()==1);
 assert(call());assert(current==1&&streams.size()==1&&live.size()==1);
 launches=downloads=syncs=0;fail_launch=1;assert(!call());assert(launches==1&&downloads==0&&syncs==1);
 launches=downloads=syncs=0;fail_launch=2;assert(!call());assert(launches==2&&downloads==0&&syncs==1);
 launches=downloads=syncs=0;fail_launch=0;assert(call());assert(launches==2&&downloads==3&&syncs==1);
 current=0;fail_free=1;assert(!call());assert(current==0&&streams.empty()&&live.empty());assert(call());
 current=1;fail_destroy=1;assert(!call());assert(current==1&&streams.empty()&&live.empty());assert(call());
 fail_free=1;assert(!ggml_cuda_low4_native_prepare(GGML_TYPE_Q4_1,0,k,v,2,1,pk,pv));assert(live.empty());
 assert(ggml_cuda_low4_native_prepare(GGML_TYPE_Q4_1,0,k,v,2,1,pk,pv));assert(live.size()==1);
 std::puts("PASS extracted CUDA host orchestration: device-switch restoration failure, retry, old-device failure, independent launch failures, free/destroy errors and scratch-grow recovery; fake APIs, no CUDA execution");
}
'''


def run(output, sanitize):
    source = extract_orchestration()
    marker = "scratch = scratch_t{};\n            const cudaError_t restore_status"
    if source.count(marker) != 1:
        raise RuntimeError("Expected exactly one reset immediately before device restoration")
    mutant = source.replace(marker, "/* removed reset */\n            const cudaError_t restore_status", 1)
    if mutant == source:
        raise RuntimeError("Lifecycle mutation was not applied")
    output.mkdir(parents=True, exist_ok=True)
    for variant, body in (("fixed", source), ("old-restoration-mutant", mutant)):
        path = output / f"lifecycle-{variant}.cpp"
        executable = output / f"lifecycle-{variant}"
        path.write_text(STUB + body + MAIN)
        flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if sanitize else []
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-g", *flags,
                        "-I" + str(ROOT / "ggml/include"), str(path), "-o", str(executable)], check=True)
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
        result = subprocess.run([str(executable)], cwd=output, env=env, capture_output=True, text=True)
        (output / f"lifecycle-{variant}.log").write_text(result.stdout + result.stderr)
        print(variant, "exit", result.returncode, result.stdout.strip())
        if variant == "fixed":
            if result.returncode != 0:
                raise RuntimeError("Fixed orchestration failed:\n" + result.stderr)
        elif result.returncode == 0 or "streams.count(stream)" not in result.stderr:
            raise RuntimeError("Mutant did not fail with the expected destroyed-stream assertion:\n" + result.stderr)
    print("PASS fixed lifecycle and negative mutation; " +
          ("ASan/UBSan enabled" if sanitize else "sanitizers explicitly disabled") + "; no CUDA execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Preserve generated C++ and logs here")
    parser.add_argument("--no-sanitize", action="store_true", help="Disable ASan/UBSan for this run")
    args = parser.parse_args()
    if args.output_dir is not None:
        run(args.output_dir.resolve(), not args.no_sanitize)
    else:
        with tempfile.TemporaryDirectory(prefix="rkva-cuda-lifecycle-") as temp:
            run(Path(temp), not args.no_sanitize)


if __name__ == "__main__":
    main()
