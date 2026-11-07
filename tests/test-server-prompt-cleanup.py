#!/usr/bin/env python3
"""Compile the production slot cleanup body against an injected memory failure."""
from pathlib import Path
import subprocess
import tempfile

source = (Path(__file__).resolve().parents[1] / 'tools/server/server-context.cpp').read_text()
start = source.index('    bool prompt_clear() {')
brace = source.index('{', start)
depth = 0
for end in range(brace, len(source)):
    if source[end] == '{':
        depth += 1
    elif source[end] == '}':
        depth -= 1
        if depth == 0:
            break
else:
    raise RuntimeError('unterminated prompt_clear')
method = source[start:end + 1]
program = r'''
#include <cassert>
#include <stdexcept>
#define SLT_TRC(...) ((void)0)
#define SRV_ERR(...) ((void)0)
struct memory {
    bool ok = true;
    int calls = 0;
    bool should_throw = false;
    bool seq_rm(int, int, int) { ++calls; if (should_throw) throw std::runtime_error("injected"); return ok; }
};
struct slot {
    memory mem;
    int id = 0;
    int forgotten = 0;
    bool prompt_cleanup_failed = false;
    void prompt_forget_memory() { ++forgotten; }
''' + method + r'''
};
int main() {
    slot healthy;
    assert(healthy.prompt_clear());
    assert(healthy.mem.calls == 1 && healthy.forgotten == 1 && !healthy.prompt_cleanup_failed);
    slot failed;
    failed.mem.ok = false;
    assert(!failed.prompt_clear());
    assert(failed.mem.calls == 1 && failed.forgotten == 0 && failed.prompt_cleanup_failed);
    failed.mem.ok = true;
    assert(!failed.prompt_clear());
    assert(failed.mem.calls == 1 && failed.forgotten == 0 && failed.prompt_cleanup_failed);
    slot exception;
    exception.mem.should_throw = true;
    assert(!exception.prompt_clear());
    assert(exception.mem.calls == 1 && exception.forgotten == 0 && exception.prompt_cleanup_failed);
}
'''
with tempfile.TemporaryDirectory(prefix='beellama-slot-cleanup-test-') as directory:
    cpp = Path(directory) / 'test.cpp'
    binary = Path(directory) / 'test'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Werror', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
print('PASS: failed clear preserves bookkeeping and quarantines subsequent cleanup')
