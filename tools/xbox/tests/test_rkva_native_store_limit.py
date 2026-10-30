#!/usr/bin/env python3
"""Check native v2 uses guarded admission and leaves legacy policy alone."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1] / 'uwp/rkva'
source = (root / 'rkva-disk-snapshot.inc').read_text()
store = (root / 'rkva-native-store.h').read_text()
assert re.search(r'constexpr uint64_t store_limit\s*=\s*8ull\s*<<\s*30\s*;', source)
assert re.search(r'constexpr size_t retained_snapshots\s*=\s*4\s*;', source)
assert re.search(r'constexpr size_t native_retained_snapshots\s*=\s*8\s*;', source)
legacy = source[source.index('inline void make_room('):source.index('} // namespace rkva_snapshot')]
assert 'count < retained_snapshots' in legacy and 'count >= retained_snapshots' in legacy
native = source[source.index('const auto root = snap::fs::u8path(local_folder) / "rkva-snapshots-v2"'):source.index('// The legacy Q4/Q8/F16 format')]
assert native.index('rkva_native_store::WriteGuard writer(root)') < native.index('rkva_native_store::check_room(') < native.index('rkva_native_store::create_stage(temporary)')
assert 'remove_all(temporary' not in native
assert 'temporary' in store and '++used.snapshots' in store
assert 'remove_all(' not in store
print('native guarded quota and preserved staging, legacy policy: PASS')
