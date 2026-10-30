#!/usr/bin/env python3
"""Check that native admission uses its limit without changing legacy eviction."""
from pathlib import Path
import re

source = (Path(__file__).resolve().parents[1] / 'uwp/rkva/rkva-disk-snapshot.inc').read_text()
assert re.search(r'constexpr uint64_t store_limit\s*=\s*8ull\s*<<\s*30\s*;', source)
assert re.search(r'constexpr size_t retained_snapshots\s*=\s*4\s*;', source)
assert re.search(r'constexpr size_t native_retained_snapshots\s*=\s*8\s*;', source)
legacy = source[source.index('inline void make_room('):source.index('inline void check_native_room(')]
native = source[source.index('inline void check_native_room('):source.index('} // namespace rkva_snapshot')]
assert 'count < retained_snapshots' in legacy and 'count >= retained_snapshots' in legacy
assert 'native_retained_snapshots' not in legacy
assert re.search(r'admit_snapshot_store\(used,\s*requested,\s*store_limit,\s*uint32_t\(native_retained_snapshots\)', native)
# Cleanup is restricted to generated tmp-* paths; the quota decision itself
# has no remove operation or victim selection.
assert native.count('fs::remove_all(') == 1
assert native.index('name.compare(0, 4, "tmp-")') < native.index('fs::remove_all(') < native.index('if (!opaque_name(name)')
assert native.count('admit_snapshot_store(') == 1
print('native quota 8, legacy 4 and no published-victim eviction: PASS')
