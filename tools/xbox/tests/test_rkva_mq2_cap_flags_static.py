#!/usr/bin/env python3
"""Check HELLO rejects conflicting flags before mutating a live session."""
from pathlib import Path

source = (Path(__file__).resolve().parents[1] / 'uwp/rkva/remote-attn.cpp').read_text()
hello = source.split('case RKVA_HELLO: {', 1)[1].split('rkva_hello_ack ack{};', 1)[0]
check = hello.index('if (!rkva_rprof::kvarn4_mq2_cap_flags_valid(mq2_cap32, mq2_cap128))')
failure = hello.index('return RKVA_ERR_STATE;', check)
assert hello.index('rkva-mq2-split-cap32.flag') < check
assert hello.index('rkva-mq2-split-cap128.flag') < check
assert failure < hello.index('state.sessions.clear();')
assert failure < hello.index('state.engine = std::make_unique<RemoteAttnEngine>();')
assert 'set_kvarn4_mq2_cap32(state.kvarn4_mq2_cap32)' in hello
assert 'set_kvarn4_mq2_cap128(state.kvarn4_mq2_cap128)' in hello
print('MQ2 conflicting flags fail HELLO before session replacement')
