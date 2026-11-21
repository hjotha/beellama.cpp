#!/usr/bin/env python3
# Mirror of moe_cache_slru::plan/move/touch/victim (src/llama-moe-cache.cpp) to check the
# invariant that the new gate relies on: with n_slots >= n_layers*n_expert, plan() never
# fails and no fill happens after the first pass over the group.
import random

class SLRU:
    def __init__(self, n_layer, n_expert, n_slots):
        self.n_expert, self.n_slots = n_expert, n_slots
        self.slot_map = [[-1]*n_expert for _ in range(n_layer)]
        self.key_of = [-1]*n_slots
        self.prev = [s-1 for s in range(n_slots)]
        self.next = [s+1 if s+1 < n_slots else -1 for s in range(n_slots)]
        self.head, self.tail = [0, -1], [n_slots-1, -1]
        self.is_prot = [False]*n_slots
        self.n_prot = 0
        self.seen = [0]*n_expert
        self.seen_gen = 0
        self.uniq = []

    def move(self, s, protect):
        frm = self.is_prot[s]
        if frm == protect and s == self.tail[frm]:
            return
        if self.prev[s] >= 0: self.next[self.prev[s]] = self.next[s]
        else:                 self.head[frm] = self.next[s]
        if self.next[s] >= 0: self.prev[self.next[s]] = self.prev[s]
        else:                 self.tail[frm] = self.prev[s]

        self.prev[s] = self.tail[protect]
        self.next[s] = -1
        if self.tail[protect] >= 0: self.next[self.tail[protect]] = s
        else:                       self.head[protect] = s
        self.tail[protect] = s
        self.is_prot[s] = protect
        self.n_prot += int(protect) - int(frm)

    def touch(self, s):
        self.move(s, True)
        if self.n_prot > self.n_slots//2:
            self.move(self.head[1], False)

    def victim(self, il):
        for seg in range(2):
            s = self.head[seg]
            while s >= 0:
                key = self.key_of[s]
                if key < 0 or key//self.n_expert != il or self.seen[key % self.n_expert] != self.seen_gen:
                    return s
                s = self.next[s]
        raise AssertionError("no SLRU victim")

    def plan(self, il, ids):
        self.seen_gen += 1
        self.uniq = []
        for e in ids:
            if self.seen[e] != self.seen_gen:
                self.seen[e] = self.seen_gen
                self.uniq.append(e)
        if len(self.uniq) > self.n_slots:
            return False
        slots = self.slot_map[il]
        n_hit = 0
        for e in self.uniq:
            if slots[e] >= 0:
                self.touch(slots[e]); n_hit += 1
        self.uniq.sort()
        fills = []
        for e in self.uniq:
            if slots[e] >= 0: continue
            s = self.victim(il)
            if self.key_of[s] >= 0:
                self.slot_map[self.key_of[s]//self.n_expert][self.key_of[s] % self.n_expert] = -1
            self.key_of[s] = il*self.n_expert + e
            slots[e] = s
            self.move(s, False)
            fills.append(e)
        return True, n_hit, fills


def ubatch_ids(n_tokens, n_expert, k, rng):
    return [rng.randrange(n_expert) for _ in range(n_tokens*k)]


def run(name, n_layers, n_expert, n_slots, decode_first, n_ub, ub):
    rng = random.Random(1234)
    slru = SLRU(n_layers, n_expert, n_slots)
    assert n_layers*n_expert <= n_slots, "gate would be false"
    per_ub = []
    if decode_first:
        for _ in range(decode_first):
            for il in range(n_layers):
                ok, hit, fills = slru.plan(il, ubatch_ids(1, n_expert, 8, rng))
                assert ok
    for u in range(n_ub):
        tot_fill = 0
        for il in range(n_layers):
            ok, hit, fills = slru.plan(il, ubatch_ids(ub, n_expert, 8, rng))
            assert ok, "plan failed"
            tot_fill += len(fills)
        per_ub.append(tot_fill)
    # decode after the prefill: every selected expert must be a hit
    d_fill = 0
    for _ in range(32):
        for il in range(n_layers):
            ok, hit, fills = slru.plan(il, ubatch_ids(1, n_expert, 8, rng))
            assert ok
            d_fill += len(fills)
    print(f"{name}: layers={n_layers} experts={n_expert} slots={n_slots} "
          f"prefill_fills_per_ubatch={per_ub[:4]}... total={sum(per_ub)} "
          f"decode_fills_after={d_fill}")
    return per_ub, d_fill


# target scenario: 2 host layers, cache 6 GiB > working set -> n_slots = 2*n_expert
for n_expert in (64, 128, 256):
    per_ub, d_fill = run("full-a", 2, n_expert, 2*n_expert, 0, 28, 1024)
    assert per_ub[0] <= 2*n_expert and all(f == 0 for f in per_ub[1:]), per_ub
    assert d_fill == 0

# decode warms the cache first, then the prefill
per_ub, d_fill = run("full-b", 2, 128, 256, 64, 28, 1024)
assert per_ub[0] <= 128 and all(f == 0 for f in per_ub[1:]), per_ub
assert d_fill == 0

# 8 host layers fully resident
per_ub, d_fill = run("full-8l", 8, 128, 8*128, 0, 28, 1024)
assert per_ub[0] <= 8*128 and all(f == 0 for f in per_ub[1:]), per_ub
assert d_fill == 0

# 1 host layer fully resident
per_ub, d_fill = run("full-1l", 1, 128, 128, 0, 28, 1024)
assert per_ub[0] <= 128 and all(f == 0 for f in per_ub[1:]), per_ub
print("OK: plan() never failed and no fill after the first pass")
