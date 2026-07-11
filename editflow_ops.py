"""
SAE-EF (Edit Flows) — pure alignment / process functions. No model code.

Adapts Havasi et al., "Edit Flows: Flow Matching with Edit Operations"
(arXiv:2506.09018) to the SAE-LEWIS editing regime (EDIT_FLOWS_PLAN.md):
X0 = x' (corrupted / source), X1 = x (clean / target), coupling given by
the corruption cache, alignment DETERMINISTIC from the min-edit matching
(difflib) instead of the paper's random alignments — natural for a
minimal-edit task where the path length equals the edit distance.

Data structures
---------------
slots : List[Tuple[Optional[int], Optional[int]]]
    The blank-augmented alignment (z0, z1) as one list of slots.
    slot = (a0, a1); None = blank (ε).
      (tok, tok)          aligned KEEP        — no op
      (tok, tok')         substitution        — sub fires tok→tok'
      (tok, None)         deletion            — del fires tok→ε
      (None, tok)         insertion           — ins fires ε→tok
    z0 = [a0 for slots if a0 is not None]  (== x0 exactly)
    z1 = [a1 for slots if a1 is not None]  (== x1 exactly)

ops : List[dict]
    One entry per non-KEEP slot: {"slot": k, "kind": KIND, "tgt": tok|None}

Forward process: each op fires independently by time t with probability
κ(t) = t³ (paper's default). z_t holds a1 at fired slots, a0 elsewhere;
x_t = tokens of z_t with blanks dropped.

Pending-op supervision (build_xt): each unfired op is mapped to a position
of x_t —
  sub/del : the x_t index currently holding its a0 token.
  ins     : the x_t index of the nearest non-blank slot to its LEFT (the
            "insert AFTER position i" gap representation; <bos> guarantees
            an anchor). When several pending ins share one anchor (a
            multi-token gap with nothing fired in between), only the
            LEFTMOST is supervised this step — left-to-right gap filling
            keeps the Q^ins target unique per (sample, position).

Loss weight: w(t) = κ̇(t)/(1−κ(t)) = 3t²/(1−t³), clipped (t→1 divergence).
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional, Sequence, Set, Tuple

KIND_INS, KIND_DEL, KIND_SUB = 0, 1, 2
KIND_NAMES = ["INS", "DEL", "SUB"]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_pair(src_ids: Sequence[int],
               tgt_ids: Sequence[int]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Min-edit blank-augmented alignment (z0, z1) as slots."""
    slots: List[Tuple[Optional[int], Optional[int]]] = []
    sm = difflib.SequenceMatcher(None, list(src_ids), list(tgt_ids),
                                 autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                slots.append((src_ids[i1 + k], tgt_ids[j1 + k]))
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                slots.append((src_ids[i1 + k], tgt_ids[j1 + k]))
            for k in range(i1 + n, i2):          # source longer → del
                slots.append((src_ids[k], None))
            for k in range(j1 + n, j2):          # target longer → ins
                slots.append((None, tgt_ids[k]))
        elif tag == "delete":
            for k in range(i1, i2):
                slots.append((src_ids[k], None))
        elif tag == "insert":
            for k in range(j1, j2):
                slots.append((None, tgt_ids[k]))
    return slots


def slot_ops(slots) -> List[Dict]:
    """All edit operations implied by the alignment."""
    ops = []
    for k, (a0, a1) in enumerate(slots):
        if a0 is None and a1 is not None:
            ops.append({"slot": k, "kind": KIND_INS, "tgt": int(a1)})
        elif a0 is not None and a1 is None:
            ops.append({"slot": k, "kind": KIND_DEL, "tgt": None})
        elif a0 is not None and a1 is not None and a0 != a1:
            ops.append({"slot": k, "kind": KIND_SUB, "tgt": int(a1)})
    return ops


def apply_all(slots) -> List[int]:
    """Fire every op → must reconstruct x1 exactly (roundtrip invariant)."""
    return [int(a1) for _, a1 in slots if a1 is not None]


def x0_of(slots) -> List[int]:
    return [int(a0) for a0, _ in slots if a0 is not None]


def cache_slots(x0, x1, editor_input, editor_target,
                mask_id: int, ins_id: int, del_id: int):
    """TRUE-alignment slots from the corruption cache's editor artifacts
    (Z2, EDIT_FLOWS_ZERO.md): the cache encodes the ops that GENERATED the
    pair — [MASK] at REPL (target = gold token), [INS] slots (target =
    inserted token), DEL kept in place with a [DEL] target — so the slot
    list can be reconstructed exactly instead of re-derived with difflib
    (which can misalign repeated tokens). Returns None when the artifacts
    don't reconstruct (x0, x1) — caller falls back to align_pair."""
    slots: List[Tuple[Optional[int], Optional[int]]] = []
    i = 0
    for j in range(len(editor_input)):
        tin, tt = int(editor_input[j]), int(editor_target[j])
        if tin == ins_id:
            slots.append((None, tt))
        else:
            if i >= len(x0):
                return None
            tok0 = int(x0[i])
            i += 1
            if tt == del_id:
                slots.append((tok0, None))
            else:
                slots.append((tok0, tt))
    if i != len(x0):
        return None
    if x0_of(slots) != [int(v) for v in x0]:
        return None
    if apply_all(slots) != [int(v) for v in x1]:
        return None
    return slots


# ---------------------------------------------------------------------------
# Forward process / training targets
# ---------------------------------------------------------------------------
def kappa(t: float) -> float:
    return t ** 3


def w_weight(t: float, w_max: float = 20.0) -> float:
    """κ̇/(1−κ) = 3t²/(1−t³), clipped for the t→1 divergence."""
    denom = 1.0 - t ** 3
    if denom <= 0:
        return w_max
    return min(3.0 * t * t / denom, w_max)


def build_xt(slots, ops: List[Dict], fired: Sequence[bool]) -> Tuple[List[int], List[Dict]]:
    """State at time t given per-op fired flags (aligned with `ops`).

    Returns (x_t token ids, pending) where each pending entry is
      {"kind": KIND, "pos": x_t index, "tgt": token or None}
    following the mapping rules in the module docstring. Pending ins whose
    anchor falls before the first token (no non-blank slot to the left) is
    dropped from supervision (cannot happen when x0 starts with <bos>).
    """
    fired_by_slot = {}
    for op, f in zip(ops, fired):
        fired_by_slot[op["slot"]] = (op, bool(f))

    x_t: List[int] = []
    slot_state_pos: List[Optional[int]] = []   # per slot: x_t index or None
    for k, (a0, a1) in enumerate(slots):
        op_f = fired_by_slot.get(k)
        state = a1 if (op_f is not None and op_f[1]) else a0
        if state is None:
            slot_state_pos.append(None)
        else:
            slot_state_pos.append(len(x_t))
            x_t.append(int(state))

    pending: List[Dict] = []
    used_ins_anchor: Set[int] = set()
    for op, f in zip(ops, fired):
        if f:
            continue
        k = op["slot"]
        if op["kind"] in (KIND_DEL, KIND_SUB):
            pos = slot_state_pos[k]           # a0 still present
            pending.append({"kind": op["kind"], "pos": pos, "tgt": op["tgt"]})
        else:                                  # KIND_INS — left anchor
            anchor = None
            for j in range(k - 1, -1, -1):
                if slot_state_pos[j] is not None:
                    anchor = slot_state_pos[j]
                    break
            if anchor is None:
                continue                       # no left token — unsupervised
            if anchor in used_ins_anchor:
                continue                       # leftmost-per-anchor rule
            used_ins_anchor.add(anchor)
            pending.append({"kind": KIND_INS, "pos": anchor, "tgt": op["tgt"]})
    return x_t, pending


# ---------------------------------------------------------------------------
# WHERE verification (λ-IoU) and gold sites
# ---------------------------------------------------------------------------
def gold_edit_positions(slots) -> Set[int]:
    """Gold edit sites in x0 coordinates: sub/del at their own position,
    ins at its left-anchor position (the 'insert after i' site)."""
    pos_of_slot: List[Optional[int]] = []
    n = 0
    for a0, _ in slots:
        if a0 is None:
            pos_of_slot.append(None)
        else:
            pos_of_slot.append(n)
            n += 1
    gold: Set[int] = set()
    for op in slot_ops(slots):
        k = op["slot"]
        if op["kind"] in (KIND_DEL, KIND_SUB):
            gold.add(pos_of_slot[k])
        else:
            for j in range(k - 1, -1, -1):
                if pos_of_slot[j] is not None:
                    gold.add(pos_of_slot[j])
                    break
    gold.discard(None)
    return gold


def lambda_iou(lam_total: Sequence[float], gold: Set[int],
               k: Optional[int] = None) -> float:
    """IoU between {top-k positions by total rate} and the gold edit sites.
    k defaults to |gold| (localization quality independent of the rate
    calibration — the count oracle; report it alongside a calibrated
    threshold variant if needed)."""
    if not gold:
        return float("nan")
    k = len(gold) if k is None else k
    order = sorted(range(len(lam_total)), key=lambda i: -float(lam_total[i]))
    pred = set(order[:k])
    inter = len(pred & gold)
    union = len(pred | gold)
    return inter / union if union else float("nan")


# ---------------------------------------------------------------------------
# Inference: apply one step's chosen ops to a token list
# ---------------------------------------------------------------------------
def apply_step_ops(ids: List[int], chosen: List[Dict]) -> List[int]:
    """Apply ops of one decode step. Each op: {"kind", "pos", "tok"} in the
    CURRENT ids' coordinates (ins = insert AFTER pos). At most one op per
    position (caller dedupes); applied right-to-left so indices stay valid."""
    out = list(ids)
    for op in sorted(chosen, key=lambda o: -o["pos"]):
        p = op["pos"]
        if p < 0 or p >= len(out):
            continue
        if op["kind"] == KIND_SUB:
            out[p] = int(op["tok"])
        elif op["kind"] == KIND_DEL:
            del out[p]
        else:                                  # KIND_INS after p
            out.insert(p + 1, int(op["tok"]))
    return out


# ---------------------------------------------------------------------------
# Self-test (CPU, no model): python editflow_ops.py
# ---------------------------------------------------------------------------
def _selftest():
    import random

    rng = random.Random(0)
    # 1) roundtrip on random pairs: fire everything → x1
    for _ in range(500):
        n = rng.randint(1, 30)
        src = [1] + [rng.randint(5, 20) for _ in range(n)]
        tgt = list(src)
        for _ in range(rng.randint(0, 6)):     # random edits
            r = rng.random()
            if r < 0.34 and len(tgt) > 2:
                del tgt[rng.randrange(1, len(tgt))]
            elif r < 0.67:
                tgt.insert(rng.randrange(1, len(tgt) + 1), rng.randint(5, 20))
            elif len(tgt) > 1:
                tgt[rng.randrange(1, len(tgt))] = rng.randint(5, 20)
        slots = align_pair(src, tgt)
        assert x0_of(slots) == src
        assert apply_all(slots) == tgt, (src, tgt)

    # 2) build_xt: nothing fired → x_t == x0; everything fired → x_t == x1
    src = [1, 10, 11, 12, 13]
    tgt = [1, 10, 99, 13, 14, 15]              # sub@2, del@3? — check below
    slots = align_pair(src, tgt)
    ops = slot_ops(slots)
    xt0, pend0 = build_xt(slots, ops, [False] * len(ops))
    assert xt0 == src
    xt1, pend1 = build_xt(slots, ops, [True] * len(ops))
    assert xt1 == tgt and pend1 == []

    # 3) pending mapping: unfired sub sits at its x_t position
    src = [1, 10, 11, 12]
    tgt = [1, 10, 99, 12]
    slots = align_pair(src, tgt)
    ops = slot_ops(slots)
    assert len(ops) == 1 and ops[0]["kind"] == KIND_SUB
    xt, pend = build_xt(slots, ops, [False])
    assert pend == [{"kind": KIND_SUB, "pos": 2, "tgt": 99}]

    # 4) multi-token insertion gap: leftmost-per-anchor; firing the first
    #    shifts the anchor of the second onto the inserted token
    src = [1, 10, 20]
    tgt = [1, 10, 30, 31, 20]
    slots = align_pair(src, tgt)
    ops = slot_ops(slots)
    assert [o["kind"] for o in ops] == [KIND_INS, KIND_INS]
    xt, pend = build_xt(slots, ops, [False, False])
    assert xt == src
    assert pend == [{"kind": KIND_INS, "pos": 1, "tgt": 30}]   # leftmost only
    xt, pend = build_xt(slots, ops, [True, False])
    assert xt == [1, 10, 30, 20]
    assert pend == [{"kind": KIND_INS, "pos": 2, "tgt": 31}]   # anchor moved
    # out-of-order fire: 2nd ins fired first — 1st still anchors at 10
    xt, pend = build_xt(slots, ops, [False, True])
    assert xt == [1, 10, 31, 20]
    assert pend == [{"kind": KIND_INS, "pos": 1, "tgt": 30}]

    # 5) deletion pending keeps its own position; earlier fired del shifts it
    src = [1, 10, 11, 12, 13]
    tgt = [1, 13]
    slots = align_pair(src, tgt)
    ops = slot_ops(slots)
    assert all(o["kind"] == KIND_DEL for o in ops) and len(ops) == 3
    xt, pend = build_xt(slots, ops, [True, False, False])
    assert xt == [1, 11, 12, 13]
    assert [(p["pos"], p["kind"]) for p in pend] == [(1, KIND_DEL), (2, KIND_DEL)]

    # 6) gold_edit_positions
    src = [1, 10, 11, 12]
    tgt = [1, 10, 99, 12, 40]                  # sub@2, ins after 3
    slots = align_pair(src, tgt)
    assert gold_edit_positions(slots) == {2, 3}

    # 7) lambda_iou: perfect / disjoint / partial
    assert lambda_iou([0, 0, 9, 8], {2, 3}) == 1.0
    assert lambda_iou([9, 8, 0, 0], {2, 3}) == 0.0
    assert abs(lambda_iou([9, 0, 8, 0], {2, 3}) - (1 / 3)) < 1e-9

    # 8) apply_step_ops: right-to-left validity, all three kinds
    ids = [1, 10, 11, 12]
    out = apply_step_ops(ids, [
        {"kind": KIND_SUB, "pos": 1, "tok": 77},
        {"kind": KIND_DEL, "pos": 2, "tok": None},
        {"kind": KIND_INS, "pos": 3, "tok": 88},
    ])
    assert out == [1, 77, 12, 88], out

    # 9) w_weight: increasing, clipped
    assert w_weight(0.0) == 0.0
    assert w_weight(0.5) < w_weight(0.9)
    assert w_weight(0.999999) == 20.0

    # 10) simulated flow: independent κ(t) firing, then fire the rest → x1
    rng2 = random.Random(1)
    for _ in range(200):
        n = rng2.randint(2, 20)
        src = [1] + [rng2.randint(5, 15) for _ in range(n)]
        tgt = list(src)
        for _ in range(rng2.randint(1, 5)):
            r = rng2.random()
            if r < 0.34 and len(tgt) > 2:
                del tgt[rng2.randrange(1, len(tgt))]
            elif r < 0.67:
                tgt.insert(rng2.randrange(1, len(tgt) + 1), rng2.randint(5, 15))
            else:
                tgt[rng2.randrange(1, len(tgt))] = rng2.randint(5, 15)
        slots = align_pair(src, tgt)
        ops = slot_ops(slots)
        t = rng2.random()
        fired = [rng2.random() < kappa(t) for _ in ops]
        xt, _ = build_xt(slots, ops, fired)
        # completing the remaining ops must still reach x1: refire all
        xt_full, pend_none = build_xt(slots, ops, [True] * len(ops))
        assert xt_full == tgt and pend_none == []
        # and x_t must be reachable: len bounds
        assert abs(len(xt) - len(src)) <= len(ops)

    print("OK: editflow_ops self-test passed")


if __name__ == "__main__":
    _selftest()
