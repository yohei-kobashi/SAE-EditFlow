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

KIND_INS, KIND_DEL, KIND_SUB, KIND_MOV = 0, 1, 2, 3
KIND_NAMES = ["INS", "DEL", "SUB", "MOV"]


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


# ---------------------------------------------------------------------------
# MOVE reinterpretation (M1): content-identical DEL/INS run pairs become
# single MOV ops — one firing instead of the DEL-fire + INS-fire +
# Q-regenerates-content product. Same rule as measure_move_headroom.py.
# ---------------------------------------------------------------------------
def del_ins_runs(slots) -> Tuple[List[Tuple[int, Tuple[int, ...]]],
                                 List[Tuple[int, Tuple[int, ...]]]]:
    """Maximal contiguous runs of pure-DEL / pure-INS slots, each as
    (start_slot, content_tuple). SUB/KEEP slots break runs."""
    d_runs, i_runs = [], []
    kind_prev: Optional[str] = None
    start = 0
    buf: List[int] = []

    def flush():
        if buf:
            (d_runs if kind_prev == "del" else i_runs).append(
                (start, tuple(buf)))

    for k, (a0, a1) in enumerate(slots):
        if a0 is not None and a1 is None:
            kind, tok = "del", int(a0)
        elif a0 is None and a1 is not None:
            kind, tok = "ins", int(a1)
        else:
            kind, tok = None, None
        if kind != kind_prev:
            flush()
            buf, start, kind_prev = [], k, kind
        if kind is not None:
            buf.append(tok)
    flush()
    return d_runs, i_runs


def match_move_runs(d_runs, i_runs) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """Greedy 1:1 pairing of DEL runs with content-identical INS runs,
    longest content first. Returns (del_start, ins_start, content)."""
    out = []
    used = set()
    for di in sorted(range(len(d_runs)), key=lambda i: -len(d_runs[i][1])):
        d_start, d_content = d_runs[di]
        for ii, (i_start, i_content) in enumerate(i_runs):
            if ii in used or i_content != d_content:
                continue
            used.add(ii)
            out.append((d_start, i_start, d_content))
            break
    return sorted(out)


def move_reinterpret(slots, ops: List[Dict]) -> List[Dict]:
    """Rewrite `ops`: each matched DEL/INS slot pair becomes one MOV op
    {"slot": del_slot, "kind": KIND_MOV, "tgt": tok, "dst": ins_slot}
    (per token — an n-token span move becomes n MOV ops with consecutive
    src/dst slots); the paired INS ops are dropped. All other ops pass
    through unchanged, order preserved by src slot."""
    matches = match_move_runs(*del_ins_runs(slots))
    dst_of: Dict[int, int] = {}
    for d_start, i_start, content in matches:
        for j in range(len(content)):
            dst_of[d_start + j] = i_start + j
    taken_ins = set(dst_of.values())
    out: List[Dict] = []
    for op in ops:
        k = op["slot"]
        if op["kind"] == KIND_DEL and k in dst_of:
            out.append({"slot": k, "kind": KIND_MOV,
                        "tgt": int(slots[k][0]), "dst": dst_of[k]})
        elif op["kind"] == KIND_INS and k in taken_ins:
            continue
        else:
            out.append(op)
    return out


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


def _pois(lam: float, rng) -> int:
    """Poisson draw (Knuth); lam is small here (λ_prop·Δt ≲ a few)."""
    if lam <= 0:
        return 0
    import math
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def sample_localized_fired(slots, ops: List[Dict], t: float,
                           lam_prop: float, rng,
                           w_max: float = 20.0) -> Tuple[List[bool], List[float]]:
    """Localized propagation sampling (paper appendix C.1, eqs 46-48).

    Every alignment slot i runs an independent source process M^i: it
    self-fires at t*_i = κ⁻¹(u_i) = u_i^{1/3} (eq 46, κ = t³); once on,
    it propagates to Nl/Nr ~ Pois(λ_prop·(t−t*_i)) neighbors left/right
    (eqs 47-48). m_j = OR over source coverages (eq 41); an op is fired
    iff m is true at its slot. Returns (fired, lam_eff), both aligned
    with `ops`; lam_eff is the per-op training weight (eqs 43-44):
        λ_eff = w(t) + λ_prop·#{sources whose coverage touches slot j±1}
    (used at PENDING ops; computed for all). With lam_prop == 0 this
    reduces exactly to independent Bernoulli κ(t) firing with weight w(t).
    """
    n = len(slots)
    tstar = [rng.random() ** (1.0 / 3.0) for _ in range(n)]
    cover: List[Tuple[int, int]] = []          # inclusive (lo, hi) per ON source
    for i in range(n):
        if tstar[i] <= t:
            dt = t - tstar[i]
            nl = _pois(lam_prop * dt, rng)
            nr = _pois(lam_prop * dt, rng)
            cover.append((max(0, i - nl), min(n - 1, i + nr)))
    m = [False] * n
    for lo, hi in cover:
        for j in range(lo, hi + 1):
            m[j] = True
    fired = [m[op["slot"]] for op in ops]
    w = w_weight(t, w_max)
    lam_eff = []
    for op in ops:
        j = op["slot"]
        cnt = sum(1 for lo, hi in cover
                  if lo <= j - 1 <= hi or lo <= j + 1 <= hi)
        lam_eff.append(w + lam_prop * cnt)
    return fired, lam_eff


def edited_marks_xt(slots, ops: List[Dict], fired: Sequence[bool]) -> Tuple[Set[int], int]:
    """x_t positions carrying a fired edit, for the S3 adjacency feature:
    fired sub/ins mark their own x_t index; a fired del marks the nearest
    LEFT surviving index (the deletion boundary is invisible in tokens).
    Mirrors the decode-side tracking of apply_step_ops(..., edited=...).
    A fired MOV marks its landing position (like ins) plus the source's
    left boundary (like del). Returns (marks, len(x_t))."""
    fired_by_slot = {}
    for op, f in zip(ops, fired):
        fired_by_slot[op["slot"]] = bool(f)
        if op["kind"] == KIND_MOV:
            fired_by_slot[op["dst"]] = bool(f)
    pos_of_slot: List[Optional[int]] = []
    x_len = 0
    for k, (a0, a1) in enumerate(slots):
        state = a1 if fired_by_slot.get(k, False) else a0
        if state is None:
            pos_of_slot.append(None)
        else:
            pos_of_slot.append(x_len)
            x_len += 1

    def left_boundary(k):
        for j in range(k - 1, -1, -1):
            if pos_of_slot[j] is not None:
                return pos_of_slot[j]
        return None

    marks: Set[int] = set()
    for op, f in zip(ops, fired):
        if not f:
            continue
        k = op["slot"]
        if op["kind"] in (KIND_SUB, KIND_INS):
            marks.add(pos_of_slot[k])
        elif op["kind"] == KIND_MOV:
            marks.add(pos_of_slot[op["dst"]])
            marks.add(left_boundary(k))
        else:                                   # DEL — left boundary
            marks.add(left_boundary(k))
    marks.discard(None)
    return marks, x_len


def adj_counts(marks: Set[int], length: int) -> List[int]:
    """Per-position count (0/1/2) of edited neighbors — the S3 model
    feature matching eq 40's off-diagonal (j±1) propagation."""
    return [(1 if j - 1 in marks else 0) + (1 if j + 1 in marks else 0)
            for j in range(length)]


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
        if op["kind"] == KIND_MOV:             # dst slot shares the flag
            fired_by_slot[op["dst"]] = (op, bool(f))

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

    def left_anchor(k):
        for j in range(k - 1, -1, -1):
            if slot_state_pos[j] is not None:
                return slot_state_pos[j]
        return None

    pending: List[Dict] = []
    used_ins_anchor: Set[int] = set()
    for oi, (op, f) in enumerate(zip(ops, fired)):
        if f:
            continue
        k = op["slot"]
        if op["kind"] in (KIND_DEL, KIND_SUB):
            pos = slot_state_pos[k]           # a0 still present
            pending.append({"kind": op["kind"], "pos": pos,
                            "tgt": op["tgt"], "op": oi})
        elif op["kind"] == KIND_MOV:           # src present; dst = anchor
            dst_pos = left_anchor(op["dst"])
            if dst_pos is None:
                continue                       # cannot point — unsupervised
            pending.append({"kind": KIND_MOV, "pos": slot_state_pos[k],
                            "tgt": op["tgt"], "op": oi,
                            "dst_pos": dst_pos})
        else:                                  # KIND_INS — left anchor
            anchor = left_anchor(k)
            if anchor is None:
                continue                       # no left token — unsupervised
            if anchor in used_ins_anchor:
                continue                       # leftmost-per-anchor rule
            used_ins_anchor.add(anchor)
            pending.append({"kind": KIND_INS, "pos": anchor,
                            "tgt": op["tgt"], "op": oi})
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
def apply_step_ops(ids: List[int], chosen: List[Dict],
                   edited: Optional[List[bool]] = None):
    """Apply ops of one decode step. Each op: {"kind", "pos", "tok"} in the
    CURRENT ids' coordinates (ins = insert AFTER pos; mov additionally has
    "dst" = insert-after position for the moved token). At most one op per
    position (caller dedupes). Implemented as a single left-to-right
    rebuild over the ORIGINAL coordinates, so any op mix — including a mov
    whose src and dst straddle other ops — stays index-valid.
    When `edited` (bool per current position, S3 adjacency tracking) is
    given, it is kept in sync — sub marks its position, ins/mov mark the
    new token, del/mov-src mark the left surviving boundary — and the
    return value becomes (out, edited_out)."""
    n = len(ids)
    sub_at: Dict[int, int] = {}
    del_at: Set[int] = set()
    ins_after: Dict[int, List[int]] = {}
    for op in chosen:
        p = op["pos"]
        if p < 0 or p >= n:
            continue
        if op["kind"] == KIND_SUB:
            sub_at[p] = int(op["tok"])
        elif op["kind"] == KIND_DEL:
            del_at.add(p)
        elif op["kind"] == KIND_INS:
            ins_after.setdefault(p, []).append(int(op["tok"]))
        elif op["kind"] == KIND_MOV:
            q = op.get("dst", -1)
            if 0 <= q < n and q != p and p not in del_at:
                del_at.add(p)
                ins_after.setdefault(q, []).append(int(ids[p]))
    out: List[int] = []
    ed: Optional[List[bool]] = [] if edited is not None else None
    for i in range(n):
        if i in del_at:
            if ed is not None and ed:
                ed[-1] = True                  # left boundary of the removal
        else:
            out.append(sub_at.get(i, int(ids[i])))
            if ed is not None:
                ed.append(True if i in sub_at else bool(edited[i]))
        for tok in ins_after.get(i, ()):
            out.append(tok)
            if ed is not None:
                ed.append(True)
    return (out, ed) if edited is not None else out


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
    assert pend == [{"kind": KIND_SUB, "pos": 2, "tgt": 99, "op": 0}]

    # 4) multi-token insertion gap: leftmost-per-anchor; firing the first
    #    shifts the anchor of the second onto the inserted token
    src = [1, 10, 20]
    tgt = [1, 10, 30, 31, 20]
    slots = align_pair(src, tgt)
    ops = slot_ops(slots)
    assert [o["kind"] for o in ops] == [KIND_INS, KIND_INS]
    xt, pend = build_xt(slots, ops, [False, False])
    assert xt == src
    assert pend == [{"kind": KIND_INS, "pos": 1, "tgt": 30, "op": 0}]   # leftmost only
    xt, pend = build_xt(slots, ops, [True, False])
    assert xt == [1, 10, 30, 20]
    assert pend == [{"kind": KIND_INS, "pos": 2, "tgt": 31, "op": 1}]   # anchor moved
    # out-of-order fire: 2nd ins fired first — 1st still anchors at 10
    xt, pend = build_xt(slots, ops, [False, True])
    assert xt == [1, 10, 31, 20]
    assert pend == [{"kind": KIND_INS, "pos": 1, "tgt": 30, "op": 0}]

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

    # 11) localized sampling (S3): lam_prop=0 reduces EXACTLY to
    #     independent kappa(t) firing (same rng stream)
    for seed in range(20):
        src = [1] + [random.Random(seed).randint(5, 15) for _ in range(8)]
        tgt = list(src)
        tgt[3] = 99
        tgt.insert(6, 77)
        slots = align_pair(src, tgt)
        ops = slot_ops(slots)
        t = 0.55
        r1, r2 = random.Random(seed + 100), random.Random(seed + 100)
        tstar = [r1.random() ** (1.0 / 3.0) for _ in range(len(slots))]
        fired, lam_eff = sample_localized_fired(slots, ops, t, 0.0, r2)
        assert fired == [tstar[o["slot"]] <= t for o in ops]
        assert all(abs(le - w_weight(t)) < 1e-12 for le in lam_eff)
    # boundary: t=0 -> nothing fires; t=1 -> everything fires
    fired0, _ = sample_localized_fired(slots, ops, 0.0, 4.0, random.Random(0))
    assert not any(fired0)
    fired1, _ = sample_localized_fired(slots, ops, 1.0, 4.0, random.Random(0))
    assert all(fired1)
    # propagation raises both firing counts and lam_eff above baseline
    rngp = random.Random(7)
    tot0 = tot4 = 0
    for _ in range(300):
        f0, _ = sample_localized_fired(slots, ops, 0.4, 0.0, rngp)
        f4, le4 = sample_localized_fired(slots, ops, 0.4, 4.0, rngp)
        tot0 += sum(f0)
        tot4 += sum(f4)
        assert all(le >= w_weight(0.4) - 1e-12 for le in le4)
    assert tot4 > tot0

    # 12) edited_marks_xt: sub marks own pos, del marks left boundary,
    #     ins marks the inserted token
    src = [1, 10, 11, 12, 13]
    tgt = [1, 99, 12, 13, 55]                 # sub@1, del@2(11), ins after 13
    slots = align_pair(src, tgt)
    ops = slot_ops(slots)
    kinds = {o["kind"] for o in ops}
    assert kinds == {KIND_SUB, KIND_DEL, KIND_INS}
    fired = [o["kind"] == KIND_SUB for o in ops]
    marks, xl = edited_marks_xt(slots, ops, fired)
    xt, _ = build_xt(slots, ops, fired)
    assert xl == len(xt) and marks == {1}     # sub at pos 1
    fired = [o["kind"] == KIND_DEL for o in ops]
    marks, xl = edited_marks_xt(slots, ops, fired)
    xt, _ = build_xt(slots, ops, fired)
    assert xl == len(xt) and marks == {1}     # boundary left of deleted 11
    fired = [o["kind"] == KIND_INS for o in ops]
    marks, xl = edited_marks_xt(slots, ops, fired)
    xt, _ = build_xt(slots, ops, fired)
    assert xt[-1] == 55 and marks == {len(xt) - 1}
    # adjacency counts: neighbors of a mark get 1, between two marks 2
    assert adj_counts({2}, 5) == [0, 1, 0, 1, 0]
    assert adj_counts({1, 3}, 5) == [1, 0, 2, 0, 1]

    # 13) apply_step_ops edited tracking stays in sync
    ids = [1, 10, 11, 12]
    out, ed = apply_step_ops(ids, [
        {"kind": KIND_SUB, "pos": 1, "tok": 77},
        {"kind": KIND_DEL, "pos": 2, "tok": None},
        {"kind": KIND_INS, "pos": 3, "tok": 88},
    ], edited=[False] * 4)
    assert out == [1, 77, 12, 88] and len(ed) == len(out)
    assert ed == [False, True, False, True]   # sub@1 (del boundary also @1), ins 88

    # 14) MOVE reinterpretation: particle-shift pair [1,2,3,4]->[1,3,4,2]
    src, tgt = [1, 2, 3, 4], [1, 3, 4, 2]
    slots = align_pair(src, tgt)
    raw = slot_ops(slots)
    assert [o["kind"] for o in raw] == [KIND_DEL, KIND_INS]
    ops = move_reinterpret(slots, raw)
    assert len(ops) == 1 and ops[0]["kind"] == KIND_MOV
    assert ops[0]["tgt"] == 2 and slots[ops[0]["dst"]] == (None, 2)
    # roundtrip both ways
    xt, pend = build_xt(slots, ops, [True])
    assert xt == tgt and pend == []
    xt, pend = build_xt(slots, ops, [False])
    assert xt == src
    assert pend == [{"kind": KIND_MOV, "pos": 1, "tgt": 2, "op": 0,
                     "dst_pos": 3}]           # move token@1 to after pos 3
    # decode application + edited sync: landing marked, src left boundary
    out, ed = apply_step_ops(src, [{"kind": KIND_MOV, "pos": 1, "dst": 3}],
                             edited=[False] * 4)
    assert out == tgt and ed == [True, False, False, True]
    # train-side adjacency agrees with the decode-side tracking
    marks, xl = edited_marks_xt(slots, ops, [True])
    assert xl == 4 and marks == {0, 3}
    # non-MOV pairs pass through move_reinterpret unchanged
    s2 = align_pair([1, 10, 11], [1, 99, 11])
    assert move_reinterpret(s2, slot_ops(s2)) == slot_ops(s2)

    # 15) MOVE roundtrip + pending validity on random pairs (incl. span
    #     moves and mixed edits); partial firing states stay consistent
    rng3 = random.Random(3)
    n_mov_pairs = 0
    for _ in range(400):
        n = rng3.randint(3, 14)
        src = [1] + [rng3.randint(5, 15) for _ in range(n)]
        tgt = list(src)
        r = rng3.random()
        if r < 0.5 and len(tgt) > 4:           # true move: cut a span, paste
            a = rng3.randrange(1, len(tgt) - 1)
            ln = min(rng3.randint(1, 2), len(tgt) - a - 1)
            span = tgt[a:a + ln]
            del tgt[a:a + ln]
            b = rng3.randrange(1, len(tgt) + 1)
            tgt[b:b] = span
        for _ in range(rng3.randint(0, 2)):    # extra unrelated edits
            rr = rng3.random()
            if rr < 0.34 and len(tgt) > 2:
                del tgt[rng3.randrange(1, len(tgt))]
            elif rr < 0.67:
                tgt.insert(rng3.randrange(1, len(tgt) + 1),
                           rng3.randint(5, 15))
            else:
                tgt[rng3.randrange(1, len(tgt))] = rng3.randint(5, 15)
        slots = align_pair(src, tgt)
        ops = move_reinterpret(slots, slot_ops(slots))
        if any(o["kind"] == KIND_MOV for o in ops):
            n_mov_pairs += 1
        # all fired -> x1 exactly, regardless of reinterpretation
        xt1, p1 = build_xt(slots, ops, [True] * len(ops))
        assert xt1 == tgt and p1 == [], (src, tgt)
        # none fired -> x0; every pending mov's pos holds its token
        xt0, p0 = build_xt(slots, ops, [False] * len(ops))
        assert xt0 == src
        for pe in p0:
            if pe["kind"] == KIND_MOV:
                assert xt0[pe["pos"]] == pe["tgt"]
                assert 0 <= pe["dst_pos"] < len(xt0)
        # random partial firing: states well-formed, pending movs valid
        fired = [rng3.random() < 0.5 for _ in ops]
        xtp, pp = build_xt(slots, ops, fired)
        for pe in pp:
            assert 0 <= pe["pos"] < len(xtp)
            if pe["kind"] == KIND_MOV:
                assert xtp[pe["pos"]] == pe["tgt"]
                assert 0 <= pe["dst_pos"] < len(xtp)
        marks, xl = edited_marks_xt(slots, ops, fired)
        assert xl == len(xtp) and all(0 <= m < xl for m in marks)
    assert n_mov_pairs > 50                    # the generator makes moves

    # 16) one-pass apply matches decode semantics for a mov among other
    #     ops in the same step (src right of dst and vice versa)
    ids = [1, 10, 11, 12, 13]
    out = apply_step_ops(ids, [
        {"kind": KIND_MOV, "pos": 3, "dst": 0},       # 12 -> after 1
        {"kind": KIND_SUB, "pos": 1, "tok": 77},
    ])
    assert out == [1, 12, 77, 11, 13], out
    out = apply_step_ops(ids, [
        {"kind": KIND_MOV, "pos": 1, "dst": 4},       # 10 -> after 13
        {"kind": KIND_DEL, "pos": 2},
    ])
    assert out == [1, 12, 13, 10], out

    print("OK: editflow_ops self-test passed")


if __name__ == "__main__":
    _selftest()
