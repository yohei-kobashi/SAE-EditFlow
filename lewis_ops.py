"""
LEWIS-style op application for SAE-LEWIS.

The 4-class tagger predicts per-token ops over {KEEP, REPL, INS, DEL}. This
module turns (token_ids, ops) into the editor's input sequence and tracks INS
gaps so that template enumeration at inference can vary slot counts per gap.

`apply_ops_for_editor` is purely deterministic given (token_ids, ops); it does
not need access to the editor or tagger. At training time the corruption
generator emits ops directly; at inference the tagger does.

Op semantics (LEWIS-faithful, with [DEL] output marker for our bidirectional
editor):

    KEEP : retain the original token in the editor input; expect identity
           at the output.
    REPL : replace the original token by [MASK] in the editor input; the
           editor predicts the replacement at the output.
    DEL  : keep the original token in the editor input ("in-place"); the
           editor predicts [DEL] at the output (dropped in post-processing).
    INS  : a [INS] slot is inserted to the LEFT of the next KEEP/REPL/DEL
           position. The editor predicts the inserted token.

Conventions:
- INS does not consume an original token; it expands the sequence.
- Consecutive INS tags at training time mean a multi-token gap; one [INS]
  token per gold INS tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


OP_KEEP = 0
OP_REPL = 1
OP_INS = 2
OP_DEL = 3
NUM_OPS = 4

OP_NAMES = ["KEEP", "REPL", "INS", "DEL"]


def op_name(op: int) -> str:
    return OP_NAMES[int(op)]


def op_id(name: str) -> int:
    return OP_NAMES.index(name)


@dataclass
class EditorInputs:
    """Deterministic editor-side artifacts derived from (token_ids, ops).

    Attributes
    ----------
    input_ids : np.ndarray (T_out,) int32
        The editor's input sequence — original tokens for KEEP/DEL, [MASK]
        for REPL, [INS] for INS slots.
    op_per_pos : np.ndarray (T_out,) int8
        Op assigned to each output position. INS positions carry OP_INS.
        KEEP/REPL/DEL positions correspond to original tokens.
    source_pos : np.ndarray (T_out,) int32
        Index into the source `token_ids` for each output position; -1 for
        INS positions (no source token).
    ins_gaps : list of (start, length)
        Start position (in T_out) and slot count for each INS gap. Used by
        template enumeration at inference.
    """

    input_ids: np.ndarray
    op_per_pos: np.ndarray
    source_pos: np.ndarray
    ins_gaps: List[Tuple[int, int]]


def apply_ops_for_editor(
    token_ids: Sequence[int],
    ops: Sequence[int],
    mask_token_id: int,
    ins_token_id: int,
) -> EditorInputs:
    """Build the editor's input sequence from (token_ids, ops).

    `ops` has the SAME LENGTH AS `token_ids`. Each source position carries one
    op; the source token at that position is implicitly preserved (KEEP) for
    INS, replaced with [MASK] for REPL, dropped at output for DEL.

    Semantics per op at source position i:
      KEEP : emit token_ids[i]
      REPL : emit [MASK] in place of token_ids[i]
      DEL  : emit token_ids[i] in place; editor learns to output [DEL]
      INS  : emit one [INS] slot, then emit token_ids[i] (implicit KEEP)

    Multi-slot gaps are handled by `expand_ins_gap`, which is called by
    template enumeration at inference.
    """
    if len(token_ids) != len(ops):
        raise ValueError(
            f"token_ids ({len(token_ids)}) and ops ({len(ops)}) length mismatch"
        )

    out_ids: List[int] = []
    out_ops: List[int] = []
    out_src: List[int] = []
    ins_gaps: List[Tuple[int, int]] = []

    for i in range(len(token_ids)):
        op = int(ops[i])
        if op == OP_KEEP:
            out_ids.append(int(token_ids[i]))
            out_ops.append(OP_KEEP)
            out_src.append(i)
        elif op == OP_REPL:
            out_ids.append(mask_token_id)
            out_ops.append(OP_REPL)
            out_src.append(i)
        elif op == OP_DEL:
            out_ids.append(int(token_ids[i]))
            out_ops.append(OP_DEL)
            out_src.append(i)
        elif op == OP_INS:
            ins_gaps.append((len(out_ids), 1))
            out_ids.append(ins_token_id)
            out_ops.append(OP_INS)
            out_src.append(-1)
            out_ids.append(int(token_ids[i]))
            out_ops.append(OP_KEEP)
            out_src.append(i)
        else:
            raise ValueError(f"unknown op id {op} at position {i}")

    return EditorInputs(
        input_ids=np.asarray(out_ids, dtype=np.int32),
        op_per_pos=np.asarray(out_ops, dtype=np.int8),
        source_pos=np.asarray(out_src, dtype=np.int32),
        ins_gaps=ins_gaps,
    )


def expand_ins_gap(
    inputs: EditorInputs,
    gap_idx: int,
    new_slot_count: int,
    ins_token_id: int,
) -> EditorInputs:
    """Return a new EditorInputs with the `gap_idx`-th INS gap resized to
    `new_slot_count` slots. Used by template enumeration at inference.
    """
    start, length = inputs.ins_gaps[gap_idx]
    if new_slot_count == length:
        return inputs

    new_input_ids = np.concatenate([
        inputs.input_ids[:start],
        np.full(new_slot_count, ins_token_id, dtype=np.int32),
        inputs.input_ids[start + length:],
    ])
    new_op_per_pos = np.concatenate([
        inputs.op_per_pos[:start],
        np.full(new_slot_count, OP_INS, dtype=np.int8),
        inputs.op_per_pos[start + length:],
    ])
    new_source_pos = np.concatenate([
        inputs.source_pos[:start],
        np.full(new_slot_count, -1, dtype=np.int32),
        inputs.source_pos[start + length:],
    ])

    delta = new_slot_count - length
    new_gaps: List[Tuple[int, int]] = []
    for i, (s, l) in enumerate(inputs.ins_gaps):
        if i == gap_idx:
            new_gaps.append((s, new_slot_count))
        elif s > start:
            new_gaps.append((s + delta, l))
        else:
            new_gaps.append((s, l))

    return EditorInputs(
        input_ids=new_input_ids,
        op_per_pos=new_op_per_pos,
        source_pos=new_source_pos,
        ins_gaps=new_gaps,
    )


def decode_with_op_mask(
    input_ids: np.ndarray,
    argmax_ids: np.ndarray,
    op_per_pos: np.ndarray,
    del_token_id: int,
) -> List[int]:
    """Reduce editor (input_ids, argmax) using the op_per_pos op mask to a
    list of output tokens. Drops every position whose final token is [DEL].

    KEEP : output = input token (identity is enforced regardless of argmax)
    REPL : output = argmax (unless argmax == [DEL], in which case dropped)
    DEL  : output = [DEL] (dropped)
    INS  : output = argmax (unless argmax == [DEL], in which case dropped)

    Per the README, any other position whose argmax is [DEL] is dropped too.
    """
    out: List[int] = []
    for pos in range(len(input_ids)):
        op = int(op_per_pos[pos])
        if op == OP_KEEP:
            tok = int(input_ids[pos])
        elif op == OP_REPL:
            tok = int(argmax_ids[pos])
        elif op == OP_DEL:
            continue
        else:  # OP_INS
            tok = int(argmax_ids[pos])
        if tok == del_token_id:
            continue
        out.append(tok)
    return out


def stats_ins_run_lengths(ops: Sequence[int]) -> List[int]:
    """Return a list of INS-run lengths (consecutive OP_INS) in `ops`."""
    out: List[int] = []
    i, n = 0, len(ops)
    while i < n:
        if int(ops[i]) == OP_INS:
            j = i
            while j < n and int(ops[j]) == OP_INS:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    return out
