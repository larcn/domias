# FILE: evaluate.py | version: 2026-01-17.forensics_truth_enriched (+rust knobs wired)
# Headless evaluator (match-to-target) for Domino AI
#
# Design:
# - engine.GameState is authoritative for counts; we keep "truth" sets for identities.
# - We NEVER overwrite st.opponent_tile_count / st.boneyard_count during play.
# - We only assert they match truth.
#
# Me policy options:
# - "rust": use domino_rs.suggest_move_ismcts(state_dict, ...) to pick moves (decision in Rust).
# - "model": use ai.model_predict (policy head) in Python.
# - "ai": use ai.suggest_moves (legacy Python analysis).
# - "greedy": greedy baseline.
#
# Telemetry:
# - When me_policy="rust", we capture metadata returned by Rust (if present),
#   and compare predicted spike risk vs truth win-now existence.
# - This file is robust if Rust does not yet return spike fields (telemetry auto-disables).
#
# Forensics additions:
# - dump_match_script.json includes:
#     * round1_opp_hand_truth
#     * round1_boneyard_order (draw via pop() from end)
#     * me_decisions: per "me" move snapshot (chosen + root_candidates if present)
# - Root candidates are enriched (for first dumped match only) with:
#     * truth_opp_best_reply_pts
#     * truth_opp_win_now_exists
#   This allows human audit of "was this move actually unsafe?"

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict, Any, Literal

import os
import struct

from engine import GameState, Board, Tile, EndName, ALL_TILES, tile_is_double, tile_pip_count, parse_tile
import ai as ai_mod
import numpy as np

# Rust module (optional unless --me rust is used)
try:
    import domino_rs  # type: ignore
except Exception:
    domino_rs = None  # type: ignore

ENDS: List[EndName] = ["right", "left", "up", "down"]

INF_MAGIC = b"INF1"
INF_FEAT_DIM = 235
INF_LABEL_SIZE = 28
INF_RECORD_SIZE = (INF_FEAT_DIM * 4) + (INF_LABEL_SIZE * 1) + 2
INF_HEADER_LEN = 16


def _tile_id(t: Tile) -> int:
    """
    Match Rust tile.rs pips_to_id(hi, lo):
      start = hi*(hi+1)/2; id = start + lo
    Assumes engine.norm_tile (hi>=lo) already holds.
    """
    hi, lo = int(t[0]), int(t[1])
    return int((hi * (hi + 1)) // 2 + lo)


class Inf1Writer:
    """
    Minimal INF1 writer (raw, no compression) compatible with tools/infer_tool.py.
    Records:
      inf_feat[235]*f32 + label[28]*i8 + opp_cnt(u8) + bone_cnt(u8)
    """
    def __init__(self, out_dir: Path, run_id: str, shard_max_samples: int = 50000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.shard_max = int(max(1000, shard_max_samples))
        self.shard_idx = 0
        self.samples_in_shard = 0
        self.total_samples = 0
        self._fh: Optional[Any] = None
        self.shards: List[Dict[str, Any]] = []
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        if self._fh is not None:
            self._close_shard()
        fn = f"{self.run_id}.infer_strat_{self.shard_idx:05d}.inf"
        path = self.out_dir / fn
        self._fh = open(path, "wb")
        header = INF_MAGIC + struct.pack("<III", INF_FEAT_DIM, INF_LABEL_SIZE, INF_RECORD_SIZE)
        assert len(header) == INF_HEADER_LEN
        self._fh.write(header)
        self.samples_in_shard = 0
        self._cur_path = path

    def _close_shard(self) -> None:
        assert self._fh is not None
        self._fh.flush()
        self._fh.close()
        b = int(self._cur_path.stat().st_size)
        self.shards.append({
            "filename": self._cur_path.name,
            "codec": "raw",
            "samples": int(self.samples_in_shard),
            "bytes_on_disk": int(b),
        })
        self._fh = None
        self.shard_idx += 1

    def write(self, inf_feat: np.ndarray, label28: np.ndarray, opp_cnt: int, bone_cnt: int) -> None:
        if self._fh is None:
            self._open_new_shard()
        if self.samples_in_shard >= self.shard_max:
            self._open_new_shard()
        assert inf_feat.shape == (INF_FEAT_DIM,)
        assert label28.shape == (INF_LABEL_SIZE,)
        self._fh.write(inf_feat.astype(np.float32).tobytes())
        self._fh.write(label28.astype(np.int8).tobytes())
        self._fh.write(bytes([int(max(0, min(28, opp_cnt)))]))
        self._fh.write(bytes([int(max(0, min(28, bone_cnt)))]))
        self.samples_in_shard += 1
        self.total_samples += 1

    def finish(self) -> Dict[str, Any]:
        if self._fh is not None:
            self._close_shard()
        return {
            "infer_codec": "raw",
            "infer_feat_dim": int(INF_FEAT_DIM),
            "infer_label_size": int(INF_LABEL_SIZE),
            "infer_record_size": int(INF_RECORD_SIZE),
            "infer_samples": int(self.total_samples),
            "infer_shards": list(self.shards),
        }


def _slice_events_current_round(sd: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Return a shallow-copied state_dict with events sliced from last round_start.
    Also returns round_start_idx used.
    """
    evs = list(sd.get("events") or [])
    start = 0
    for i, ev in enumerate(evs):
        if isinstance(ev, dict) and ev.get("type") == "round_start":
            start = i
    sd2 = dict(sd)
    sd2["events"] = evs[start:]
    return sd2, start


def _update_endchoice_from_play(
    board_before: Board,
    tile: Tile,
    chosen_end: EndName,
    prefer: List[int],
    avoid: List[int],
    prev_was_opp_draw: bool,
) -> None:
    """
    Mirror Rust signal logic:
      - only if not forced immediately after opponent draw
      - only if >1 legal end for this tile on this board
      - prefer[new_open_value_of_chosen_end]++
      - avoid[new_open_value_of_other_ends]++
    """
    if prev_was_opp_draw:
        return
    ends = list(board_before.legal_ends_for_tile(tile))
    if len(ends) <= 1:
        return
    if chosen_end not in ends:
        return
    # chosen new value
    ov = int((board_before.ends.get(chosen_end) or (None, False))[0] or 0)
    nv = int(ai_mod.other_value(tile, ov)) if ai_mod.tile_has(tile, ov) else ov
    if 0 <= nv <= 6:
        prefer[nv] += 1
    # alternatives
    for e in ends:
        if e == chosen_end:
            continue
        ov2 = int((board_before.ends.get(e) or (None, False))[0] or 0)
        nv2 = int(ai_mod.other_value(tile, ov2)) if ai_mod.tile_has(tile, ov2) else ov2
        if nv2 != nv and 0 <= nv2 <= 6:
            avoid[nv2] += 1


def _build_inf_feat235(
    feat193: np.ndarray,
    opp_played_mask: int,
    prefer: List[int],
    avoid: List[int],
) -> np.ndarray:
    out = np.zeros((INF_FEAT_DIM,), np.float32)
    out[0:193] = feat193.astype(np.float32)
    off = 193
    for i in range(28):
        out[off + i] = 1.0 if (opp_played_mask & (1 << i)) != 0 else 0.0
    off += 28
    for v in range(7):
        out[off + v] = float(min(8, int(prefer[v]))) / 8.0
    off += 7
    for v in range(7):
        out[off + v] = float(min(8, int(avoid[v]))) / 8.0
    return out


def _label_opp_hand_mask28(opp_hand_truth: Set[Tile]) -> np.ndarray:
    y = np.zeros((28,), np.int8)
    for t in opp_hand_truth:
        tid = _tile_id(t)
        if 0 <= tid < 28:
            y[tid] = 1
    return y


# ----------------------------
# Truth consistency
# ----------------------------

def assert_consistency(st: GameState, opp_hand: Set[Tile], boneyard: List[Tile]) -> None:
    if int(st.opponent_tile_count) != int(len(opp_hand)):
        raise RuntimeError(f"opp_count mismatch: state={st.opponent_tile_count} truth={len(opp_hand)}")
    if int(st.boneyard_count) != int(len(boneyard)):
        raise RuntimeError(f"bone_count mismatch: state={st.boneyard_count} truth={len(boneyard)}")
    total = int(len(st.my_hand) + len(st.board.played_set) + len(opp_hand) + len(boneyard))
    if total != 28:
        raise RuntimeError(f"Invariant broken: {total} != 28")


# ----------------------------
# Baselines / helpers
# ----------------------------

def best_opening_tile(hand: Set[Tile]) -> Tile:
    doubles = [t for t in hand if tile_is_double(t)]
    if doubles:
        return max(doubles, key=lambda t: t[0])
    return max(hand, key=lambda t: (t[0] + t[1], t[0], t[1]))


def key_open(t: Tile) -> Tuple[int, int, int]:
    return (1 if tile_is_double(t) else 0, t[0] + t[1], t[0])


def legal_moves_for_hand(board: Board, hand: Set[Tile]) -> List[Tuple[Tile, EndName]]:
    out: List[Tuple[Tile, EndName]] = []
    for t in hand:
        for e in board.legal_ends_for_tile(t):
            out.append((t, e))
    return out


def immediate_points(board: Board, tile: Tile, end: EndName) -> int:
    return int(ai_mod.immediate_points(board, tile, end))


def best_opp_reply_points_max(board: Board, hand: Set[Tile]) -> int:
    """
    Existence-based oracle: maximum immediate points available from truth-hand
    (hand-only; does NOT model draw-then-play).
    """
    moves = legal_moves_for_hand(board, hand)
    best = 0
    for t, e in moves:
        pts = int(immediate_points(board, t, e))
        if pts > best:
            best = pts
    return int(best)


def opp_has_win_now_reply(board: Board, hand: Set[Tile], opp_score: int, target: int) -> bool:
    """
    True if opponent has ANY immediate move from truth-hand that reaches target now.
    (hand-only; does NOT model draw-then-play).
    """
    need = int(target) - int(opp_score)
    if need <= 0:
        return False
    for t, e in legal_moves_for_hand(board, hand):
        pts = int(immediate_points(board, t, e))
        if pts > 0 and pts >= need:
            return True
    return False


def _enrich_root_candidates_with_truth(
    board_before: Board,
    opp_hand_truth: Set[Tile],
    opp_score: int,
    target: int,
    root_candidates: Any,
) -> Any:
    """
    Forensics only:
    Add truth-based opponent reply metrics to each root candidate:
      - truth_opp_best_reply_pts
      - truth_opp_win_now_exists
    """
    if not isinstance(root_candidates, list):
        return root_candidates

    out: List[Dict[str, Any]] = []
    for c in root_candidates:
        if not isinstance(c, dict):
            continue
        tile_s = c.get("tile")
        end_s = c.get("end")
        if not isinstance(tile_s, str) or not isinstance(end_s, str) or end_s not in ENDS:
            out.append(dict(c))
            continue

        try:
            t = parse_tile(tile_s)
            b2 = board_before.clone()
            b2.play(t, end_s)
            best_pts = best_opp_reply_points_max(b2, opp_hand_truth)
            wn = opp_has_win_now_reply(b2, opp_hand_truth, int(opp_score), int(target))
            cc = dict(c)
            cc["truth_opp_best_reply_pts"] = int(best_pts)
            cc["truth_opp_win_now_exists"] = bool(wn)
            out.append(cc)
        except Exception:
            out.append(dict(c))
    return out


def has_safe_option_against_truth_win_now(
    board_before: Board,
    legal_before: List[Tuple[Tile, EndName]],
    opp_hand_truth: Set[Tile],
    opp_score: int,
    target: int,
) -> bool:
    """
    True if there exists a legal ME move such that after playing it,
    opponent does NOT have an immediate hand-only win-now from their TRUE hand.
    """
    for (t, e) in legal_before:
        b2 = board_before.clone()
        try:
            b2.play(t, e)
        except Exception:
            continue
        if not opp_has_win_now_reply(b2, opp_hand_truth, int(opp_score), int(target)):
            return True
    return False


def pick_move_greedy(board: Board, hand: Set[Tile]) -> Optional[Tuple[Tile, EndName]]:
    moves = legal_moves_for_hand(board, hand)
    if not moves:
        return None
    best = None
    best_pts = -1
    best_pips = 10**9
    for t, e in moves:
        pts = immediate_points(board, t, e)
        pips = tile_pip_count(t)
        if pts > best_pts or (pts == best_pts and pips < best_pips):
            best = (t, e)
            best_pts = pts
            best_pips = pips
    return best


def pick_move_strategic(board: Board, hand: Set[Tile]) -> Optional[Tuple[Tile, EndName]]:
    moves = legal_moves_for_hand(board, hand)
    if not moves:
        return None
    best = None
    best_sc = -1e18
    for t, e in moves:
        sc = ai_mod.score_opponent_move(board, t, e, hand)
        if sc > best_sc:
            best_sc = sc
            best = (t, e)
    return best


class MatchDumper:
    """
    Collect a replayable script + optional truth/meta for a SINGLE match.
    Script format matches state.rs/apply_script:
      play|me|6-4|right
      draw_me|6-1
      draw|opponent|1|certain
      pass|opponent|certain
      finalize_out|23
      declare_locked|17
      start_new_round|t1,t2,t3,t4,t5,t6,t7
    """
    def __init__(self, match_target: int, first_hand: List[Tile]):
        self.match_target = int(match_target)
        self.first_hand = list(first_hand)
        self.round1_truth: Optional[Dict[str, Any]] = None
        self.script: List[str] = []
        self.me_decisions: List[Dict[str, Any]] = []

    def tile_str(self, t: Tile) -> str:
        return f"{t[0]}-{t[1]}"

    def record_me_decision(self, entry: Dict[str, Any]) -> None:
        try:
            self.me_decisions.append(entry)
        except Exception:
            pass

    def start_new_round(self, my_hand_list: List[Tile]) -> None:
        csv = ",".join(self.tile_str(t) for t in my_hand_list)
        self.script.append(f"start_new_round|{csv}")

    def play(self, player: str, t: Tile, e: EndName) -> None:
        self.script.append(f"play|{player}|{self.tile_str(t)}|{e}")

    def draw_me(self, t: Tile) -> None:
        self.script.append(f"draw_me|{self.tile_str(t)}")

    def draw_opp(self, count: int, certainty: str = "certain") -> None:
        self.script.append(f"draw|opponent|{int(count)}|{certainty}")

    def pass_(self, player: str, certainty: str = "certain") -> None:
        self.script.append(f"pass|{player}|{certainty}")

    def finalize_out(self, opp_pips: int) -> None:
        self.script.append(f"finalize_out|{int(opp_pips)}")

    def declare_locked(self, opp_pips: int) -> None:
        self.script.append(f"declare_locked|{int(opp_pips)}")

    def set_round1_truth(self, my_hand: List[Tile], opp_hand: List[Tile], boneyard_order: List[Tile]) -> None:
        self.round1_truth = {
            "round1_my_hand": [self.tile_str(t) for t in my_hand],
            "round1_opp_hand_truth": [self.tile_str(t) for t in opp_hand],
            # IMPORTANT: evaluator draws via boneyard.pop() (from the end).
            "round1_boneyard_order": [self.tile_str(t) for t in boneyard_order],
            "draw_pop": "end",
        }

    def to_json(self) -> Dict[str, Any]:
        out = {
            "match_target": int(self.match_target),
            "my_hand": [self.tile_str(t) for t in self.first_hand],
            "script": list(self.script),
            "me_decisions": list(self.me_decisions),
        }
        if self.round1_truth is not None:
            out.update(self.round1_truth)
        return out


def best_opp_reply_points(board: Board, opp_hand: Set[Tile], policy: str) -> int:
    mv = pick_move_strategic(board, opp_hand) if policy == "strategic" else pick_move_greedy(board, opp_hand)
    if mv is None:
        return 0
    t, e = mv
    return int(immediate_points(board, t, e))


def pick_me_move_model(st: GameState) -> Tuple[Tile, EndName]:
    legal = st.legal_moves_me()
    if not legal:
        raise RuntimeError("me has no legal moves")

    feats = ai_mod.features_small(st)
    mask = ai_mod.legal_mask_state(st)
    policy, _v = ai_mod.model_predict(feats, mask)

    best_idx = None
    best_p = -1.0
    for (t, e) in legal:
        idx = ai_mod.encode_action(t, e)
        p = float(policy[idx])
        if p > best_p:
            best_p = p
            best_idx = (t, e)

    return best_idx if best_idx is not None else legal[0]


def _clear_last_rust_meta(st: GameState) -> None:
    try:
        st._last_rust_res = None  # type: ignore[attr-defined]
    except Exception:
        pass


def pick_me_move_rust(
    st: GameState,
    rng: random.Random,
    det: int,
    think_ms: int,
    rust_opp_mix_greedy: float = 1.0,
    rust_me_mix_greedy: float = 1.0,
    rust_leaf_value_weight: float = 0.0,
    rust_gift_penalty_weight: float = 0.0,
    rust_pessimism_alpha_max: float = 0.0,
    return_root_stats: bool = False,
    root_stats_top_k: int = 12,
) -> Tuple[Tile, EndName]:
    if domino_rs is None:
        raise RuntimeError("domino_rs not available (install/build Rust extension first)")

    sd = st.to_dict()

    res = domino_rs.suggest_move_ismcts(  # type: ignore[attr-defined]
        sd,
        det=int(det),
        think_ms=int(think_ms),
        seed=int(rng.getrandbits(64) or 1),
        model_path=None,  # default: model.json
        opp_mix_greedy=float(rust_opp_mix_greedy),
        leaf_value_weight=float(rust_leaf_value_weight),
        me_mix_greedy=float(rust_me_mix_greedy),
        gift_penalty_weight=float(rust_gift_penalty_weight),
        pessimism_alpha_max=float(rust_pessimism_alpha_max),
        enable_guard=True,          # Rust may ignore this if spike path is active
        guard_top_k=25,
        guard_worlds=64,
        guard_close_threshold=50,
        return_root_stats=bool(return_root_stats),
        root_stats_top_k=int(root_stats_top_k),
    )

    try:
        st._last_rust_res = res  # type: ignore[attr-defined]
    except Exception:
        pass

    tile_s = res.get("tile")
    end_s = res.get("end")
    if not isinstance(tile_s, str) or not isinstance(end_s, str):
        _clear_last_rust_meta(st)
        return pick_me_move_model(st)

    if end_s not in ENDS:
        _clear_last_rust_meta(st)
        return pick_me_move_model(st)

    try:
        return parse_tile(tile_s), end_s
    except Exception:
        _clear_last_rust_meta(st)
        return pick_me_move_model(st)


def pick_me_move_ai(st: GameState, rng: random.Random, analysis_level: str, det: int, think_ms: int) -> Tuple[Tile, EndName]:
    legal = st.legal_moves_me()
    if not legal:
        raise RuntimeError("me has no legal moves")

    out = ai_mod.suggest_moves(
        st,
        top_n=1,
        determinizations=int(det),
        seed=int(rng.randrange(1, 2**31 - 1)),
        analysis_level=str(analysis_level),
        think_ms=int(think_ms),
    )
    sug = (out.get("suggestions") or [])
    if sug:
        try:
            t = ai_mod.parse_tile(sug[0]["tile"])
            e = sug[0]["end"]
            if e in ENDS:
                return t, e
        except Exception:
            pass

    return max(legal, key=lambda m: immediate_points(st.board, m[0], m[1]))


@dataclass
class EvalConfig:
    matches: int = 2000
    target: int = 150
    base_seed: int = 12345

    me_policy: Literal["model", "ai", "greedy", "rust"] = "model"
    analysis_level: str = "quick"
    det: int = 8
    think_ms: int = 120

    rust_opp_mix_greedy: float = 1.0
    rust_me_mix_greedy: float = 1.0
    rust_leaf_value_weight: float = 0.0
    rust_gift_penalty_weight: float = 0.0
    rust_pessimism_alpha_max: float = 0.0

    opp_policy: Literal["greedy", "strategic"] = "greedy"

    max_rounds_per_match: int = 200
    max_plies_per_round: int = 500
    strict_asserts: bool = True
    assert_every: int = 1

    # dump first match (optional)
    dump_match_json: Optional[str] = None
    dump_match_script: Optional[str] = None
    dump_match_text: Optional[str] = None

    # IM4.2: optional inference dataset output directory (INF1 raw)
    infer_out_dir: Optional[str] = None
    # internal writer (not serialized)
    _infer_writer: Any = None
    _infer_errs: int = 0


def finalize_if_out_me_pending(st: GameState, opp_hand: Set[Tile], dumper: Optional[MatchDumper] = None) -> None:
    if st.round_over and st.round_end_reason == "out_me_pending":
        opp_pips = sum(tile_pip_count(t) for t in opp_hand)
        if dumper is not None:
            dumper.finalize_out(int(opp_pips))
        st.finalize_out_with_opponent_pips(int(opp_pips))


def maybe_declare_locked(st: GameState, opp_hand: Set[Tile], dumper: Optional[MatchDumper] = None) -> bool:
    if int(st.boneyard_count) != 0:
        return False
    me_no = (len(st.legal_moves_me()) == 0)
    opp_no = (pick_move_greedy(st.board, opp_hand) is None)
    if not (me_no and opp_no):
        return False

    st.current_turn = "opponent"
    if dumper is not None:
        dumper.pass_("opponent", "certain")
    st.record_pass("opponent", certainty="certain")  # flips to me

    st.current_turn = "me"
    if not st.must_pass_me():
        return False

    opp_pips = sum(tile_pip_count(t) for t in opp_hand)
    if dumper is not None:
        dumper.declare_locked(int(opp_pips))
    st.declare_locked(int(opp_pips))
    return True


def _pick_me_move_by_policy(st: GameState, rng: random.Random, cfg: EvalConfig) -> Tuple[Tile, EndName]:
    if cfg.me_policy != "rust":
        _clear_last_rust_meta(st)

    if cfg.me_policy == "rust":
        want = bool(getattr(st, "_want_root_stats", False))
        return pick_me_move_rust(
            st,
            rng,
            cfg.det,
            cfg.think_ms,
            rust_opp_mix_greedy=float(cfg.rust_opp_mix_greedy),
            rust_me_mix_greedy=float(cfg.rust_me_mix_greedy),
            rust_leaf_value_weight=float(cfg.rust_leaf_value_weight),
            rust_gift_penalty_weight=float(cfg.rust_gift_penalty_weight),
            rust_pessimism_alpha_max=float(cfg.rust_pessimism_alpha_max),
            return_root_stats=want,
            root_stats_top_k=12,
        )
    if cfg.me_policy == "ai":
        return pick_me_move_ai(st, rng, cfg.analysis_level, cfg.det, cfg.think_ms)
    if cfg.me_policy == "greedy":
        mv = pick_move_greedy(st.board, set(st.my_hand))
        if mv is not None:
            return mv
        legal = st.legal_moves_me()
        return legal[0]
    return pick_me_move_model(st)


def _extract_spike_fields(res: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(res, dict):
        return None
    out: Dict[str, Any] = {}
    out["model_has_spike"] = bool(res.get("model_has_spike")) if "model_has_spike" in res else None
    out["spike_used"] = bool(res.get("spike_used")) if "spike_used" in res else None
    out["spike_p_chosen"] = res.get("spike_p_chosen", None)
    out["spike_p_best_visit"] = res.get("spike_p_best_visit", None)
    out["spike_safe_exists"] = res.get("spike_safe_exists", None)
    out["chosen_by"] = res.get("chosen_by", None)
    return out


def play_one_round(
    st: GameState,
    rng: random.Random,
    cfg: EvalConfig,
    first_round: bool,
    strict_asserts: bool,
    dumper: Optional[MatchDumper] = None,
) -> Tuple[List[Tile], List[Tile], List[Tile]]:
    tiles = list(ALL_TILES)
    rng.shuffle(tiles)

    my_hand_list = [tiles[i] for i in range(7)]
    opp_hand_list = [tiles[i] for i in range(7, 14)]
    opp_hand: Set[Tile] = set(opp_hand_list)
    boneyard: List[Tile] = list(tiles[14:])
    boneyard_init: List[Tile] = list(boneyard)

    prev_reason = st.round_end_reason

    if first_round:
        st.start_new_game(my_hand_list, match_target=int(cfg.target))
        prev_reason = None
    else:
        st.start_new_round(my_hand_list)
        if dumper is not None:
            dumper.start_new_round(my_hand_list)

    _clear_last_rust_meta(st)

    # enable root stats only when dumping match (first match only)
    try:
        st._want_root_stats = bool(dumper is not None)  # type: ignore[attr-defined]
    except Exception:
        pass

    if strict_asserts:
        assert_consistency(st, opp_hand, boneyard)

    opening_free = (not first_round) and (prev_reason in ("out_me", "out_opponent"))

    if opening_free:
        opener: str = "me" if prev_reason == "out_me" else "opponent"

        if opener == "me":
            st.current_turn = "me"
            t, e = _pick_me_move_by_policy(st, rng, cfg)
            if dumper is not None:
                dumper.play("me", t, "right")
            st.play_tile("me", t, "right")
            if strict_asserts:
                assert_consistency(st, opp_hand, boneyard)
            finalize_if_out_me_pending(st, opp_hand, dumper)
        else:
            st.current_turn = "opponent"
            mv = pick_move_strategic(st.board, opp_hand) if cfg.opp_policy == "strategic" else pick_move_greedy(st.board, opp_hand)
            if mv is None:
                mv = legal_moves_for_hand(st.board, opp_hand)[0]
            t, e = mv
            opp_hand.remove(t)
            if dumper is not None:
                dumper.play("opponent", t, "right")
            st.play_tile("opponent", t, "right")
            if strict_asserts:
                assert_consistency(st, opp_hand, boneyard)
    else:
        my_open = best_opening_tile(set(my_hand_list))
        opp_open = best_opening_tile(set(opp_hand))

        if key_open(opp_open) > key_open(my_open):
            st.current_turn = "opponent"
            opp_hand.remove(opp_open)
            if dumper is not None:
                dumper.play("opponent", opp_open, "right")
            st.play_tile("opponent", opp_open, "right")
        else:
            st.current_turn = "me"
            if dumper is not None:
                dumper.play("me", my_open, "right")
            st.play_tile("me", my_open, "right")

        if strict_asserts:
            assert_consistency(st, opp_hand, boneyard)

    # ----------------------------
    # IM4.2: inference dataset trackers (opponent signals within current round)
    # ----------------------------
    opp_played_mask: int = 0
    opp_endpref: List[int] = [0] * 7
    opp_endavoid: List[int] = [0] * 7
    prev_was_opp_draw: bool = False

    plies = 0
    while not st.round_over and plies < int(cfg.max_plies_per_round):
        plies += 1

        if strict_asserts:
            assert_consistency(st, opp_hand, boneyard)

        if maybe_declare_locked(st, opp_hand, dumper):
            break
        if st.round_over:
            break

        if st.current_turn == "me":
            forced_play_done = False

            while st.must_draw_me():
                if not boneyard:
                    break
                drawn = boneyard.pop()
                if dumper is not None:
                    dumper.draw_me(drawn)
                st.record_draw_me(drawn)
                if strict_asserts:
                    assert_consistency(st, opp_hand, boneyard)

                if st.forced_play_tile is not None:
                    ft = st.forced_play_tile
                    ends = st.board.legal_ends_for_tile(ft)
                    if ends:
                        chosen_end = max(ends, key=lambda e: immediate_points(st.board, ft, e))
                        _clear_last_rust_meta(st)

                        if dumper is not None:
                            dumper.play("me", ft, chosen_end)
                        st.play_tile("me", ft, chosen_end)
                        if strict_asserts:
                            assert_consistency(st, opp_hand, boneyard)
                        finalize_if_out_me_pending(st, opp_hand, dumper)
                        forced_play_done = True
                        break

            if forced_play_done:
                continue

            if st.round_over:
                finalize_if_out_me_pending(st, opp_hand, dumper)
                break

            if st.must_pass_me():
                _clear_last_rust_meta(st)
                if dumper is not None:
                    dumper.pass_("me", "certain")
                st.record_pass("me", certainty="certain")
                continue

            legal = st.legal_moves_me()
            if not legal:
                _clear_last_rust_meta(st)
                if dumper is not None:
                    dumper.pass_("me", "certain")
                st.record_pass("me", certainty="certain")
                continue

            # --- snapshot before choosing the move (truth-safe analysis) ---
            pre_board = st.board.clone()
            pre_legal = list(st.legal_moves_me())
            pre_forced = (st.forced_play_tile is not None)
            pre_bone = int(st.boneyard_count)
            pre_opp_score = int(st.opponent_score)

            # choose move
            t, e = _pick_me_move_by_policy(st, rng, cfg)

            # IM4.2: record inference sample at decision-time (only when Rust decides a move)
            # This is open-loop collection: gameplay is unchanged.
            if getattr(st, "_infer_writer", None) is not None and cfg.me_policy == "rust":
                try:
                    iw: Inf1Writer = getattr(st, "_infer_writer")  # type: ignore[assignment]
                    sd_full = st.to_dict()
                    # slice events to current round for feature parity
                    sd_round, _ = _slice_events_current_round(sd_full)
                    if domino_rs is None:
                        raise RuntimeError("domino_rs required for inference dataset (features_from_state_dict)")
                    feat_b, _mask_b = domino_rs.features_from_state_dict(sd_round)  # type: ignore[attr-defined]
                    feat193 = np.frombuffer(feat_b, dtype=np.float32, count=193).copy()
                    inf_feat = _build_inf_feat235(feat193, opp_played_mask, opp_endpref, opp_endavoid)
                    label = _label_opp_hand_mask28(set(opp_hand))
                    iw.write(inf_feat, label, int(st.opponent_tile_count), int(st.boneyard_count))
                except Exception:
                    # dataset collection must never break evaluation
                    cfg._infer_errs += 1

            # If dumping and rust policy, capture root candidates BEFORE playing
            if dumper is not None and cfg.me_policy == "rust":
                rust_meta = getattr(st, "_last_rust_res", None)
                board_before = pre_board.clone()
                opp_score_before = int(pre_opp_score)
                try:
                    entry: Dict[str, Any] = {
                        "ply": int(st.ply()),
                        "my_score": int(st.my_score),
                        "opp_score": int(st.opponent_score),
                        "opp_cnt": int(st.opponent_tile_count),
                        "bone_cnt": int(st.boneyard_count),
                        "ends_sum": int(st.board.ends_sum()),
                        "score_now": int(st.board.score_now()),
                        "my_hand": [f"{x[0]}-{x[1]}" for x in sorted(list(st.my_hand))],
                        "chosen": {"tile": f"{t[0]}-{t[1]}", "end": e},
                    }
                    if isinstance(rust_meta, dict):
                        for k in ("chosen_by", "spike_p_chosen", "spike_p_best_visit", "spike_safe_exists", "model_has_spike", "spike_used"):
                            if k in rust_meta:
                                entry[k] = rust_meta.get(k)
                        if "root_candidates" in rust_meta:
                            entry["root_candidates"] = _enrich_root_candidates_with_truth(
                                board_before=board_before,
                                opp_hand_truth=set(opp_hand),
                                opp_score=int(opp_score_before),
                                target=int(cfg.target),
                                root_candidates=rust_meta.get("root_candidates"),
                            )
                    dumper.record_me_decision(entry)
                except Exception:
                    pass

            # play it
            if dumper is not None:
                dumper.play("me", t, e)
            st.play_tile("me", t, e)
            if strict_asserts:
                assert_consistency(st, opp_hand, boneyard)
            finalize_if_out_me_pending(st, opp_hand, dumper)

            # --- init counters on first usage ---
            if not hasattr(st, "_m_me_moves_total"):
                st._m_me_moves_total = 0
                st._m_gift10 = 0
                st._m_gift15 = 0
                st._m_gift_win_now = 0
                st._m_gift_win_now_exists = 0
                st._m_end_me_moves = 0
                st._m_end_gift10 = 0
                st._m_end_gift15 = 0
                st._m_end_gift_win_now = 0
                st._m_end_gift_win_now_exists = 0
                st._m_sum_best_opp_reply_pts = 0
                st._m_end_wn_total = 0
                st._m_end_wn_bone0 = 0
                st._m_end_wn_forced = 0
                st._m_end_wn_single = 0
                st._m_end_wn_safeopt = 0

                # Spike telemetry counters
                st._m_spike_moves = 0
                st._m_spike_p_sum = 0.0
                st._m_spike_best_p_sum = 0.0
                st._m_spike_safe_exists = 0
                st._m_spike_tp = 0
                st._m_spike_fp = 0
                st._m_spike_tn = 0
                st._m_spike_fn = 0

            # If round ended by target/out, skip (no opponent reply exists).
            if not st.round_over:
                pts = best_opp_reply_points(st.board, opp_hand, cfg.opp_policy)
                win_now_exists = opp_has_win_now_reply(st.board, opp_hand, int(st.opponent_score), int(cfg.target))

                st._m_me_moves_total += 1
                st._m_sum_best_opp_reply_pts += int(pts)

                if pts >= 10:
                    st._m_gift10 += 1
                if pts >= 15:
                    st._m_gift15 += 1
                if int(st.opponent_score) + int(pts) >= int(cfg.target) and pts > 0:
                    st._m_gift_win_now += 1
                if win_now_exists:
                    st._m_gift_win_now_exists += 1

                # -------- Spike telemetry (only if rust meta exists + has fields) --------
                rust_meta = getattr(st, "_last_rust_res", None)
                sp = _extract_spike_fields(rust_meta)

                if sp is not None:
                    p_ch = sp.get("spike_p_chosen", None)
                    p_bv = sp.get("spike_p_best_visit", None)
                    safe_exists = sp.get("spike_safe_exists", None)
                    if isinstance(p_ch, (int, float)):
                        pch = float(p_ch)
                        pbv = float(p_bv) if isinstance(p_bv, (int, float)) else 0.0
                        pred_unsafe = (pch >= 0.5)
                        truth_unsafe = bool(win_now_exists)

                        st._m_spike_moves += 1
                        st._m_spike_p_sum += float(pch)
                        st._m_spike_best_p_sum += float(pbv)
                        if safe_exists is True:
                            st._m_spike_safe_exists += 1

                        if pred_unsafe and truth_unsafe:
                            st._m_spike_tp += 1
                        elif pred_unsafe and (not truth_unsafe):
                            st._m_spike_fp += 1
                        elif (not pred_unsafe) and truth_unsafe:
                            st._m_spike_fn += 1
                        else:
                            st._m_spike_tn += 1

                _clear_last_rust_meta(st)

                danger = (max(int(st.my_score), int(st.opponent_score)) >= 130) or (int(cfg.target) - int(st.opponent_score) <= 20)
                if danger:
                    st._m_end_me_moves += 1
                    if pts >= 10:
                        st._m_end_gift10 += 1
                    if pts >= 15:
                        st._m_end_gift15 += 1
                    if int(st.opponent_score) + int(pts) >= int(cfg.target) and pts > 0:
                        st._m_end_gift_win_now += 1

                    if win_now_exists:
                        st._m_end_gift_win_now_exists += 1
                        st._m_end_wn_total += 1
                        if pre_bone == 0:
                            st._m_end_wn_bone0 += 1
                        if pre_forced:
                            st._m_end_wn_forced += 1
                        if len(pre_legal) == 1:
                            st._m_end_wn_single += 1
                        if has_safe_option_against_truth_win_now(pre_board, pre_legal, opp_hand, pre_opp_score, int(cfg.target)):
                            st._m_end_wn_safeopt += 1
            else:
                _clear_last_rust_meta(st)

            continue

        # opponent turn
        mv = pick_move_strategic(st.board, opp_hand) if cfg.opp_policy == "strategic" else pick_move_greedy(st.board, opp_hand)

        if mv is not None:
            t, e = mv
            # IM4.2: update opponent signals before applying the move (board_before is needed)
            board_before = st.board.clone()
            opp_hand.remove(t)
            if dumper is not None:
                dumper.play("opponent", t, e)
            st.play_tile("opponent", t, e)
            # signals
            try:
                opp_played_mask |= 1 << _tile_id(t)
                _update_endchoice_from_play(board_before, t, e, opp_endpref, opp_endavoid, prev_was_opp_draw)
            except Exception:
                pass
            prev_was_opp_draw = False
            if strict_asserts:
                assert_consistency(st, opp_hand, boneyard)
            continue

        # no move: draw until play or pass if boneyard empty
        drew_any = False
        played_after_draw = False
        while boneyard and not st.round_over:
            drew_any = True
            drawn = boneyard.pop()
            opp_hand.add(drawn)
            if dumper is not None:
                dumper.draw_opp(1, "certain")
            st.record_draw("opponent", count=1, certainty="certain")
            if strict_asserts:
                assert_consistency(st, opp_hand, boneyard)
            prev_was_opp_draw = True

            ends = st.board.legal_ends_for_tile(drawn)
            if ends:
                best_end = max(ends, key=lambda e: immediate_points(st.board, drawn, e))
                # IM4.2: signals: forced play after draw => skip endchoice update, but still counts as played tile
                board_before = st.board.clone()
                opp_hand.remove(drawn)
                if dumper is not None:
                    dumper.play("opponent", drawn, best_end)
                st.play_tile("opponent", drawn, best_end)
                try:
                    opp_played_mask |= 1 << _tile_id(drawn)
                except Exception:
                    pass
                prev_was_opp_draw = False
                if strict_asserts:
                    assert_consistency(st, opp_hand, boneyard)
                played_after_draw = True
                break

        if played_after_draw:
            continue

        if st.round_over:
            break

        if (not drew_any) or (not boneyard):
            if dumper is not None:
                dumper.pass_("opponent", "certain")
            st.record_pass("opponent", certainty="certain")
            if strict_asserts:
                assert_consistency(st, opp_hand, boneyard)

    finalize_if_out_me_pending(st, opp_hand, dumper)
    _clear_last_rust_meta(st)
    try:
        st._want_root_stats = False  # type: ignore[attr-defined]
    except Exception:
        pass
    if dumper is not None and first_round:
        dumper.set_round1_truth(my_hand_list, opp_hand_list, boneyard_init)
    return my_hand_list, opp_hand_list, boneyard_init


def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
    wins = losses = ties = 0
    margins: List[int] = []
    rounds_used: List[int] = []

    t0 = time.perf_counter()

    me_moves_total = 0
    gift10 = 0
    gift15 = 0
    gift_win_now = 0
    gift_win_now_exists = 0
    end_me_moves = 0
    end_gift10 = 0
    end_gift15 = 0
    end_gift_win_now = 0
    end_gift_win_now_exists = 0
    sum_best_opp_reply_pts = 0

    end_wn_total = 0
    end_wn_bone0 = 0
    end_wn_forced = 0
    end_wn_single = 0
    end_wn_safeopt = 0

    spike_moves = 0
    spike_p_sum = 0.0
    spike_best_p_sum = 0.0
    spike_safe_exists = 0
    spike_tp = spike_fp = spike_tn = spike_fn = 0

    assert_every = int(cfg.assert_every)
    if assert_every < 0:
        assert_every = 0
    asserted_matches = 0

    dump_json = cfg.dump_match_json
    dump_script = cfg.dump_match_script
    dump_text = cfg.dump_match_text
    dumper: Optional[MatchDumper] = None

    for i in range(int(cfg.matches)):
        rng = random.Random(int(cfg.base_seed + i * 10007))
        st = GameState()
        # IM4.2: attach optional inference writer to state for collection
        try:
            setattr(st, "_infer_writer", cfg._infer_writer)  # type: ignore[attr-defined]
        except Exception:
            pass

        do_assert = bool(cfg.strict_asserts) and (assert_every > 0) and (i % assert_every == 0)
        if do_assert:
            asserted_matches += 1

        rounds = 0
        first = True
        while int(st.my_score) < int(cfg.target) and int(st.opponent_score) < int(cfg.target):
            rounds += 1
            if dumper is None and i == 0 and (dump_json or dump_script or dump_text):
                first_hand, _opp_hand_list, _boneyard_init = play_one_round(st, rng, cfg, first_round=True, strict_asserts=do_assert, dumper=None)
                dumper = MatchDumper(match_target=int(cfg.target), first_hand=first_hand)
                rng = random.Random(int(cfg.base_seed + i * 10007))
                st = GameState()
                do_assert = bool(cfg.strict_asserts) and (assert_every > 0) and (i % assert_every == 0)
                play_one_round(st, rng, cfg, first_round=True, strict_asserts=do_assert, dumper=dumper)
                first = False
                continue

            play_one_round(st, rng, cfg, first_round=first, strict_asserts=do_assert, dumper=dumper if i == 0 else None)
            first = False
            if rounds >= int(cfg.max_rounds_per_match):
                break

        me_moves_total += int(getattr(st, "_m_me_moves_total", 0))
        gift10 += int(getattr(st, "_m_gift10", 0))
        gift15 += int(getattr(st, "_m_gift15", 0))
        gift_win_now += int(getattr(st, "_m_gift_win_now", 0))
        gift_win_now_exists += int(getattr(st, "_m_gift_win_now_exists", 0))
        end_me_moves += int(getattr(st, "_m_end_me_moves", 0))
        end_gift10 += int(getattr(st, "_m_end_gift10", 0))
        end_gift15 += int(getattr(st, "_m_end_gift15", 0))
        end_gift_win_now += int(getattr(st, "_m_end_gift_win_now", 0))
        end_gift_win_now_exists += int(getattr(st, "_m_end_gift_win_now_exists", 0))
        sum_best_opp_reply_pts += int(getattr(st, "_m_sum_best_opp_reply_pts", 0))

        end_wn_total += int(getattr(st, "_m_end_wn_total", 0))
        end_wn_bone0 += int(getattr(st, "_m_end_wn_bone0", 0))
        end_wn_forced += int(getattr(st, "_m_end_wn_forced", 0))
        end_wn_single += int(getattr(st, "_m_end_wn_single", 0))
        end_wn_safeopt += int(getattr(st, "_m_end_wn_safeopt", 0))

        spike_moves += int(getattr(st, "_m_spike_moves", 0))
        spike_p_sum += float(getattr(st, "_m_spike_p_sum", 0.0))
        spike_best_p_sum += float(getattr(st, "_m_spike_best_p_sum", 0.0))
        spike_safe_exists += int(getattr(st, "_m_spike_safe_exists", 0))
        spike_tp += int(getattr(st, "_m_spike_tp", 0))
        spike_fp += int(getattr(st, "_m_spike_fp", 0))
        spike_tn += int(getattr(st, "_m_spike_tn", 0))
        spike_fn += int(getattr(st, "_m_spike_fn", 0))

        rounds_used.append(rounds)
        margin = int(st.my_score - st.opponent_score)
        margins.append(margin)

        me_win = int(st.my_score) >= int(cfg.target) and int(st.opponent_score) < int(cfg.target)
        opp_win = int(st.opponent_score) >= int(cfg.target) and int(st.my_score) < int(cfg.target)

        if me_win:
            wins += 1
        elif opp_win:
            losses += 1
        else:
            ties += 1

        # Dump first match after it finishes
        if i == 0 and (dump_json or dump_script or dump_text):
            try:
                if dump_json:
                    Path(dump_json).write_text(json.dumps(st.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                if dump_script and dumper is not None:
                    Path(dump_script).write_text(json.dumps(dumper.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
                if dump_text:
                    d = st.to_dict()
                    lines: List[str] = []
                    lines.append("=== MATCH DUMP ===")
                    lines.append(json.dumps(d.get("meta", {}), ensure_ascii=False))
                    lines.append("")
                    lines.append("=== EVENTS ===")
                    for ev in (d.get("events") or []):
                        lines.append(json.dumps(ev, ensure_ascii=False))
                    Path(dump_text).write_text("\n".join(lines), encoding="utf-8")
                print(f"[dump] wrote match_json={dump_json} script={dump_script} text={dump_text}", flush=True)
            except Exception as e:
                print(f"[dump] failed: {e}", flush=True)

    elapsed = float(time.perf_counter() - t0)
    n = int(cfg.matches) or 1
    tie_adjusted = (float(wins) + 0.5 * float(ties)) / float(n)
    den_no_ties = float(wins + losses)
    win_no_ties = (float(wins) / den_no_ties) if den_no_ties > 0 else 0.0

    denom = float(me_moves_total) if me_moves_total > 0 else 1.0
    end_denom = float(end_me_moves) if end_me_moves > 0 else 1.0

    spike_denom = float(spike_moves) if spike_moves > 0 else 1.0
    spike_conf_total = float(spike_tp + spike_fp + spike_tn + spike_fn) if (spike_tp + spike_fp + spike_tn + spike_fn) > 0 else 1.0

    return {
        "ok": True,
        "config": {
            "matches": cfg.matches,
            "target": cfg.target,
            "base_seed": cfg.base_seed,
            "me_policy": cfg.me_policy,
            "analysis_level": cfg.analysis_level,
            "det": cfg.det,
            "think_ms": cfg.think_ms,
            "rust_opp_mix_greedy": float(cfg.rust_opp_mix_greedy),
            "rust_me_mix_greedy": float(cfg.rust_me_mix_greedy),
            "rust_leaf_value_weight": float(cfg.rust_leaf_value_weight),
            "rust_gift_penalty_weight": float(cfg.rust_gift_penalty_weight),
            "rust_pessimism_alpha_max": float(cfg.rust_pessimism_alpha_max),
            "opp_policy": cfg.opp_policy,
            "strict_asserts": cfg.strict_asserts,
            "assert_every": int(cfg.assert_every),
            "ai_file": getattr(ai_mod, "__file__", ""),
        },
        "results": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(float(wins) / float(n), 4),
            "tie_adjusted": round(tie_adjusted, 4),
            "win_no_ties": round(win_no_ties, 4),
            "avg_margin": round(float(sum(margins)) / float(n), 2),
            "avg_rounds_per_match": round(float(sum(rounds_used)) / float(n), 2),
            "elapsed_sec": round(elapsed, 3),
            "asserted_matches": int(asserted_matches),

            "me_moves_total": int(me_moves_total),
            "gift10_rate": round(float(gift10) / denom, 4),
            "gift15_rate": round(float(gift15) / denom, 4),
            "gift_win_now_rate": round(float(gift_win_now) / denom, 4),
            "gift_win_now_exists_rate": round(float(gift_win_now_exists) / denom, 4),
            "avg_best_opp_reply_pts_after_me": round(float(sum_best_opp_reply_pts) / denom, 3),

            "endgame_me_moves": int(end_me_moves),
            "endgame_gift10_rate": round(float(end_gift10) / end_denom, 4),
            "endgame_gift15_rate": round(float(end_gift15) / end_denom, 4),
            "endgame_gift_win_now_rate": round(float(end_gift_win_now) / end_denom, 4),
            "endgame_gift_win_now_exists_rate": round(float(end_gift_win_now_exists) / end_denom, 4),

            "endgame_wn_total": int(end_wn_total),
            "endgame_wn_bone0_rate": round(float(end_wn_bone0) / float(max(1, end_wn_total)), 4),
            "endgame_wn_forced_rate": round(float(end_wn_forced) / float(max(1, end_wn_total)), 4),
            "endgame_wn_single_legal_rate": round(float(end_wn_single) / float(max(1, end_wn_total)), 4),
            "endgame_wn_safe_option_rate": round(float(end_wn_safeopt) / float(max(1, end_wn_total)), 4),

            "spike_moves": int(spike_moves),
            "spike_used_rate": round(float(spike_moves) / float(max(1, me_moves_total)), 4),
            "avg_spike_p_chosen": round(float(spike_p_sum) / spike_denom, 4),
            "avg_spike_p_best_visit": round(float(spike_best_p_sum) / spike_denom, 4),
            "spike_safe_exists_rate": round(float(spike_safe_exists) / spike_denom, 4),

            "spike_tp": int(spike_tp),
            "spike_fp": int(spike_fp),
            "spike_tn": int(spike_tn),
            "spike_fn": int(spike_fn),
            "spike_fn_rate": round(float(spike_fn) / spike_conf_total, 4),
            "spike_fp_rate": round(float(spike_fp) / spike_conf_total, 4),
        }
    }


def _merge_reports(reps: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins = losses = ties = 0
    asserted_matches = 0
    margins_sum = 0.0
    rounds_sum = 0.0
    elapsed_sum = 0.0

    me_moves_total = 0
    gift10 = 0
    gift15 = 0
    gift_win_now = 0
    gift_win_now_exists = 0
    sum_best_opp_reply_pts = 0

    endgame_me_moves = 0
    endgame_gift10 = 0
    endgame_gift15 = 0
    endgame_gift_win_now = 0
    endgame_gift_win_now_exists = 0

    endgame_wn_total = 0
    endgame_wn_bone0 = 0
    endgame_wn_forced = 0
    endgame_wn_single = 0
    endgame_wn_safeopt = 0

    spike_moves = 0
    spike_p_sum = 0.0
    spike_best_p_sum = 0.0
    spike_safe_exists = 0
    spike_tp = spike_fp = spike_tn = spike_fn = 0

    total_matches = 0
    base_cfg = None

    for r in reps:
        if not r.get("ok", False):
            continue
        if base_cfg is None:
            base_cfg = r.get("config", {})
        res = r.get("results", {})
        n = int((r.get("config") or {}).get("matches", 0) or 0)
        total_matches += n

        wins += int(res.get("wins", 0))
        losses += int(res.get("losses", 0))
        ties += int(res.get("ties", 0))
        asserted_matches += int(res.get("asserted_matches", 0))

        margins_sum += float(res.get("avg_margin", 0.0)) * float(n)
        rounds_sum += float(res.get("avg_rounds_per_match", 0.0)) * float(n)
        elapsed_sum += float(res.get("elapsed_sec", 0.0))

        mm = int(res.get("me_moves_total", 0))
        me_moves_total += mm
        gift10 += int(round(float(res.get("gift10_rate", 0.0)) * float(mm)))
        gift15 += int(round(float(res.get("gift15_rate", 0.0)) * float(mm)))
        gift_win_now += int(round(float(res.get("gift_win_now_rate", 0.0)) * float(mm)))
        gift_win_now_exists += int(round(float(res.get("gift_win_now_exists_rate", 0.0)) * float(mm)))
        sum_best_opp_reply_pts += int(round(float(res.get("avg_best_opp_reply_pts_after_me", 0.0)) * float(mm)))

        em = int(res.get("endgame_me_moves", 0))
        endgame_me_moves += em
        endgame_gift10 += int(round(float(res.get("endgame_gift10_rate", 0.0)) * float(em)))
        endgame_gift15 += int(round(float(res.get("endgame_gift15_rate", 0.0)) * float(em)))
        endgame_gift_win_now += int(round(float(res.get("endgame_gift_win_now_rate", 0.0)) * float(em)))
        endgame_gift_win_now_exists += int(round(float(res.get("endgame_gift_win_now_exists_rate", 0.0)) * float(em)))

        ewt = int(res.get("endgame_wn_total", 0))
        endgame_wn_total += ewt
        endgame_wn_bone0 += int(round(float(res.get("endgame_wn_bone0_rate", 0.0)) * float(ewt)))
        endgame_wn_forced += int(round(float(res.get("endgame_wn_forced_rate", 0.0)) * float(ewt)))
        endgame_wn_single += int(round(float(res.get("endgame_wn_single_legal_rate", 0.0)) * float(ewt)))
        endgame_wn_safeopt += int(round(float(res.get("endgame_wn_safe_option_rate", 0.0)) * float(ewt)))

        spike_moves += int(res.get("spike_moves", 0))
        spike_p_sum += float(res.get("avg_spike_p_chosen", 0.0)) * float(max(1, int(res.get("spike_moves", 0))))
        spike_best_p_sum += float(res.get("avg_spike_p_best_visit", 0.0)) * float(max(1, int(res.get("spike_moves", 0))))
        spike_safe_exists += int(round(float(res.get("spike_safe_exists_rate", 0.0)) * float(max(1, int(res.get("spike_moves", 0))))))

        spike_tp += int(res.get("spike_tp", 0))
        spike_fp += int(res.get("spike_fp", 0))
        spike_tn += int(res.get("spike_tn", 0))
        spike_fn += int(res.get("spike_fn", 0))

    n = total_matches if total_matches > 0 else 1
    tie_adjusted = (float(wins) + 0.5 * float(ties)) / float(n)
    den_no_ties = float(wins + losses)
    win_no_ties = (float(wins) / den_no_ties) if den_no_ties > 0 else 0.0

    denom = float(me_moves_total) if me_moves_total > 0 else 1.0
    end_denom = float(endgame_me_moves) if endgame_me_moves > 0 else 1.0
    spike_denom = float(spike_moves) if spike_moves > 0 else 1.0
    spike_conf_total = float(spike_tp + spike_fp + spike_tn + spike_fn) if (spike_tp + spike_fp + spike_tn + spike_fn) > 0 else 1.0

    return {
        "ok": True,
        "config": dict(base_cfg or {}),
        "results": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(float(wins) / float(n), 4),
            "tie_adjusted": round(float(tie_adjusted), 4),
            "win_no_ties": round(float(win_no_ties), 4),
            "avg_margin": round(float(margins_sum) / float(n), 2),
            "avg_rounds_per_match": round(float(rounds_sum) / float(n), 2),
            "elapsed_sec": round(float(elapsed_sum), 3),
            "asserted_matches": int(asserted_matches),

            "me_moves_total": int(me_moves_total),
            "gift10_rate": round(float(gift10) / denom, 4),
            "gift15_rate": round(float(gift15) / denom, 4),
            "gift_win_now_rate": round(float(gift_win_now) / denom, 4),
            "gift_win_now_exists_rate": round(float(gift_win_now_exists) / denom, 4),
            "avg_best_opp_reply_pts_after_me": round(float(sum_best_opp_reply_pts) / denom, 3),

            "endgame_me_moves": int(endgame_me_moves),
            "endgame_gift10_rate": round(float(endgame_gift10) / end_denom, 4),
            "endgame_gift15_rate": round(float(endgame_gift15) / end_denom, 4),
            "endgame_gift_win_now_rate": round(float(endgame_gift_win_now) / end_denom, 4),
            "endgame_gift_win_now_exists_rate": round(float(endgame_gift_win_now_exists) / end_denom, 4),

            "endgame_wn_total": int(endgame_wn_total),
            "endgame_wn_bone0_rate": round(float(endgame_wn_bone0) / float(max(1, endgame_wn_total)), 4),
            "endgame_wn_forced_rate": round(float(endgame_wn_forced) / float(max(1, endgame_wn_total)), 4),
            "endgame_wn_single_legal_rate": round(float(endgame_wn_single) / float(max(1, endgame_wn_total)), 4),
            "endgame_wn_safe_option_rate": round(float(endgame_wn_safeopt) / float(max(1, endgame_wn_total)), 4),

            "spike_moves": int(spike_moves),
            "spike_used_rate": round(float(spike_moves) / float(max(1, me_moves_total)), 4),
            "avg_spike_p_chosen": round(float(spike_p_sum) / spike_denom, 4),
            "avg_spike_p_best_visit": round(float(spike_best_p_sum) / spike_denom, 4),
            "spike_safe_exists_rate": round(float(spike_safe_exists) / spike_denom, 4),
            "spike_tp": int(spike_tp),
            "spike_fp": int(spike_fp),
            "spike_tn": int(spike_tn),
            "spike_fn": int(spike_fn),
            "spike_fn_rate": round(float(spike_fn) / spike_conf_total, 4),
            "spike_fp_rate": round(float(spike_fp) / spike_conf_total, 4),
        }
    }


def run_eval_parallel(cfg: EvalConfig, jobs: int, progress_every: int) -> Dict[str, Any]:
    jobs = max(1, int(jobs))
    matches = int(cfg.matches)
    # IM4.2: dataset collection requires single-process to avoid file contention
    if getattr(cfg, "infer_out_dir", None):
        jobs = 1
    if jobs == 1 or matches < 50:
        return run_eval(cfg)

    jobs = min(jobs, matches)
    per = matches // jobs
    rem = matches % jobs

    chunks: List[EvalConfig] = []
    base = int(cfg.base_seed)
    start = 0
    for j in range(jobs):
        n = per + (1 if j < rem else 0)
        c = EvalConfig(**{**cfg.__dict__})
        c.matches = n
        c.base_seed = base + start * 10007
        chunks.append(c)
        start += n

    t0 = time.perf_counter()
    reps: List[Dict[str, Any]] = []

    with mp.get_context("spawn").Pool(processes=jobs) as pool:
        done = 0
        for rep in pool.imap_unordered(run_eval, chunks, chunksize=1):
            reps.append(rep)
            done += int((rep.get("config") or {}).get("matches", 0) or 0)
            if progress_every > 0 and (done % progress_every == 0 or done == matches):
                dt = time.perf_counter() - t0
                mps = done / max(1e-9, dt)
                print(f"[eval] progress matches_done={done}/{matches} mps={mps:.2f}", flush=True)

    merged = _merge_reports(reps)
    # IMPORTANT:
    # When jobs>1, each worker chunk has its own base_seed (cfg.base_seed + start*10007).
    # _merge_reports currently takes the first returned worker's config as "base_cfg",
    # which makes merged["config"]["base_seed"] appear random/unrelated to the requested seed.
    #
    # The panel (A/B) groups results by config.base_seed, so we MUST restore the requested
    # base_seed here to keep reports stable and comparable.
    merged.setdefault("config", {})
    merged["config"]["matches"] = matches
    merged["config"]["base_seed"] = int(cfg.base_seed)
    merged["config"]["target"] = int(cfg.target)
    merged["results"]["elapsed_sec"] = round(float(time.perf_counter() - t0), 3)
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=500)
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--me", choices=["model", "ai", "greedy", "rust"], default="model")
    ap.add_argument("--level", choices=["quick", "standard", "deep", "ismcts"], default="quick")
    ap.add_argument("--det", type=int, default=8)
    ap.add_argument("--think_ms", type=int, default=120)
    ap.add_argument("--opp", choices=["greedy", "strategic"], default="greedy")
    ap.add_argument("--no_asserts", action="store_true")
    ap.add_argument("--assert_every", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers for eval (Windows uses spawn).")
    ap.add_argument("--progress_every", type=int, default=50, help="print progress every N matches (0=off).")
    ap.add_argument("--dump_match_json", type=str, default=None, help="Write first match full state JSON to path.")
    ap.add_argument("--dump_match_script", type=str, default=None, help="Write first match replay script JSON to path.")
    ap.add_argument("--dump_match_text", type=str, default=None, help="Write first match readable events text to path.")
    ap.add_argument("--rust_opp_mix_greedy", type=float, default=1.0)
    ap.add_argument("--rust_me_mix_greedy", type=float, default=1.0)
    ap.add_argument("--rust_leaf_value_weight", type=float, default=0.0)
    ap.add_argument("--rust_gift_penalty_weight", type=float, default=0.0)
    ap.add_argument("--rust_pessimism_alpha_max", type=float, default=0.0)
    ap.add_argument("--infer_out_dir", type=str, default=None, help="IM4.2: write INF1 dataset here (requires --jobs 1).")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    cfg = EvalConfig(
        matches=int(args.matches),
        target=int(args.target),
        base_seed=int(args.seed),
        me_policy=str(args.me),  # type: ignore[arg-type]
        analysis_level=str(args.level),
        det=int(args.det),
        think_ms=int(args.think_ms),
        rust_opp_mix_greedy=float(args.rust_opp_mix_greedy),
        rust_me_mix_greedy=float(args.rust_me_mix_greedy),
        rust_leaf_value_weight=float(args.rust_leaf_value_weight),
        rust_gift_penalty_weight=float(args.rust_gift_penalty_weight),
        rust_pessimism_alpha_max=float(args.rust_pessimism_alpha_max),
        opp_policy=str(args.opp),  # type: ignore[arg-type]
        strict_asserts=(not bool(args.no_asserts)),
        assert_every=int(args.assert_every),
        dump_match_json=args.dump_match_json,
        dump_match_script=args.dump_match_script,
        dump_match_text=args.dump_match_text,
        infer_out_dir=args.infer_out_dir,
    )

    # IM4.2: init writer if requested
    if cfg.infer_out_dir:
        if domino_rs is None:
            raise RuntimeError("infer_out_dir requires domino_rs (features_from_state_dict)")
        # recommended: open-loop collection
        if str(os.environ.get("DOMINO_INFER", "0")).strip() not in ("0", "", "false", "False"):
            print("[WARN] infer_out_dir: recommended to collect with DOMINO_INFER=0 (open-loop)", flush=True)
        out_dir = Path(str(cfg.infer_out_dir)).resolve()
        run_id = f"infer_strat_{int(time.time())}_{int(cfg.base_seed)}"
        cfg._infer_writer = Inf1Writer(out_dir, run_id=run_id, shard_max_samples=50000)
        print(json.dumps({"ok": True, "op": "infer_out_init", "out_dir": str(out_dir), "run_id": run_id}, ensure_ascii=False), flush=True)

    rep = run_eval_parallel(cfg, jobs=int(args.jobs), progress_every=int(args.progress_every))

    # IM4.2: finalize writer + write manifest
    if cfg._infer_writer is not None:
        try:
            man = cfg._infer_writer.finish()
            out_dir = Path(str(cfg.infer_out_dir)).resolve()
            mp = out_dir / f"{cfg._infer_writer.run_id}.manifest.json"
            mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({
                "ok": True,
                "op": "infer_out_done",
                "manifest": str(mp),
                "infer_samples": int(man.get("infer_samples", 0)),
                "infer_out_errors": int(getattr(cfg, "_infer_errs", 0)),
            }, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "op": "infer_out_done", "error": str(e)}, ensure_ascii=False), flush=True)

    print(json.dumps(rep, ensure_ascii=False), flush=True)           # panel-friendly
    print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True) # readable


if __name__ == "__main__":
    main()
