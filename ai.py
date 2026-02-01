# FILE: ai.py | version: 2025-12-25.v9
# (Final routing: analysis_level="deep" runs Deep v2 Root-PUCT; fixed reward; fixed draw-until-play in rollout;
#  standard remains model-agnostic; keeps model_predict for model-only)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Set, NamedTuple
import math
import random
import json
import os
import threading
import secrets
import numpy as np
from itertools import combinations
from functools import lru_cache
import time

from engine import (
    GameState, Board, Tile, EndName, PlayerName,
    ALL_TILES, parse_tile, tile_str, tile_is_double, tile_has,
    tile_pip_count, other_value, round_to_nearest_5,
    match_target_reason
)

ENDS: List[EndName] = ["right", "left", "up", "down"]
TILES_SORTED: List[Tile] = sorted(ALL_TILES, key=lambda t: (t[0], t[1]))
TILE_TO_IDX: Dict[Tile, int] = {t: i for i, t in enumerate(TILES_SORTED)}

ACTION_SIZE = 112
FEAT_DIM = 193

# analysis
ENUM_MAX_HANDS = 5000
SAMPLES_PER_MOVE = 8

MAX_ROLLOUTS_PER_WORLD = int(os.environ.get("DOMINO_MAX_ROLLOUTS_PER_WORLD", "120"))
ROLLOUT_MS_PER = int(os.environ.get("DOMINO_ROLLOUT_MS_PER", "120"))

# move eval weights (2-ply)
IMMEDIATE_POINTS_FACTOR = 3.5
OPPONENT_AVG_PENALTY = 2.5
OPPONENT_WORST_PENALTY = 0.5
MY_REPLY_AVG_BONUS = 2.0
MY_REPLY_BEST_BONUS = 0.5
OPPONENT_PASS_BONUS = 15.0
OPPONENT_DRAW_FACTOR = 2.0
DOUBLE_TILE_BONUS = 3.0

# kept for compatibility; Stage-B disables their use in STANDARD
PRIOR_LOG_WEIGHT = 0.5
VALUE_NET_SCALE = 25.0

# opponent strategic params
OPPONENT_IMMEDIATE_FACTOR = 3.0
OPPONENT_DOUBLE_BONUS = 6.0
OPPONENT_PIP_PENALTY = 0.5
OPPONENT_DIST_TO_5_FACTOR = 0.8
OPPONENT_DIVERSITY_BONUS = 0.4
OPPONENT_OPEN_DANGER_PENALTY = 2.0

# belief params
DECAY_RATE = 0.85
MAX_CUT_PROBABILITY = 0.98
CERTAIN_WEIGHT = 1.0
PROBABLE_WEIGHT = 0.6
POSSIBLE_WEIGHT = 0.35

# solver flag
USE_SOLVER = os.environ.get("DOMINO_USE_SOLVER", "0").strip() == "1"

# =============================================================================
# Solver toggle (OFF by default) - solver can hang with spinner/4-ends and is not match-aware unless rewritten.
# =============================================================================
USE_ENDGAME_SOLVER = os.environ.get("DOMINO_USE_SOLVER", "0").strip() == "1"
SOLVER_MAX_TILES = int(os.environ.get("DOMINO_SOLVER_MAX_TILES", "8"))  # strict
SOLVER_ALLOW_SPINNER = os.environ.get("DOMINO_SOLVER_ALLOW_SPINNER", "0").strip() == "1"


def encode_action(tile: Tile, end: EndName) -> int:
    return TILE_TO_IDX[tile] * 4 + ENDS.index(end)


def legal_mask_state(st: GameState) -> np.ndarray:
    mask = np.zeros((ACTION_SIZE,), dtype=np.int8)
    for (t, e) in st.legal_moves_me():
        mask[encode_action(t, e)] = 1
    return mask


def _move_seed(base_seed: int, tile: Tile, end: EndName, salt: int = 0) -> int:
    a = int(encode_action(tile, end))
    x = (int(base_seed) ^ (a * 2654435761) ^ (int(salt) * 97531)) & 0xFFFFFFFF
    return int(x if x != 0 else 1)


def _rank_move_for_search(board: Board, tile: Tile, end: EndName) -> float:
    ip = int(immediate_points(board, tile, end))
    sc = float(ip) * 100.0
    if tile_is_double(tile):
        sc += 12.0
    sc -= 0.20 * float(tile_pip_count(tile))
    return float(sc)


# =============================================================================
# Belief
# =============================================================================

@dataclass
class Belief:
    cut_prob: List[float] = field(default_factory=lambda: [0.0] * 7)
    last_ply: List[int] = field(default_factory=lambda: [0] * 7)
    hard_forbid: List[bool] = field(default_factory=lambda: [False] * 7)  # Phase0-rescue
    decay: float = DECAY_RATE
    max_prob: float = MAX_CUT_PROBABILITY

    def apply_decay_to(self, value: int, current_ply: int) -> None:
        if bool(self.hard_forbid[value]):
            return  # do not decay hard constraints
        ply_diff = current_ply - self.last_ply[value]
        if ply_diff <= 0:
            return
        self.cut_prob[value] *= (self.decay ** ply_diff)
        self.last_ply[value] = current_ply

    def update_from_event(self, ev: Dict[str, Any]) -> None:
        if ev.get("player") != "opponent":
            return

        event_type = ev.get("type")
        ply = int(ev.get("ply", 0) or 0)

        raw_ends = ev.get("open_ends", []) or []
        certainty = ev.get("certainty", "probable")

        # Robust: open_ends may contain junk (e.g. "down"). We only keep ints 0..6.
        ends: List[int] = []
        for x in raw_ends:
            try:
                v = int(x)
            except Exception:
                continue
            if 0 <= v <= 6:
                ends.append(v)

        if event_type == "pass":
            w = {"certain": 0.92, "probable": 0.75, "possible": 0.55}.get(certainty, 0.75)
            for v in ends:
                self.apply_decay_to(v, ply)
                # Phase0-rescue: opponent certain-pass implies (by engine semantics)
                # boneyard empty at that time => no future draws => HARD constraint.
                if certainty == "certain":
                    self.cut_prob[v] = 1.0
                    self.hard_forbid[v] = True
                else:
                    self.cut_prob[v] = min(self.max_prob, max(self.cut_prob[v], float(w)))
                self.last_ply[v] = ply

        elif event_type == "draw":
            weight = {"certain": CERTAIN_WEIGHT, "probable": PROBABLE_WEIGHT, "possible": POSSIBLE_WEIGHT}.get(certainty, PROBABLE_WEIGHT)
            adjustment = weight * 0.30
            for v in ends:
                self.apply_decay_to(v, ply)
                cur = self.cut_prob[v]
                self.cut_prob[v] = min(self.max_prob, cur + adjustment * (1.0 - cur))
                self.last_ply[v] = ply

    def to_dict(self) -> Dict[str, Any]:
        return {"cut_prob": [round(float(x), 4) for x in self.cut_prob]}


def build_belief_from_state(st: GameState) -> Belief:
    belief = Belief()

    for e in (st.events or []):
        ev = {
            "type": getattr(e, "type", None),
            "ply": int(getattr(e, "ply", 0) or 0),
            "player": getattr(e, "player", None),
            "open_ends": list(getattr(e, "open_ends", []) or []),
            "certainty": getattr(e, "certainty", "probable"),
        }
        belief.update_from_event(ev)

    # Phase0-rescue: keep hard_forbid pinned.
    for v in range(7):
        if belief.hard_forbid[v]:
            belief.cut_prob[v] = 1.0

    visible = st.visible_tiles()
    for value in range(7):
        tiles_with_value = [t for t in ALL_TILES if t[0] == value or t[1] == value]
        visible_count = sum(1 for t in tiles_with_value if t in visible)
        hidden_count = 7 - visible_count

        if hidden_count == 0:
            belief.cut_prob[value] = 1.0
        elif hidden_count == 1:
            belief.cut_prob[value] = max(belief.cut_prob[value], 0.75)
        elif hidden_count == 2:
            belief.cut_prob[value] = max(belief.cut_prob[value], 0.45)

    current_ply = len(st.events)
    for value in range(7):
        if belief.hard_forbid[value]:
            continue
        belief.apply_decay_to(value, current_ply)

    return belief


# =============================================================================
# Model (thread-safe load)
# =============================================================================

MODEL: Optional[Dict[str, np.ndarray]] = None
MODEL_MTIME: Optional[float] = None
MODEL_LOCK = threading.Lock()


def _softmax_with_mask(logits: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    scores = logits.astype(np.float64)
    if mask is not None:
        scores = scores.copy()
        scores[mask == 0] = -1e9
    max_score = float(np.max(scores))
    exp_scores = np.exp(scores - max_score)
    return (exp_scores / (float(np.sum(exp_scores)) + 1e-12)).astype(np.float32)


def ensure_model_loaded(path: str = "model.json") -> None:
    global MODEL, MODEL_MTIME

    with MODEL_LOCK:
        try:
            file_mtime = os.path.getmtime(path)
        except OSError:
            MODEL = None
            MODEL_MTIME = None
            return

        if MODEL is not None and MODEL_MTIME == file_mtime:
            return

        try:
            with open(path, "rb") as f:
                raw = f.read()
            model_data = json.loads(raw.decode("utf-8"))

            if model_data.get("type") != "mlp_pv_v1":
                MODEL = None
                MODEL_MTIME = file_mtime
                return

            if int(model_data.get("action_size", -1)) != int(ACTION_SIZE):
                MODEL = None
                MODEL_MTIME = file_mtime
                return

            if int(model_data.get("feat_dim", -1)) != int(FEAT_DIM):
                MODEL = None
                MODEL_MTIME = file_mtime
                return

            MODEL = {
                "W1": np.array(model_data["W1"], dtype=np.float32),
                "b1": np.array(model_data["b1"], dtype=np.float32),
                "Wp": np.array(model_data["Wp"], dtype=np.float32),
                "bp": np.array(model_data["bp"], dtype=np.float32),
                "Wv": np.array(model_data["Wv"], dtype=np.float32),
                "bv": np.array(model_data["bv"], dtype=np.float32),
            }
            MODEL_MTIME = file_mtime

        except Exception:
            MODEL = None
            MODEL_MTIME = None
            return


def model_predict(features: np.ndarray, legal_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    ensure_model_loaded("model.json")

    if features.shape != (FEAT_DIM,):
        raise ValueError(f"features shape mismatch: expected {(FEAT_DIM,)}, got {features.shape}")

    if MODEL is None:
        if legal_mask is None:
            policy = np.ones((ACTION_SIZE,), dtype=np.float32) / float(ACTION_SIZE)
        else:
            legal_count = max(1, int(legal_mask.sum()))
            policy = legal_mask.astype(np.float32) / float(legal_count)
        return policy, 0.0

    W1, b1 = MODEL["W1"], MODEL["b1"]
    Wp, bp = MODEL["Wp"], MODEL["bp"]
    Wv, bv = MODEL["Wv"], MODEL["bv"]

    z1 = features @ W1.T + b1
    h1 = np.maximum(z1, 0.0)

    logits = h1 @ Wp.T + bp
    policy = _softmax_with_mask(logits, legal_mask)

    value_raw = float(h1 @ Wv[0] + bv[0])
    value = float(np.tanh(value_raw))
    return policy, value


# =============================================================================
# Features (FEAT_DIM=193) — must match train.py
# =============================================================================

def features_small(st: GameState) -> np.ndarray:
    belief = build_belief_from_state(st)
    cut = belief.cut_prob[:] if belief and belief.cut_prob else [0.0] * 7

    out = np.zeros((FEAT_DIM,), dtype=np.float32)
    off = 0

    for t in st.my_hand:
        out[off + TILE_TO_IDX[t]] = 1.0
    off += 28

    for t in st.board.played_set:
        out[off + TILE_TO_IDX[t]] = 1.0
    off += 28

    ct = st.board.center_tile
    if ct is None:
        out[off + 28] = 1.0
    else:
        out[off + TILE_TO_IDX[ct]] = 1.0
    off += 29

    ends = st.board.ends or {}
    for end_name in ["right", "left", "up", "down"]:
        if end_name in ends:
            v = int(ends[end_name][0])
            idx = v if 0 <= v <= 6 else 7
        else:
            idx = 7
        out[off + idx] = 1.0
        off += 8

    for i, end_name in enumerate(["right", "left", "up", "down"]):
        if end_name in ends:
            out[off + i] = 1.0 if bool(ends[end_name][1]) else 0.0
        else:
            out[off + i] = 0.0
    off += 4

    arms = st.board.arms or {"right": [], "left": [], "up": [], "down": []}
    for i, end_name in enumerate(["right", "left", "up", "down"]):
        ln = len(arms.get(end_name, []) or [])
        out[off + i] = float(min(12, ln)) / 12.0
    off += 4

    out[off + 0] = 1.0 if (ct is not None and tile_is_double(ct)) else 0.0
    out[off + 1] = 1.0 if bool(st.board.spinner_sides_open) else 0.0
    off += 2

    ft = st.forced_play_tile
    if ft is None:
        out[off + 28] = 1.0
    else:
        out[off + TILE_TO_IDX[ft]] = 1.0
    off += 29

    out[off + 0] = 1.0 if st.must_draw_me() else 0.0
    out[off + 1] = 1.0 if st.must_pass_me() else 0.0
    off += 2

    out[off + 0] = float(len(st.my_hand)) / 7.0
    out[off + 1] = float(st.opponent_tile_count) / 7.0
    out[off + 2] = float(st.boneyard_count) / 14.0
    off += 3

    tgt = max(50, int(st.match_target))
    out[off + 0] = float(st.my_score) / float(tgt)
    out[off + 1] = float(st.opponent_score) / float(tgt)
    off += 2

    ends_sum = float(st.board.ends_sum())
    out[off + 0] = ends_sum / 30.0
    out[off + 1] = float(int(ends_sum) % 5) / 5.0
    out[off + 2] = float(st.board.score_now()) / 20.0
    off += 3

    out[off + 0] = float(min(50, int(st.round_index))) / 50.0
    out[off + 1] = float(min(300, int(st.match_target))) / 300.0
    off += 2

    for i in range(7):
        out[off + i] = float(max(0.0, min(1.0, float(cut[i]))))
    off += 7

    for i, end_name in enumerate(["right", "left", "up", "down"]):
        if end_name in ends:
            v = int(ends[end_name][0])
            out[off + i] = float(max(0.0, min(1.0, float(cut[v])))) if 0 <= v <= 6 else 0.0
        else:
            out[off + i] = 0.0
    off += 4

    my_value_dist = np.zeros(7, dtype=np.float32)
    for t in st.my_hand:
        my_value_dist[t[0]] += 1.0
        my_value_dist[t[1]] += 1.0
    my_value_dist /= 7.0

    visible_value_dist = np.zeros(7, dtype=np.float32)
    visible_tiles = st.visible_tiles()
    for t in visible_tiles:
        visible_value_dist[t[0]] += 1.0
        visible_value_dist[t[1]] += 1.0
    visible_value_dist /= 28.0

    out[off:off + 7] = my_value_dist
    off += 7
    out[off:off + 7] = visible_value_dist
    off += 7

    if off != FEAT_DIM:
        raise RuntimeError(f"features offset mismatch: {off} != {FEAT_DIM}")
    return out


# =============================================================================
# Determinization
# =============================================================================

def _tile_weight_from_cut(tile: Tile, belief: Belief) -> float:
    a, b = tile
    cut_probability = max(float(belief.cut_prob[a]), float(belief.cut_prob[b]))
    return max(0.01, 1.0 - 0.90 * cut_probability)


class Determinization(NamedTuple):
    opp_hand: Set[Tile]
    boneyard: List[Tile]


def weighted_sample_without_replacement(items: List[Tile], weights: List[float], sample_size: int, rng: random.Random) -> List[Tile]:
    if sample_size <= 0 or not items:
        return []
    if sample_size >= len(items):
        result = list(items)
        rng.shuffle(result)
        return result

    remaining_items = list(items)
    remaining_weights = list(weights)
    selected: List[Tile] = []

    for _ in range(sample_size):
        if not remaining_items:
            break
        total_weight = float(sum(remaining_weights))
        if total_weight <= 1e-12:
            idx = rng.randint(0, len(remaining_items) - 1)
            selected.append(remaining_items[idx])
            remaining_items.pop(idx)
            remaining_weights.pop(idx)
            continue

        r = rng.random() * total_weight
        acc = 0.0
        for idx, w in enumerate(remaining_weights):
            acc += float(w)
            if acc >= r:
                selected.append(remaining_items[idx])
                remaining_items.pop(idx)
                remaining_weights.pop(idx)
                break

    return selected


def determinize_hidden(st: GameState, belief: Belief, rng: random.Random) -> Determinization:
    visible_tiles = st.visible_tiles()
    hidden_tiles = [t for t in ALL_TILES if t not in visible_tiles]
    if not hidden_tiles:
        return Determinization(set(), [])

    tile_weights = [_tile_weight_from_cut(t, belief) for t in hidden_tiles]
    opp_hand_size = min(int(st.opponent_tile_count), len(hidden_tiles))

    opp_hand_list = weighted_sample_without_replacement(hidden_tiles, tile_weights, opp_hand_size, rng)
    opp_hand_set = set(opp_hand_list)

    boneyard_list = [t for t in hidden_tiles if t not in opp_hand_set]
    rng.shuffle(boneyard_list)
    boneyard_list = boneyard_list[:int(st.boneyard_count)]

    return Determinization(opp_hand_set, boneyard_list)


# =============================================================================
# Helpers: scoring / board clones
# =============================================================================

def immediate_points(board: Board, tile: Tile, end: EndName) -> int:
    b2 = board.clone()
    try:
        return int(b2.play(tile, end))
    except Exception:
        return 0


def board_after_move(board: Board, tile: Tile, end: EndName) -> Optional[Board]:
    b2 = board.clone()
    try:
        b2.play(tile, end)
        return b2
    except Exception:
        return None


def out_points_from_opponent_hand(opponent_hand: Set[Tile]) -> int:
    total = sum(tile_pip_count(t) for t in opponent_hand)
    return int(round_to_nearest_5(int(total)))


def locked_delta_points(my_hand: Set[Tile], opponent_hand: Set[Tile]) -> int:
    my_total = sum(tile_pip_count(t) for t in my_hand)
    opp_total = sum(tile_pip_count(t) for t in opponent_hand)
    diff = abs(int(my_total) - int(opp_total))
    pts = int(round_to_nearest_5(diff))
    if pts == 0:
        return 0
    if my_total < opp_total:
        return pts
    if opp_total < my_total:
        return -pts
    return 0


# =============================================================================
# Opponent models
# =============================================================================

def score_opponent_move(board: Board, tile: Tile, end: EndName, opponent_hand: Set[Tile]) -> float:
    points_earned = immediate_points(board, tile, end)
    score = float(points_earned) * OPPONENT_IMMEDIATE_FACTOR

    if tile_is_double(tile):
        score += OPPONENT_DOUBLE_BONUS

    score += float(tile_pip_count(tile)) * OPPONENT_PIP_PENALTY

    board_after = board_after_move(board, tile, end)
    if board_after:
        new_sum = board_after.ends_sum()
        distance_to_5 = min(new_sum % 5, 5 - (new_sum % 5))
        score += float(distance_to_5) * OPPONENT_DIST_TO_5_FACTOR

    remaining_tiles = opponent_hand - {tile}
    unique_values: Set[int] = set()
    for t in remaining_tiles:
        unique_values.add(t[0]); unique_values.add(t[1])
    score += float(len(unique_values)) * OPPONENT_DIVERSITY_BONUS

    if board_after:
        for open_value in board_after.open_end_values():
            has_matching_tile = any(tile_has(t, open_value) for t in remaining_tiles)
            if not has_matching_tile:
                score -= OPPONENT_OPEN_DANGER_PENALTY

    return float(score)


def _collect_legal_moves(board: Board, hand: Set[Tile]) -> List[Tuple[Tile, EndName]]:
    out: List[Tuple[Tile, EndName]] = []
    for t in hand:
        for e in board.legal_ends_for_tile(t):
            out.append((t, e))
    return out

def _pick_opp_move_greedy(board: Board, opp_hand: Set[Tile]) -> Optional[Tuple[Tile, EndName]]:
    moves = _collect_legal_moves(board, opp_hand)
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


def _best_end_for_forced_tile(board: Board, tile: Tile, opponent_hand: Set[Tile]) -> Tuple[EndName, int]:
    ends = board.legal_ends_for_tile(tile)
    if not ends:
        return "right", 0
    best_end = ends[0]
    best_score = -1e9
    best_points = 0
    for e in ends:
        sc = score_opponent_move(board, tile, e, opponent_hand)
        if sc > best_score:
            best_score = sc
            best_end = e
            best_points = immediate_points(board, tile, e)
    return best_end, int(best_points)


def simulate_opponent_turn_strategic(board_after_my_move: Board, determinization: Determinization, rng: random.Random) -> Dict[str, Any]:
    opponent_hand = set(determinization.opp_hand)
    boneyard = list(determinization.boneyard)
    rng.shuffle(boneyard)
    drew_count = 0

    legal_moves = _collect_legal_moves(board_after_my_move, opponent_hand)
    if legal_moves:
        best_score = -1e18
        best_move: Optional[Tuple[Tile, EndName]] = None
        best_points = 0
        for tile, end in legal_moves:
            move_score = score_opponent_move(board_after_my_move, tile, end, opponent_hand)
            if move_score > best_score:
                best_score = move_score
                best_move = (tile, end)
                best_points = immediate_points(board_after_my_move, tile, end)
        assert best_move is not None
        t, e = best_move
        opponent_hand.discard(t)

        return {
            "played": True, "passed": False, "drew": 0, "points": int(best_points),
            "move": (t, e),
            "opp_hand_after": set(opponent_hand),
            "boneyard_after": list(boneyard),
            "forced_play": False,
        }

    while boneyard:
        drawn = boneyard.pop()
        opponent_hand.add(drawn)
        drew_count += 1

        if board_after_my_move.legal_ends_for_tile(drawn):
            end, pts = _best_end_for_forced_tile(board_after_my_move, drawn, opponent_hand)
            opponent_hand.discard(drawn)
            return {
                "played": True, "passed": False, "drew": int(drew_count), "points": int(pts),
                "move": (drawn, end),
                "opp_hand_after": set(opponent_hand),
                "boneyard_after": list(boneyard),
                "forced_play": True,
            }

    return {
        "played": False, "passed": True, "drew": int(drew_count), "points": 0, "move": None,
        "opp_hand_after": set(opponent_hand),
        "boneyard_after": list(boneyard),
        "forced_play": False,
    }


def _simulate_forced_draw_for_me(board: Board, my_hand: Set[Tile], boneyard: List[Tile], rng: random.Random) -> Dict[str, Any]:
    hand = set(my_hand)
    by = list(boneyard)
    rng.shuffle(by)

    if _collect_legal_moves(board, hand):
        return {"drew": 0, "played": False, "move": None, "points": 0, "my_hand_after": hand, "boneyard_after": by}

    drew = 0
    while by:
        drawn = by.pop()
        drew += 1
        hand.add(drawn)
        ends = board.legal_ends_for_tile(drawn)
        if ends:
            best_end = max(ends, key=lambda e: immediate_points(board, drawn, e))
            pts = immediate_points(board, drawn, best_end)
            board2 = board_after_move(board, drawn, best_end)
            if board2 is None:
                return {"drew": drew, "played": False, "move": None, "points": 0, "my_hand_after": hand, "boneyard_after": by}
            hand.remove(drawn)
            return {
                "drew": int(drew), "played": True, "move": (drawn, best_end), "points": int(pts),
                "my_hand_after": set(hand), "boneyard_after": list(by),
            }

    return {"drew": int(drew), "played": False, "move": None, "points": 0, "my_hand_after": hand, "boneyard_after": by}


# =============================================================================
# 2-ply evaluation (teacher core)
# =============================================================================

def evaluate_move_2ply(state: GameState, tile: Tile, end: EndName, belief: Belief, rng: random.Random, num_samples: int = SAMPLES_PER_MOVE) -> Dict[str, Any]:
    st2 = state.fast_clone(keep_events=False)
    st2.current_turn = "me"

    my_points = int(st2.play_tile("me", tile, end).score_gained)
    board_after_my_move = st2.board.clone()
    my_hand_after_play = set(st2.my_hand)

    if len(my_hand_after_play) == 0:
        total_out = 0.0
        best_out = 0
        for _ in range(int(num_samples)):
            det = determinize_hidden(st2, belief, rng)
            out_pts = out_points_from_opponent_hand(det.opp_hand)
            total_out += float(out_pts)
            best_out = max(best_out, int(out_pts))
        denom = float(max(1, int(num_samples)))
        return {
            "immediate_points": int(my_points),
            "opponent_avg_points": 0.0,
            "opponent_worst_points": 0,
            "my_reply_avg_points": float(total_out / denom),
            "my_reply_best_points": int(best_out),
            "opponent_pass_rate": 0.0,
            "opponent_draw_avg": 0.0,
        }

    total_opponent_points = 0.0
    total_my_reply_points = 0.0
    total_opponent_pass = 0.0
    total_opponent_draw = 0.0
    worst_opponent_points = 0
    best_my_reply = 0

    for _ in range(int(num_samples)):
        det = determinize_hidden(st2, belief, rng)

        opponent_result = simulate_opponent_turn_strategic(board_after_my_move, det, rng)
        total_opponent_draw += float(opponent_result["drew"])
        if opponent_result["passed"]:
            total_opponent_pass += 1.0

        opp_pts = int(opponent_result["points"])
        total_opponent_points += float(opp_pts)
        worst_opponent_points = max(worst_opponent_points, opp_pts)

        board_after_opp = board_after_my_move
        opp_hand_after = set(opponent_result["opp_hand_after"])
        boneyard_after = list(opponent_result["boneyard_after"])

        if opponent_result["move"]:
            opp_tile, opp_end = opponent_result["move"]
            tmp = board_after_move(board_after_my_move, opp_tile, opp_end)
            if tmp is not None:
                board_after_opp = tmp

            if len(opp_hand_after) == 0:
                out_pts = round_to_nearest_5(sum(tile_pip_count(t) for t in my_hand_after_play))
                total_opponent_points += float(out_pts)
                worst_opponent_points = max(worst_opponent_points, int(out_pts))
                continue

        my_moves: List[Tuple[Tile, EndName]] = []
        for my_tile in my_hand_after_play:
            for reply_end in board_after_opp.legal_ends_for_tile(my_tile):
                my_moves.append((my_tile, reply_end))

        if my_moves:
            reply_points = max(immediate_points(board_after_opp, t, e) for t, e in my_moves)
            total_my_reply_points += float(reply_points)
            best_my_reply = max(best_my_reply, int(reply_points))
        else:
            draw_out = _simulate_forced_draw_for_me(board_after_opp, my_hand_after_play, boneyard_after, rng)
            if draw_out["played"]:
                total_my_reply_points += float(draw_out["points"])
                best_my_reply = max(best_my_reply, int(draw_out["points"]))
            else:
                if len(boneyard_after) == 0 and opponent_result["passed"]:
                    delta = locked_delta_points(my_hand_after_play, opp_hand_after)
                    if delta > 0:
                        total_my_reply_points += float(delta)
                        best_my_reply = max(best_my_reply, int(delta))
                    elif delta < 0:
                        total_opponent_points += float(abs(delta))
                        worst_opponent_points = max(worst_opponent_points, int(abs(delta)))

    denom = float(max(1, int(num_samples)))
    return {
        "immediate_points": int(my_points),
        "opponent_avg_points": float(total_opponent_points / denom),
        "opponent_worst_points": int(worst_opponent_points),
        "my_reply_avg_points": float(total_my_reply_points / denom),
        "my_reply_best_points": int(best_my_reply),
        "opponent_pass_rate": float(total_opponent_pass / denom),
        "opponent_draw_avg": float(total_opponent_draw / denom),
    }


def analyze_opponent_threats(board_after_move_: Board, belief: Belief, my_hand_after_play: Set[Tile]) -> List[Dict[str, Any]]:
    visible_tiles = my_hand_after_play | board_after_move_.played_set
    hidden_tiles = [t for t in ALL_TILES if t not in visible_tiles]
    if not hidden_tiles:
        return []

    weights = np.array([_tile_weight_from_cut(t, belief) for t in hidden_tiles], dtype=np.float64)
    wsum = float(np.sum(weights)) + 1e-12

    threats: List[Dict[str, Any]] = []
    for tile, w in zip(hidden_tiles, weights):
        for end in board_after_move_.legal_ends_for_tile(tile):
            points = immediate_points(board_after_move_, tile, end)
            if points >= 5:
                prob_est = float(w) / wsum
                threats.append({
                    "tile": tile_str(tile),
                    "end": end,
                    "points": int(points),
                    "probability": round(prob_est, 4),
                    "warning": f"الخصم قد ينزل {tile_str(tile)} ويكسب {points} نقطة"
                })

    threats.sort(key=lambda x: float(x["points"]) * float(x["probability"]), reverse=True)
    return threats[:3]


# =============================================================================
# Enumeration helpers (used in standard when small enough)
# =============================================================================

def _softmax_logweights(logw: List[float]) -> List[float]:
    if not logw:
        return []
    m = max(logw)
    exps = [math.exp(x - m) for x in logw]
    s = float(sum(exps))
    if s <= 0:
        return [1.0 / len(exps)] * len(exps)
    return [float(x) / s for x in exps]


def _weighted_percentile(values: List[float], weights: List[float], q: float) -> float:
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = float(sum(w for _, w in pairs))
    if total <= 0:
        return float(pairs[int(q * (len(pairs) - 1))][0])
    acc = 0.0
    for v, w in pairs:
        acc += float(w)
        if acc / total >= q:
            return float(v)
    return float(pairs[-1][0])


def _enumerate_opponent_hands_with_weights(state: GameState, belief: Belief) -> Tuple[List[Tuple[Tile, ...]], List[float]]:
    hidden = state.hidden_tiles()
    n = len(hidden)
    k = min(int(state.opponent_tile_count), n)
    all_hands = list(combinations(hidden, k))
    if not all_hands:
        return [], []

    logw: List[float] = []
    for hand in all_hands:
        lw = 0.0
        for t in hand:
            lw += math.log(_tile_weight_from_cut(t, belief))
        logw.append(lw)

    probs = _softmax_logweights(logw)
    return all_hands, probs


def _evaluate_move_against_fixed_hand(base_state: GameState, tile: Tile, end: EndName, opp_hand: Set[Tile], boneyard_list: List[Tile], rng: random.Random) -> Dict[str, float]:
    st2 = base_state.fast_clone(keep_events=False)
    st2.current_turn = "me"

    my_points = int(st2.play_tile("me", tile, end).score_gained)
    board_after_my = st2.board.clone()
    my_hand_after = set(st2.my_hand)

    if len(my_hand_after) == 0:
        out_pts = out_points_from_opponent_hand(set(opp_hand))
        return {"my_points": float(my_points), "opp_points": 0.0, "opp_passed": 0.0, "opp_drew": 0.0, "my_reply": float(out_pts), "my_reply_best": float(out_pts)}

    det = Determinization(set(opp_hand), list(boneyard_list))
    opp = simulate_opponent_turn_strategic(board_after_my, det, rng)

    opp_points = float(opp["points"])
    opp_drew = float(opp["drew"])
    opp_passed = 1.0 if opp["passed"] else 0.0

    board_after_opp = board_after_my
    opp_hand_after = set(opp["opp_hand_after"])
    boneyard_after = list(opp["boneyard_after"])

    if opp["move"]:
        opp_tile, opp_end = opp["move"]
        tmp = board_after_move(board_after_my, opp_tile, opp_end)
        if tmp is not None:
            board_after_opp = tmp

        if len(opp_hand_after) == 0:
            my_pips = sum(tile_pip_count(t) for t in my_hand_after)
            opp_out = int(round_to_nearest_5(int(my_pips)))
            return {"my_points": float(my_points), "opp_points": float(opp_points + float(opp_out)), "opp_passed": float(opp_passed), "opp_drew": float(opp_drew), "my_reply": 0.0, "my_reply_best": 0.0}

    replies: List[Tuple[Tile, EndName]] = []
    for t2 in my_hand_after:
        for e2 in board_after_opp.legal_ends_for_tile(t2):
            replies.append((t2, e2))

    my_reply_points = 0.0
    my_reply_best = 0.0

    if replies:
        rp = max(immediate_points(board_after_opp, t2, e2) for t2, e2 in replies)
        my_reply_points += float(rp)
        my_reply_best = max(my_reply_best, float(rp))
    else:
        draw_out = _simulate_forced_draw_for_me(board_after_opp, my_hand_after, boneyard_after, rng)
        if draw_out["played"]:
            my_reply_points += float(draw_out["points"])
            my_reply_best = max(my_reply_best, float(draw_out["points"]))
        else:
            if opp["passed"] and len(boneyard_after) == 0:
                delta = locked_delta_points(my_hand_after, opp_hand_after)
                if delta > 0:
                    my_reply_points += float(delta)
                    my_reply_best = max(my_reply_best, float(delta))
                elif delta < 0:
                    opp_points += float(abs(delta))

    return {"my_points": float(my_points), "opp_points": float(opp_points), "opp_passed": float(opp_passed), "opp_drew": float(opp_drew), "my_reply": float(my_reply_points), "my_reply_best": float(my_reply_best)}


def _evaluate_move_by_enumeration(state: GameState, tile: Tile, end: EndName, belief: Belief, hands: List[Tuple[Tile, ...]], hand_probs: List[float], rng_seed: int, boneyard_samples_per_hand: int) -> Dict[str, Any]:
    hidden = state.hidden_tiles()
    hidden_set = set(hidden)

    exp_opp_points = 0.0
    exp_opp_draw = 0.0
    exp_opp_pass = 0.0
    exp_my_reply = 0.0

    opp_points_samples: List[float] = []
    opp_points_weights: List[float] = []
    my_reply_samples: List[float] = []
    my_reply_weights: List[float] = []
    my_reply_best_global = 0.0

    immediate_pts = immediate_points(state.board, tile, end)

    for idx, (hand, p_hand) in enumerate(zip(hands, hand_probs)):
        if p_hand <= 0:
            continue

        opp_hand = set(hand)
        boneyard_list = list(hidden_set - opp_hand)
        if len(boneyard_list) > int(state.boneyard_count):
            boneyard_list = boneyard_list[:int(state.boneyard_count)]

        local_opp_points = 0.0
        local_opp_draw = 0.0
        local_opp_pass = 0.0
        local_my_reply = 0.0
        local_my_reply_best = 0.0

        s_count = max(1, int(boneyard_samples_per_hand))
        for s in range(s_count):
            rr = random.Random((rng_seed * 1315423911 + idx * 2654435761 + s) & 0xFFFFFFFF)
            out = _evaluate_move_against_fixed_hand(state, tile, end, opp_hand, boneyard_list, rr)

            local_opp_points += float(out["opp_points"])
            local_opp_draw += float(out["opp_drew"])
            local_opp_pass += float(out["opp_passed"])
            local_my_reply += float(out["my_reply"])
            local_my_reply_best = max(local_my_reply_best, float(out["my_reply_best"]))

        denom = float(s_count)
        local_opp_points /= denom
        local_opp_draw /= denom
        local_opp_pass /= denom
        local_my_reply /= denom

        exp_opp_points += float(p_hand) * local_opp_points
        exp_opp_draw += float(p_hand) * local_opp_draw
        exp_opp_pass += float(p_hand) * local_opp_pass
        exp_my_reply += float(p_hand) * local_my_reply

        opp_points_samples.append(local_opp_points)
        opp_points_weights.append(float(p_hand))
        my_reply_samples.append(local_my_reply)
        my_reply_weights.append(float(p_hand))
        my_reply_best_global = max(my_reply_best_global, float(local_my_reply_best))

    opp_tail = _weighted_percentile(opp_points_samples, opp_points_weights, 0.90)
    my_reply_tail_best = _weighted_percentile(my_reply_samples, my_reply_weights, 0.90)

    return {
        "immediate_points": int(immediate_pts),
        "opponent_avg_points": float(exp_opp_points),
        "opponent_worst_points": int(round(opp_tail)),
        "my_reply_avg_points": float(exp_my_reply),
        "my_reply_best_points": int(round(max(my_reply_best_global, my_reply_tail_best))),
        "opponent_pass_rate": float(exp_opp_pass),
        "opponent_draw_avg": float(exp_opp_draw),
    }


# =============================================================================
# Endgame solver (perfect-info when boneyard empty)
# =============================================================================

def _legal_moves_for_hand(board: Board, hand: Set[Tile]) -> List[Tuple[Tile, EndName]]:
    return _collect_legal_moves(board, hand)


def _board_key(board: Board) -> str:
    snap = board.snapshot()
    k = {
        "center_tile": snap.get("center_tile"),
        "spinner_value": snap.get("spinner_value"),
        "spinner_sides_open": snap.get("spinner_sides_open"),
        "ends": snap.get("ends"),
        "arms": snap.get("arms"),
    }
    return json.dumps(k, sort_keys=True)


@lru_cache(maxsize=200000)
def _solve_no_boneyard_delta_cached(board_json: str, my_hand_t: Tuple[Tile, ...], opp_hand_t: Tuple[Tile, ...], turn: str) -> Tuple[int, Tuple[Tuple[str, str], ...]]:
    board = Board.from_snapshot(json.loads(board_json))
    my_hand = set(my_hand_t)
    opp_hand = set(opp_hand_t)

    if len(my_hand) == 0:
        pts = out_points_from_opponent_hand(set(opp_hand))
        return int(pts), tuple()
    if len(opp_hand) == 0:
        pts = round_to_nearest_5(sum(tile_pip_count(t) for t in my_hand))
        return -int(pts), tuple()

    my_moves = _legal_moves_for_hand(board, my_hand)
    opp_moves = _legal_moves_for_hand(board, opp_hand)

    if (not my_moves) and (not opp_moves):
        delta = locked_delta_points(set(my_hand), set(opp_hand))
        return int(delta), tuple()

    if turn == "me" and not my_moves:
        return _solve_no_boneyard_delta_cached(board_json, tuple(sorted(my_hand)), tuple(sorted(opp_hand)), "opponent")
    if turn == "opponent" and not opp_moves:
        return _solve_no_boneyard_delta_cached(board_json, tuple(sorted(my_hand)), tuple(sorted(opp_hand)), "me")

    if turn == "me":
        best = -10**9
        best_pv: Tuple[Tuple[str, str], ...] = tuple()
        for (t, e) in my_moves:
            b2 = board.clone()
            pts = int(b2.play(t, e))
            my2 = set(my_hand); my2.discard(t)

            if len(my2) == 0:
                out_pts = out_points_from_opponent_hand(set(opp_hand))
                cand = int(pts + out_pts)
                pv = ((tile_str(t), e),)
            else:
                d2, pv2 = _solve_no_boneyard_delta_cached(_board_key(b2), tuple(sorted(my2)), tuple(sorted(opp_hand)), "opponent")
                cand = int(pts + d2)
                pv = ((tile_str(t), e),) + pv2

            if cand > best:
                best = cand
                best_pv = pv
        return int(best), best_pv

    best = 10**9
    best_pv = tuple()
    for (t, e) in opp_moves:
        b2 = board.clone()
        pts = int(b2.play(t, e))
        opp2 = set(opp_hand); opp2.discard(t)

        if len(opp2) == 0:
            out_pts = round_to_nearest_5(sum(tile_pip_count(x) for x in my_hand))
            cand = int(-pts - int(out_pts))
            pv = ((tile_str(t), e),)
        else:
            d2, pv2 = _solve_no_boneyard_delta_cached(_board_key(b2), tuple(sorted(my_hand)), tuple(sorted(opp2)), "me")
            cand = int(d2 - pts)
            pv = ((tile_str(t), e),) + pv2

        if cand < best:
            best = cand
            best_pv = pv

    return int(best), best_pv


def solve_no_boneyard_delta(board: Board, my_hand: Set[Tile], opp_hand: Set[Tile], turn: PlayerName) -> Tuple[int, List[str]]:
    d, pv = _solve_no_boneyard_delta_cached(
        _board_key(board),
        tuple(sorted(my_hand)),
        tuple(sorted(opp_hand)),
        "me" if turn == "me" else "opponent"
    )
    pv_str = [f"{t}@{e}" for (t, e) in pv[:10]]
    return int(d), pv_str


def _target_terminal_v(base: GameState, my_pts_acc: int, opp_pts_acc: int, last_player: PlayerName) -> Optional[float]:
    """
    Returns terminal value v in [-1,+1] if target reached, else None.
    Uses engine.match_target_reason as the single source of truth.
    """
    reason = match_target_reason(
        my_score=int(base.my_score + my_pts_acc),
        opp_score=int(base.opponent_score + opp_pts_acc),
        target=int(base.match_target),
        last_player=last_player,
    )
    if reason == "target_me":
        return 1.0
    if reason == "target_opponent":
        return -1.0
    return None


def _match_win_prob_from_scores(my_score: int, opp_score: int, target: int) -> float:
    my_win = (int(my_score) >= int(target))
    opp_win = (int(opp_score) >= int(target))

    if my_win and not opp_win:
        return 1.0
    if opp_win and not my_win:
        return 0.0
    if my_win and opp_win:
        # Should not happen if target-immediate is enforced everywhere.
        return 0.5

    diff = float(int(my_score) - int(opp_score))
    return float(1.0 / (1.0 + math.exp(-diff / 18.0)))


# =============================================================================
# Tactical Guard Helpers (Exact enumeration for root threat assessment)
# =============================================================================

def _comb_count(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return int(math.comb(int(n), int(k)))

def _can_exact_enumerate_hidden(n: int, k: int, cap: int = 5000) -> bool:
    """
    Use exact enumeration only when combinations count is small enough.
    This makes it fast and deterministic.
    """
    c = _comb_count(n, k)
    return (c > 0) and (c <= int(cap))

def _opp_best_immediate_points(board: Board, opp_hand: Set[Tile]) -> int:
    best = 0
    for t in opp_hand:
        for e in board.legal_ends_for_tile(t):
            pts = int(immediate_points(board, t, e))
            if pts > best:
                best = pts
    return int(best)

def _root_spike_stats_exact(state: GameState, board_after: Board) -> Dict[str, float]:
    """
    Exact enumeration over opponent hands (uniform over C(n,k)).

    Returns:
      p_ge20: P(best immediate opp points >= 20)
      p_ge15: P(best immediate opp points >= 15)
      p_ge10: P(best immediate opp points >= 10)
      exp_best: E[best immediate opp points)
    """
    hidden = list(state.hidden_tiles())
    n = len(hidden)
    k = int(state.opponent_tile_count)

    if n <= 0 or k <= 0 or k > n:
        return {"p_ge20": 0.0, "p_ge15": 0.0, "p_ge10": 0.0, "exp_best": 0.0}

    if not _can_exact_enumerate_hidden(n, k, cap=5000):
        return {"p_ge20": 0.0, "p_ge15": 0.0, "p_ge10": 0.0, "exp_best": 0.0}

    total = 0
    ge20 = ge15 = ge10 = 0
    s_best = 0.0

    for hand_t in combinations(hidden, k):
        total += 1
        opp_hand = set(hand_t)
        best = _opp_best_immediate_points(board_after, opp_hand)

        s_best += float(best)
        if best >= 20: ge20 += 1
        if best >= 15: ge15 += 1
        if best >= 10: ge10 += 1

    if total <= 0:
        return {"p_ge20": 0.0, "p_ge15": 0.0, "p_ge10": 0.0, "exp_best": 0.0}

    return {
        "p_ge20": ge20 / float(total),
        "p_ge15": ge15 / float(total),
        "p_ge10": ge10 / float(total),
        "exp_best": s_best / float(total),
    }

def _apply_root_tactical_guard(state: GameState, sugs: List[Dict[str, Any]]) -> None:
    """
    Risk-sensitive guard at root. Penalize moves that allow large immediate
    opponent points on the next ply.

    Applies only when exact enumeration is feasible (C(n,k) small).
    """
    hidden_n = len(state.hidden_tiles())
    k = int(state.opponent_tile_count)

    if not _can_exact_enumerate_hidden(hidden_n, k, cap=5000):
        return

    for s in sugs:
        try:
            t = parse_tile(s["tile"])
            e = s["end"]
        except Exception:
            continue
        if e not in ENDS:
            continue

        b2 = board_after_move(state.board, t, e)
        if b2 is None:
            continue

        st = _root_spike_stats_exact(state, b2)
        p20 = float(st["p_ge20"])
        p15 = float(st["p_ge15"])
        p10 = float(st["p_ge10"])
        exp_best = float(st["exp_best"])

        # Compute a risk penalty, but DO NOT modify q or win_prob (keep semantics valid).
        penalty = 0.0
        penalty += 0.70 * p20
        penalty += 0.25 * max(0.0, (p15 - p20))
        penalty += 0.08 * max(0.0, (p10 - p15))
        penalty += (exp_best / 60.0)

        q = float(s.get("q", 0.0))
        # rank is used for ordering/score display only (UI safety). q remains pure.
        rank = float(q) - float(penalty)
        s["risk_penalty"] = float(penalty)
        s["rank"] = float(rank)
        # keep win_prob based on q (not rank)
        s["win_prob"] = round((float(q) + 1.0) / 2.0, 3)
        # score used for sorting in UI can be rank-based
        s["score"] = float(((rank + 1.0) / 2.0) * 40.0)

        rs = s.get("reasons") or []
        rs.append(
            f"tactical_guard: C({hidden_n},{k})={_comb_count(hidden_n,k)} "
            f"p20={p20:.2f} p15={p15:.2f} expBest={exp_best:.1f} pen={penalty:.3f}"
        )
        s["reasons"] = rs


# =============================================================================
# Deep v2: Root-PUCT + policy-guided rollout + corrected draw/reward
# =============================================================================

class _PUCTStats:
    __slots__ = ("N", "W", "Q", "P", "best_pv", "best_score")
    def __init__(self, P: float):
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = float(P)
        self.best_pv: List[str] = []
        self.best_score = -1e18


def _sample_action_from_policy(st: GameState, policy: np.ndarray, rng: random.Random, temperature: float = 0.8) -> Tuple[Tile, EndName]:
    legal = st.legal_moves_me()
    if not legal:
        raise RuntimeError("no legal moves")
    acts = [encode_action(t, e) for (t, e) in legal]
    probs = np.array([float(policy[a]) for a in acts], dtype=np.float64)
    s = float(probs.sum())
    if s <= 1e-12:
        return legal[int(rng.randrange(0, len(legal)))]

    probs /= s
    if temperature <= 1e-6:
        idx = int(np.argmax(probs))
        return legal[idx]

    probs = np.power(probs, 1.0 / float(temperature))
    probs = probs / (float(probs.sum()) + 1e-12)

    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if acc >= r:
            return legal[i]
    return legal[-1]


def _policy_for_state(st: GameState) -> Tuple[np.ndarray, float]:
    feats = features_small(st)
    mask = legal_mask_state(st)
    policy, value = model_predict(feats, mask)
    return policy, float(value)


def _make_world_state_for_policy(board: Board, my_hand: Set[Tile], opp_cnt: int, bone_cnt: int, base: GameState) -> GameState:
    st = GameState()
    st.board = board.clone()
    st.my_hand = set(my_hand)
    st.opponent_tile_count = int(opp_cnt)
    st.boneyard_count = int(bone_cnt)

    st.my_score = int(base.my_score)
    st.opponent_score = int(base.opponent_score)
    st.match_target = int(base.match_target)
    st.round_index = int(base.round_index)

    st.current_turn = "me"
    st.started_from_beginning = False   # critical: avoid opening-rule restriction in rollouts
    st.forced_play_tile = None
    st.round_over = False
    st.round_end_reason = None
    st.pending_out_opponent_pips = False
    st.events = []
    return st


def _rollout_round_from_world_policy_guided(
    base: GameState,
    board: Board,
    my_hand: Set[Tile],
    opp_hand: Set[Tile],
    boneyard: List[Tile],
    turn: PlayerName,
    rng: random.Random,
    max_plies: int = 120,
    me_temp: float = 0.8,
    opp_mix_greedy: float = 0.60,
    my_pts_pre: int = 0,
    opp_pts_pre: int = 0,
) -> Tuple[int, int, List[str], int, int]:
    """
    Perfect-info rollout of a round:
      - me uses model policy sampling
      - opponent uses a mix of greedy & strategic
      - draw-until-play enforced
      - optional solver when boneyard empty and small

    IMPORTANT FIX:
      The rollout must respect match_target when there are already points accumulated
      before entering the rollout (e.g., root move points or traversal points).
      So target checks use: base.score + pre_accum + rollout_accum.
    """
    my_pts = 0
    opp_pts = 0
    pv: List[str] = []
    opp_passes = 0
    opp_draws = 0

    target = int(base.match_target)

    def my_total() -> int:
        return int(base.my_score) + int(my_pts_pre) + int(my_pts)

    def opp_total() -> int:
        return int(base.opponent_score) + int(opp_pts_pre) + int(opp_pts)

    # If target already reached before rollout starts, stop immediately.
    if my_total() >= target or opp_total() >= target:
        return my_pts, opp_pts, pv, opp_passes, opp_draws

    # Optional endgame solver (kept as in your code: gated by flags)
    if USE_ENDGAME_SOLVER and len(boneyard) == 0 and (len(my_hand) + len(opp_hand) <= SOLVER_MAX_TILES):
        if SOLVER_ALLOW_SPINNER or (board.spinner_value is None):
            delta, pv_s = solve_no_boneyard_delta(board, set(my_hand), set(opp_hand), turn)
            if delta >= 0:
                my_pts += int(delta)
            else:
                opp_pts += int(-delta)
            pv.extend(pv_s)
            return my_pts, opp_pts, pv, opp_passes, opp_draws

    for _ in range(int(max_plies)):
        # Target check at top of ply
        if my_total() >= target or opp_total() >= target:
            return my_pts, opp_pts, pv, opp_passes, opp_draws

        # Out checks
        if len(my_hand) == 0:
            out_pts = out_points_from_opponent_hand(set(opp_hand))
            my_pts += int(out_pts)
            # target-immediate after out award
            tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="me")
            if tv is not None:
                return my_pts, opp_pts, pv, opp_passes, opp_draws
            return my_pts, opp_pts, pv, opp_passes, opp_draws

        if len(opp_hand) == 0:
            opp_pts += int(round_to_nearest_5(sum(tile_pip_count(t) for t in my_hand)))
            tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="opponent")
            if tv is not None:
                return my_pts, opp_pts, pv, opp_passes, opp_draws
            return my_pts, opp_pts, pv, opp_passes, opp_draws

        if turn == "me":
            st_for_policy = _make_world_state_for_policy(board, my_hand, len(opp_hand), len(boneyard), base)
            legal = st_for_policy.legal_moves_me()

            if not legal:
                # draw-until-play (forced)
                while boneyard:
                    drawn = boneyard.pop()
                    my_hand.add(drawn)
                    ends = board.legal_ends_for_tile(drawn)
                    if ends:
                        e = max(ends, key=lambda ee: immediate_points(board, drawn, ee))
                        pts = int(board.play(drawn, e))
                        my_pts += pts
                        tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="me")
                        if tv is not None:
                            return my_pts, opp_pts, pv, opp_passes, opp_draws
                        my_hand.discard(drawn)
                        pv.append(f"{tile_str(drawn)}@{e}")
                        break
                turn = "opponent"
                continue

            pol, _v = _policy_for_state(st_for_policy)
            t, e = _sample_action_from_policy(st_for_policy, pol, rng, temperature=float(me_temp))
            pts = int(board.play(t, e))
            my_pts += pts
            tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="me")
            if tv is not None:
                return my_pts, opp_pts, pv, opp_passes, opp_draws
            my_hand.discard(t)
            pv.append(f"{tile_str(t)}@{e}")
            turn = "opponent"
            continue

        # opponent turn
        opp_moves = _legal_moves_for_hand(board, opp_hand)
        if opp_moves:
            if rng.random() < float(opp_mix_greedy):
                mv = _pick_opp_move_greedy(board, opp_hand)
            else:
                mv = None
                best_sc = -1e18
                for t, e in opp_moves:
                    sc = score_opponent_move(board, t, e, opp_hand)
                    if sc > best_sc:
                        best_sc = sc
                        mv = (t, e)
            if mv is None:
                mv = opp_moves[0]

            t, e = mv
            pts = int(board.play(t, e))
            opp_pts += pts
            tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="opponent")
            if tv is not None:
                return my_pts, opp_pts, pv, opp_passes, opp_draws
            opp_hand.discard(t)
            pv.append(f"{tile_str(t)}@{e}")
            turn = "me"
        else:
            # draw-until-play then pass
            played = False
            while boneyard:
                drawn = boneyard.pop()
                opp_hand.add(drawn)
                opp_draws += 1
                ends = board.legal_ends_for_tile(drawn)
                if ends:
                    e = max(ends, key=lambda ee: immediate_points(board, drawn, ee))
                    pts = int(board.play(drawn, e))
                    opp_pts += pts
                    tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="opponent")
                    if tv is not None:
                        return my_pts, opp_pts, pv, opp_passes, opp_draws
                    opp_hand.discard(drawn)
                    pv.append(f"{tile_str(drawn)}@{e}")
                    played = True
                    break
            if not played:
                opp_passes += 1
            turn = "me"
            continue

        # locked check when boneyard empty and both have no moves
        if not boneyard:
            my_moves2 = _legal_moves_for_hand(board, my_hand)
            opp_moves2 = _legal_moves_for_hand(board, opp_hand)
            if (not my_moves2) and (not opp_moves2):
                delta = locked_delta_points(set(my_hand), set(opp_hand))
                if delta > 0:
                    my_pts += int(delta)
                    tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="me")
                    if tv is not None:
                        return my_pts, opp_pts, pv, opp_passes, opp_draws
                elif delta < 0:
                    opp_pts += int(-delta)
                    tv = _target_terminal_v(base, my_pts_pre + my_pts, opp_pts_pre + opp_pts, last_player="opponent")
                    if tv is not None:
                        return my_pts, opp_pts, pv, opp_passes, opp_draws
                return my_pts, opp_pts, pv, opp_passes, opp_draws

    return my_pts, opp_pts, pv, opp_passes, opp_draws


def _deep_v2_root_puct(
    state: GameState,
    belief: Belief,
    base_seed: int,
    det: int,
    think_ms: int,
    c_puct: float = 1.6,
    rollout_temp: float = 0.8
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    legal = sorted(state.legal_moves_me(), key=lambda m: encode_action(m[0], m[1]))
    if not legal:
        return [], {"mode": "deep_v2_no_moves"}

    # Root priors from model; if model missing => uniform over legal
    st_root = state.fast_clone(keep_events=True)
    st_root.current_turn = "me"
    pol_root, _v0 = _policy_for_state(st_root)

    priors: Dict[Tuple[Tile, EndName], float] = {}
    psum = 0.0
    for (t, e) in legal:
        p = float(pol_root[encode_action(t, e)])
        priors[(t, e)] = p
        psum += p

    if psum <= 1e-12:
        for k in priors:
            priors[k] = 1.0 / float(len(priors))
    else:
        for k in priors:
            priors[k] /= psum

    stats: Dict[Tuple[Tile, EndName], _PUCTStats] = {m: _PUCTStats(P=priors[m]) for m in priors}

    worlds = max(6, min(int(det), 28))
    rollouts_per_world = max(2, min(MAX_ROLLOUTS_PER_WORLD, int(int(think_ms) // ROLLOUT_MS_PER)))
    total_sims = int(worlds * rollouts_per_world)

    for s in range(total_sims):
        N_total = sum(st.N for st in stats.values())
        sqrtN = math.sqrt(float(max(1, N_total)))

        best_m: Optional[Tuple[Tile, EndName]] = None
        best_ucb = -1e18
        for m, stt in stats.items():
            u = float(stt.Q) + float(c_puct) * float(stt.P) * (sqrtN / float(1 + stt.N))
            if u > best_ucb:
                best_ucb = u
                best_m = m
        assert best_m is not None
        tile, end = best_m

        rr = random.Random(_move_seed(base_seed, tile, end, salt=s))
        det_world = determinize_hidden(state, belief, rr)
        opp_hand = set(det_world.opp_hand)
        boneyard = list(det_world.boneyard)
        rr.shuffle(boneyard)

        board = state.board.clone()
        my_hand = set(state.my_hand)
        my_hand.discard(tile)

        try:
            my0 = int(board.play(tile, end))
        except Exception:
            reward = 0.0
            pv = [f"{tile_str(tile)}@{end}"]
            stt = stats[best_m]
            stt.N += 1
            stt.W += reward
            stt.Q = stt.W / float(stt.N)
            continue

        if len(my_hand) == 0:
            my0 += out_points_from_opponent_hand(set(opp_hand))
            my_final = int(state.my_score + my0)
            opp_final = int(state.opponent_score)
            reward = _match_win_prob_from_scores(my_final, opp_final, int(state.match_target))
            pv = [f"{tile_str(tile)}@{end}"]
        else:
            mp, op, pv_tail, op_pass, op_draws = _rollout_round_from_world_policy_guided(
                base=state,
                board=board,
                my_hand=my_hand,
                opp_hand=opp_hand,
                boneyard=boneyard,
                turn="opponent",
                rng=rr,
                max_plies=120,
                me_temp=float(rollout_temp),
                opp_mix_greedy=0.60,
                my_pts_pre=int(my0),
                opp_pts_pre=0,
            )
            # IMPORTANT: correct reward (no delta trick)
            my_final = int(state.my_score + my0 + mp)
            opp_final = int(state.opponent_score + op)
            reward = _match_win_prob_from_scores(my_final, opp_final, int(state.match_target))
            pv = [f"{tile_str(tile)}@{end}"] + pv_tail[:7]

        stt = stats[best_m]
        stt.N += 1
        stt.W += float(reward)
        stt.Q = float(stt.W) / float(stt.N)

        if float(reward) > stt.best_score:
            stt.best_score = float(reward)
            stt.best_pv = list(pv)

    suggestions: List[Dict[str, Any]] = []
    for (t, e) in legal:
        stt = stats[(t, e)]
        ip = immediate_points(state.board, t, e)
        suggestions.append({
            "tile": tile_str(t),
            "end": e,
            "score": float(stt.Q * 40.0),
            "immediate_points": int(ip),
            "win_prob": round(float(stt.Q), 3),
            "visits": int(stt.N),
            "q": float(stt.Q),
            "prior": float(stt.P),
            "pv": " → ".join(stt.best_pv[:8]) if stt.best_pv else "",
            "analysis_depth": "PUCT(root)+policy-rollout",
            "v_next": round(float(stt.Q * 2.0 - 1.0), 3),
            "pressure": 0.0,
            "opponent_reply_points": 0.0,
            "reasons": [f"visits={stt.N}"],
            "threats": [],
        })

    suggestions.sort(key=lambda x: float(x["score"]), reverse=True)
    
    visits_sum = int(sum(st.N for st in stats.values()))
    
    meta = {
        "mode": "deep_v2_root_puct",
        "enum_used": False,
        "hands_count": int(worlds),
        "analysis_level": "deep",
        "think_ms": int(think_ms),
        "rollouts_per_world": int(rollouts_per_world),
        "total_sims": int(total_sims),
        "total_moves": int(len(legal)),
        "model_loaded": bool(MODEL is not None),
        "visits_sum": int(visits_sum),
    }
    return suggestions, meta


# =============================================================================
# Deep v3: ISMCTS (multi-ply) with opponent re-determinization (RIS-style approx)
# =============================================================================

_ISAction = Tuple[Tile, EndName]
_ISKey = Tuple[str, Tuple[Tile, ...], int, int, Optional[Tile], PlayerName]  # (board_json, my_hand, opp_cnt, bone_cnt, forced, turn)


@dataclass
class _ISEdge:
    N: int = 0
    W: float = 0.0      # from MY perspective in [-1, +1]
    Q: float = 0.0      # W/N
    P: float = 1.0      # prior weight (doesn't need to sum to 1 for non-root)
    best_score: float = -1e18
    best_pv: List[str] = None

    def __post_init__(self):
        if self.best_pv is None:
            self.best_pv = []


@dataclass
class _ISNode:
    key: _ISKey
    edges: Dict[_ISAction, _ISEdge]
    def __init__(self, key: _ISKey):
        self.key = key
        self.edges = {}


def _is_key(board: Board, my_hand: Set[Tile], opp_cnt: int, bone_cnt: int, forced: Optional[Tile], turn: PlayerName) -> _ISKey:
    return (_board_key(board), tuple(sorted(my_hand)), int(opp_cnt), int(bone_cnt), forced, turn)


def _legal_moves_me_simple(board: Board, my_hand: Set[Tile], forced: Optional[Tile]) -> List[_ISAction]:
    tiles = [forced] if (forced is not None and forced in my_hand) else list(my_hand)
    out: List[_ISAction] = []
    for t in tiles:
        for e in board.legal_ends_for_tile(t):
            out.append((t, e))
    return out


def _legal_moves_for_opp(board: Board, opp_hand: Set[Tile]) -> List[_ISAction]:
    return _collect_legal_moves(board, opp_hand)


def _redeterminize_opponent_hand(
    base_state: GameState,
    belief: Belief,
    board: Board,
    my_hand: Set[Tile],
    opp_cnt: int,
    bone_cnt: int,
    rng: random.Random
) -> Determinization:
    st = GameState()
    st.board = board.clone()
    st.my_hand = set(my_hand)
    st.opponent_tile_count = int(opp_cnt)
    st.boneyard_count = int(bone_cnt)

    # carry meta (not required for determinization, but keeps functions safe)
    st.my_score = int(base_state.my_score)
    st.opponent_score = int(base_state.opponent_score)
    st.match_target = int(base_state.match_target)
    st.round_index = int(base_state.round_index)

    st.current_turn = "me"
    st.started_from_beginning = False
    st.forced_play_tile = None
    st.events = []  # belief is passed explicitly

    return determinize_hidden(st, belief, rng)


def _ismcts_root_search(
    state: GameState,
    belief: Belief,
    base_seed: int,
    det: int,
    think_ms: int,
    max_depth: int = 6,
    c_puct: float = 1.6,
    redet_opp: bool = True,
    apply_guard: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Multi-ply ISMCTS (approx RIS-style by re-sampling opp_hand/boneyard at opponent nodes).
    - Tree is on information sets (public board + my hand + counts + forced).
    - Underlying world is determinized; at opponent nodes we may re-determinize.
    - Values backprop are from MY perspective in [-1,+1]; opponent selection minimizes my value.
    """
    if state.current_turn != "me":
        return [], {"mode": "ismcts_not_my_turn"}

    think_ms = max(150, min(int(think_ms), 15000))

    root_board = state.board.clone()
    root_my_hand = set(state.my_hand)
    root_forced = state.forced_play_tile if state.forced_play_tile in root_my_hand else None

    root_opp_cnt = int(state.opponent_tile_count)
    root_bone_cnt = int(state.boneyard_count)

    # NEW: target for match termination
    target = int(state.match_target)

    root_key = _is_key(root_board, root_my_hand, root_opp_cnt, root_bone_cnt, root_forced, "me")
    nodes: Dict[_ISKey, _ISNode] = {root_key: _ISNode(root_key)}
    root = nodes[root_key]

    # Root priors from model (only at root to avoid event/belief mismatch deeper)
    st_root = state.fast_clone(keep_events=True)
    st_root.current_turn = "me"
    pol_root, _v0 = _policy_for_state(st_root)

    root_legal = sorted(state.legal_moves_me(), key=lambda m: encode_action(m[0], m[1]))
    if not root_legal:
        return [], {"mode": "ismcts_no_moves"}

    root_priors: Dict[_ISAction, float] = {}
    ps = 0.0
    for (t, e) in root_legal:
        p = float(pol_root[encode_action(t, e)])
        root_priors[(t, e)] = p
        ps += p
    if ps <= 1e-12:
        for k in root_priors:
            root_priors[k] = 1.0 / float(len(root_priors))
    else:
        for k in root_priors:
            root_priors[k] /= ps

    # --- prior-mix: reduce sensitivity to a possibly-misaligned model ---
    mix = 0.85  # 85% model prior + 15% uniform
    u = 1.0 / float(len(root_priors))
    for k in root_priors:
        root_priors[k] = mix * float(root_priors[k]) + (1.0 - mix) * u

    # renormalize
    s = float(sum(root_priors.values()))
    for k in root_priors:
        root_priors[k] /= (s + 1e-12)

    # convert think_ms + det to a fixed simulation budget (like deep v2 does)
    worlds = max(6, min(int(det), 28))
    sims_per_world = max(6, min(26, int(think_ms) // 60))   # 1200ms -> 13, 900ms -> 10, 300ms -> 4
    total_sims = int(worlds * sims_per_world)

    max_sims = min(6000, total_sims)  # keep hard cap
    sims = 0
    redets = 0

    for sim_id in range(1, int(max_sims) + 1):
        sims += 1

        # --- World determinization at root ---
        rr = random.Random((int(base_seed) * 1315423911 + sim_id * 2654435761) & 0xFFFFFFFF)
        det_world = determinize_hidden(state, belief, rr)
        opp_hand = set(det_world.opp_hand)
        boneyard = list(det_world.boneyard)
        rr.shuffle(boneyard)

        board = root_board.clone()
        my_hand = set(root_my_hand)

        opp_cnt = int(root_opp_cnt)
        bone_cnt = int(root_bone_cnt)

        forced = root_forced
        turn: PlayerName = "me"

        my_pts_acc = 0
        opp_pts_acc = 0

        # path to update: list of (node_key, action)
        path: List[Tuple[_ISKey, _ISAction]] = []

        # --- Traverse/Expand ---
        depth = 0
        node_key = root_key
        
        # NEW: terminal_v for immediate match termination during traversal
        terminal_v: Optional[float] = None

        while depth < int(max_depth):
            node = nodes.get(node_key)
            if node is None:
                node = _ISNode(node_key)
                nodes[node_key] = node

            # Opponent re-determinization (RIS-style approx)
            if redet_opp and turn == "opponent":
                dd = _redeterminize_opponent_hand(state, belief, board, my_hand, opp_cnt, bone_cnt, rr)
                opp_hand = set(dd.opp_hand)
                boneyard = list(dd.boneyard)
                rr.shuffle(boneyard)
                redets += 1

            # legal moves (play only). draws/passes handled in rollout at leaf.
            if turn == "me":
                legal = _legal_moves_me_simple(board, my_hand, forced)
            else:
                legal = _legal_moves_for_opp(board, opp_hand)

            if not legal:
                break

            # ensure edges exist for seen legal actions
            for a in legal:
                if a not in node.edges:
                    if node_key == root_key:
                        P = float(root_priors.get(a, 1e-6))
                    else:
                        P = 1.0
                    node.edges[a] = _ISEdge(P=P)

            # pick an untried action first (expansion), else PUCT select
            untried = [a for a in legal if node.edges[a].N == 0]
            if untried:
                if node_key == root_key:
                    # pick by prior (deterministic-ish)
                    a = max(untried, key=lambda x: float(node.edges[x].P))
                else:
                    a = untried[int(rr.randrange(0, len(untried)))]
            else:
                # PUCT selection:
                parent_N = sum(node.edges[a].N for a in legal)
                sqrtN = math.sqrt(float(max(1, parent_N)))

                best_a = legal[0]
                best_ucb = -1e18
                for a2 in legal:
                    ed = node.edges[a2]
                    U = float(c_puct) * float(ed.P) * (sqrtN / float(1 + ed.N))
                    # opponent minimizes my value => maximize (-Q) + U
                    val = (-float(ed.Q)) if turn == "opponent" else float(ed.Q)
                    ucb = val + U
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_a = a2
                a = best_a

            # apply action to world
            t, e = a
            try:
                pts = int(board.play(t, e))
            except Exception:
                break

            if turn == "me":
                if t not in my_hand:
                    break
                my_hand.discard(t)
                my_pts_acc += pts
                if forced == t:
                    forced = None
                turn = "opponent"
            else:
                if t not in opp_hand:
                    break
                opp_hand.discard(t)
                opp_pts_acc += pts
                opp_cnt = max(0, opp_cnt - 1)
                turn = "me"

            # NEW: immediate match termination DURING traversal
            tv = _target_terminal_v(state, my_pts_acc, opp_pts_acc, last_player=("opponent" if turn == "me" else "me"))
            if tv is not None:
                terminal_v = float(tv)
                break

            # update child key and path
            child_key = _is_key(board, my_hand, opp_cnt, bone_cnt, forced, turn)
            path.append((node_key, a))
            node_key = child_key
            depth += 1

            # quick terminal checks (optional optimization)
            if len(my_hand) == 0 or len(opp_hand) == 0:
                break

        # --- Rollout finish from leaf (perfect-info world) ---
        if terminal_v is None:
            mp, op, pv_tail, _op_pass, _op_draw = _rollout_round_from_world_policy_guided(
                base=state,
                board=board,
                my_hand=my_hand,
                opp_hand=opp_hand,
                boneyard=boneyard,
                turn=turn,
                rng=rr,
                max_plies=70,
                me_temp=0.8,
                opp_mix_greedy=0.60,
                my_pts_pre=int(my_pts_acc),
                opp_pts_pre=int(opp_pts_acc),
            )

            my_final = int(state.my_score + my_pts_acc + mp)
            opp_final = int(state.opponent_score + opp_pts_acc + op)
            win_prob = float(_match_win_prob_from_scores(my_final, opp_final, int(state.match_target)))
            v = float(2.0 * win_prob - 1.0)  # in [-1,+1]
            pv_full = [f"{tile_str(t)}@{e}" for (_k, (t, e)) in path[:8]] + list(pv_tail[:8])
        else:
            v = float(terminal_v)
            pv_full = [f"{tile_str(t)}@{e}" for (_k, (t, e)) in path[:8]]

        # --- Backprop along path (MY perspective) ---
        # update edges on visited nodes
        # also keep best pv on root edge only
        for (k, a) in path:
            node = nodes.get(k)
            if node is None:
                continue
            ed = node.edges.get(a)
            if ed is None:
                continue
            ed.N += 1
            ed.W += float(v)
            ed.Q = float(ed.W) / float(ed.N)

        if path:
            rk, ra = path[0]
            if rk == root_key:
                ed0 = root.edges.get(ra)
                if ed0 and float(v) > float(ed0.best_score):
                    ed0.best_score = float(v)
                    ed0.best_pv = list(pv_full)

    # --- Build root suggestions ---
    sugs: List[Dict[str, Any]] = []
    for (t, e) in root_legal:
        ed = root.edges.get((t, e), _ISEdge(P=float(root_priors.get((t, e), 0.0))))
        winp = float((ed.Q + 1.0) / 2.0)
        sugs.append({
            "tile": tile_str(t),
            "end": e,
            "visits": int(ed.N),
            "q": float(ed.Q),
            "prior": float(ed.P),
            "win_prob": round(winp, 3),
            "score": float(winp * 40.0),
            "immediate_points": int(immediate_points(state.board, t, e)),
            "pv": " → ".join(ed.best_pv[:8]) if ed.best_pv else "",
            "analysis_depth": f"ISMCTS(depth={max_depth})",
            "reasons": [f"visits={ed.N}", f"q={ed.Q:.3f}"],
            "threats": [],
        })

    # Optional UI safety: apply guard to compute rank/score without corrupting q
    if apply_guard:
        _apply_root_tactical_guard(state, sugs)

    # Sort: prefer rank if present (guarded UI), else fall back to q
    sugs.sort(
        key=lambda x: (
            float(x.get("rank", x.get("q", 0.0))),
            float(x.get("q", 0.0)),
            int(x.get("visits", 0)),
        ),
        reverse=True
    )

    meta = {
        "mode": "ismcts",
        "analysis_level": "ismcts",
        "think_ms": int(think_ms),
        "det": int(det),
        "max_depth": int(max_depth),
        "c_puct": float(c_puct),
        "sims": int(sims),
        "nodes": int(len(nodes)),
        "redets": int(redets),
        "model_loaded": bool(MODEL is not None),
        "apply_guard": bool(apply_guard),
    }
    return sugs, meta


# =============================================================================
# Main suggestions
# =============================================================================

def suggest_moves(
    state: GameState,
    top_n: int = 5,
    determinizations: int = 12,
    seed: Optional[int] = None,
    analysis_level: str = "standard",
    think_ms: int = 900,
) -> Dict[str, Any]:
    """
    analysis_level:
      - quick: fast
      - standard: enum if possible else 2-ply sampling (model-agnostic)
      - deep: Deep v2 (Root-PUCT + policy rollout)
      - ismcts: ISMCTS multi-ply with opponent re-determinization
    """
    if state.current_turn != "me":
        belief = build_belief_from_state(state)
        return {
            "meta": {"mode": "not_my_turn", "analysis_level": (analysis_level or "standard"), "enum_used": False, "hands_count": 0},
            "belief": belief.to_dict(),
            "suggestions": []
        }

    legal_moves = state.legal_moves_me()
    belief = build_belief_from_state(state)

    if not legal_moves:
        meta = {"mode": "no_moves", "analysis_level": (analysis_level or "standard"), "enum_used": False, "hands_count": 0}
        return {"meta": meta, "belief": belief.to_dict(), "suggestions": []}

    base_seed = int(seed) if seed is not None else secrets.randbits(31)

    think_ms = int(think_ms or 900)
    think_ms = max(150, min(think_ms, 15000))

    level = (analysis_level or "standard").lower().strip()
    if level not in ("quick", "standard", "deep", "ismcts"):
        level = "standard"

    legal_sorted = sorted(legal_moves, key=lambda m: encode_action(m[0], m[1]))

    # ---------------- ISMCTS ----------------
    if level == "ismcts":
        suggestions, meta = _ismcts_root_search(
            state=state,
            belief=belief,
            base_seed=base_seed,
            det=int(determinizations),
            think_ms=int(think_ms),
            max_depth=6,
            c_puct=1.6,
            redet_opp=True,
            apply_guard=True,
        )
        return {
            "meta": meta,
            "belief": belief.to_dict(),
            "suggestions": suggestions[:int(top_n)]
        }

    # ---------------- DEEP (v2) ----------------
    if level == "deep":
        suggestions, meta = _deep_v2_root_puct(
            state=state,
            belief=belief,
            base_seed=base_seed,
            det=int(determinizations),
            think_ms=int(think_ms),
            c_puct=1.6,
            rollout_temp=0.8
        )
        return {
            "meta": meta,
            "belief": belief.to_dict(),
            "suggestions": suggestions[:int(top_n)]
        }

    # ---------------- QUICK / STANDARD ----------------
    if level == "quick":
        deep_evaluation_count = min(8, len(legal_moves))
        samples_per_move = max(4, min(int(determinizations // 2 or 4), 10))
    else:
        deep_evaluation_count = len(legal_moves) if len(legal_moves) <= 28 else 18
        samples_per_move = max(18, min(int(determinizations) * 2, 36))

    can_enum, hands_count = can_enumerate(state)
    evaluated_moves: List[Dict[str, Any]] = []

    # ENUM (standard only)
    if can_enum and level != "quick":
        hands, hand_probs = _enumerate_opponent_hands_with_weights(state, belief)

        boneyard_samples_per_hand = 1
        if int(state.boneyard_count) > 0:
            boneyard_samples_per_hand = max(1, min(4, int(determinizations // 4) or 1))

        for (tile, end) in legal_sorted:
            board_after = board_after_move(state.board, tile, end)
            if board_after is None:
                continue

            enum_eval = _evaluate_move_by_enumeration(
                state=state,
                tile=tile,
                end=end,
                belief=belief,
                hands=hands,
                hand_probs=hand_probs,
                rng_seed=base_seed,
                boneyard_samples_per_hand=boneyard_samples_per_hand
            )

            total_score = (
                float(enum_eval["immediate_points"]) * IMMEDIATE_POINTS_FACTOR
                - float(enum_eval["opponent_avg_points"]) * OPPONENT_AVG_PENALTY
                - float(enum_eval["opponent_worst_points"]) * OPPONENT_WORST_PENALTY
                + float(enum_eval["my_reply_avg_points"]) * MY_REPLY_AVG_BONUS
                + float(enum_eval["my_reply_best_points"]) * MY_REPLY_BEST_BONUS
                + float(enum_eval["opponent_pass_rate"]) * OPPONENT_PASS_BONUS
                + float(enum_eval["opponent_draw_avg"]) * OPPONENT_DRAW_FACTOR
                + (DOUBLE_TILE_BONUS if tile_is_double(tile) else 0.0)
            )

            threats = analyze_opponent_threats(board_after, belief, state.my_hand - {tile})
            threat_warnings = [t["warning"] for t in threats[:2]]

            evaluated_moves.append({
                "tile": tile_str(tile),
                "end": end,
                "score": float(total_score),
                "immediate_points": int(enum_eval["immediate_points"]),
                "opponent_avg_points": round(float(enum_eval["opponent_avg_points"]), 2),
                "opponent_worst_points": int(enum_eval["opponent_worst_points"]),
                "my_reply_avg": round(float(enum_eval["my_reply_avg_points"]), 2),
                "my_reply_best": int(enum_eval["my_reply_best_points"]),
                "opponent_pass_rate": round(float(enum_eval["opponent_pass_rate"]), 3),
                "opponent_draw_avg": round(float(enum_eval["opponent_draw_avg"]), 2),
                "reasons": [],
                "threats": threat_warnings,
                "analysis_depth": "ENUM+2ply",
                "v_next": 0.0,
                "pressure": float(enum_eval["opponent_pass_rate"]),
                "opponent_reply_points": float(enum_eval["opponent_avg_points"]),
            })

        evaluated_moves.sort(key=lambda x: x["score"], reverse=True)
        return {
            "meta": {
                "mode": "enumeration",
                "enum_used": True,
                "hands_count": int(hands_count),
                "model_loaded": bool(MODEL is not None),
                "boneyard_samples_per_hand": int(boneyard_samples_per_hand),
                "analysis_level": level,
                "total_moves": int(len(legal_moves)),
                "rng_per_move": True,
            },
            "belief": belief.to_dict(),
            "suggestions": evaluated_moves[:int(top_n)]
        }

    # fallback 2-ply sampling
    ranked_moves: List[Tuple[Tuple[Tile, EndName], float]] = []
    for (t, e) in legal_sorted:
        ranked_moves.append(((t, e), _rank_move_for_search(state.board, t, e)))
    ranked_moves.sort(key=lambda x: x[1], reverse=True)

    for idx, (move, _rk) in enumerate(ranked_moves):
        tile, end = move
        immediate_pts = immediate_points(state.board, tile, end)
        board_after = board_after_move(state.board, tile, end)

        if idx < deep_evaluation_count and board_after:
            mv_rng = random.Random(_move_seed(base_seed, tile, end, salt=0))
            eval_result = evaluate_move_2ply(state, tile, end, belief, mv_rng, num_samples=samples_per_move)

            total_score = (
                float(eval_result["immediate_points"]) * IMMEDIATE_POINTS_FACTOR
                - float(eval_result["opponent_avg_points"]) * OPPONENT_AVG_PENALTY
                - float(eval_result["opponent_worst_points"]) * OPPONENT_WORST_PENALTY
                + float(eval_result["my_reply_avg_points"]) * MY_REPLY_AVG_BONUS
                + float(eval_result["my_reply_best_points"]) * MY_REPLY_BEST_BONUS
                + float(eval_result["opponent_pass_rate"]) * OPPONENT_PASS_BONUS
                + float(eval_result["opponent_draw_avg"]) * OPPONENT_DRAW_FACTOR
                + (DOUBLE_TILE_BONUS if tile_is_double(tile) else 0.0)
            )

            evaluated_moves.append({
                "tile": tile_str(tile),
                "end": end,
                "score": float(total_score),
                "immediate_points": int(immediate_pts),
                "opponent_avg_points": round(float(eval_result["opponent_avg_points"]), 2),
                "opponent_worst_points": int(eval_result["opponent_worst_points"]),
                "my_reply_avg": round(float(eval_result["my_reply_avg_points"]), 2),
                "my_reply_best": int(eval_result["my_reply_best_points"]),
                "opponent_pass_rate": round(float(eval_result["opponent_pass_rate"]), 2),
                "opponent_draw_avg": round(float(eval_result["opponent_draw_avg"]), 2),
                "reasons": [],
                "threats": [],
                "analysis_depth": "2-ply",
                "v_next": 0.0,
                "pressure": float(eval_result["opponent_pass_rate"]),
                "opponent_reply_points": float(eval_result["opponent_avg_points"]),
            })
        else:
            quick_score = float(immediate_pts) * 2.5 + (2.0 if tile_is_double(tile) else 0.0)
            evaluated_moves.append({
                "tile": tile_str(tile),
                "end": end,
                "score": float(quick_score),
                "immediate_points": int(immediate_pts),
                "opponent_avg_points": 0.0,
                "opponent_worst_points": 0,
                "my_reply_avg": 0.0,
                "my_reply_best": 0,
                "opponent_pass_rate": 0.0,
                "opponent_draw_avg": 0.0,
                "reasons": [],
                "threats": [],
                "analysis_depth": "1-ply",
                "v_next": 0.0,
                "pressure": 0.0,
                "opponent_reply_points": 0.0,
            })

    evaluated_moves.sort(key=lambda x: x["score"], reverse=True)

    return {
        "meta": {
            "mode": "2-ply",
            "enum_used": False,
            "hands_count": int(determinizations),
            "model_loaded": bool(MODEL is not None),
            "deep_evaluated": int(deep_evaluation_count),
            "samples_per_move": int(samples_per_move),
            "analysis_level": level,
            "think_ms": int(think_ms),
            "total_moves": int(len(legal_moves)),
            "rng_per_move": True,
        },
        "belief": belief.to_dict(),
        "suggestions": evaluated_moves[:int(top_n)]
    }


# =============================================================================
# What-If
# =============================================================================

def what_if(state: GameState, tile_string: str, end: EndName) -> Dict[str, Any]:
    tile = parse_tile(tile_string)
    belief_before = build_belief_from_state(state)

    before_sum = int(state.board.ends_sum())
    before_score = int(state.board.score_now())

    before_state = {
        "sum": before_sum,
        "score_now": before_score,
        "ends_sum": before_sum,
        "current_score": before_score,
        "cut_prob": belief_before.to_dict()["cut_prob"]
    }

    state_copy = GameState.from_dict(state.to_dict())
    state_copy.current_turn = "me"
    event = state_copy.play_tile("me", tile, end)

    belief_after = build_belief_from_state(state_copy)
    threats = analyze_opponent_threats(state_copy.board, belief_after, state_copy.my_hand)

    after_sum = int(state_copy.board.ends_sum())
    after_score = int(state_copy.board.score_now())

    after_state = {
        "sum": after_sum,
        "score_now": after_score,
        "ends_sum": after_sum,
        "current_score": after_score,
        "cut_prob": belief_after.to_dict()["cut_prob"],
        "threats": threats
    }

    return {"move": event.to_dict(), "before": before_state, "after": after_state}


# =============================================================================
# Enumeration utility
# =============================================================================

def nCk(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def can_enumerate(state: GameState) -> Tuple[bool, int]:
    hidden = state.hidden_tiles()
    n = len(hidden)
    k = min(int(state.opponent_tile_count), n)
    combinations_count = nCk(n, k)
    return (0 < combinations_count <= ENUM_MAX_HANDS), int(combinations_count)