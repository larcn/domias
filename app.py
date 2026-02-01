# FILE: app.py | version: 2025-12-20.v5
# (FEATURES_V2_FULL compat: trains hidden=256; job cleanup uses completed_at; strict-turn; safe suggest think_ms fallback)

from __future__ import annotations

from flask import Flask, request, jsonify, Response
from typing import Dict, Any, Optional, List, Set, Tuple
import json
import os
import threading
import uuid
import time
import secrets
import traceback
from collections import OrderedDict

from engine import GameState, Board, GameEvent, parse_tile, tile_str
import ai
import storage
import train

app = Flask(__name__)

# Rust extension (optional; UI can fallback to python suggestions)
try:
    import domino_rs  # type: ignore
except Exception:
    domino_rs = None  # type: ignore

os.environ.setdefault("DOMINO_INFER", "0")
os.environ.setdefault("DOMINO_AVOID_BETA", "0")

# =============================================================================
# Sessions (LRU + TTL) + per-session locks
# =============================================================================

class SessionStore:
    """Thread-safe session store with TTL+LRU and per-session locks."""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 6 * 3600):
        self.max_size = int(max_size)
        self.ttl_seconds = int(ttl_seconds)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, GameState]" = OrderedDict()
        self._ts: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = {}

    def _get_or_create_session_lock(self, key: str) -> threading.Lock:
        lk = self._locks.get(key)
        if lk is None:
            lk = threading.Lock()
            self._locks[key] = lk
        return lk

    def lock_for(self, key: str) -> threading.Lock:
        with self._lock:
            return self._get_or_create_session_lock(key)

    def get(self, key: str) -> Optional[GameState]:
        now = time.time()
        with self._lock:
            st = self._data.get(key)
            if st is None:
                return None

            ts = self._ts.get(key, 0.0)
            if now - ts > self.ttl_seconds:
                self._data.pop(key, None)
                self._ts.pop(key, None)
                self._locks.pop(key, None)
                return None

            self._data.move_to_end(key)
            self._ts[key] = now
            self._get_or_create_session_lock(key)
            return st

    def set(self, key: str, value: GameState) -> None:
        now = time.time()
        with self._lock:
            if key in self._data:
                self._data[key] = value
                self._data.move_to_end(key)
                self._ts[key] = now
                self._get_or_create_session_lock(key)
                return

            while len(self._data) >= self.max_size:
                oldest_key, _ = self._data.popitem(last=False)
                self._ts.pop(oldest_key, None)
                self._locks.pop(oldest_key, None)

            self._data[key] = value
            self._ts[key] = now
            self._get_or_create_session_lock(key)

    def cleanup(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            keys = list(self._data.keys())
            for k in keys:
                ts = self._ts.get(k, 0.0)
                if now - ts > self.ttl_seconds:
                    self._data.pop(k, None)
                    self._ts.pop(k, None)
                    self._locks.pop(k, None)
                    removed += 1
        return removed


SESSIONS = SessionStore(max_size=200, ttl_seconds=6 * 3600)
SESSION_CLEANUP_INTERVAL = 900
_LAST_SESSION_CLEANUP = time.time()


def cleanup_old_sessions() -> None:
    global _LAST_SESSION_CLEANUP
    now = time.time()
    if now - _LAST_SESSION_CLEANUP < SESSION_CLEANUP_INTERVAL:
        return
    SESSIONS.cleanup()
    _LAST_SESSION_CLEANUP = now


# =============================================================================
# Training jobs
# =============================================================================

JOBS: Dict[str, Dict[str, Any]] = {}
JLOCK = threading.Lock()

JOB_CLEANUP_INTERVAL = 3600
LAST_CLEANUP = time.time()

# Single-flight: prevent concurrent training jobs in-process (SQLite + CPU stability).
TRAIN_SINGLEFLIGHT = threading.Lock()


def cleanup_old_jobs() -> None:
    """
    Cleanup finished jobs based on completed_at (not created_at),
    so long-running jobs are not deleted immediately after finishing.
    """
    global LAST_CLEANUP
    now = time.time()
    if now - LAST_CLEANUP < JOB_CLEANUP_INTERVAL:
        return

    with JLOCK:
        to_delete = []
        for job_id, job in JOBS.items():
            status = job.get("status", "")
            created_at = float(job.get("created_at", 0) or 0)
            completed_at = float(job.get("completed_at", 0) or 0)
            finished_at = completed_at or created_at

            if status in ("done", "error") and now - finished_at > 3600:
                to_delete.append(job_id)
            elif status == "queued" and now - created_at > 86400:
                to_delete.append(job_id)

        for job_id in to_delete:
            JOBS.pop(job_id, None)

    LAST_CLEANUP = now


# =============================================================================
# Helpers
# =============================================================================

def ok(payload: Dict[str, Any] | None = None):
    return jsonify({"ok": True, **(payload or {})})


def err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def sid() -> str:
    data = request.json if request.is_json else {}
    return (data or {}).get("session_id") or request.args.get("session_id") or "default"


def get_state(session_id: str) -> Optional[GameState]:
    return SESSIONS.get(session_id)


def _normalize_save_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    return name if name.endswith(".json") else name + ".json"


def _parse_tiles_list(x: Any, field: str) -> List[Tuple[int, int]]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"{field} must be a list")
    tiles = [parse_tile(s) for s in x]
    if len(set(tiles)) != len(tiles):
        raise ValueError(f"{field} contains duplicate tiles")
    return tiles


def _build_board_from_parts(center_tile: Optional[str], arms_raw: Dict[str, Any]) -> Board:
    """
    Build a Board from midgame inputs by replaying:
      center -> right -> left -> up -> down
    """
    b = Board()
    if not center_tile:
        return b

    ct = parse_tile(center_tile)

    arms = arms_raw or {}
    for k in ["right", "left", "up", "down"]:
        if k not in arms:
            arms[k] = []
        if not isinstance(arms[k], list):
            raise ValueError(f"arms.{k} must be a list")

    right = [parse_tile(s) for s in arms["right"]]
    left = [parse_tile(s) for s in arms["left"]]
    up = [parse_tile(s) for s in arms["up"]]
    down = [parse_tile(s) for s in arms["down"]]

    all_arm_tiles = right + left + up + down
    if len(set(all_arm_tiles)) != len(all_arm_tiles):
        raise ValueError("arms contains duplicate tiles")

    b.play(ct, "right")
    for t in right:
        b.play(t, "right")
    for t in left:
        b.play(t, "left")
    for t in up:
        b.play(t, "up")
    for t in down:
        b.play(t, "down")

    return b


def _assert_invariant_28(my_hand: Set[Tuple[int, int]], played: Set[Tuple[int, int]], opp_cnt: int, bone_cnt: int) -> None:
    total = int(len(my_hand) + len(played) + int(opp_cnt) + int(bone_cnt))
    if total != 28:
        raise ValueError(
            f"Invariant broken: my_hand({len(my_hand)}) + played({len(played)}) + opp({opp_cnt}) + bone({bone_cnt}) = {total} != 28"
        )


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def index() -> Response:
    with open("ui.html", "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html; charset=utf-8")


@app.get("/api/rules")
def api_rules() -> Response:
    if not os.path.exists("RULES.md"):
        return Response("RULES.md not found", status=404, mimetype="text/plain; charset=utf-8")
    with open("RULES.md", "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/plain; charset=utf-8")


@app.post("/api/new_game")
def api_new_game():
    """Start a new Match (reset scores) + Round #1."""
    try:
        data = request.json or {}
        session_id = sid()
        hand = data.get("my_hand", [])

        match_target = int(data.get("match_target", 150))
        match_target = max(50, min(match_target, 1000))

        if not isinstance(hand, list) or len(hand) != 7:
            return err("my_hand must be list of 7 tiles")

        try:
            tiles = [parse_tile(x) for x in hand]
        except ValueError as e:
            return err(f"Invalid tile: {str(e)}")

        if len(set(tiles)) != 7:
            return err("my_hand contains duplicate tiles")

        st = GameState()
        st.start_new_game(tiles, match_target=match_target)
        SESSIONS.set(session_id, st)
        return ok({"session_id": session_id, "state": st.to_dict()})
    except Exception as e:
        return err(str(e))


@app.post("/api/new_round")
def api_new_round():
    """Start a new Round in same Match (keep scores)."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        hand = data.get("my_hand", [])
        if not isinstance(hand, list) or len(hand) != 7:
            return err("my_hand must be list of 7 tiles")

        try:
            tiles = [parse_tile(x) for x in hand]
        except ValueError as e:
            return err(f"Invalid tile: {str(e)}")

        if len(set(tiles)) != 7:
            return err("my_hand contains duplicate tiles")

        with SESSIONS.lock_for(session_id):
            st.start_new_round(tiles)
            return ok({"session_id": session_id, "state": st.to_dict()})

    except Exception as e:
        return err(str(e))


@app.post("/api/setup_midgame")
def api_setup_midgame():
    """
    Midgame setup for analysis.
    """
    try:
        data = request.json or {}
        session_id = sid()

        my_hand_in = data.get("my_hand", [])
        my_tiles = _parse_tiles_list(my_hand_in, "my_hand")
        my_hand: Set[Tuple[int, int]] = set(my_tiles)
        if len(my_hand) == 0:
            return err("my_hand is required (enter your remaining tiles midgame)")

        current_turn = str(data.get("current_turn", "me"))
        if current_turn not in ("me", "opponent"):
            return err("current_turn must be me/opponent")

        opp_cnt = int(data.get("opponent_tile_count", 7))
        if opp_cnt < 0 or opp_cnt > 28:
            return err("opponent_tile_count must be in [0..28]")

        board_snapshot = data.get("board_snapshot")
        board_parts = data.get("board") or {}
        try:
            if board_snapshot is not None:
                if not isinstance(board_snapshot, dict):
                    return err("board_snapshot must be an object")
                board_obj = Board.from_snapshot(board_snapshot)
            else:
                center_tile = board_parts.get("center_tile")
                arms_raw = board_parts.get("arms") or {}
                if arms_raw and not isinstance(arms_raw, dict):
                    return err("board.arms must be an object")
                board_obj = _build_board_from_parts(center_tile, arms_raw)
        except Exception as e:
            return err(f"Invalid board: {str(e)}")

        played = set(board_obj.played_set)

        if my_hand & played:
            inter = sorted([tile_str(t) for t in (my_hand & played)])
            return err(f"Invalid setup: tiles appear in both my_hand and board.played: {inter}")

        computed_bone = 28 - len(my_hand) - len(played) - int(opp_cnt)
        if computed_bone < 0:
            return err(
                f"Invariant impossible: 28 - my_hand({len(my_hand)}) - played({len(played)}) - opp({opp_cnt}) = {computed_bone} < 0"
            )

        provided_bone = data.get("boneyard_count", None)
        if provided_bone is not None:
            try:
                provided_bone_i = int(provided_bone)
            except Exception:
                return err("boneyard_count must be an integer if provided")
            if provided_bone_i != int(computed_bone):
                return err(
                    f"boneyard_count mismatch: provided={provided_bone_i} but computed={computed_bone} "
                    f"(based on invariant 28). Fix opponent_tile_count or board/my_hand."
                )

        match_target = int(data.get("match_target", 150))
        match_target = max(50, min(match_target, 1000))

        round_index = int(data.get("round_index", 1))
        round_index = max(1, min(round_index, 9999))

        my_score = max(0, int(data.get("my_score", 0)))
        opp_score = max(0, int(data.get("opponent_score", 0)))

        forced_play_tile = data.get("forced_play_tile", None)
        forced_tile_parsed: Optional[Tuple[int, int]] = None
        if forced_play_tile:
            try:
                forced_tile_parsed = parse_tile(str(forced_play_tile))
            except Exception as e:
                return err(f"Invalid forced_play_tile: {str(e)}")
            if forced_tile_parsed not in my_hand:
                return err("forced_play_tile must be in my_hand for a consistent state")

        st = GameState()
        st.board = board_obj
        st.my_hand = set(my_hand)

        st.match_target = int(match_target)
        st.round_index = int(round_index)
        st.my_score = int(my_score)
        st.opponent_score = int(opp_score)

        st.current_turn = current_turn
        st.started_from_beginning = False

        st.opponent_tile_count = int(opp_cnt)
        st.boneyard_count = int(computed_bone)

        st.forced_play_tile = forced_tile_parsed

        st.round_over = False
        st.round_end_reason = None
        st.pending_out_opponent_pips = False

        # Seed events (for belief)
        st.events = [
            GameEvent(type="match_start", ply=0),
            GameEvent(type="round_start", ply=1),
            GameEvent(type="snapshot_load", ply=2),
        ]
        ply = 3
        open_ends = sorted(list(st.board.open_end_values()))

        opp_draw_count = int(data.get("opponent_draw_count", 0) or 0)
        opp_draw_cert = str(data.get("opponent_draw_certainty", "probable") or "probable")
        if opp_draw_cert not in ("certain", "probable", "possible"):
            opp_draw_cert = "probable"

        if opp_draw_count > 0:
            st.events.append(GameEvent(
                type="draw",
                ply=ply,
                player="opponent",
                draw_count=int(opp_draw_count),
                certainty=opp_draw_cert,
                open_ends=open_ends,
            ))
            ply += 1

        opp_passed = bool(data.get("opponent_passed", False))
        if opp_passed:
            st.events.append(GameEvent(
                type="pass",
                ply=ply,
                player="opponent",
                certainty="certain",
                open_ends=open_ends,
            ))
            ply += 1

        _assert_invariant_28(st.my_hand, st.board.played_set, st.opponent_tile_count, st.boneyard_count)

        st._undo.clear()
        st._redo.clear()

        SESSIONS.set(session_id, st)
        return ok({"session_id": session_id, "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.post("/api/import_state")
def api_import_state():
    """Import a full GameState dict into a session."""
    try:
        data = request.json or {}
        session_id = sid()

        state_dict = data.get("state")
        if not isinstance(state_dict, dict):
            return err("state must be an object (GameState JSON)")

        st = GameState.from_dict(state_dict)

        total = int(len(st.my_hand) + len(st.board.played_set) + int(st.opponent_tile_count) + int(st.boneyard_count))
        if total != 28:
            return err(f"Invariant broken in imported state: {total} != 28")

        SESSIONS.set(session_id, st)
        return ok({"session_id": session_id, "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.get("/api/state")
def api_state():
    session_id = request.args.get("session_id", "default")
    st = get_state(session_id)
    if st is None:
        return err("no active session", 404)

    with SESSIONS.lock_for(session_id):
        payload = st.to_dict()

    return ok({"session_id": session_id, "state": payload})


@app.post("/api/play")
def api_play():
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            player = data.get("player", "me")
            if player not in ("me", "opponent"):
                return err("player must be me/opponent")

            try:
                tile = parse_tile(data.get("tile", ""))
            except ValueError as e:
                return err(f"Invalid tile: {str(e)}")

            end = data.get("end", "right")
            if end not in ("right", "left", "up", "down"):
                return err("invalid end")

            if st.board.is_empty() and end != "right":
                return err("First move must be played on end='right'")

            ev = st.play_tile(player, tile, end)
            return ok({"event": ev.to_dict(), "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.post("/api/draw")
def api_draw():
    """Record opponent draw only (unknown tiles)."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            player = data.get("player", "opponent")
            if player != "opponent":
                return err("Only opponent draw is supported here. Use /api/draw_me for player 'me'.")

            if st.current_turn != "opponent":
                return err(f"Not opponent's turn (current_turn={st.current_turn}). Use 'set turn' first if needed.")

            count = int(data.get("count", 1))
            certainty = data.get("certainty", "probable")
            if certainty not in ("certain", "probable", "possible"):
                return err("certainty must be certain/probable/possible")

            ev = st.record_draw("opponent", count=count, certainty=certainty)
            return ok({"event": ev.to_dict(), "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.post("/api/draw_me")
def api_draw_me():
    """Legal draw for me (tile known)."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            t = (data.get("tile") or "").strip()
            if not t:
                return err("tile required, e.g. '6-5'")

            try:
                tile = parse_tile(t)
            except ValueError as e:
                return err(f"Invalid tile: {str(e)}")

            ev = st.record_draw_me(tile)
            return ok({"event": ev.to_dict(), "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.post("/api/pass")
def api_pass():
    """Record pass (me/opponent)."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            player = data.get("player", "opponent")
            if player not in ("me", "opponent"):
                return err("player must be me/opponent")

            certainty = (data.get("certainty") or "certain")
            if certainty not in ("certain", "probable", "possible"):
                certainty = "certain"

            ev = st.record_pass(player, certainty=certainty)
            return ok({"event": ev.to_dict(), "state": st.to_dict()})

    except Exception as e:
        return err(str(e))


@app.post("/api/declare_locked")
def api_declare_locked():
    """Declare locked (requires opponent_pips)."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        opponent_pips = int(data.get("opponent_pips", -1))
        if opponent_pips < 0:
            return err("opponent_pips required (>=0)")

        with SESSIONS.lock_for(session_id):
            ev = st.declare_locked(opponent_pips=opponent_pips)
            return ok({"event": ev.to_dict(), "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.post("/api/finalize_out")
def api_finalize_out():
    """Finalize out when state is out_me_pending."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        opponent_pips = int(data.get("opponent_pips", -1))
        if opponent_pips < 0:
            return err("opponent_pips required (>=0)")

        with SESSIONS.lock_for(session_id):
            ev = st.finalize_out_with_opponent_pips(opponent_pips=opponent_pips)
            return ok({"event": ev.to_dict(), "state": st.to_dict()})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.post("/api/undo")
def api_undo():
    session_id = sid()
    st = get_state(session_id)
    if st is None:
        return err("no active session", 404)

    with SESSIONS.lock_for(session_id):
        if not st.undo():
            return err("nothing to undo", 400)
        return ok({"state": st.to_dict()})


@app.post("/api/redo")
def api_redo():
    session_id = sid()
    st = get_state(session_id)
    if st is None:
        return err("no active session", 404)

    with SESSIONS.lock_for(session_id):
        if not st.redo():
            return err("nothing to redo", 400)
        return ok({"state": st.to_dict()})


@app.post("/api/set_turn")
def api_set_turn():
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            turn = data.get("current_turn", "me")
            if turn not in ("me", "opponent"):
                return err("current_turn must be me/opponent")
            st.current_turn = turn
            return ok({"state": st.to_dict()})

    except Exception as e:
        return err(str(e))


def _immediate_points_after_move(st: GameState, tile_s: str, end_s: str) -> int:
    """
    Compute immediate points for a move on the current board snapshot.
    (Used for UI display when Rust returns solver-only decision without root_candidates.)
    """
    try:
        t = parse_tile(tile_s)
        b2 = st.board.clone()
        pts = b2.play(t, end_s)
        return int(pts)
    except Exception:
        return 0


def _rust_pick_with_root_candidates(
    st_snapshot: GameState,
    top_n: int,
    det: int,
    think_ms: int,
) -> Dict[str, Any]:
    """
    Rust runtime suggestion + UI-friendly candidates list.
    Uses domino_rs.suggest_move_ismcts(...) with return_root_stats=True.
    """
    if domino_rs is None:
        raise RuntimeError("domino_rs not available")

    sd = st_snapshot.to_dict()

    res = domino_rs.suggest_move_ismcts(  # type: ignore[attr-defined]
        sd,
        det=int(det),
        think_ms=int(think_ms),
        seed=int(secrets.randbits(64) or 1),
        model_path=None,
        opp_mix_greedy=1.0,
        leaf_value_weight=0.0,
        me_mix_greedy=1.0,
        gift_penalty_weight=0.15,
        pessimism_alpha_max=0.0,
        enable_guard=True,
        guard_top_k=25,
        guard_worlds=64,
        guard_close_threshold=50,
        return_root_stats=True,
        root_stats_top_k=max(12, int(top_n)),
    )

    sugs: List[Dict[str, Any]] = []
    cands = res.get("root_candidates") or []
    if isinstance(cands, list):
        for c in cands[:max(1, int(top_n))]:
            if not isinstance(c, dict):
                continue
            tile = c.get("tile")
            end = c.get("end")
            if not isinstance(tile, str) or not isinstance(end, str):
                continue
            sugs.append({
                "tile": tile,
                "end": end,
                "visits": int(c.get("visits") or 0),
                "immediate_points": int(c.get("immediate_points") or 0),
                # Keep ordering intuitive in UI
                "score": float(c.get("visits") or 0),
                "analysis_depth": "rust_ismcts",
                "win_prob": None,
                "pv": "",
            })

    try:
        belief = ai.build_belief_from_state(st_snapshot).to_dict()
    except Exception:
        belief = {}

    meta = {
        "engine": "rust",
        "analysis_level": "rust",
        "think_ms": int(think_ms),
        "det": int(det),
        "chosen": {"tile": res.get("tile"), "end": res.get("end")},
        "chosen_by": res.get("chosen_by", None),
        "rust_version": (domino_rs.version() if domino_rs is not None else None),  # type: ignore[attr-defined]
    }

    return {
        "state": st_snapshot.to_dict(),
        "meta": meta,
        "belief": belief,
        "suggestions": sugs,
    }


@app.post("/api/suggest")
def api_suggest():
    """Suggestions with analysis_level + optional think_ms (fallback-safe)."""
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            top_n = int(data.get("top_n", 5))
            det = int(data.get("determinizations", 12))
            top_n = max(1, min(top_n, 10))
            det = max(2, min(det, 60))

            analysis_level = str(data.get("analysis_level", "rust") or "rust").strip().lower()
            if analysis_level not in ("rust", "quick", "standard", "deep", "ismcts"):
                analysis_level = "rust"

            think_ms = int(data.get("think_ms", 900) or 900)
            think_ms = max(150, min(think_ms, 15000))

            # Analyze a snapshot to avoid holding the lock during heavy compute
            st_snapshot = st.fast_clone(keep_events=True)

        # --- Rust runtime path (supports solver endgames) ---
        if analysis_level == "rust" and domino_rs is not None:
            try:
                res = domino_rs.suggest_move_ismcts(  # type: ignore[attr-defined]
                    st_snapshot.to_dict(),
                    det=int(det),
                    think_ms=int(think_ms),
                    seed=int(secrets.randbits(64) or 1),
                    model_path=None,
                    opp_mix_greedy=1.0,
                    leaf_value_weight=0.0,
                    me_mix_greedy=1.0,
                    gift_penalty_weight=0.15,
                    pessimism_alpha_max=0.0,
                    enable_guard=True,
                    guard_top_k=25,
                    guard_worlds=64,
                    guard_close_threshold=50,
                    return_root_stats=True,
                    root_stats_top_k=max(12, int(top_n)),
                )

                # Belief for UI
                try:
                    belief = ai.build_belief_from_state(st_snapshot).to_dict()
                except Exception:
                    belief = {}

                sugs: List[Dict[str, Any]] = []
                cands = res.get("root_candidates") or []
                if isinstance(cands, list):
                    for c in cands[:top_n]:
                        if not isinstance(c, dict):
                            continue
                        tile = c.get("tile")
                        end = c.get("end")
                        if not isinstance(tile, str) or not isinstance(end, str):
                            continue
                        sugs.append({
                            "tile": tile,
                            "end": end,
                            "visits": int(c.get("visits") or 0),
                            "immediate_points": int(c.get("immediate_points") or 0),
                            "score": float(c.get("visits") or 0),
                            "analysis_depth": str(res.get("chosen_by") or "rust"),
                            "win_prob": None,
                            "pv": "",
                        })

                # IMPORTANT FIX:
                # Solver path returns early from Rust WITHOUT root_candidates.
                # In that case, show at least the chosen move as a single suggestion.
                if not sugs:
                    tile = res.get("tile")
                    end = res.get("end")
                    if isinstance(tile, str) and isinstance(end, str):
                        sugs.append({
                            "tile": tile,
                            "end": end,
                            "visits": 0,
                            "immediate_points": _immediate_points_after_move(st_snapshot, tile, end),
                            "score": 999999.0,  # keep it at top in UI
                            "analysis_depth": str(res.get("chosen_by") or "solver"),
                            "win_prob": None,
                            "pv": "",
                        })

                meta = {
                    "engine": "rust",
                    "analysis_level": "rust",
                    "think_ms": int(think_ms),
                    "det": int(det),
                    "chosen_by": res.get("chosen_by", None),
                }

                return ok({
                    "state": st_snapshot.to_dict(),
                    "meta": meta,
                    "belief": belief,
                    "suggestions": sugs,
                })
            except Exception:
                # fallback to python below
                pass

        # heavy compute outside lock
        try:
            r = ai.suggest_moves(
                st_snapshot,
                top_n=top_n,
                determinizations=det,
                seed=None,
                analysis_level=analysis_level,
                think_ms=think_ms,
            )
        except TypeError:
            r = ai.suggest_moves(
                st_snapshot,
                top_n=top_n,
                determinizations=det,
                seed=None,
                analysis_level=analysis_level,
            )

        meta = dict(r.get("meta", {}) or {})
        meta["analysis_level"] = analysis_level
        meta["think_ms"] = think_ms
        meta["engine"] = "python"
        meta["rust_available"] = bool(domino_rs is not None)

        return ok({
            "state": st_snapshot.to_dict(),
            "meta": meta,
            "belief": r.get("belief", {}),
            "suggestions": r.get("suggestions", [])
        })

    except Exception as e:
        return err(str(e))


@app.post("/api/play_best")
def api_play_best():
    """
    Execute best move for ME using Rust runtime (max strength).
    This makes the UI actually 'play' by the engine.
    """
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)
        if domino_rs is None:
            return err("domino_rs not available (build Rust extension first)", 409)

        det = int(data.get("determinizations", 28))
        think_ms = int(data.get("think_ms", 4000))
        det = max(2, min(det, 60))
        think_ms = max(150, min(think_ms, 15000))

        with SESSIONS.lock_for(session_id):
            if st.round_over:
                return err("round is over", 400)
            if st.current_turn != "me":
                return err("not me's turn", 400)

            # Snapshot for decision
            st_snapshot = st.fast_clone(keep_events=True)

        # Decide outside lock
        rep = _rust_pick_with_root_candidates(st_snapshot, top_n=12, det=det, think_ms=think_ms)
        chosen = (rep.get("meta") or {}).get("chosen") or {}
        tile_s = chosen.get("tile")
        end_s = chosen.get("end")
        if not isinstance(tile_s, str) or not isinstance(end_s, str):
            return err("rust did not return a valid move", 500)
        if end_s not in ("right", "left", "up", "down"):
            return err("invalid end from rust", 500)

        t = parse_tile(tile_s)

        with SESSIONS.lock_for(session_id):
            ev = st.play_tile("me", t, end_s)
            return ok({
                "event": ev.to_dict(),
                "state": st.to_dict(),
                "meta": rep.get("meta", {}),
                "suggestions": rep.get("suggestions", []),
            })

    except Exception as e:
        return err(str(e))


@app.post("/api/what_if")
def api_what_if():
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        tile = data.get("tile", "")
        end = data.get("end", "right")
        if end not in ("right", "left", "up", "down"):
            return err("invalid end")

        if st.board.is_empty() and end != "right":
            return err("First move must be played on end='right'")

        with SESSIONS.lock_for(session_id):
            return ok({"result": ai.what_if(st, tile, end)})

    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@app.get("/api/model_status")
def api_model_status():
    if not os.path.exists("model.json"):
        return ok({"exists": False})

    try:
        with open("model.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        return ok({
            "exists": True,
            "type": d.get("type"),
            "hidden": d.get("hidden"),
            "feat_dim": d.get("feat_dim"),
        })
    except Exception:
        return ok({"exists": True, "type": "unknown"})


@app.post("/api/save")
def api_save():
    try:
        data = request.json or {}
        session_id = sid()
        st = get_state(session_id)
        if st is None:
            return err("no active session", 404)

        with SESSIONS.lock_for(session_id):
            name = (data.get("name") or "").strip() or None
            overwrite = bool(data.get("overwrite", False))

            if overwrite:
                if not name:
                    return err("name required for overwrite=true")
                fn = storage.save_game_as(st, filename=name, overwrite=True)
            else:
                # normal "new save" mode (unique timestamped)
                fn = storage.save_game(st, name=name)

            return ok({"saved_as": fn, "saves": storage.list_saves()})

    except FileNotFoundError as e:
        return err(str(e), 404)
    except FileExistsError as e:
        return err(str(e), 409)
    except Exception as e:
        return err(str(e))


@app.post("/api/load")
def api_load():
    try:
        data = request.json or {}
        session_id = sid()
        name = (data.get("name") or "").strip()
        if not name:
            return err("name required")

        normalized = storage._normalize_save_filename(name)
        saves = storage.list_saves()
        if normalized not in saves:
            return err("save not found", 404)

        st = storage.load_game(name)
        SESSIONS.set(session_id, st)
        return ok({"state": st.to_dict(), "saves": saves})

    except FileNotFoundError as e:
        return err(str(e), 404)
    except Exception as e:
        return err(str(e))


@app.get("/api/list_saves")
def api_list_saves():
    return ok({"saves": storage.list_saves()})


@app.post("/api/delete_save")
def api_delete_save():
    try:
        data = request.json or {}
        name = (data.get("name") or "").strip()
        if not name:
            return err("name required")
        deleted = storage.delete_save(name)
        return ok({"deleted": deleted, "saves": storage.list_saves()})
    except FileNotFoundError as e:
        return err(str(e), 404)
    except Exception as e:
        return err(str(e))


@app.post("/api/delete_saves")
def api_delete_saves():
    try:
        data = request.json or {}
        names = data.get("names")
        if not isinstance(names, list) or not names:
            return err("names must be a non-empty list")
        rep = storage.delete_saves([str(x) for x in names])
        return ok({"report": rep, "saves": storage.list_saves()})
    except Exception as e:
        return err(str(e))


@app.get("/api/export_state")
def api_export_state():
    session_id = request.args.get("session_id", "default")
    st = get_state(session_id)
    if st is None:
        return err("no active session", 404)

    with SESSIONS.lock_for(session_id):
        payload = st.to_dict()

    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        mimetype="application/json; charset=utf-8"
    )


@app.get("/api/export_log")
def api_export_log():
    session_id = request.args.get("session_id", "default")
    st = get_state(session_id)
    if st is None:
        return err("no active session", 404)

    with SESSIONS.lock_for(session_id):
        txt = storage.export_log_text(st)

    return Response(txt, mimetype="text/plain; charset=utf-8")


# =============================================================================
# Training thread
# =============================================================================

def _train_job(job_id: str, games: int, det: int, epochs: int) -> None:
    with TRAIN_SINGLEFLIGHT:
        try:
            with JLOCK:
                JOBS[job_id]["status"] = "running"
                JOBS[job_id]["progress"] = 5

            seed = secrets.randbits(31)

            added = train.generate_samples(
                games=games,
                determinizations=det,
                temperature=0.85,
                max_moves=500,      # per-round safety cap in rebuild
                seed=seed,
                think_ms=1200,
            )

            with JLOCK:
                JOBS[job_id]["progress"] = 55
                JOBS[job_id]["added_samples"] = int(added)

            rep = train.train_model(
                epochs=epochs,
                batch_size=512,
                learning_rate=0.0015,
                l2_lambda=1e-4,
                hidden_size=256,     # IMPORTANT for FEAT_DIM=193
                seed=None,
                model_path="model.json"
            )

            with JLOCK:
                JOBS[job_id]["status"] = "done" if rep.get("ok") else "error"
                JOBS[job_id]["progress"] = 100
                JOBS[job_id]["report"] = rep
                JOBS[job_id]["completed_at"] = time.time()
                if not rep.get("ok"):
                    JOBS[job_id]["error"] = rep.get("error")

        except Exception as e:
            tb = traceback.format_exc()
            with JLOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["progress"] = 100
                JOBS[job_id]["error"] = str(e)
                JOBS[job_id]["traceback"] = tb
                JOBS[job_id]["completed_at"] = time.time()


@app.post("/api/train_start")
def api_train_start():
    try:
        cleanup_old_jobs()

        with JLOCK:
            for job in JOBS.values():
                if job.get("status") in ("queued", "running"):
                    return err("Training already running. Try again after it finishes.", 409)

        data = request.json or {}
        games = int(data.get("games", 200))      # self-play match is heavier; adjust as needed
        det = int(data.get("det", 12))
        epochs = int(data.get("epochs", 3))

        games = max(50, min(games, 50000))
        det = max(6, min(det, 30))
        epochs = max(1, min(epochs, 60))

        job_id = uuid.uuid4().hex[:10]

        with JLOCK:
            JOBS[job_id] = {
                "status": "queued",
                "progress": 0,
                "games": games,
                "det": det,
                "epochs": epochs,
                "added_samples": 0,
                "report": None,
                "error": None,
                "traceback": None,
                "created_at": time.time(),
                "completed_at": None
            }

        th = threading.Thread(target=_train_job, args=(job_id, games, det, epochs), daemon=True)
        th.start()
        return ok({"job_id": job_id})

    except ValueError as e:
        return err(f"Invalid parameter: {str(e)}")
    except Exception as e:
        return err(str(e))


@app.get("/api/train_status")
def api_train_status():
    job_id = request.args.get("job_id", "")
    if not job_id:
        return err("job_id required")

    cleanup_old_jobs()

    with JLOCK:
        job = JOBS.get(job_id)
        if not job:
            return err("job not found", 404)

        payload = dict(job)
        payload["job_id"] = job_id

    return ok(payload)


@app.before_request
def before_request():
    cleanup_old_jobs()
    cleanup_old_sessions()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)