# FILE: engine.py | version: 2025-12-21.v7
# (spinner upgraded: FIRST double played becomes spinner even if not first tile;
#  reroots board around that double; strict-turn + invariant 28 preserved)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Literal, Any
from datetime import datetime

EndName = Literal["right", "left", "up", "down"]
PlayerName = Literal["me", "opponent"]
Certainty = Literal["certain", "probable", "possible"]
Tile = Tuple[int, int]
OpeningMode = Literal["forced_best", "free"]

# =============================================================================
# Ruleset identity (bump when semantics change)
# =============================================================================
RULESET_ID = "fives_house_v3_target_immediate"


def match_target_reason(my_score: int, opp_score: int, target: int, last_player: PlayerName) -> Optional[str]:
    """
    Returns a terminal reason if match target has been reached.
    Assumption: "target ends immediately" and the last_player just scored.
    """
    t = int(target)
    my_win = int(my_score) >= t
    opp_win = int(opp_score) >= t

    if my_win and not opp_win:
        return "target_me"
    if opp_win and not my_win:
        return "target_opponent"

    # Should not normally happen when enforced immediately, but keep deterministic behavior:
    if my_win and opp_win:
        # If both reached in a weird state, award to the side who just played.
        return "target_me" if last_player == "me" else "target_opponent"

    return None


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def round_to_nearest_5(x: int) -> int:
    # Nearest-5: round(x/5)*5 (works with RULES.md examples)
    return int(round(float(int(x)) / 5.0) * 5)


def norm_tile(a: int, b: int) -> Tile:
    if a < b:
        a, b = b, a
    if not (0 <= a <= 6 and 0 <= b <= 6):
        raise ValueError(f"Tile out of range: {a}-{b}")
    return (a, b)


def parse_tile(s: str) -> Tile:
    s = (s or "").strip().replace("[", "").replace("]", "").replace(" ", "")
    s = s.replace("|", "-").replace(",", "-")
    if "-" in s:
        a, b = s.split("-", 1)
        return norm_tile(int(a), int(b))
    if len(s) == 2 and s.isdigit():
        return norm_tile(int(s[0]), int(s[1]))
    raise ValueError(f"Cannot parse tile: {s}")


def tile_str(t: Tile) -> str:
    return f"{t[0]}-{t[1]}"


def tile_is_double(t: Tile) -> bool:
    return t[0] == t[1]


def best_opening_tile(hand: Set[Tile]) -> Optional[Tile]:
    """
    Opening rule:
      - If you have doubles: must open with the highest double.
      - Else: must open with the highest tile by (pip_sum, hi, lo).
    """
    if not hand:
        return None
    doubles = [t for t in hand if tile_is_double(t)]
    if doubles:
        return max(doubles, key=lambda t: t[0])  # 6-6 > 5-5 > ...
    return max(hand, key=lambda t: (t[0] + t[1], t[0], t[1]))


def tile_has(t: Tile, v: int) -> bool:
    return t[0] == v or t[1] == v


def tile_pip_count(t: Tile) -> int:
    return t[0] + t[1]


def other_value(t: Tile, v: int) -> int:
    if t[0] == v:
        return t[1]
    if t[1] == v:
        return t[0]
    raise ValueError(f"{tile_str(t)} does not contain {v}")


def all_tiles() -> List[Tile]:
    out: List[Tile] = []
    for hi in range(7):
        for lo in range(hi + 1):
            out.append((hi, lo))
    return out


ALL_TILES: List[Tile] = all_tiles()
ALL_SET: Set[Tile] = set(ALL_TILES)


@dataclass
class Board:
    """
    Domino Fives board with a spinner.

    NEW RULE (2025-12-21):
      - Spinner is created by the FIRST double that is played in the round,
        even if it is not the first tile.
      - When that happens, the board is re-rooted around that double (it becomes center_tile).
      - Up/Down open only after BOTH left and right arms from spinner have started (phase2).
    """

    center_tile: Optional[Tile] = None

    # Spinner state:
    spinner_value: Optional[int] = None
    spinner_sides_open: bool = False

    # ends: end_name -> (open_value, is_double_open_end)
    ends: Dict[EndName, Tuple[int, bool]] = field(default_factory=dict)
    arms: Dict[EndName, List[Tile]] = field(
        default_factory=lambda: {"right": [], "left": [], "up": [], "down": []}
    )

    played_set: Set[Tile] = field(default_factory=set)
    played_order: List[Tile] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.center_tile is None

    def open_end_values(self) -> Set[int]:
        """
        Return numeric open-end values (0..6), not end names.

        This feeds GameEvent.open_ends and therefore directly affects:
          - Belief updates from pass/draw events
          - Any downstream inference relying on open_ends
        """
        # ends: Dict[EndName, Tuple[int, bool]]
        return {int(val) for (_name, (val, _is_dbl)) in self.ends.items()}

    def legal_ends_for_tile(self, t: Tile) -> List[EndName]:
        if self.is_empty():
            return ["right"]
        out: List[EndName] = []
        for e, (v, _d) in self.ends.items():
            if tile_has(t, v):
                out.append(e)
        return out

    def _arm_started(self, e: EndName) -> bool:
        return len(self.arms.get(e, [])) > 0

    # -------------------------
    # Scoring / ends sum
    # -------------------------
    def ends_sum(self) -> int:
        if self.is_empty() or self.center_tile is None:
            return 0

        # Spinner scoring applies only if we have a spinner and center_tile is the spinner double.
        if self.spinner_value is not None and tile_is_double(self.center_tile):
            sv = int(self.spinner_value)

            started_right = self._arm_started("right")
            started_left = self._arm_started("left")
            started_up = self._arm_started("up")
            started_down = self._arm_started("down")
            started_any = started_right or started_left or started_up or started_down

            # spinner alone
            if not started_any:
                return sv * 2

            # phase1: only one of right/left started
            if (started_right and not started_left) or (started_left and not started_right):
                started_end: EndName = "right" if started_right else "left"
                if started_end not in self.ends:
                    return sv * 2
                val, is_dbl = self.ends[started_end]
                return int(sv * 2 + ((int(val) * 2) if bool(is_dbl) else int(val)))

            # phase2: right and left started => spinner not counted; up/down only if played
            total = 0
            for end_name, (value, is_dbl) in self.ends.items():
                if end_name in ("up", "down") and not self._arm_started(end_name):
                    continue
                total += (int(value) * 2) if bool(is_dbl) else int(value)
            return int(total)

        # non-spinner (or inconsistent snapshot): sum all open ends
        total2 = 0
        for _end_name, (value, is_dbl) in self.ends.items():
            total2 += (int(value) * 2) if bool(is_dbl) else int(value)
        return int(total2)

    def score_now(self) -> int:
        s = int(self.ends_sum())
        return int(s) if (s % 5 == 0) else 0

    # -------------------------
    # Core play
    # -------------------------
    def play(self, t: Tile, end: EndName) -> int:
        if t in self.played_set:
            raise ValueError(f"Tile already played: {tile_str(t)}")

        if self.is_empty():
            # First tile: does not necessarily create spinner unless it's a double
            pts = self._play_first(t)
            return int(pts)

        if end not in self.ends:
            raise ValueError(f"End not open: {end}")

        end_val, _ = self.ends[end]
        if not tile_has(t, end_val):
            raise ValueError(f"Illegal: {tile_str(t)} cannot go on {end}({end_val})")

        new_val = other_value(t, end_val)

        # apply move normally
        self._add_played(t)
        self.arms[end].append(t)
        self.ends[end] = (int(new_val), tile_is_double(t))

        # NEW: if this is the first double of the round, promote it to spinner
        if self.spinner_value is None and tile_is_double(t):
            self._promote_first_double_to_spinner(spinner_tile=t, played_end=end)

        # open up/down if spinner exists and phase2 achieved
        self._check_open_spinner_sides()

        return int(self.score_now())

    def _play_first(self, t: Tile) -> int:
        self.center_tile = t
        self._add_played(t)
        self.arms = {"right": [], "left": [], "up": [], "down": []}

        if tile_is_double(t):
            # First tile is a double => it becomes spinner immediately
            self.spinner_value = int(t[0])
            self.spinner_sides_open = False
            self.ends = {"right": (int(t[0]), False), "left": (int(t[0]), False)}
        else:
            self.spinner_value = None
            self.spinner_sides_open = False
            self.ends = {"right": (int(t[0]), False), "left": (int(t[1]), False)}

        return int(self.score_now())

    def _check_open_spinner_sides(self) -> None:
        # open up/down only if spinner exists AND right+left arms started
        if self.spinner_value is None or self.spinner_sides_open:
            return
        if self._arm_started("right") and self._arm_started("left"):
            self.spinner_sides_open = True
            self.ends["up"] = (int(self.spinner_value), False)
            self.ends["down"] = (int(self.spinner_value), False)

    def _add_played(self, t: Tile) -> None:
        self.played_set.add(t)
        self.played_order.append(t)

    # -------------------------
    # Spinner promotion (reroot)
    # -------------------------
    def _promote_first_double_to_spinner(self, spinner_tile: Tile, played_end: EndName) -> None:
        """
        Promote the first double played (not necessarily first tile) to be the spinner (center).

        Current board must be a line (no spinner yet), so the double must have been played on 'right' or 'left'.
        We rebuild a new Board around the spinner tile and replay the existing chain on the opposite arm.
        """
        if self.center_tile is None:
            return
        if self.spinner_value is not None:
            return
        if not tile_is_double(spinner_tile):
            return
        if played_end not in ("right", "left"):
            # If somehow called with up/down (shouldn't happen when spinner_value is None), do nothing.
            return

        # Snapshot pieces we must preserve (chronological played order)
        old_order = list(self.played_order)

        ct0 = self.center_tile
        if ct0 is None:
            return

        arms0 = {k: list(v) for k, v in self.arms.items()}

        path = arms0.get(played_end, [])
        if not path or path[-1] != spinner_tile:
            # Unexpected, refuse to reroot
            return

        opp_end: EndName = "left" if played_end == "right" else "right"

        # chain away from spinner goes to the opposite direction
        chain_arm: EndName = opp_end  # if spinner was on right, chain goes left; if on left, chain goes right
        free_arm: EndName = played_end

        path_wo = path[:-1]  # exclude spinner itself
        opp_arm_tiles = arms0.get(opp_end, [])

        # Order from spinner outward:
        #   reverse(tiles between old center and spinner) + old center + tiles on opposite arm
        chain_tiles: List[Tile] = list(reversed(path_wo)) + [ct0] + list(opp_arm_tiles)

        # Build new board centered on spinner
        b2 = Board()
        b2._play_first(spinner_tile)  # sets spinner_value, ends right/left=v

        # Replay chain on chain_arm
        for t in chain_tiles:
            b2.play(t, chain_arm)

        # Ensure free arm exists and is open at spinner value (it will be, via _play_first)
        # Ensure up/down are not open until phase2
        b2.spinner_sides_open = False
        b2.ends.pop("up", None)
        b2.ends.pop("down", None)

        # Copy rebuilt topology back
        self.center_tile = b2.center_tile
        self.spinner_value = b2.spinner_value
        self.spinner_sides_open = b2.spinner_sides_open
        self.ends = dict(b2.ends)
        self.arms = {k: list(v) for k, v in b2.arms.items()}

        # Preserve original chronological played order
        self.played_order = old_order
        self.played_set = set(old_order)

        # Sanity: ensure line arm assignment matches expectation
        # (chain arm should be started; free arm may be empty)
        _ = chain_arm
        _ = free_arm

    # -------------------------
    # Snapshot / clone
    # -------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "ruleset": RULESET_ID,
            "center_tile": tile_str(self.center_tile) if self.center_tile else None,
            "spinner_value": self.spinner_value,
            "spinner_sides_open": self.spinner_sides_open,
            "ends": {e: [v, bool(d)] for e, (v, d) in self.ends.items()},
            "arms": {e: [tile_str(x) for x in self.arms[e]] for e in ["right", "left", "up", "down"]},
            "played_tiles": [tile_str(x) for x in self.played_order],
            "ends_sum": self.ends_sum(),
            "current_score": self.score_now(),
            "is_empty": self.is_empty(),
        }

    @classmethod
    def from_snapshot(cls, d: Dict[str, Any]) -> "Board":
        b = cls()
        ct = d.get("center_tile")
        b.center_tile = parse_tile(ct) if ct else None
        b.spinner_value = d.get("spinner_value", None)
        b.spinner_sides_open = bool(d.get("spinner_sides_open", False))

        ends_raw = d.get("ends", {}) or {}
        b.ends = {}
        for k, v in ends_raw.items():
            if k in ("right", "left", "up", "down"):
                b.ends[k] = (int(v[0]), bool(v[1]) if len(v) > 1 else False)

        arms_raw = d.get("arms", {}) or {}
        b.arms = {"right": [], "left": [], "up": [], "down": []}
        for k in b.arms.keys():
            b.arms[k] = [parse_tile(s) for s in (arms_raw.get(k, []) or [])]

        played = d.get("played_tiles", []) or []
        b.played_order = [parse_tile(s) for s in played]
        b.played_set = set(b.played_order)
        return b

    def clone(self) -> "Board":
        b = Board()
        b.center_tile = self.center_tile
        b.spinner_value = self.spinner_value
        b.spinner_sides_open = self.spinner_sides_open
        b.ends = dict(self.ends)
        b.arms = {k: list(v) for k, v in self.arms.items()}
        b.played_set = set(self.played_set)
        b.played_order = list(self.played_order)
        return b


@dataclass
class GameEvent:
    type: Literal["match_start", "round_start", "snapshot_load", "play", "draw", "pass", "round_end"]
    ts: str = field(default_factory=now_ts)
    ply: int = 0
    player: Optional[PlayerName] = None
    tile: Optional[str] = None
    end: Optional[EndName] = None
    draw_count: int = 0
    open_ends: List[int] = field(default_factory=list)
    certainty: Certainty = "certain"
    score_gained: int = 0

    end_reason: Optional[str] = None
    end_award_me: int = 0
    end_award_opp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "ts": self.ts,
            "ply": self.ply,
            "player": self.player,
            "tile": self.tile,
            "end": self.end,
            "draw_count": self.draw_count,
            "open_ends": list(self.open_ends),
            "certainty": self.certainty,
            "score_gained": self.score_gained,
            "end_reason": self.end_reason,
            "end_award_me": int(self.end_award_me),
            "end_award_opp": int(self.end_award_opp),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameEvent":
        return cls(
            type=d["type"],
            ts=d.get("ts", now_ts()),
            ply=int(d.get("ply", 0)),
            player=d.get("player"),
            tile=d.get("tile"),
            end=d.get("end"),
            draw_count=int(d.get("draw_count", 0)),
            open_ends=list(d.get("open_ends", [])),
            certainty=d.get("certainty", "certain"),
            score_gained=int(d.get("score_gained", 0)),
            end_reason=d.get("end_reason"),
            end_award_me=int(d.get("end_award_me", 0)),
            end_award_opp=int(d.get("end_award_opp", 0)),
        )


@dataclass
class GameState:
    board: Board = field(default_factory=Board)
    my_hand: Set[Tile] = field(default_factory=set)
    opponent_tile_count: int = 7
    boneyard_count: int = 14

    # match score (persist across rounds)
    my_score: int = 0
    opponent_score: int = 0
    match_target: int = 150
    round_index: int = 1

    current_turn: PlayerName = "me"
    started_from_beginning: bool = True

    forced_play_tile: Optional[Tile] = None

    # round end
    round_over: bool = False
    round_end_reason: Optional[str] = None
    pending_out_opponent_pips: bool = False

    events: List[GameEvent] = field(default_factory=list)

    # NEW: disable undo snapshots for AI simulation clones (performance)
    undo_enabled: bool = True

    _undo: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _redo: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    # NEW: opening mode (forced_best/free)
    opening_mode: OpeningMode = "forced_best"

    def ply(self) -> int:
        return len(self.events)

    def visible_tiles(self) -> Set[Tile]:
        return set(self.my_hand) | set(self.board.played_set)

    def hidden_tiles(self) -> List[Tile]:
        visible = self.visible_tiles()
        return [t for t in ALL_TILES if t not in visible]

    def tile_conservation_total(self) -> int:
        return int(len(self.my_hand) + len(self.board.played_set) + int(self.opponent_tile_count) + int(self.boneyard_count))

    def my_hand_pips(self) -> int:
        return int(sum(tile_pip_count(t) for t in self.my_hand))

    def _recalc_boneyard(self) -> None:
        rem = 28 - len(self.visible_tiles()) - int(self.opponent_tile_count)
        self.boneyard_count = max(0, int(rem))

    def _assert_round_active(self) -> None:
        if self.round_over:
            raise ValueError(f"Round is over ({self.round_end_reason}). Start a new round to continue.")

    # ---- match/round lifecycle ----

    def start_new_game(self, my_hand: List[Tile], match_target: int = 150) -> None:
        if len(my_hand) != 7:
            raise ValueError("New match requires exactly 7 tiles")
        if len(set(my_hand)) != 7:
            raise ValueError("Duplicate tile in hand")

        self.match_target = int(match_target) if int(match_target) > 0 else 150
        self.my_score = 0
        self.opponent_score = 0
        self.round_index = 1
        self.events = [GameEvent(type="match_start", ply=0)]
        self.opening_mode = "forced_best"
        self._start_round_internal(my_hand)

    def start_new_round(self, my_hand: List[Tile]) -> None:
        if len(my_hand) != 7:
            raise ValueError("New round requires exactly 7 tiles")
        if len(set(my_hand)) != 7:
            raise ValueError("Duplicate tile in hand")

        prev_reason = self.round_end_reason  # <-- مهم

        self.round_index += 1

        # out/target: treat as "winner starts free" (even though match should end, this keeps semantics sane)
        if prev_reason in ("out_me", "out_opponent", "target_me", "target_opponent"):
            self.opening_mode = "free"
        else:
            # locked_* / locked_tie / أي شيء آخر: treat as new game opening rule
            self.opening_mode = "forced_best"

        self._start_round_internal(my_hand)

    def _start_round_internal(self, my_hand: List[Tile]) -> None:
        self.board = Board()
        self.my_hand = set(my_hand)
        self.opponent_tile_count = 7
        self.forced_play_tile = None

        self.round_over = False
        self.round_end_reason = None
        self.pending_out_opponent_pips = False

        self.current_turn = "me"
        self.started_from_beginning = True
        self._recalc_boneyard()

        self._undo.clear()
        self._redo.clear()

        self.events.append(GameEvent(type="round_start", ply=self.ply()))

    # ---- undo/redo ----

    def push_undo(self) -> None:
        if not self.undo_enabled:
            return
        self._undo.append(self.to_dict())
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        cur = self.to_dict()
        prev = self._undo.pop()
        self._redo.append(cur)
        self._restore(prev)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        cur = self.to_dict()
        nxt = self._redo.pop()
        self._undo.append(cur)
        self._restore(nxt)
        return True

    def fast_clone(self, keep_events: bool = False) -> "GameState":
        st = GameState()
        st.board = self.board.clone()
        st.my_hand = set(self.my_hand)

        st.opponent_tile_count = int(self.opponent_tile_count)
        st.boneyard_count = int(self.boneyard_count)

        st.my_score = int(self.my_score)
        st.opponent_score = int(self.opponent_score)
        st.match_target = int(self.match_target)
        st.round_index = int(self.round_index)

        st.current_turn = self.current_turn
        st.started_from_beginning = self.started_from_beginning

        st.forced_play_tile = self.forced_play_tile

        st.round_over = self.round_over
        st.round_end_reason = self.round_end_reason
        st.pending_out_opponent_pips = self.pending_out_opponent_pips

        st.events = list(self.events) if keep_events else []
        st._undo = []
        st._redo = []
        st.undo_enabled = False
        st.opening_mode = self.opening_mode
        return st

    # ---- legal moves/draw flags ----

    def legal_moves_me(self) -> List[Tuple[Tile, EndName]]:
        if self.forced_play_tile is not None and self.forced_play_tile not in self.my_hand:
            self.forced_play_tile = None

        # Enforce opening rule when starting from beginning and board is empty.
        opening_tile: Optional[Tile] = None
        if (
            self.opening_mode == "forced_best"
            and self.started_from_beginning
            and self.current_turn == "me"
            and self.board.is_empty()
            and self.forced_play_tile is None
        ):
            opening_tile = best_opening_tile(self.my_hand)

        tiles_iter = (
            [opening_tile] if opening_tile is not None
            else ([self.forced_play_tile] if self.forced_play_tile is not None else list(self.my_hand))
        )

        out: List[Tuple[Tile, EndName]] = []
        for t in tiles_iter:
            if t is None:
                continue
            for e in self.board.legal_ends_for_tile(t):
                out.append((t, e))
        return out

    def must_draw_me(self) -> bool:
        if self.round_over or self.current_turn != "me":
            return False
        if self.forced_play_tile is not None:
            return False
        return (len(self.legal_moves_me()) == 0) and (int(self.boneyard_count) > 0)

    def must_pass_me(self) -> bool:
        if self.round_over or self.current_turn != "me":
            return False
        if self.forced_play_tile is not None:
            return False
        return (len(self.legal_moves_me()) == 0) and (int(self.boneyard_count) == 0)

    # ---- round end helpers ----

    def _finalize_round(self, reason: str, award_me: int = 0, award_opp: int = 0) -> GameEvent:
        self.round_over = True
        self.round_end_reason = reason
        self.pending_out_opponent_pips = False

        self.my_score += int(award_me)
        self.opponent_score += int(award_opp)

        ev = GameEvent(
            type="round_end",
            ply=self.ply(),
            end_reason=reason,
            end_award_me=int(award_me),
            end_award_opp=int(award_opp),
            open_ends=sorted(list(self.board.open_end_values())),
        )
        self.events.append(ev)
        return ev

    def finalize_out_with_opponent_pips(self, opponent_pips: int) -> GameEvent:
        if not self.round_over or self.round_end_reason != "out_me_pending":
            raise ValueError("No pending out to finalize")
        pts = int(round_to_nearest_5(int(opponent_pips)))
        return self._finalize_round("out_me", award_me=pts, award_opp=0)

    def declare_locked(self, opponent_pips: int) -> GameEvent:
        self._assert_round_active()

        if int(self.boneyard_count) != 0:
            raise ValueError("Cannot declare locked while boneyard is not empty")

        if not self.must_pass_me():
            raise ValueError("Cannot declare locked: you still have a legal move")

        opp_pass_seen = any((e.type == "pass" and e.player == "opponent") for e in self.events)
        if not opp_pass_seen:
            raise ValueError("Cannot declare locked: record opponent pass first (as evidence)")

        my_p = self.my_hand_pips()
        opp_p = int(opponent_pips)
        diff = abs(int(my_p) - int(opp_p))
        pts = int(round_to_nearest_5(diff))

        if pts <= 0:
            return self._finalize_round("locked_tie", 0, 0)
        if my_p < opp_p:
            return self._finalize_round("locked_me_wins", award_me=pts, award_opp=0)
        if opp_p < my_p:
            return self._finalize_round("locked_opp_wins", award_me=0, award_opp=pts)
        return self._finalize_round("locked_tie", 0, 0)

    # ---- core actions ----

    def play_tile(self, player: PlayerName, t: Tile, end: EndName) -> GameEvent:
        self._assert_round_active()

        if player != self.current_turn:
            raise ValueError(f"Not {player}'s turn (current_turn={self.current_turn})")

        # Enforce opening rule on actual play as well (API safety)
        if (
            player == "me"
            and self.opening_mode == "forced_best"
            and self.started_from_beginning
            and self.board.is_empty()
            and self.forced_play_tile is None
        ):
            req = best_opening_tile(self.my_hand)
            if req is not None and t != req:
                raise ValueError(f"Opening rule: must start with {tile_str(req)}")

        if player == "me":
            if t not in self.my_hand:
                raise ValueError(f"You don't have tile: {tile_str(t)}")
            if self.forced_play_tile is not None and t != self.forced_play_tile:
                raise ValueError(f"Must play drawn tile first: {tile_str(self.forced_play_tile)}")
        else:
            if t in self.my_hand:
                raise ValueError("Opponent played a tile that is in your hand")
            if self.opponent_tile_count <= 0:
                raise ValueError("Opponent tile count already 0")

        if not self.board.is_empty():
            if end not in self.board.ends:
                raise ValueError(f"End not open: {end}")
            end_val, _ = self.board.ends[end]
            if not tile_has(t, end_val):
                raise ValueError(f"Illegal: {tile_str(t)} cannot go on {end}({end_val})")

        if t in self.board.played_set:
            raise ValueError(f"Tile already played: {tile_str(t)}")

        self.push_undo()

        pts = int(self.board.play(t, end))
        if player == "me":
            self.my_hand.remove(t)
            self.my_score += pts
            if self.forced_play_tile == t:
                self.forced_play_tile = None
        else:
            self.opponent_tile_count -= 1
            self.opponent_score += pts

        # Record the play event first (chronological correctness)
        ev = GameEvent(
            type="play",
            ply=self.ply(),
            player=player,
            tile=tile_str(t),
            end=end,
            score_gained=int(pts),
            open_ends=sorted(list(self.board.open_end_values())),
        )
        self.events.append(ev)

        # NEW: target ends immediately (mid-round)
        reason = match_target_reason(self.my_score, self.opponent_score, self.match_target, last_player=player)
        if reason is not None:
            # Do NOT go into out_me_pending even if hand is empty. Target is final.
            self._finalize_round(reason, award_me=0, award_opp=0)
            return ev

        self._recalc_boneyard()
        self.current_turn = "opponent" if player == "me" else "me"

        # Out logic
        if player == "me" and len(self.my_hand) == 0:
            self.round_over = True
            self.round_end_reason = "out_me_pending"
            self.pending_out_opponent_pips = True
            return ev

        if player == "opponent" and int(self.opponent_tile_count) == 0:
            my_pips = int(round_to_nearest_5(self.my_hand_pips()))
            self._finalize_round("out_opponent", award_me=0, award_opp=my_pips)
            return ev

        return ev

    def record_draw(self, player: PlayerName, count: int = 1, certainty: Certainty = "probable") -> GameEvent:
        self._assert_round_active()

        if player != "opponent":
            raise ValueError("Use record_draw_me(tile) for player='me' draws")

        if self.current_turn != "opponent":
            raise ValueError("Not opponent's turn")

        if count <= 0:
            raise ValueError("count must be >= 1")
        if self.boneyard_count < count:
            raise ValueError("Not enough boneyard tiles according to state")

        if certainty not in ("certain", "probable", "possible"):
            certainty = "probable"

        self.push_undo()

        self.opponent_tile_count += int(count)
        self.boneyard_count -= int(count)

        ev = GameEvent(
            type="draw",
            ply=self.ply(),
            player=player,
            draw_count=int(count),
            certainty=certainty,
            open_ends=sorted(list(self.board.open_end_values())),
        )
        self.events.append(ev)
        return ev

    def record_draw_me(self, tile: Tile) -> GameEvent:
        self._assert_round_active()

        if self.current_turn != "me":
            raise ValueError("Not your turn")
        if self.forced_play_tile is not None:
            raise ValueError(f"Must play drawn tile first: {tile_str(self.forced_play_tile)}")
        if self.boneyard_count <= 0:
            raise ValueError("Boneyard is empty")
        if len(self.legal_moves_me()) > 0:
            raise ValueError("Draw not allowed: you already have a legal move")
        if tile in self.visible_tiles():
            raise ValueError(f"Drawn tile is already visible: {tile_str(tile)}")

        self.push_undo()

        self.my_hand.add(tile)
        self.boneyard_count -= 1
        self._recalc_boneyard()

        if len(self.board.legal_ends_for_tile(tile)) > 0:
            self.forced_play_tile = tile

        ev = GameEvent(
            type="draw",
            ply=self.ply(),
            player="me",
            tile=tile_str(tile),
            draw_count=1,
            certainty="certain",
            open_ends=sorted(list(self.board.open_end_values())),
        )
        self.events.append(ev)
        return ev

    def record_pass(self, player: PlayerName, certainty: Certainty = "certain") -> GameEvent:
        self._assert_round_active()

        if player == "me":
            if self.current_turn != "me":
                raise ValueError("Not your turn")
            if not self.must_pass_me():
                raise ValueError("Pass not allowed: you still have a legal move or boneyard not empty")
            certainty = "certain"

        else:
            if self.current_turn != "opponent":
                raise ValueError("Not opponent's turn")
            if certainty not in ("certain", "probable", "possible"):
                certainty = "probable"
            if int(self.boneyard_count) > 0 and certainty == "certain":
                certainty = "possible"

        self.push_undo()
        self.current_turn = "opponent" if player == "me" else "me"

        ev = GameEvent(
            type="pass",
            ply=self.ply(),
            player=player,
            certainty=certainty,
            open_ends=sorted(list(self.board.open_end_values())),
        )
        self.events.append(ev)
        return ev

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "ruleset": RULESET_ID,
                "match_target": int(self.match_target),
                "round_index": int(self.round_index),

                "started_from_beginning": self.started_from_beginning,
                "current_turn": self.current_turn,

                "my_score": int(self.my_score),
                "opponent_score": int(self.opponent_score),

                "opponent_tile_count": int(self.opponent_tile_count),
                "boneyard_count": int(self.boneyard_count),

                "forced_play_tile": tile_str(self.forced_play_tile) if self.forced_play_tile else None,
                "must_draw_me": bool(self.must_draw_me()),
                "must_pass_me": bool(self.must_pass_me()),

                "round_over": bool(self.round_over),
                "round_end_reason": self.round_end_reason,
                "pending_out_opponent_pips": bool(self.pending_out_opponent_pips),

                "tile_total": int(self.tile_conservation_total()),

                "opening_mode": self.opening_mode,   # NEW
            },
            "board": self.board.snapshot(),
            "my_hand": [tile_str(t) for t in sorted(self.my_hand)],
            "events": [e.to_dict() for e in self.events],
        }

    def _restore(self, d: Dict[str, Any]) -> None:
        st = GameState.from_dict(d)
        self.board = st.board
        self.my_hand = st.my_hand
        self.opponent_tile_count = st.opponent_tile_count
        self.boneyard_count = st.boneyard_count
        self.my_score = st.my_score
        self.opponent_score = st.opponent_score
        self.match_target = st.match_target
        self.round_index = st.round_index
        self.current_turn = st.current_turn
        self.started_from_beginning = st.started_from_beginning
        self.forced_play_tile = st.forced_play_tile
        self.round_over = st.round_over
        self.round_end_reason = st.round_end_reason
        self.pending_out_opponent_pips = st.pending_out_opponent_pips
        self.events = st.events
        self.undo_enabled = True
        self.opening_mode = st.opening_mode

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameState":
        st = cls()
        meta = d.get("meta", {}) or {}

        st.match_target = int(meta.get("match_target", 150))
        st.round_index = int(meta.get("round_index", 1))

        st.started_from_beginning = bool(meta.get("started_from_beginning", False))
        st.current_turn = meta.get("current_turn", "me")
        if st.current_turn not in ("me", "opponent"):
            st.current_turn = "me"

        st.my_score = int(meta.get("my_score", 0))
        st.opponent_score = int(meta.get("opponent_score", 0))

        st.opponent_tile_count = int(meta.get("opponent_tile_count", 7))
        st.boneyard_count = int(meta.get("boneyard_count", 14))

        st.forced_play_tile = parse_tile(meta["forced_play_tile"]) if meta.get("forced_play_tile") else None

        st.round_over = bool(meta.get("round_over", False))
        st.round_end_reason = meta.get("round_end_reason")
        st.pending_out_opponent_pips = bool(meta.get("pending_out_opponent_pips", False))

        st.opening_mode = meta.get("opening_mode", "forced_best")
        if st.opening_mode not in ("forced_best", "free"):
            st.opening_mode = "forced_best"

        st.board = Board.from_snapshot(d.get("board", {}) or {})
        st.my_hand = {parse_tile(s) for s in (d.get("my_hand", []) or [])}
        st.events = [GameEvent.from_dict(x) for x in (d.get("events", []) or [])]

        st.undo_enabled = True
        return st