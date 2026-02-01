// FILE: src/state.rs | version: 2026-01-03.rc3
// CHANGELOG:
// - RC3: Fix lifetime issue in apply_script by converting &str to &'static str using match.
// - RC2: Add boundary-only script runner (apply_script) so lib.rs stays exports-only.
// - RC2: Add boundary-only ops: setup_board, set_hand_counts, set_opening_mode, set_started.
// - RC2: Conformance now can test target-immediate and out_me_pending deterministically.

use std::collections::BTreeMap;

use crate::board::{Board, End, RULESET_ID};
use crate::board::serde_like::Value as V;
use crate::tile::Tile;

pub type PlayerName = &'static str;   // "me" | "opponent"
pub type Certainty = &'static str;    // "certain" | "probable" | "possible"
pub type OpeningMode = &'static str;  // "forced_best" | "free"

#[derive(Clone, Debug)]
pub struct GameEvent {
    pub typ: &'static str,
    pub ply: i32,
    pub player: Option<PlayerName>,
    pub tile: Option<String>,
    pub end: Option<&'static str>,
    pub draw_count: i32,
    pub open_ends: Vec<u8>,
    pub certainty: Certainty,
    pub score_gained: i32,
    pub end_reason: Option<String>,
    pub end_award_me: i32,
    pub end_award_opp: i32,
}

impl GameEvent {
    fn to_value(&self) -> V {
        let mut m = BTreeMap::<String, V>::new();
        m.insert("type".into(), V::Str(self.typ.into()));
        m.insert("ply".into(), V::Int(self.ply as i64));
        m.insert(
            "player".into(),
            match self.player {
                Some(p) => V::Str(p.into()),
                None => V::Null,
            },
        );
        m.insert(
            "tile".into(),
            match &self.tile {
                Some(t) => V::Str(t.clone()),
                None => V::Null,
            },
        );
        m.insert(
            "end".into(),
            match self.end {
                Some(e) => V::Str(e.into()),
                None => V::Null,
            },
        );
        m.insert("draw_count".into(), V::Int(self.draw_count as i64));
        m.insert(
            "open_ends".into(),
            V::List(self.open_ends.iter().map(|x| V::Int(*x as i64)).collect()),
        );
        m.insert("certainty".into(), V::Str(self.certainty.into()));
        m.insert("score_gained".into(), V::Int(self.score_gained as i64));
        m.insert(
            "end_reason".into(),
            match &self.end_reason {
                Some(r) => V::Str(r.clone()),
                None => V::Null,
            },
        );
        m.insert("end_award_me".into(), V::Int(self.end_award_me as i64));
        m.insert("end_award_opp".into(), V::Int(self.end_award_opp as i64));
        V::Map(m)
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub board: Board,
    pub my_hand: Vec<Tile>, // RC1 correctness-first; later can become bitset internally (no API change)
    pub opponent_tile_count: i32,
    pub boneyard_count: i32,

    pub my_score: i32,
    pub opponent_score: i32,
    pub match_target: i32,
    pub round_index: i32,

    pub current_turn: PlayerName,
    pub started_from_beginning: bool,
    pub forced_play_tile: Option<Tile>,

    pub round_over: bool,
    pub round_end_reason: Option<String>,
    pub pending_out_opponent_pips: bool,

    pub events: Vec<GameEvent>,
    pub opening_mode: OpeningMode,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            my_hand: Vec::new(),
            opponent_tile_count: 7,
            boneyard_count: 14,
            my_score: 0,
            opponent_score: 0,
            match_target: 150,
            round_index: 1,
            current_turn: "me",
            started_from_beginning: true,
            forced_play_tile: None,
            round_over: false,
            round_end_reason: None,
            pending_out_opponent_pips: false,
            events: Vec::new(),
            opening_mode: "forced_best",
        }
    }

    #[inline]
    pub fn ply(&self) -> i32 {
        self.events.len() as i32
    }

    fn my_hand_contains(&self, t: Tile) -> bool {
        self.my_hand.iter().any(|x| *x == t)
    }

    fn my_hand_remove(&mut self, t: Tile) -> bool {
        if let Some(i) = self.my_hand.iter().position(|x| *x == t) {
            self.my_hand.swap_remove(i);
            true
        } else {
            false
        }
    }

    fn played_len(&self) -> i32 {
        // For conformance we can use board's internal knowledge by reading snapshot played_tiles length.
        // This is boundary-only; not hot.
        match self.board.snapshot().get("played_tiles") {
            Some(V::List(xs)) => xs.len() as i32,
            _ => 0,
        }
    }

    fn visible_count(&self) -> i32 {
        (self.my_hand.len() as i32) + self.played_len()
    }

    fn recalc_boneyard(&mut self) {
        let rem = 28 - self.visible_count() - self.opponent_tile_count;
        self.boneyard_count = rem.max(0);
    }

    fn tile_total(&self) -> i32 {
        (self.my_hand.len() as i32) + self.played_len() + self.opponent_tile_count + self.boneyard_count
    }

    fn assert_round_active(&self) -> Result<(), String> {
        if self.round_over {
            return Err(format!(
                "round is over ({})",
                self.round_end_reason.as_deref().unwrap_or("?")
            ));
        }
        Ok(())
    }

    pub fn start_new_game(&mut self, my_hand: Vec<Tile>, match_target: i32) -> Result<(), String> {
        if my_hand.len() != 7 {
            return Err("new match requires exactly 7 tiles".into());
        }
        self.match_target = if match_target > 0 { match_target } else { 150 };
        self.my_score = 0;
        self.opponent_score = 0;
        self.round_index = 1;

        self.events.clear();
        self.events.push(GameEvent {
            typ: "match_start",
            ply: 0,
            player: None,
            tile: None,
            end: None,
            draw_count: 0,
            open_ends: vec![],
            certainty: "certain",
            score_gained: 0,
            end_reason: None,
            end_award_me: 0,
            end_award_opp: 0,
        });

        self.opening_mode = "forced_best";
        self.start_round_internal(my_hand)
    }

    pub fn start_new_round(&mut self, my_hand: Vec<Tile>) -> Result<(), String> {
        if my_hand.len() != 7 {
            return Err("new round requires exactly 7 tiles".into());
        }
        let prev_reason = self.round_end_reason.clone().unwrap_or_default();
        self.round_index += 1;

        if matches!(
            prev_reason.as_str(),
            "out_me" | "out_opponent" | "target_me" | "target_opponent"
        ) {
            self.opening_mode = "free";
        } else {
            self.opening_mode = "forced_best";
        }

        self.start_round_internal(my_hand)
    }

    fn start_round_internal(&mut self, my_hand: Vec<Tile>) -> Result<(), String> {
        self.board = Board::new();
        self.my_hand = my_hand;
        self.opponent_tile_count = 7;
        self.forced_play_tile = None;

        self.round_over = false;
        self.round_end_reason = None;
        self.pending_out_opponent_pips = false;

        self.current_turn = "me";
        self.started_from_beginning = true;
        self.recalc_boneyard();

        self.events.push(GameEvent {
            typ: "round_start",
            ply: self.ply(),
            player: None,
            tile: None,
            end: None,
            draw_count: 0,
            open_ends: vec![],
            certainty: "certain",
            score_gained: 0,
            end_reason: None,
            end_award_me: 0,
            end_award_opp: 0,
        });

        Ok(())
    }

    fn round_to_nearest_5(x: i32) -> i32 {
        ((x as f64) / 5.0).round() as i32 * 5
    }

    fn best_opening_tile(hand: &[Tile]) -> Option<Tile> {
        if hand.is_empty() {
            return None;
        }
        // highest double if any
        let mut best_double: Option<Tile> = None;
        for &t in hand {
            if t.is_double() {
                best_double = match best_double {
                    None => Some(t),
                    Some(cur) => {
                        let (a, _) = t.pips();
                        let (c, _) = cur.pips();
                        if a > c { Some(t) } else { Some(cur) }
                    }
                };
            }
        }
        if let Some(d) = best_double {
            return Some(d);
        }
        // else max by (pip_sum, hi, lo)
        let mut best = hand[0];
        for &t in hand.iter().skip(1) {
            let (a, b) = t.pips();
            let (ca, cb) = best.pips();
            let key = (a as i32 + b as i32, a as i32, b as i32);
            let ckey = (ca as i32 + cb as i32, ca as i32, cb as i32);
            if key > ckey {
                best = t;
            }
        }
        Some(best)
    }

    pub fn legal_moves_me(&mut self) -> Vec<(Tile, End)> {
        if let Some(ft) = self.forced_play_tile {
            if !self.my_hand_contains(ft) {
                self.forced_play_tile = None;
            }
        }

        // Match Python legal_moves_me EXACTLY (including current odd condition):
        // opening_tile is only set if forced_play_tile is NOT None.
        let mut opening_tile: Option<Tile> = None;
        if self.opening_mode == "forced_best"
            && self.started_from_beginning
            && self.current_turn == "me"
            && self.board.is_empty()
            && self.forced_play_tile.is_none()
        {
            opening_tile = Self::best_opening_tile(&self.my_hand);
        }

        let tiles: Vec<Tile> = if let Some(ot) = opening_tile {
            vec![ot]
        } else if let Some(ft) = self.forced_play_tile {
            vec![ft]
        } else {
            self.my_hand.clone()
        };

        let mut out = Vec::new();
        for t in tiles {
            for e in self.board.legal_ends_for_tile(t) {
                out.push((t, e));
            }
        }
        out
    }

    pub fn must_draw_me(&mut self) -> bool {
        if self.round_over || self.current_turn != "me" {
            return false;
        }
        if self.forced_play_tile.is_some() {
            return false;
        }
        self.legal_moves_me().is_empty() && self.boneyard_count > 0
    }

    pub fn must_pass_me(&mut self) -> bool {
        if self.round_over || self.current_turn != "me" {
            return false;
        }
        if self.forced_play_tile.is_some() {
            return false;
        }
        self.legal_moves_me().is_empty() && self.boneyard_count == 0
    }

    fn match_target_reason(my_score: i32, opp_score: i32, target: i32, last_player: PlayerName) -> Option<&'static str> {
        let my_win = my_score >= target;
        let opp_win = opp_score >= target;

        if my_win && !opp_win {
            return Some("target_me");
        }
        if opp_win && !my_win {
            return Some("target_opponent");
        }
        if my_win && opp_win {
            return Some(if last_player == "me" { "target_me" } else { "target_opponent" });
        }
        None
    }

    fn finalize_round(&mut self, reason: &str, award_me: i32, award_opp: i32) {
        self.round_over = true;
        self.round_end_reason = Some(reason.to_string());
        self.pending_out_opponent_pips = false;

        self.my_score += award_me;
        self.opponent_score += award_opp;

        let mut open_ends = self.board.open_end_values();
        open_ends.sort_unstable();

        self.events.push(GameEvent {
            typ: "round_end",
            ply: self.ply(),
            player: None,
            tile: None,
            end: None,
            draw_count: 0,
            open_ends,
            certainty: "certain",
            score_gained: 0,
            end_reason: Some(reason.to_string()),
            end_award_me: award_me,
            end_award_opp: award_opp,
        });
    }

    pub fn finalize_out_with_opponent_pips(&mut self, opponent_pips: i32) -> Result<(), String> {
        if !self.round_over || self.round_end_reason.as_deref() != Some("out_me_pending") {
            return Err("no pending out to finalize".into());
        }
        let pts = Self::round_to_nearest_5(opponent_pips.max(0));
        self.finalize_round("out_me", pts, 0);
        Ok(())
    }

    pub fn declare_locked(&mut self, opponent_pips: i32) -> Result<(), String> {
        self.assert_round_active()?;
        if self.boneyard_count != 0 {
            return Err("cannot declare locked while boneyard is not empty".into());
        }
        if !self.must_pass_me() {
            return Err("cannot declare locked: you still have a legal move".into());
        }
        let opp_pass_seen = self.events.iter().any(|e| e.typ == "pass" && e.player == Some("opponent"));
        if !opp_pass_seen {
            return Err("cannot declare locked: record opponent pass first (as evidence)".into());
        }

        let my_pips: i32 = self.my_hand.iter().map(|t| t.pip_sum() as i32).sum();
        let opp_p = opponent_pips.max(0);
        let diff = (my_pips - opp_p).abs();
        let pts = Self::round_to_nearest_5(diff);

        if pts <= 0 {
            self.finalize_round("locked_tie", 0, 0);
            return Ok(());
        }
        if my_pips < opp_p {
            self.finalize_round("locked_me_wins", pts, 0);
            return Ok(());
        }
        if opp_p < my_pips {
            self.finalize_round("locked_opp_wins", 0, pts);
            return Ok(());
        }
        self.finalize_round("locked_tie", 0, 0);
        Ok(())
    }

    pub fn record_draw_me(&mut self, tile: Tile) -> Result<(), String> {
        self.assert_round_active()?;
        if self.current_turn != "me" {
            return Err("not your turn".into());
        }
        if let Some(ft) = self.forced_play_tile {
            return Err(format!("must play drawn tile first: {}", ft.to_str()));
        }
        if self.boneyard_count <= 0 {
            return Err("boneyard is empty".into());
        }
        if !self.legal_moves_me().is_empty() {
            return Err("draw not allowed: you already have a legal move".into());
        }
        if self.my_hand_contains(tile) {
            return Err(format!("drawn tile is already visible: {}", tile.to_str()));
        }

        self.my_hand.push(tile);
        self.boneyard_count -= 1;
        self.recalc_boneyard();

        if !self.board.legal_ends_for_tile(tile).is_empty() {
            self.forced_play_tile = Some(tile);
        }

        let mut open_ends = self.board.open_end_values();
        open_ends.sort_unstable();

        self.events.push(GameEvent {
            typ: "draw",
            ply: self.ply(),
            player: Some("me"),
            tile: Some(tile.to_str()),
            end: None,
            draw_count: 1,
            open_ends,
            certainty: "certain",
            score_gained: 0,
            end_reason: None,
            end_award_me: 0,
            end_award_opp: 0,
        });

        Ok(())
    }

    pub fn record_draw_opponent(&mut self, count: i32, mut certainty: Certainty) -> Result<(), String> {
        self.assert_round_active()?;
        if self.current_turn != "opponent" {
            return Err("not opponent's turn".into());
        }
        if count <= 0 {
            return Err("count must be >= 1".into());
        }
        if self.boneyard_count < count {
            return Err("not enough boneyard tiles according to state".into());
        }
        if certainty != "certain" && certainty != "probable" && certainty != "possible" {
            certainty = "probable";
        }

        self.opponent_tile_count += count;
        self.boneyard_count -= count;

        let mut open_ends = self.board.open_end_values();
        open_ends.sort_unstable();

        self.events.push(GameEvent {
            typ: "draw",
            ply: self.ply(),
            player: Some("opponent"),
            tile: None,
            end: None,
            draw_count: count,
            open_ends,
            certainty,
            score_gained: 0,
            end_reason: None,
            end_award_me: 0,
            end_award_opp: 0,
        });
        Ok(())
    }

    pub fn record_pass(&mut self, player: PlayerName, mut certainty: Certainty) -> Result<(), String> {
        self.assert_round_active()?;

        if player == "me" {
            if self.current_turn != "me" {
                return Err("not your turn".into());
            }
            if !self.must_pass_me() {
                return Err("pass not allowed: you still have a legal move or boneyard not empty".into());
            }
            certainty = "certain";
        } else {
            if self.current_turn != "opponent" {
                return Err("not opponent's turn".into());
            }
            if certainty != "certain" && certainty != "probable" && certainty != "possible" {
                certainty = "probable";
            }
            if self.boneyard_count > 0 && certainty == "certain" {
                certainty = "possible";
            }
        }

        let mut open_ends = self.board.open_end_values();
        open_ends.sort_unstable();

        self.events.push(GameEvent {
            typ: "pass",
            ply: self.ply(),
            player: Some(player),
            tile: None,
            end: None,
            draw_count: 0,
            open_ends,
            certainty,
            score_gained: 0,
            end_reason: None,
            end_award_me: 0,
            end_award_opp: 0,
        });

        self.current_turn = if player == "me" { "opponent" } else { "me" };
        Ok(())
    }

    pub fn play_tile(&mut self, player: PlayerName, t: Tile, end: End) -> Result<i32, String> {
        self.assert_round_active()?;
        if player != self.current_turn {
            return Err(format!("not {player}'s turn (current_turn={})", self.current_turn));
        }

        // Opening rule enforcement on actual play (matches Python)
        if player == "me"
            && self.opening_mode == "forced_best"
            && self.started_from_beginning
            && self.board.is_empty()
            && self.forced_play_tile.is_none()
        {
            if let Some(req) = Self::best_opening_tile(&self.my_hand) {
                if t != req {
                    return Err(format!("opening rule: must start with {}", req.to_str()));
                }
            }
        }

        if player == "me" {
            if !self.my_hand_contains(t) {
                return Err(format!("you don't have tile: {}", t.to_str()));
            }
            if let Some(ft) = self.forced_play_tile {
                if t != ft {
                    return Err(format!("must play drawn tile first: {}", ft.to_str()));
                }
            }
        } else {
            if self.my_hand_contains(t) {
                return Err("opponent played a tile that is in your hand".into());
            }
            if self.opponent_tile_count <= 0 {
                return Err("opponent tile count already 0".into());
            }
        }

        let pts = self.board.play(t, end)?;

        if player == "me" {
            self.my_hand_remove(t);
            self.my_score += pts;
            if self.forced_play_tile == Some(t) {
                self.forced_play_tile = None;
            }
        } else {
            self.opponent_tile_count -= 1;
            self.opponent_score += pts;
        }

        let mut open_ends = self.board.open_end_values();
        open_ends.sort_unstable();

        self.events.push(GameEvent {
            typ: "play",
            ply: self.ply(),
            player: Some(player),
            tile: Some(t.to_str()),
            end: Some(end.as_str()),
            draw_count: 0,
            open_ends,
            certainty: "certain",
            score_gained: pts,
            end_reason: None,
            end_award_me: 0,
            end_award_opp: 0,
        });

        // target-immediate
        if let Some(reason) = Self::match_target_reason(self.my_score, self.opponent_score, self.match_target, player) {
            self.finalize_round(reason, 0, 0);
            return Ok(pts);
        }

        self.recalc_boneyard();
        self.current_turn = if player == "me" { "opponent" } else { "me" };

        // Out logic
        if player == "me" && self.my_hand.is_empty() {
            self.round_over = true;
            self.round_end_reason = Some("out_me_pending".into());
            self.pending_out_opponent_pips = true;
            return Ok(pts);
        }

        if player == "opponent" && self.opponent_tile_count == 0 {
            let my_pips: i32 = self.my_hand.iter().map(|x| x.pip_sum() as i32).sum();
            let award = Self::round_to_nearest_5(my_pips);
            self.finalize_round("out_opponent", 0, award);
            return Ok(pts);
        }

        Ok(pts)
    }

    // --- Boundary-only helpers for conformance scripting ---
    fn setup_board(&mut self, center: Tile, arms: [Vec<Tile>; 4]) -> Result<(), String> {
        let mut b = Board::new();
        // replay: center -> right -> left -> up -> down (same as app.py midgame builder)
        b.play(center, End::Right)?;
        for t in arms[End::Right.idx()].iter().copied() {
            b.play(t, End::Right)?;
        }
        for t in arms[End::Left.idx()].iter().copied() {
            b.play(t, End::Left)?;
        }
        for t in arms[End::Up.idx()].iter().copied() {
            b.play(t, End::Up)?;
        }
        for t in arms[End::Down.idx()].iter().copied() {
            b.play(t, End::Down)?;
        }
        self.board = b;
        Ok(())
    }

    fn set_hand_counts(&mut self, hand: Vec<Tile>, opp_cnt: i32, bone_cnt: i32) {
        self.my_hand = hand;
        self.opponent_tile_count = opp_cnt.max(0);
        self.boneyard_count = bone_cnt.max(0);
        if let Some(ft) = self.forced_play_tile {
            if !self.my_hand_contains(ft) {
                self.forced_play_tile = None;
            }
        }
    }

    pub fn snapshot_state(&mut self) -> BTreeMap<String, V> {
        let mut root = BTreeMap::<String, V>::new();

        let mut meta = BTreeMap::<String, V>::new();
        meta.insert("ruleset".into(), V::Str(RULESET_ID.into()));
        meta.insert("match_target".into(), V::Int(self.match_target as i64));
        meta.insert("round_index".into(), V::Int(self.round_index as i64));

        meta.insert("started_from_beginning".into(), V::Bool(self.started_from_beginning));
        meta.insert("current_turn".into(), V::Str(self.current_turn.into()));

        meta.insert("my_score".into(), V::Int(self.my_score as i64));
        meta.insert("opponent_score".into(), V::Int(self.opponent_score as i64));

        meta.insert("opponent_tile_count".into(), V::Int(self.opponent_tile_count as i64));
        meta.insert("boneyard_count".into(), V::Int(self.boneyard_count as i64));

        meta.insert(
            "forced_play_tile".into(),
            match self.forced_play_tile {
                Some(t) => V::Str(t.to_str()),
                None => V::Null,
            },
        );

        meta.insert("must_draw_me".into(), V::Bool(self.must_draw_me()));
        meta.insert("must_pass_me".into(), V::Bool(self.must_pass_me()));

        meta.insert("round_over".into(), V::Bool(self.round_over));
        meta.insert(
            "round_end_reason".into(),
            match &self.round_end_reason {
                Some(r) => V::Str(r.clone()),
                None => V::Null,
            },
        );
        meta.insert("pending_out_opponent_pips".into(), V::Bool(self.pending_out_opponent_pips));
        meta.insert("tile_total".into(), V::Int(self.tile_total() as i64));
        meta.insert("opening_mode".into(), V::Str(self.opening_mode.into()));

        root.insert("meta".into(), V::Map(meta));

        // board snapshot
        let b = self.board.snapshot();
        let mut bm = BTreeMap::<String, V>::new();
        for (k, v) in b {
            bm.insert(k, v);
        }
        root.insert("board".into(), V::Map(bm));

        // my_hand
        let mut hand = self.my_hand.iter().map(|t| V::Str(t.to_str())).collect::<Vec<_>>();
        hand.sort_by(|a, b| match (a, b) {
            (V::Str(sa), V::Str(sb)) => sa.cmp(sb),
            _ => std::cmp::Ordering::Equal,
        });
        root.insert("my_hand".into(), V::List(hand));

        // events
        root.insert("events".into(), V::List(self.events.iter().map(|e| e.to_value()).collect()));

        root
    }
}

/// Boundary-only script runner for conformance.
/// This keeps lib.rs thin and prevents drift.
///
/// Supported ops (strings):
/// - play|me|6-6|right
/// - play|opponent|6-5|left
/// - pass|me|certain
/// - pass|opponent|probable
/// - draw_me|6-1
/// - draw|opponent|2|certain
/// - finalize_out|23
/// - declare_locked|17
/// - set_scores|140|145
/// - set_turn|me / set_turn|opponent
/// - start_new_round|t1,t2,t3,t4,t5,t6,t7
/// - setup_board|center|right_csv|left_csv|up_csv|down_csv
/// - set_hand_counts|hand_csv|opp_cnt|bone_cnt
/// - set_opening_mode|forced_best/free
/// - set_started|0/1
pub fn apply_script(my_hand: Vec<Tile>, match_target: i32, script: &[String]) -> Result<BTreeMap<String, V>, String> {
    if my_hand.len() != 7 {
        return Err("my_hand must have exactly 7 tiles".into());
    }

    let mut st = GameState::new();
    st.start_new_game(my_hand, match_target)?;

    for line in script {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "play" => {
                if parts.len() != 4 { return Err("play expects: play|player|tile|end".into()); }
                let player = match parts[1] {
                    "me" => "me",
                    "opponent" => "opponent",
                    _ => return Err("bad player".into()),
                };
                let t = Tile::parse(parts[2])?;
                let e = End::from_str(parts[3]).ok_or_else(|| "bad end".to_string())?;
                st.play_tile(player, t, e)?;
            }
            "pass" => {
                if parts.len() != 3 { return Err("pass expects: pass|player|certainty".into()); }
                let player = match parts[1] {
                    "me" => "me",
                    "opponent" => "opponent",
                    _ => return Err("bad player".into()),
                };
                let cert = match parts[2] {
                    "certain" => "certain",
                    "probable" => "probable",
                    "possible" => "possible",
                    _ => "probable",
                };
                st.record_pass(player, cert)?;
            }
            "draw_me" => {
                if parts.len() != 2 { return Err("draw_me expects: draw_me|tile".into()); }
                let t = Tile::parse(parts[1])?;
                st.record_draw_me(t)?;
            }
            "draw" => {
                if parts.len() != 4 { return Err("draw expects: draw|opponent|count|certainty".into()); }
                if parts[1] != "opponent" { return Err("draw supports opponent only".into()); }
                let count: i32 = parts[2].parse().map_err(|_| "bad count".to_string())?;
                let cert = match parts[3] {
                    "certain" => "certain",
                    "probable" => "probable",
                    "possible" => "possible",
                    _ => "probable",
                };
                st.record_draw_opponent(count, cert)?;
            }
            "finalize_out" => {
                if parts.len() != 2 { return Err("finalize_out expects: finalize_out|opponent_pips".into()); }
                let p: i32 = parts[1].parse().map_err(|_| "bad pips".to_string())?;
                st.finalize_out_with_opponent_pips(p)?;
            }
            "declare_locked" => {
                if parts.len() != 2 { return Err("declare_locked expects: declare_locked|opponent_pips".into()); }
                let p: i32 = parts[1].parse().map_err(|_| "bad pips".to_string())?;
                st.declare_locked(p)?;
            }
            "set_scores" => {
                if parts.len() != 3 { return Err("set_scores expects: set_scores|my|opp".into()); }
                st.my_score = parts[1].parse().map_err(|_| "bad my_score".to_string())?;
                st.opponent_score = parts[2].parse().map_err(|_| "bad opp_score".to_string())?;
            }
            "set_turn" => {
                if parts.len() != 2 { return Err("set_turn expects: set_turn|me/opponent".into()); }
                let t = match parts[1] {
                    "me" => "me",
                    "opponent" => "opponent",
                    _ => return Err("bad turn".into()),
                };
                st.current_turn = t;
            }
            "start_new_round" => {
                if parts.len() != 2 { return Err("start_new_round expects: start_new_round|tile,tile,...(7)".into()); }
                let tiles_s: Vec<&str> = parts[1].split(',').filter(|x| !x.trim().is_empty()).collect();
                if tiles_s.len() != 7 { return Err("start_new_round requires 7 tiles".into()); }
                let mut hand2 = Vec::with_capacity(7);
                for s in tiles_s {
                    hand2.push(Tile::parse(s)?);
                }
                st.start_new_round(hand2)?;
            }
            "setup_board" => {
                if parts.len() != 6 { return Err("setup_board expects: setup_board|center|right_csv|left_csv|up_csv|down_csv".into()); }
                let center = Tile::parse(parts[1])?;
                let parse_csv = |s: &str| -> Result<Vec<Tile>, String> {
                    let mut out = Vec::new();
                    let ss = s.trim();
                    if ss.is_empty() { return Ok(out); }
                    for tok in ss.split(',') {
                        let tt = tok.trim();
                        if tt.is_empty() { continue; }
                        out.push(Tile::parse(tt)?);
                    }
                    Ok(out)
                };
                let right = parse_csv(parts[2])?;
                let left = parse_csv(parts[3])?;
                let up = parse_csv(parts[4])?;
                let down = parse_csv(parts[5])?;
                st.setup_board(center, [right, left, up, down])?;
            }
            "set_hand_counts" => {
                if parts.len() != 4 { return Err("set_hand_counts expects: set_hand_counts|hand_csv|opp_cnt|bone_cnt".into()); }
                let hand_csv = parts[1];
                let mut hand = Vec::new();
                if !hand_csv.trim().is_empty() {
                    for tok in hand_csv.split(',') {
                        let tt = tok.trim();
                        if tt.is_empty() { continue; }
                        hand.push(Tile::parse(tt)?);
                    }
                }
                let opp_cnt: i32 = parts[2].parse().map_err(|_| "bad opp_cnt".to_string())?;
                let bone_cnt: i32 = parts[3].parse().map_err(|_| "bad bone_cnt".to_string())?;
                st.set_hand_counts(hand, opp_cnt, bone_cnt);
            }
            "set_opening_mode" => {
                if parts.len() != 2 { return Err("set_opening_mode expects: set_opening_mode|forced_best/free".into()); }
                let m = match parts[1] {
                    "forced_best" => "forced_best",
                    "free" => "free",
                    _ => return Err("bad opening_mode".into()),
                };
                st.opening_mode = m;
            }
            "set_started" => {
                if parts.len() != 2 { return Err("set_started expects: set_started|0/1".into()); }
                st.started_from_beginning = match parts[1] {
                    "0" => false,
                    "1" => true,
                    _ => return Err("bad set_started value (use 0/1)".into()),
                };
            }
            _ => return Err(format!("unknown script op: {}", parts[0])),
        }
    }

    Ok(st.snapshot_state())
}