// FILE: src/features.rs | version: 2026-01-17.stage2_5
//
// CHANGELOG:
// - Stage-2.5: Extend EventView with optional `tile: Option<Tile>` (NOT consumed by features yet).
// - RC2: Fix opening legal mask to match Python: at empty board in forced_best start, only best opening tile is legal.
// - RC1: Implement FEAT_DIM=193 features + ACTION_SIZE=112 legal mask matching Python ai.features_small/legal_mask_state.
//
// IMPORTANT:
// - This file still contains NO JSON hashing and NO per-move I/O.
// - Parsing is done only in lib.rs boundary helper.

use crate::tile::Tile;

pub const ACTION_SIZE: usize = 112;
pub const FEAT_DIM: usize = 193;

// Match ai.py ENDS order
pub const END_RIGHT: usize = 0;
pub const END_LEFT: usize = 1;
pub const END_UP: usize = 2;
pub const END_DOWN: usize = 3;

#[inline]
pub fn encode_action(tile: Tile, end_idx: usize) -> usize {
    (tile.id() as usize) * 4 + end_idx
}

// ------------------------------
// Belief (matches ai.py exactly)
// ------------------------------
const DECAY_RATE: f64 = 0.85;
const MAX_CUT_PROB: f64 = 0.98;

const CERTAIN_WEIGHT: f64 = 1.0;
const PROBABLE_WEIGHT: f64 = 0.6;
const POSSIBLE_WEIGHT: f64 = 0.35;

#[derive(Clone, Debug)]
pub struct Belief {
    pub cut_prob: [f64; 7],
    pub last_ply: [i32; 7],
    pub hard_forbid: [bool; 7], // Phase0-rescue: certain-pass (no-boneyard) is a hard constraint
}

impl Belief {
    pub fn new() -> Self {
        Self {
            cut_prob: [0.0; 7],
            last_ply: [0; 7],
            hard_forbid: [false; 7],
        }
    }

    fn apply_decay_to(&mut self, value: usize, current_ply: i32) {
        if self.hard_forbid[value] {
            return; // do not decay hard constraints
        }
        let last = self.last_ply[value];
        let diff = current_ply - last;
        if diff <= 0 {
            return;
        }
        self.cut_prob[value] *= DECAY_RATE.powi(diff as i32);
        self.last_ply[value] = current_ply;
    }

    fn update_from_event(&mut self, ev: &EventView) {
        if ev.player != Some(Player::Opponent) {
            return;
        }
        let ply = ev.ply;

        let ends: Vec<usize> = ev
            .open_ends
            .iter()
            .copied()
            .filter(|&v| v <= 6)
            .map(|v| v as usize)
            .collect();

        match ev.typ {
            EventType::Pass => {
                let w = match ev.certainty {
                    Certainty::Certain => 0.92,
                    Certainty::Probable => 0.75,
                    Certainty::Possible => 0.55,
                };
                for v in ends {
                    self.apply_decay_to(v, ply);
                    // Phase0-rescue: opponent certain-pass implies boneyard empty => HARD constraint.
                    if ev.certainty == Certainty::Certain {
                        self.cut_prob[v] = 1.0;
                        self.hard_forbid[v] = true;
                    } else {
                        let cur = self.cut_prob[v];
                        self.cut_prob[v] = cur.max(w).min(MAX_CUT_PROB);
                    }
                    self.last_ply[v] = ply;
                }
            }
            EventType::Draw => {
                let weight = match ev.certainty {
                    Certainty::Certain => CERTAIN_WEIGHT,
                    Certainty::Probable => PROBABLE_WEIGHT,
                    Certainty::Possible => POSSIBLE_WEIGHT,
                };
                let adjustment = weight * 0.30;
                for v in ends {
                    self.apply_decay_to(v, ply);
                    let cur = self.cut_prob[v];
                    let upd = cur + adjustment * (1.0 - cur);
                    self.cut_prob[v] = upd.min(MAX_CUT_PROB);
                    self.last_ply[v] = ply;
                }
            }
            _ => {}
        }
    }
}

// ------------------------------
// Views built by lib.rs (boundary)
// ------------------------------
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Player {
    Me,
    Opponent,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Certainty {
    Certain,
    Probable,
    Possible,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EventType {
    Play,
    Draw,
    Pass,
    RoundStart,
    MatchStart,
    RoundEnd,
    Other,
}

#[derive(Clone, Debug)]
pub struct EventView {
    pub typ: EventType,
    pub ply: i32,
    pub player: Option<Player>,
    pub open_ends: Vec<u8>,
    pub certainty: Certainty,

    // Stage-2.5: optional play tile (NOT used by features_193 yet).
    pub tile: Option<Tile>,

    // IM3: optional end index for play events (0..3 = right/left/up/down).
    // Not used by features_193.
    pub end_idx: Option<u8>,

    // IM3: optional score gained (engine telemetry). Not used by features_193.
    pub score_gained: i32,
}

#[derive(Clone, Debug)]
pub struct BoardView {
    pub center_tile: Option<Tile>,
    pub spinner_sides_open: bool,

    // ends: for each end idx 0..3: (open_value 0..6, is_double_end)
    pub ends: [Option<(u8, bool)>; 4],
    // arms lengths only
    pub arm_lens: [usize; 4],

    // already computed by Python engine (we trust oracle values)
    pub ends_sum: i32,
    pub score_now: i32,

    // played tiles (public info)
    pub played_tiles: Vec<Tile>,
}

#[derive(Clone, Debug)]
pub struct StateView {
    pub my_hand: Vec<Tile>,
    pub forced_play_tile: Option<Tile>,
    pub current_turn_me: bool,
    pub started_from_beginning: bool,
    pub opening_mode_forced_best: bool,
    pub round_over: bool,

    pub opponent_tile_count: i32,
    pub boneyard_count: i32,

    pub my_score: i32,
    pub opp_score: i32,
    pub match_target: i32,
    pub round_index: i32,

    pub board: BoardView,
    pub events: Vec<EventView>,
}

// ------------------------------
// Legal moves + mask (match engine.py legal_moves_me semantics)
// ------------------------------
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

fn legal_ends_for_tile_on_board(b: &BoardView, t: Tile) -> Vec<usize> {
    if b.center_tile.is_none() {
        return vec![END_RIGHT];
    }
    let mut out = Vec::new();
    for end_idx in 0..4 {
        if let Some((v, _is_dbl)) = b.ends[end_idx] {
            if t.has(v) {
                out.push(end_idx);
            }
        }
    }
    out
}

fn legal_moves_me(st: &StateView) -> Vec<(Tile, usize)> {
    // Match engine.GameState.legal_moves_me behavior observed from oracle:
    // At round start (forced_best + started_from_beginning + empty board + me to move),
    // only BEST opening tile is legal (mask_sum=1), unless forced_play_tile exists.

    // If forced tile exists but is not in hand, ignore it (matches engine safety behavior)
    let forced = match st.forced_play_tile {
        Some(ft) if st.my_hand.iter().any(|x| *x == ft) => Some(ft),
        _ => None,
    };

    let board_empty = st.board.center_tile.is_none();

    let tiles: Vec<Tile> = if st.opening_mode_forced_best
        && st.started_from_beginning
        && st.current_turn_me
        && board_empty
        && forced.is_none()
    {
        match best_opening_tile(&st.my_hand) {
            Some(t) => vec![t],
            None => vec![],
        }
    } else if let Some(ft) = forced {
        vec![ft]
    } else {
        st.my_hand.clone()
    };

    let mut out = Vec::new();
    for t in tiles {
        for e in legal_ends_for_tile_on_board(&st.board, t) {
            out.push((t, e));
        }
    }
    out
}

fn must_draw_me(st: &StateView) -> bool {
    if st.round_over || !st.current_turn_me {
        return false;
    }
    if st.forced_play_tile.is_some() {
        return false;
    }
    legal_moves_me(st).is_empty() && st.boneyard_count > 0
}

fn must_pass_me(st: &StateView) -> bool {
    if st.round_over || !st.current_turn_me {
        return false;
    }
    if st.forced_play_tile.is_some() {
        return false;
    }
    legal_moves_me(st).is_empty() && st.boneyard_count == 0
}

pub fn legal_mask_112(st: &StateView) -> [i8; ACTION_SIZE] {
    let mut mask = [0i8; ACTION_SIZE];
    for (t, eidx) in legal_moves_me(st) {
        let a = encode_action(t, eidx);
        if a < ACTION_SIZE {
            mask[a] = 1;
        }
    }
    mask
}

// ------------------------------
// Features 193 (match ai.features_small)
// ------------------------------
pub fn features_193(st: &StateView) -> [f32; FEAT_DIM] {
    // Build belief from events + visible tiles logic
    let belief = build_belief(st);

    let mut out = [0f32; FEAT_DIM];
    let mut off: usize = 0;

    // visible sets
    let mut visible_mask: u32 = 0;
    for t in st.my_hand.iter().copied() {
        visible_mask |= 1u32 << (t.id() as u32);
    }
    for t in st.board.played_tiles.iter().copied() {
        visible_mask |= 1u32 << (t.id() as u32);
    }

    // 1) my_hand one-hot (28)
    for t in st.my_hand.iter().copied() {
        out[off + (t.id() as usize)] = 1.0;
    }
    off += 28;

    // 2) played_set one-hot (28)
    for t in st.board.played_tiles.iter().copied() {
        out[off + (t.id() as usize)] = 1.0;
    }
    off += 28;

    // 3) center_tile one-hot + empty slot (29)
    if let Some(ct) = st.board.center_tile {
        out[off + (ct.id() as usize)] = 1.0;
    } else {
        out[off + 28] = 1.0;
    }
    off += 29;

    // 4) ends open values one-hot (8 each) for right,left,up,down => 32
    for end_idx in 0..4 {
        let idx = match st.board.ends[end_idx] {
            Some((v, _)) if v <= 6 => v as usize,
            _ => 7usize,
        };
        out[off + idx] = 1.0;
        off += 8;
    }

    // 5) ends is_double_open_end flags (4)
    for end_idx in 0..4 {
        let flag = match st.board.ends[end_idx] {
            Some((_v, is_dbl)) => is_dbl,
            None => false,
        };
        out[off + end_idx] = if flag { 1.0 } else { 0.0 };
    }
    off += 4;

    // 6) arms lengths normalized (4) (min(12,len)/12)
    for end_idx in 0..4 {
        let ln = st.board.arm_lens[end_idx].min(12) as f32;
        out[off + end_idx] = ln / 12.0;
    }
    off += 4;

    // 7) center is double + spinner_sides_open (2)
    let center_is_double = st.board.center_tile.map(|t| t.is_double()).unwrap_or(false);
    out[off + 0] = if center_is_double { 1.0 } else { 0.0 };
    out[off + 1] = if st.board.spinner_sides_open { 1.0 } else { 0.0 };
    off += 2;

    // 8) forced_play_tile one-hot + empty slot (29)
    if let Some(ft) = st.forced_play_tile {
        out[off + (ft.id() as usize)] = 1.0;
    } else {
        out[off + 28] = 1.0;
    }
    off += 29;

    // 9) must_draw_me, must_pass_me (2)
    out[off + 0] = if must_draw_me(st) { 1.0 } else { 0.0 };
    out[off + 1] = if must_pass_me(st) { 1.0 } else { 0.0 };
    off += 2;

    // 10) counts: my_hand/7, opp_cnt/7, bone/14 (3)
    out[off + 0] = (st.my_hand.len() as f32) / 7.0;
    out[off + 1] = (st.opponent_tile_count as f32) / 7.0;
    out[off + 2] = (st.boneyard_count as f32) / 14.0;
    off += 3;

    // 11) scores normalized by tgt=max(50,target) (2)
    let tgt = st.match_target.max(50) as f32;
    out[off + 0] = (st.my_score as f32) / tgt;
    out[off + 1] = (st.opp_score as f32) / tgt;
    off += 2;

    // 12) ends_sum/30, ends_sum%5/5, score_now/20 (3)
    let ends_sum = st.board.ends_sum as f32;
    out[off + 0] = ends_sum / 30.0;
    out[off + 1] = ((st.board.ends_sum.rem_euclid(5)) as f32) / 5.0;
    out[off + 2] = (st.board.score_now as f32) / 20.0;
    off += 3;

    // 13) round_index/min50, match_target/min300 (2)
    out[off + 0] = (st.round_index.min(50) as f32) / 50.0;
    out[off + 1] = (st.match_target.min(300) as f32) / 300.0;
    off += 2;

    // 14) cut_prob 7
    for i in 0..7 {
        let v = belief.cut_prob[i].clamp(0.0, 1.0) as f32;
        out[off + i] = v;
    }
    off += 7;

    // 15) ends cut_prob (4)
    for end_idx in 0..4 {
        let val = match st.board.ends[end_idx] {
            Some((v, _)) if v <= 6 => v as usize,
            _ => 999,
        };
        out[off + end_idx] = if val <= 6 {
            belief.cut_prob[val].clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
    }
    off += 4;

    // 16) my_value_dist (7), visible_value_dist (7)
    let mut my_val = [0f32; 7];
    for t in st.my_hand.iter().copied() {
        let (a, b) = t.pips();
        my_val[a as usize] += 1.0;
        my_val[b as usize] += 1.0;
    }
    for i in 0..7 {
        my_val[i] /= 7.0;
        out[off + i] = my_val[i];
    }
    off += 7;

    let mut vis_val = [0f32; 7];
    for id in 0u8..28u8 {
        if (visible_mask & (1u32 << (id as u32))) == 0 {
            continue;
        }
        let t = Tile(id);
        let (a, b) = t.pips();
        vis_val[a as usize] += 1.0;
        vis_val[b as usize] += 1.0;
    }
    for i in 0..7 {
        vis_val[i] /= 28.0;
        out[off + i] = vis_val[i];
    }
    off += 7;

    debug_assert!(off == FEAT_DIM);
    out
}

fn build_belief(st: &StateView) -> Belief {
    let mut b = Belief::new();

    // 1) events-derived
    for ev in st.events.iter() {
        b.update_from_event(ev);
    }

    // Phase0-rescue: if hard_forbid is set, keep cut_prob pinned to 1.0.
    for v in 0..7usize {
        if b.hard_forbid[v] {
            b.cut_prob[v] = 1.0;
        }
    }

    // 2) visible-based adjustments
    let mut visible_mask: u32 = 0;
    for t in st.my_hand.iter().copied() {
        visible_mask |= 1u32 << (t.id() as u32);
    }
    for t in st.board.played_tiles.iter().copied() {
        visible_mask |= 1u32 << (t.id() as u32);
    }

    for value in 0..7 {
        let mut visible_count = 0;
        for id in 0u8..28u8 {
            if (visible_mask & (1u32 << (id as u32))) == 0 {
                continue;
            }
            let t = Tile(id);
            let (a, c) = t.pips();
            if a as usize == value || c as usize == value {
                visible_count += 1;
            }
        }
        let hidden_count = 7 - visible_count;
        if hidden_count == 0 {
            b.cut_prob[value] = 1.0;
        } else if hidden_count == 1 {
            b.cut_prob[value] = b.cut_prob[value].max(0.75);
        } else if hidden_count == 2 {
            b.cut_prob[value] = b.cut_prob[value].max(0.45);
        }
    }

    // 3) decay at end (use max ply + 1 when events carry absolute ply indices)
    let mut current_ply = st.events.len() as i32;
    if !st.events.is_empty() {
        let mut max_ply = 0i32;
        for ev in st.events.iter() {
            if ev.ply > max_ply {
                max_ply = ev.ply;
            }
        }
        current_ply = current_ply.max(max_ply.saturating_add(1));
    }
    for value in 0..7 {
        if b.hard_forbid[value as usize] {
            continue;
        }
        b.apply_decay_to(value as usize, current_ply);
    }

    b
}