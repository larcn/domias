// FILE: src/ismcts.rs | version: 2026-01-17.stage2_5
// CHANGELOG:
// - Stage1 MVP: add opp_played_tiles to InfoState and use it to bias determinization.
// - Stage2 MVP: add opp_avoided_open_values[7] to InfoState and use it to penalize tiles containing avoided values.
// - Stage2 tuning via env var:
//     DOMINO_AVOID_BETA=0.0 disables Stage2 penalty (default=0.10).
// - Stage1 tuning via env var:
//     DOMINO_PLAY_BIAS_ALPHA=0.0 disables Stage1 bias (default=0.12).
//
// NOTES:
// - Hot-path; no PyO3.
// - determinize/opp_has_win_now_exists are crate-internal helpers used by lib.rs.
// - P0 solver hook remains: solve_no_boneyard_delta + root_solve_best_move.

use rand::prelude::*;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::board::{Board, End};
use crate::features::{self, ACTION_SIZE, FEAT_DIM};
use crate::hash;
use crate::mlp::MlpModel;
use crate::tile::Tile;
use crate::infer_model;

// -----------------------------
// Public params
// -----------------------------
#[derive(Copy, Clone, Debug)]
pub struct IsmctsParams {
    pub sims: u32,              // total simulations (already budgeted by caller)
    pub c_puct: f32,            // exploration
    pub temperature: f32,       // PI temperature (applied by caller; we provide visits)
    pub max_plies: u32,         // rollout cap (safety)
    pub opp_mix_greedy: f32,    // opponent policy mix in rollout
    pub leaf_value_weight: f32, // [0..1] mix model leaf value into simulation value (0=off)

    // Strategic knobs:
    pub me_mix_greedy: f32,       // [0..1] "me" greedy ratio in rollouts (1.0 keeps old behavior)
    pub gift_penalty_weight: f32, // [0..1] 0 disables; >0 penalizes opening big immediate scores for opponent
    pub pessimism_alpha_max: f32, // [0..1] 0 disables; >0 mixes min into Q near endgame
}

impl Default for IsmctsParams {
    fn default() -> Self {
        Self {
            sims: 800,
            c_puct: 1.6,
            temperature: 1.0,
            max_plies: 160,
            opp_mix_greedy: 0.60,
            leaf_value_weight: 0.0,
            me_mix_greedy: 1.0,
            gift_penalty_weight: 0.0,
            pessimism_alpha_max: 0.0,
        }
    }
}

// -----------------------------
// Info-state used for root search
// (NO hidden identities, only counts)
// -----------------------------
#[derive(Clone, Debug)]
pub struct InfoState {
    pub board: Board,
    pub my_hand: Vec<Tile>,
    pub opponent_tile_count: i32,
    pub boneyard_count: i32,
    pub forced_play_tile: Option<Tile>,

    pub my_score: i32,
    pub opp_score: i32,
    pub match_target: i32,

    pub current_turn_me: bool,
    pub ply: i32,         // absolute ply within CURRENT ROUND (for belief decay)
    pub round_index: i32, // 1..N within match (feature signal)

    // For belief: only opponent draw/pass events matter, with numeric open_ends.
    pub events: Vec<BeliefEvent>,

    // Stage1 MVP: public info signal used for determinization bias (midgame awareness).
    pub opp_played_tiles: Vec<Tile>,

    // Stage2 MVP: negative evidence counts per value 0..6 (how often opponent "avoided" an open value).
    // Used only to bias determinization; no rules change.
    pub opp_avoided_open_values: [u8; 7],

    // IM3: opponent-hand inference tile probabilities, computed in boundary (lib.rs).
    // If DOMINO_INFER=0 or model missing, this will be all zeros and determinize behaves as before.
    pub opp_infer_tile_p: [f32; 28],
}

#[derive(Copy, Clone, Debug)]
pub enum BeliefEventType {
    Draw,
    Pass,
}

#[derive(Copy, Clone, Debug)]
pub enum BeliefCertainty {
    Certain,
    Probable,
    Possible,
}

#[derive(Clone, Debug)]
pub struct BeliefEvent {
    pub typ: BeliefEventType,
    pub ply: i32,
    pub open_ends: [u8; 4], // up to 4 ends; unused slots = 255
    pub len: u8,
    pub certainty: BeliefCertainty,
}

// -----------------------------
// Belief (cut_prob[0..6]) matches Python ai.py logic
// -----------------------------
const DECAY_RATE: f64 = 0.85;
const MAX_CUT_PROB: f64 = 0.98;

fn apply_decay(cut: &mut [f64; 7], last_ply: &mut [i32; 7], v: usize, ply: i32) {
    let diff = ply - last_ply[v];
    if diff <= 0 {
        return;
    }
    cut[v] *= DECAY_RATE.powi(diff as i32);
    last_ply[v] = ply;
}

fn build_cut_prob(st: &InfoState) -> [f64; 7] {
    let mut cut = [0.0f64; 7];
    let mut last = [0i32; 7];

    for ev in st.events.iter() {
        let ply = ev.ply;
        let w_pass = match ev.certainty {
            BeliefCertainty::Certain => 0.92,
            BeliefCertainty::Probable => 0.75,
            BeliefCertainty::Possible => 0.55,
        };
        let weight_draw = match ev.certainty {
            BeliefCertainty::Certain => 1.0,
            BeliefCertainty::Probable => 0.6,
            BeliefCertainty::Possible => 0.35,
        };
        let adj = weight_draw * 0.30;

        for i in 0..(ev.len as usize) {
            let v = ev.open_ends[i] as i32;
            if !(0..=6).contains(&v) {
                continue;
            }
            let vv = v as usize;
            apply_decay(&mut cut, &mut last, vv, ply);

            match ev.typ {
                BeliefEventType::Pass => {
                    cut[vv] = cut[vv].max(w_pass).min(MAX_CUT_PROB);
                    last[vv] = ply;
                }
                BeliefEventType::Draw => {
                    let cur = cut[vv];
                    cut[vv] = (cur + adj * (1.0 - cur)).min(MAX_CUT_PROB);
                    last[vv] = ply;
                }
            }
        }
    }

    // visible-based adjustment (like ai.build_belief_from_state)
    let visible_mask: u32 = hash::hand_mask(&st.my_hand) | st.board.played_mask();
    for value in 0..7usize {
        let mut visible_count = 0;
        for id in 0u8..28u8 {
            if (visible_mask & (1u32 << (id as u32))) == 0 {
                continue;
            }
            let t = Tile(id);
            let (a, b) = t.pips();
            if a as usize == value || b as usize == value {
                visible_count += 1;
            }
        }
        let hidden_count = 7 - visible_count;
        if hidden_count == 0 {
            cut[value] = 1.0;
        } else if hidden_count == 1 {
            cut[value] = cut[value].max(0.75);
        } else if hidden_count == 2 {
            cut[value] = cut[value].max(0.45);
        }
    }

    // decay at end
    let current_ply = st.ply.max(0);
    for value in 0..7 {
        apply_decay(&mut cut, &mut last, value, current_ply);
        cut[value] = cut[value].clamp(0.0, 1.0);
    }

    cut
}

// CERTAIN opponent pass evidence => values they could NOT play on.
fn forbidden_values_from_certain_pass(st: &InfoState) -> [bool; 7] {
    let mut forbid = [false; 7];
    for ev in st.events.iter() {
        if !matches!(ev.typ, BeliefEventType::Pass) {
            continue;
        }
        if !matches!(ev.certainty, BeliefCertainty::Certain) {
            continue;
        }
        for i in 0..(ev.len as usize) {
            let v = ev.open_ends[i];
            if v <= 6 {
                forbid[v as usize] = true;
            }
        }
    }
    forbid
}

// Weight per tile from cut_prob (matches ai._tile_weight_from_cut)
fn tile_weight_from_cut(t: Tile, cut: &[f64; 7]) -> f64 {
    let (a, b) = t.pips();
    let cp = cut[a as usize].max(cut[b as usize]);
    (1.0 - 0.90 * cp).max(0.01)
}

// -----------------------------
// Stage1/Stage2 inference bias helpers
// -----------------------------
fn play_bias_alpha() -> f64 {
    static ALPHA: OnceLock<f64> = OnceLock::new();
    *ALPHA.get_or_init(|| {
        let d = 0.12f64;
        match std::env::var("DOMINO_PLAY_BIAS_ALPHA") {
            Ok(s) => s.parse::<f64>().ok().unwrap_or(d).clamp(0.0, 1.0),
            Err(_) => d,
        }
    })
}

fn avoid_beta() -> f64 {
    static BETA: OnceLock<f64> = OnceLock::new();
    *BETA.get_or_init(|| {
        let d = 0.10f64;
        match std::env::var("DOMINO_AVOID_BETA") {
            Ok(s) => s.parse::<f64>().ok().unwrap_or(d).clamp(0.0, 1.0),
            Err(_) => d,
        }
    })
}

#[inline]
fn played_value_counts(tiles: &[Tile]) -> [u8; 7] {
    let mut c = [0u8; 7];
    for &t in tiles.iter() {
        let (a, b) = t.pips();
        c[a as usize] = c[a as usize].saturating_add(1);
        c[b as usize] = c[b as usize].saturating_add(1);
    }
    c
}

#[inline]
fn played_bias_mult(t: Tile, cnt: &[u8; 7]) -> f64 {
    let alpha = play_bias_alpha();
    if alpha <= 1e-12 {
        return 1.0;
    }
    let (a, b) = t.pips();
    let m = cnt[a as usize].max(cnt[b as usize]) as f64;
    (1.0 + alpha * m).min(2.2)
}

#[inline]
fn avoid_bias_mult(t: Tile, avoid: &[u8; 7]) -> f64 {
    let beta = avoid_beta();
    if beta <= 1e-12 {
        return 1.0;
    }
    let (a, b) = t.pips();
    let m = avoid[a as usize].max(avoid[b as usize]) as f64;
    (1.0 - beta * m).clamp(0.05, 1.0)
}

#[inline]
fn infer_bias_mult(t: Tile, p: &[f32; 28]) -> f64 {
    if !infer_model::infer_enabled() {
        return 1.0;
    }
    let x = p[t.id() as usize].clamp(0.0, 1.0) as f64;
    // Soft weighting only, conservative:
    // - We avoid crushing "unlikely" tiles too hard (negative tail risk).
    // - Range becomes [0.75 .. 1.25] instead of [0.25 .. 1.25].
    // This keeps determinization diverse while still benefiting from signal.
    let m = 0.75f64 + 0.50f64 * x;
    m.clamp(0.05, 2.0)
}

// Weighted sample without replacement (O(n*k), small n=28)
fn weighted_sample_no_replace(items: &[Tile], weights: &[f64], k: usize, rng: &mut StdRng) -> Vec<Tile> {
    if k == 0 || items.is_empty() {
        return vec![];
    }
    if k >= items.len() {
        let mut v = items.to_vec();
        v.shuffle(rng);
        return v;
    }
    let mut rem_items: Vec<Tile> = items.to_vec();
    let mut rem_w: Vec<f64> = weights.to_vec();
    let mut out: Vec<Tile> = Vec::with_capacity(k);

    for _ in 0..k {
        let sum: f64 = rem_w.iter().sum();
        if sum <= 1e-12 {
            let idx = rng.gen_range(0..rem_items.len());
            out.push(rem_items.swap_remove(idx));
            rem_w.swap_remove(idx);
            continue;
        }
        let mut r = rng.gen::<f64>() * sum;
        let mut pick = rem_items.len() - 1;
        for (i, w) in rem_w.iter().enumerate() {
            r -= *w;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        out.push(rem_items.swap_remove(pick));
        rem_w.swap_remove(pick);
    }

    out
}

// Determinize hidden: choose opp_hand and boneyard from remaining tiles.
// Exposed as crate-internal for lib.rs win-now guard (evaluation-time).
pub(crate) fn determinize(st: &InfoState, rng: &mut StdRng) -> (Vec<Tile>, Vec<Tile>) {
    let visible_mask = hash::hand_mask(&st.my_hand) | st.board.played_mask();
    let mut hidden: Vec<Tile> = Vec::new();
    hidden.reserve(28);
    for id in 0u8..28u8 {
        if (visible_mask & (1u32 << (id as u32))) == 0 {
            hidden.push(Tile(id));
        }
    }
    if hidden.is_empty() {
        return (vec![], vec![]);
    }

    let cut = build_cut_prob(st);
    let forbid = forbidden_values_from_certain_pass(st);

    let cnt = played_value_counts(&st.opp_played_tiles);
    let avoid = st.opp_avoided_open_values;
    let infer_p = st.opp_infer_tile_p;

    // prefer "hard constraint" (exclude forbidden values) if feasible
    let mut allowed_hidden: Vec<Tile> = Vec::new();
    allowed_hidden.reserve(hidden.len());
    for &t in hidden.iter() {
        let (a, b) = t.pips();
        if forbid[a as usize] || forbid[b as usize] {
            continue;
        }
        allowed_hidden.push(t);
    }

    let k = (st.opponent_tile_count.max(0) as usize).min(hidden.len());
    let opp: Vec<Tile> = if k == 0 {
        vec![]
    } else if allowed_hidden.len() >= k {
        let weights: Vec<f64> = allowed_hidden
            .iter()
            .map(|&t| {
                let mut w = tile_weight_from_cut(t, &cut);
                w *= played_bias_mult(t, &cnt);
                w *= avoid_bias_mult(t, &avoid);
                w *= infer_bias_mult(t, &infer_p);
                w
            })
            .collect();
        weighted_sample_no_replace(&allowed_hidden, &weights, k, rng)
    } else {
        // fallback: sample from all hidden but penalize forbidden tiles heavily
        let mut weights: Vec<f64> = Vec::with_capacity(hidden.len());
        for &t in hidden.iter() {
            let mut wt = tile_weight_from_cut(t, &cut);
            let (a, b) = t.pips();
            if forbid[a as usize] || forbid[b as usize] {
                wt *= 0.08;
            }
            wt *= played_bias_mult(t, &cnt);
            wt *= avoid_bias_mult(t, &avoid);
            wt *= infer_bias_mult(t, &infer_p);
            weights.push(wt.max(0.0001));
        }
        weighted_sample_no_replace(&hidden, &weights, k, rng)
    };

    // boneyard from remaining hidden (ALWAYS from full hidden set)
    let mut opp_mask: u32 = 0;
    for t in opp.iter().copied() {
        opp_mask |= 1u32 << (t.id() as u32);
    }

    let mut bone: Vec<Tile> = hidden
        .into_iter()
        .filter(|t| (opp_mask & (1u32 << (t.id() as u32))) == 0)
        .collect();
    bone.shuffle(rng);
    let bc = (st.boneyard_count.max(0) as usize).min(bone.len());
    bone.truncate(bc);

    (opp, bone)
}

/// Existence-based: does opponent have ANY immediate move from their HAND that wins the match now?
/// This matches evaluate.py's "win-now exists" metric semantics (hand-only; no draw-then-play).
pub(crate) fn opp_has_win_now_exists(board: &Board, opp_hand: &[Tile], opp_score: i32, target: i32) -> bool {
    let need = target - opp_score;
    if need <= 0 {
        return false;
    }
    for &t in opp_hand.iter() {
        for e in board.legal_ends_for_tile(t) {
            let mut b2 = board.clone();
            let pts = b2.play(t, e).unwrap_or(0);
            if pts > 0 && pts >= need {
                return true;
            }
        }
    }
    false
}

// -----------------------------
// World rollout (perfect info within determinization)
// -----------------------------
#[derive(Clone)]
struct World {
    board: Board,
    me_hand: Vec<Tile>,
    opp_hand: Vec<Tile>,
    boneyard: Vec<Tile>,
    me_score: i32,
    opp_score: i32,
    target: i32,
    turn_me: bool,
}

fn round_to_nearest_5(x: i32) -> i32 {
    ((x as f64) / 5.0).round() as i32 * 5
}

fn out_points_from_hand(hand: &[Tile]) -> i32 {
    round_to_nearest_5(hand.iter().map(|t| t.pip_sum() as i32).sum())
}

fn locked_delta_points(me_hand: &[Tile], opp_hand: &[Tile]) -> i32 {
    let my = me_hand.iter().map(|t| t.pip_sum() as i32).sum::<i32>();
    let op = opp_hand.iter().map(|t| t.pip_sum() as i32).sum::<i32>();
    let diff = (my - op).abs();
    let pts = round_to_nearest_5(diff);
    if pts == 0 {
        return 0;
    }
    if my < op {
        pts
    } else if op < my {
        -pts
    } else {
        0
    }
}

// -----------------------------
// Phase0-rescue: No-boneyard exact solver (minimax on perfect-info within a determinized world)
// -----------------------------
const SOLVER_MAX_TILES: usize = 6;
const SOLVER_CACHE_MAX: usize = 200_000;

thread_local! {
    static SOLVER_CACHE: RefCell<HashMap<u128, i32>> = RefCell::new(HashMap::new());
}

fn solver_key(board: &Board, me_hand: &[Tile], opp_hand: &[Tile], turn_me: bool) -> u128 {
    let bh = hash::board_hash64(board) as u128;
    let mm = (hash::hand_mask(me_hand) as u128) & ((1u128 << 28) - 1);
    let om = (hash::hand_mask(opp_hand) as u128) & ((1u128 << 28) - 1);
    (bh << 57) | (mm << 29) | (om << 1) | (turn_me as u128)
}

fn remove_tile_from_vec(hand: &mut Vec<Tile>, t: Tile) -> bool {
    if let Some(i) = hand.iter().position(|x| *x == t) {
        hand.swap_remove(i);
        true
    } else {
        false
    }
}

fn solve_no_boneyard_delta_rec(
    board: &Board,
    me_hand: &[Tile],
    opp_hand: &[Tile],
    turn_me: bool,
    cache: &mut HashMap<u128, i32>,
) -> i32 {
    if me_hand.is_empty() {
        return out_points_from_hand(opp_hand);
    }
    if opp_hand.is_empty() {
        return -out_points_from_hand(me_hand);
    }

    let me_moves = legal_moves_for_hand(board, me_hand);
    let op_moves = legal_moves_for_hand(board, opp_hand);

    if me_moves.is_empty() && op_moves.is_empty() {
        return locked_delta_points(me_hand, opp_hand);
    }

    let key = solver_key(board, me_hand, opp_hand, turn_me);
    if let Some(v) = cache.get(&key).copied() {
        return v;
    }

    let v = if turn_me {
        if me_moves.is_empty() {
            solve_no_boneyard_delta_rec(board, me_hand, opp_hand, false, cache)
        } else {
            let mut best = i32::MIN / 4;
            for (t, e) in me_moves {
                let mut b2 = board.clone();
                let pts = b2.play(t, e).unwrap_or(0);
                let mut me2 = me_hand.to_vec();
                let mut op2 = opp_hand.to_vec();
                let _ = remove_tile_from_vec(&mut me2, t);

                let cand = if me2.is_empty() {
                    pts + out_points_from_hand(&op2)
                } else {
                    pts + solve_no_boneyard_delta_rec(&b2, &me2, &op2, false, cache)
                };
                if cand > best {
                    best = cand;
                }
            }
            best
        }
    } else {
        if op_moves.is_empty() {
            solve_no_boneyard_delta_rec(board, me_hand, opp_hand, true, cache)
        } else {
            let mut best = i32::MAX / 4;
            for (t, e) in op_moves {
                let mut b2 = board.clone();
                let pts = b2.play(t, e).unwrap_or(0);
                let mut me2 = me_hand.to_vec();
                let mut op2 = opp_hand.to_vec();
                let _ = remove_tile_from_vec(&mut op2, t);

                let cand = if op2.is_empty() {
                    -(pts + out_points_from_hand(&me2))
                } else {
                    -pts + solve_no_boneyard_delta_rec(&b2, &me2, &op2, true, cache)
                };
                if cand < best {
                    best = cand;
                }
            }
            best
        }
    };

    cache.insert(key, v);
    v
}

pub(crate) fn solve_no_boneyard_delta(board: &Board, me_hand: &[Tile], opp_hand: &[Tile], turn_me: bool) -> i32 {
    SOLVER_CACHE.with(|c| {
        let mut hm = c.borrow_mut();
        if hm.len() > SOLVER_CACHE_MAX {
            hm.clear();
        }
        solve_no_boneyard_delta_rec(board, me_hand, opp_hand, turn_me, &mut hm)
    })
}

/// Root-level perfect-information solver hook (P0).
pub(crate) fn root_solve_best_move(st: &InfoState) -> Option<(Tile, End)> {
    if !st.current_turn_me {
        return None;
    }
    if st.boneyard_count != 0 {
        return None;
    }
    if st.opponent_tile_count <= 0 {
        return None;
    }

    let visible_mask: u32 = hash::hand_mask(&st.my_hand) | st.board.played_mask();
    let mut hidden: Vec<Tile> = Vec::new();
    hidden.reserve(28);
    for id in 0u8..28u8 {
        if (visible_mask & (1u32 << (id as u32))) == 0 {
            hidden.push(Tile(id));
        }
    }

    if hidden.len() as i32 != st.opponent_tile_count {
        return None;
    }

    let my_len = st.my_hand.len();
    let opp_len = hidden.len();
    if my_len + opp_len > 12 {
        return None;
    }

    let opp_hand_exact: Vec<Tile> = hidden;

    let legal = legal_moves_root(st);
    if legal.is_empty() {
        return None;
    }

    let mut best_my_win: Option<(Tile, End, i32)> = None;
    let mut best_safe: Option<(Tile, End, i32)> = None;
    let mut best_any: Option<(Tile, End, i32)> = None;

    for (t, e) in legal.into_iter() {
        let mut board2 = st.board.clone();
        let pts_immediate = match board2.play(t, e) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let my_win_now = st.my_score + pts_immediate >= st.match_target;
        let opp_win_now = opp_has_win_now_exists(&board2, &opp_hand_exact, st.opp_score, st.match_target);

        let mut my_hand_after = st.my_hand.clone();
        let _ = remove_tile_from_vec(&mut my_hand_after, t);

        let delta_rest = solve_no_boneyard_delta(&board2, &my_hand_after, &opp_hand_exact, false);
        let total_delta = pts_immediate + delta_rest;

        if my_win_now {
            match best_my_win {
                None => best_my_win = Some((t, e, total_delta)),
                Some((_bt, _be, cur)) => {
                    if total_delta > cur {
                        best_my_win = Some((t, e, total_delta));
                    }
                }
            }
            continue;
        }

        if !opp_win_now {
            match best_safe {
                None => best_safe = Some((t, e, total_delta)),
                Some((_bt, _be, cur)) => {
                    if total_delta > cur {
                        best_safe = Some((t, e, total_delta));
                    }
                }
            }
        }

        match best_any {
            None => best_any = Some((t, e, total_delta)),
            Some((_bt, _be, cur)) => {
                if total_delta > cur {
                    best_any = Some((t, e, total_delta));
                }
            }
        }
    }

    if let Some((t, e, _)) = best_my_win {
        return Some((t, e));
    }
    if let Some((t, e, _)) = best_safe {
        return Some((t, e));
    }
    if let Some((t, e, _)) = best_any {
        return Some((t, e));
    }
    None
}

// Phase0: Hard gating to prevent catastrophic slowdown when enabling rollout policy.
fn allow_policy_in_rollout(w: &World) -> bool {
    let my_to = (w.target - w.me_score).max(0);
    let op_to = (w.target - w.opp_score).max(0);
    let close = my_to.min(op_to);
    if close > 25 {
        return false;
    }
    let total_tiles = w.me_hand.len() + w.opp_hand.len();
    if total_tiles > 12 {
        return false;
    }
    true
}

// Endgame-aware win proxy: distance-to-target
fn match_win_prob(my_score: i32, opp_score: i32, target: i32) -> f32 {
    let my_to = (target - my_score).max(0) as f32;
    let op_to = (target - opp_score).max(0) as f32;

    if my_to <= 0.0 && op_to > 0.0 {
        return 1.0;
    }
    if op_to <= 0.0 && my_to > 0.0 {
        return 0.0;
    }
    if op_to <= 0.0 && my_to <= 0.0 {
        return 0.5;
    }

    let close = my_to.min(op_to);
    let scale = (6.0 + 0.15 * close).clamp(6.0, 14.0);
    let x = (op_to - my_to) / scale;
    1.0 / (1.0 + (-x).exp())
}

fn legal_moves_for_hand(board: &Board, hand: &[Tile]) -> Vec<(Tile, End)> {
    let mut out = Vec::new();
    for &t in hand {
        for e in board.legal_ends_for_tile(t) {
            out.push((t, e));
        }
    }
    out
}

fn pick_greedy(board: &Board, hand: &[Tile]) -> Option<(Tile, End)> {
    let moves = legal_moves_for_hand(board, hand);
    if moves.is_empty() {
        return None;
    }
    let mut best = moves[0];
    let mut best_pts = -1;
    let mut best_pips = i32::MAX;
    for (t, e) in moves {
        let mut b2 = board.clone();
        let pts = b2.play(t, e).unwrap_or(0);
        let pips = t.pip_sum() as i32;
        if pts > best_pts || (pts == best_pts && pips < best_pips) {
            best = (t, e);
            best_pts = pts;
            best_pips = pips;
        }
    }
    Some(best)
}

fn best_reply_points(board: &Board, hand: &[Tile]) -> i32 {
    if let Some((t, e)) = pick_greedy(board, hand) {
        let mut b2 = board.clone();
        b2.play(t, e).unwrap_or(0)
    } else {
        0
    }
}

fn gift_penalty_from_reply(opp_reply_pts: i32, opp_score: i32, target: i32) -> f32 {
    let to_target = (target - opp_score).max(0);
    if to_target > 0 && opp_reply_pts >= to_target {
        return 0.95;
    }
    if opp_reply_pts >= 20 {
        return 0.45;
    }
    if opp_reply_pts >= 15 {
        return 0.30;
    }
    if opp_reply_pts >= 10 {
        return 0.18;
    }
    if opp_reply_pts >= 5 {
        return 0.08;
    }
    0.0
}

fn pick_model_policy_argmax(
    board: &Board,
    hand: &[Tile],
    my_score: i32,
    opp_score: i32,
    target: i32,
    opp_cnt: i32,
    bone_cnt: i32,
    model: &MlpModel,
    root_events: &[BeliefEvent],
    ply: i32,
    round_index: i32,
) -> Option<(Tile, End)> {
    let moves = legal_moves_for_hand(board, hand);
    if moves.is_empty() {
        return None;
    }
    let mut mask = [0i8; ACTION_SIZE];
    for (t, e) in moves.iter().copied() {
        mask[features::encode_action(t, e.idx())] = 1;
    }
    let st = InfoState {
        board: board.clone(),
        my_hand: hand.to_vec(),
        opponent_tile_count: opp_cnt,
        boneyard_count: bone_cnt,
        forced_play_tile: None,
        my_score,
        opp_score,
        match_target: target,
        current_turn_me: true,
        ply,
        round_index,
        events: root_events.to_vec(),
        opp_played_tiles: Vec::new(),
        opp_avoided_open_values: [0u8; 7],
        opp_infer_tile_p: [0.0f32; 28],
    };
    let feat = state_features_193(&st);
    let (pol, _v) = model.predict(&feat, &mask);

    let mut best = moves[0];
    let mut bestp = -1.0f32;
    for (t, e) in moves {
        let a = features::encode_action(t, e.idx());
        let p = pol[a];
        if p > bestp {
            bestp = p;
            best = (t, e);
        }
    }
    Some(best)
}

// model value in [-1,+1] from ROOT perspective
fn model_value_from_world_value_only(
    model: &MlpModel,
    w: &World,
    root_events: &[BeliefEvent],
    ply: i32,
    round_index: i32,
    scratch: &mut [f32],
) -> f32 {
    let (my_hand, opp_cnt, my_score, opp_score, sign) = if w.turn_me {
        (
            w.me_hand.clone(),
            w.opp_hand.len() as i32,
            w.me_score,
            w.opp_score,
            1.0f32,
        )
    } else {
        (
            w.opp_hand.clone(),
            w.me_hand.len() as i32,
            w.opp_score,
            w.me_score,
            -1.0f32,
        )
    };

    let st = InfoState {
        board: w.board.clone(),
        my_hand,
        opponent_tile_count: opp_cnt,
        boneyard_count: w.boneyard.len() as i32,
        forced_play_tile: None,
        my_score,
        opp_score,
        match_target: w.target,
        current_turn_me: true,
        ply,
        round_index,
        events: root_events.to_vec(),
        opp_played_tiles: Vec::new(),
        opp_avoided_open_values: [0u8; 7],
        opp_infer_tile_p: [0.0f32; 28],
    };

    let feat = state_features_193(&st);
    sign * model.predict_value_only(&feat, scratch)
}

fn play_from_hand(board: &mut Board, hand: &mut Vec<Tile>, mv: (Tile, End)) -> i32 {
    let (t, e) = mv;
    let pts = board.play(t, e).unwrap_or(0);
    if let Some(i) = hand.iter().position(|x| *x == t) {
        hand.swap_remove(i);
    }
    pts
}

fn rollout_one_round(
    mut w: World,
    rng: &mut StdRng,
    model: Option<&MlpModel>,
    max_plies: u32,
    opp_mix_greedy: f32,
    me_mix_greedy: f32,
    root_events: &[BeliefEvent],
    root_ply: i32,
    root_round_index: i32,
) -> (i32, i32) {
    let mut plies = 0u32;

    if w.boneyard.is_empty() {
        let total_tiles = w.me_hand.len() + w.opp_hand.len();
        if total_tiles > 0 && total_tiles <= SOLVER_MAX_TILES {
            let delta = solve_no_boneyard_delta(&w.board, &w.me_hand, &w.opp_hand, w.turn_me);
            if delta > 0 {
                w.me_score += delta;
            } else if delta < 0 {
                w.opp_score += -delta;
            }
            return (w.me_score, w.opp_score);
        }
    }

    loop {
        if w.me_score >= w.target || w.opp_score >= w.target {
            break;
        }
        if plies >= max_plies {
            break;
        }
        plies += 1;

        if w.me_hand.is_empty() {
            w.me_score += out_points_from_hand(&w.opp_hand);
            break;
        }
        if w.opp_hand.is_empty() {
            w.opp_score += out_points_from_hand(&w.me_hand);
            break;
        }

        if w.boneyard.is_empty() {
            let total_tiles = w.me_hand.len() + w.opp_hand.len();
            if total_tiles > 0 && total_tiles <= SOLVER_MAX_TILES {
                let delta = solve_no_boneyard_delta(&w.board, &w.me_hand, &w.opp_hand, w.turn_me);
                if delta > 0 {
                    w.me_score += delta;
                } else if delta < 0 {
                    w.opp_score += -delta;
                }
                break;
            }
        }

        if w.boneyard.is_empty() {
            let me_moves = legal_moves_for_hand(&w.board, &w.me_hand);
            let op_moves = legal_moves_for_hand(&w.board, &w.opp_hand);
            if me_moves.is_empty() && op_moves.is_empty() {
                let delta = locked_delta_points(&w.me_hand, &w.opp_hand);
                if delta > 0 {
                    w.me_score += delta;
                } else if delta < 0 {
                    w.opp_score += -delta;
                }
                break;
            }
        }

        if w.turn_me {
            let mv: Option<(Tile, End)> = if rng.gen::<f32>() < me_mix_greedy {
                pick_greedy(&w.board, &w.me_hand)
            } else if let Some(m) = model {
                if !allow_policy_in_rollout(&w) {
                    pick_greedy(&w.board, &w.me_hand)
                } else {
                    pick_model_policy_argmax(
                        &w.board,
                        &w.me_hand,
                        w.me_score,
                        w.opp_score,
                        w.target,
                        w.opp_hand.len() as i32,
                        w.boneyard.len() as i32,
                        m,
                        root_events,
                        root_ply,
                        root_round_index,
                    )
                }
            } else {
                pick_greedy(&w.board, &w.me_hand)
            };

            if let Some(mv) = mv {
                let pts = play_from_hand(&mut w.board, &mut w.me_hand, mv);
                w.me_score += pts;
                w.turn_me = false;
                continue;
            }

            // draw-until-play
            let mut played = false;
            while !w.boneyard.is_empty() {
                let drawn = w.boneyard.pop().unwrap();
                w.me_hand.push(drawn);

                if !w.board.legal_ends_for_tile(drawn).is_empty() {
                    let ends = w.board.legal_ends_for_tile(drawn);
                    let mut best_end = ends[0];
                    let mut best_pts = -1;
                    for e in ends {
                        let mut b2 = w.board.clone();
                        let pts = b2.play(drawn, e).unwrap_or(0);
                        if pts > best_pts {
                            best_pts = pts;
                            best_end = e;
                        }
                    }
                    let pts = play_from_hand(&mut w.board, &mut w.me_hand, (drawn, best_end));
                    w.me_score += pts;
                    played = true;
                    break;
                }
            }
            if played {
                w.turn_me = false;
                continue;
            }
            w.turn_me = false;
        } else {
            let mv: Option<(Tile, End)> = if rng.gen::<f32>() < opp_mix_greedy {
                pick_greedy(&w.board, &w.opp_hand)
            } else if let Some(m) = model {
                if !allow_policy_in_rollout(&w) {
                    pick_greedy(&w.board, &w.opp_hand)
                } else {
                    pick_model_policy_argmax(
                        &w.board,
                        &w.opp_hand,
                        w.opp_score,
                        w.me_score,
                        w.target,
                        w.me_hand.len() as i32,
                        w.boneyard.len() as i32,
                        m,
                        root_events,
                        root_ply,
                        root_round_index,
                    )
                }
            } else {
                pick_greedy(&w.board, &w.opp_hand)
            };

            if let Some(mv) = mv {
                let pts = play_from_hand(&mut w.board, &mut w.opp_hand, mv);
                w.opp_score += pts;
                w.turn_me = true;
                continue;
            }

            // draw-until-play
            let mut played = false;
            while !w.boneyard.is_empty() {
                let drawn = w.boneyard.pop().unwrap();
                w.opp_hand.push(drawn);

                if !w.board.legal_ends_for_tile(drawn).is_empty() {
                    let ends = w.board.legal_ends_for_tile(drawn);
                    let mut best_end = ends[0];
                    let mut best_pts = -1;
                    for e in ends {
                        let mut b2 = w.board.clone();
                        let pts = b2.play(drawn, e).unwrap_or(0);
                        if pts > best_pts {
                            best_pts = pts;
                            best_end = e;
                        }
                    }
                    let pts = play_from_hand(&mut w.board, &mut w.opp_hand, (drawn, best_end));
                    w.opp_score += pts;
                    played = true;
                    break;
                }
            }
            if played {
                w.turn_me = true;
                continue;
            }
            w.turn_me = true;
        }
    }

    (w.me_score, w.opp_score)
}

// -----------------------------
// Root ISMCTS: visits only
// -----------------------------
pub fn ismcts_root_visits(st: &InfoState, model: Option<&MlpModel>, params: IsmctsParams, seed: u64) -> [u32; ACTION_SIZE] {
    let mut visits = [0u32; ACTION_SIZE];

    if !st.current_turn_me {
        return visits;
    }

    let legal = legal_moves_root(st);
    if legal.is_empty() {
        return visits;
    }

    // match-pressure schedule: near target => more sims
    let sims_eff: u32 = {
        let base = params.sims.max(1);
        let my_to = (st.match_target - st.my_score).max(0);
        let op_to = (st.match_target - st.opp_score).max(0);
        let close = my_to.min(op_to);
        let mult: f32 = if close <= 20 { 1.8 } else if close <= 40 { 1.4 } else { 1.0 };
        let s = ((base as f32) * mult).round() as u32;
        s.clamp(100, 12000)
    };

    let (priors, _legal_mask) = root_priors(st, &legal, model);

    let m = legal.len();
    let mut n: Vec<u32> = vec![0; m];
    let mut wsum: Vec<f32> = vec![0.0; m];
    let mut q: Vec<f32> = vec![0.0; m];
    let mut qmin: Vec<f32> = vec![1.0; m];

    let mut rng = StdRng::seed_from_u64(seed);

    let mut value_scratch: Vec<f32> = Vec::new();
    if let Some(mdl) = model {
        if params.leaf_value_weight > 1e-6 {
            value_scratch = vec![0.0; mdl.hidden];
        }
    }

    let alpha_max = params.pessimism_alpha_max.clamp(0.0, 1.0);
    let alpha: f32 = if alpha_max > 1e-6 {
        let my_to = (st.match_target - st.my_score).max(0);
        let op_to = (st.match_target - st.opp_score).max(0);
        let close = my_to.min(op_to);
        let raw = if close <= 10 {
            1.0
        } else if close <= 20 {
            0.6
        } else if close <= 35 {
            0.35
        } else {
            0.10
        };
        (alpha_max * raw).clamp(0.0, alpha_max)
    } else {
        0.0
    };

    for _sim in 0..sims_eff {
        let total_n: u32 = n.iter().sum();
        let sqrt_n = ((total_n.max(1)) as f32).sqrt();

        let mut best_i = 0usize;
        let mut best_ucb = -1.0e30f32;
        for i in 0..m {
            let u = params.c_puct * priors[i] * (sqrt_n / ((1 + n[i]) as f32));
            let q_eff = (1.0 - alpha) * q[i] + alpha * qmin[i];
            let ucb = q_eff + u;
            if ucb > best_ucb {
                best_ucb = ucb;
                best_i = i;
            }
        }

        let (opp_hand, boneyard) = determinize(st, &mut rng);

        let (root_tile, root_end) = legal[best_i];
        let mut world = World {
            board: st.board.clone(),
            me_hand: st.my_hand.clone(),
            opp_hand,
            boneyard,
            me_score: st.my_score,
            opp_score: st.opp_score,
            target: st.match_target,
            turn_me: true,
        };

        // apply root move
        let pts = world.board.play(root_tile, root_end).unwrap_or(0);
        world.me_score += pts;
        if let Some(pos) = world.me_hand.iter().position(|x| *x == root_tile) {
            world.me_hand.swap_remove(pos);
        }

        let terminal_now = if world.me_hand.is_empty() {
            world.me_score += out_points_from_hand(&world.opp_hand);
            true
        } else {
            false
        };

        let gp_w = params.gift_penalty_weight.clamp(0.0, 1.0);
        let gift_pen: f32 = if gp_w > 1e-6 {
            let opp_reply_pts = best_reply_points(&world.board, &world.opp_hand);
            gp_w * gift_penalty_from_reply(opp_reply_pts, world.opp_score, st.match_target)
        } else {
            0.0
        };

        let v_rollout: f32 = if terminal_now {
            let wp = match_win_prob(world.me_score, world.opp_score, st.match_target);
            2.0 * wp - 1.0
        } else {
            world.turn_me = false;
            let (mf, of) = rollout_one_round(
                world.clone(),
                &mut rng,
                model,
                params.max_plies,
                params.opp_mix_greedy,
                params.me_mix_greedy.clamp(0.0, 1.0),
                &st.events,
                st.ply,
                st.round_index,
            );
            let wp = match_win_prob(mf, of, st.match_target);
            2.0 * wp - 1.0
        };

        let mut v = v_rollout;
        if !terminal_now {
            if let Some(mdl) = model {
                let wgt = params.leaf_value_weight.clamp(0.0, 1.0);
                if wgt > 1e-6 {
                    let vm = model_value_from_world_value_only(
                        mdl,
                        &world,
                        &st.events,
                        st.ply,
                        st.round_index,
                        &mut value_scratch,
                    );
                    v = (1.0 - wgt) * v_rollout + wgt * vm;
                }
            }
        }

        if gift_pen > 0.0 {
            v = (v - gift_pen).clamp(-1.0, 1.0);
        }

        n[best_i] += 1;
        wsum[best_i] += v;
        q[best_i] = wsum[best_i] / (n[best_i] as f32);
        qmin[best_i] = qmin[best_i].min(v);
    }

    for (i, (t, e)) in legal.iter().copied().enumerate() {
        let a = features::encode_action(t, e.idx());
        visits[a] = n[i];
    }

    visits
}

fn legal_moves_root(st: &InfoState) -> Vec<(Tile, End)> {
    let tiles: Vec<Tile> = if let Some(ft) = st.forced_play_tile {
        vec![ft]
    } else {
        st.my_hand.clone()
    };
    let mut out = Vec::new();
    for t in tiles {
        for e in st.board.legal_ends_for_tile(t) {
            out.push((t, e));
        }
    }
    out
}

fn root_priors(st: &InfoState, legal: &[(Tile, End)], model: Option<&MlpModel>) -> (Vec<f32>, [i8; ACTION_SIZE]) {
    let mut mask = [0i8; ACTION_SIZE];
    for &(t, e) in legal.iter() {
        let a = features::encode_action(t, e.idx());
        mask[a] = 1;
    }

    let mut priors = vec![0.0f32; legal.len()];
    if let Some(m) = model {
        let feat = state_features_193(st);
        let (pol, _v) = m.predict(&feat, &mask);
        let mut sum = 0.0f32;
        for (i, &(t, e)) in legal.iter().enumerate() {
            let a = features::encode_action(t, e.idx());
            priors[i] = pol[a];
            sum += priors[i];
        }
        if sum <= 1e-12 {
            let u = 1.0 / (legal.len() as f32);
            for x in priors.iter_mut() {
                *x = u;
            }
        } else {
            for x in priors.iter_mut() {
                *x /= sum;
            }
        }
        // prior-mix (reduce sensitivity)
        let u = 1.0 / (legal.len() as f32);
        for x in priors.iter_mut() {
            *x = 0.85 * (*x) + 0.15 * u;
        }
        let s: f32 = priors.iter().sum();
        for x in priors.iter_mut() {
            *x /= s.max(1e-12);
        }
    } else {
        let u = 1.0 / (legal.len() as f32);
        for x in priors.iter_mut() {
            *x = u;
        }
    }

    (priors, mask)
}

fn state_features_193(st: &InfoState) -> [f32; FEAT_DIM] {
    let mut ends: [Option<(u8, bool)>; 4] = [None, None, None, None];
    for e in [End::Right, End::Left, End::Up, End::Down] {
        if let Some(es) = st.board.ends_raw()[e.idx()] {
            ends[e.idx()] = Some((es.open_value, es.is_double_end));
        }
    }
    let arms = st.board.arms_raw();
    let arm_lens = [arms[0].len(), arms[1].len(), arms[2].len(), arms[3].len()];

    let board_view = features::BoardView {
        center_tile: st.board.center_tile,
        spinner_sides_open: st.board.spinner_sides_open,
        ends,
        arm_lens,
        ends_sum: st.board.ends_sum(),
        score_now: st.board.score_now(),
        played_tiles: st.board.played_order().to_vec(),
    };

    let mut evs: Vec<features::EventView> = Vec::with_capacity(st.events.len());
    for ev in st.events.iter() {
        let player = Some(features::Player::Opponent);
        let certainty = match ev.certainty {
            BeliefCertainty::Certain => features::Certainty::Certain,
            BeliefCertainty::Probable => features::Certainty::Probable,
            BeliefCertainty::Possible => features::Certainty::Possible,
        };
        let typ = match ev.typ {
            BeliefEventType::Draw => features::EventType::Draw,
            BeliefEventType::Pass => features::EventType::Pass,
        };
        let mut open_ends: Vec<u8> = Vec::with_capacity(ev.len as usize);
        for i in 0..(ev.len as usize) {
            open_ends.push(ev.open_ends[i]);
        }
        evs.push(features::EventView {
            typ,
            ply: ev.ply,
            player,
            open_ends,
            certainty,
            tile: None,
            end_idx: None,
            score_gained: 0,
        });
    }

    let sv = features::StateView {
        my_hand: st.my_hand.clone(),
        forced_play_tile: st.forced_play_tile,
        current_turn_me: st.current_turn_me,
        started_from_beginning: false,
        opening_mode_forced_best: false,
        round_over: false,
        opponent_tile_count: st.opponent_tile_count,
        boneyard_count: st.boneyard_count,
        my_score: st.my_score,
        opp_score: st.opp_score,
        match_target: st.match_target,
        round_index: st.round_index.max(1),
        board: board_view,
        events: evs,
    };

    features::features_193(&sv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, End};

    #[test]
    fn determinize_avoid_penalizes_avoided_values() {
        // If opponent "avoided" value 6 many times, sampled hands should contain fewer 6-tiles on average.
        // Note: env is cached per process in avoid_beta(); tests run in same process => set before first call.
        std::env::set_var("DOMINO_AVOID_BETA", "0.10");
        std::env::set_var("DOMINO_PLAY_BIAS_ALPHA", "0.0");

        let mut b = Board::new();
        let _ = b.play(Tile::parse("2-2").unwrap(), End::Right);

        let my_hand = vec![
            Tile::parse("0-0").unwrap(),
            Tile::parse("1-0").unwrap(),
            Tile::parse("2-0").unwrap(),
            Tile::parse("3-0").unwrap(),
            Tile::parse("4-0").unwrap(),
            Tile::parse("5-0").unwrap(),
            Tile::parse("6-0").unwrap(),
        ];

        let base = InfoState {
            board: b.clone(),
            my_hand: my_hand.clone(),
            opponent_tile_count: 7,
            boneyard_count: 14,
            forced_play_tile: None,
            my_score: 0,
            opp_score: 0,
            match_target: 150,
            current_turn_me: true,
            ply: 1,
            round_index: 1,
            events: Vec::new(),
            opp_played_tiles: Vec::new(),
            opp_avoided_open_values: [0u8; 7],
            opp_infer_tile_p: [0.0f32; 28],
        };

        let mut avoided = base.clone();
        avoided.opp_avoided_open_values[6] = 6;

        fn count_six(hand: &[Tile]) -> usize {
            hand.iter().filter(|t| t.has(6)).count()
        }

        let mut sum_base = 0usize;
        let mut sum_avoid = 0usize;
        let samples = 200usize;
        for i in 0..samples {
            let mut r1 = StdRng::seed_from_u64(3000 + i as u64);
            let (h1, _) = determinize(&base, &mut r1);
            sum_base += count_six(&h1);

            let mut r2 = StdRng::seed_from_u64(4000 + i as u64);
            let (h2, _) = determinize(&avoided, &mut r2);
            sum_avoid += count_six(&h2);
        }

        assert!(
            sum_avoid < sum_base,
            "expected avoid bias to reduce six count (base={sum_base} avoid={sum_avoid})"
        );
    }
}