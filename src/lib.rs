// FILE: src/lib.rs | version: 2026-01-17.stage2_runtime
//
// Bridge-only + Rust move picker API for evaluation (state_dict -> best move).
//
// Key points:
// - This module is boundary-only (PyO3); hot path is in ismcts.rs.
// - Board is built directly from Python snapshot parts (NO replay) to avoid spinner timeline bugs.
// - P0: Root solver hook for perfect-info endgames (boneyard=0 + full opp hand inferable).
// - P4-A3: For .json model_path, always load JSON (weights.bin v1 cannot represent spike/quantile heads).
// - P4-B1: If model.has_spike, pick root move using spike head instead of manual sampling guard.
// - Forensics: optional root candidate dump (return_root_stats/root_stats_top_k).
// - Stage1/Stage2 runtime inference wiring:
//     * collect opponent played tiles
//     * collect soft negative evidence: open values repeatedly "avoided" by opponent plays (skip forced-play after draw)
//
// Exports:
// - version
// - generate_and_save
// - state_apply_script
// - features_from_state_dict
// - suggest_move_ismcts

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use rand::prelude::*;
use rand::seq::SliceRandom;

// --- modules ---
mod tile;
mod board;
mod state;
mod features;
mod hash;
mod mlp;
mod infer_model;
mod ismcts;
mod shards;
mod inf_shards;
mod factory;

use features::{BoardView, Certainty, EventType, EventView, Player, StateView};

#[pyfunction]
fn version() -> &'static str {
    concat!("domino_rs/", env!("CARGO_PKG_VERSION"), " (p0+p4+stage2)")
}

#[pyfunction]
fn generate_and_save(py: Python<'_>, config_path: &str, output_dir: &str) -> PyResult<String> {
    py.allow_threads(|| factory::generate_and_save(config_path, output_dir))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
fn state_apply_script(
    py: Python<'_>,
    my_hand: Vec<String>,
    match_target: i32,
    script: Vec<String>,
) -> PyResult<PyObject> {
    if my_hand.len() != 7 {
        return Err(PyValueError::new_err("my_hand must have exactly 7 tiles"));
    }
    let mut hand_tiles = Vec::with_capacity(7);
    for s in my_hand.iter() {
        hand_tiles.push(tile::Tile::parse(s).map_err(PyValueError::new_err)?);
    }
    let snap = state::apply_script(hand_tiles, match_target, &script).map_err(PyValueError::new_err)?;
    sval_to_pydict(py, &snap)
}

#[pyfunction]
fn features_from_state_dict(py: Python<'_>, state_dict: &Bound<'_, PyAny>) -> PyResult<(PyObject, PyObject)> {
    let st = parse_state_view(state_dict)?;
    let feat = features::features_193(&st);
    let mask = features::legal_mask_112(&st);

    let feat_bytes = PyBytes::new(py, cast_f32_to_u8(&feat));
    let mask_bytes = PyBytes::new(py, cast_i8_to_u8(&mask));
    Ok((feat_bytes.into(), mask_bytes.into()))
}

fn cast_f32_to_u8(x: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}
fn cast_i8_to_u8(x: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len()) }
}

fn dict_get_required<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    let opt = d.get_item(key)?;
    opt.ok_or_else(|| PyValueError::new_err(format!("missing key: {}", key)))
}

#[derive(Clone, Debug)]
struct OppInferenceSignals {
    played_tiles: Vec<tile::Tile>,
    avoided_open_values: [u8; 7],
}

#[derive(Clone, Debug)]
struct OppEndChoiceSignals {
    played_mask28: u32,
    prefer_new_open_values: [u8; 7],
    avoid_new_open_values: [u8; 7],
}

fn extract_opp_inference_signals(events: &[EventView], played_mask_round: u32) -> OppInferenceSignals {
    let mut played: Vec<tile::Tile> = Vec::new();
    let mut avoided: [u8; 7] = [0u8; 7];

    let mut last_open: Vec<u8> = Vec::new(); // open_ends AFTER previous event = BEFORE current event
    let mut prev_was_opp_draw: bool = false; // immediate previous event was opponent draw

    for ev in events.iter() {
        if ev.player == Some(Player::Opponent) && matches!(ev.typ, EventType::Play) {
            if let Some(t) = ev.tile {
                if (played_mask_round & (1u32 << (t.id() as u32))) != 0 {
                    played.push(t);
                    if !prev_was_opp_draw {
                        let (a, b) = t.pips();
                        for &v in last_open.iter() {
                            if v <= 6 && v != a && v != b {
                                avoided[v as usize] = avoided[v as usize].saturating_add(1);
                            }
                        }
                    }
                }
            }
            prev_was_opp_draw = false;
        } else {
            prev_was_opp_draw = ev.player == Some(Player::Opponent) && matches!(ev.typ, EventType::Draw);
        }

        last_open = ev.open_ends.clone();
    }

    OppInferenceSignals { played_tiles: played, avoided_open_values: avoided }
}

fn end_from_idx(idx: u8) -> Option<board::End> {
    match idx {
        0 => Some(board::End::Right),
        1 => Some(board::End::Left),
        2 => Some(board::End::Up),
        3 => Some(board::End::Down),
        _ => None,
    }
}

// IM3: compute current-round slice (events are match-long; we want signals only for current round).
fn current_round_events(events: &[EventView]) -> &[EventView] {
    let mut start = 0usize;
    for (i, ev) in events.iter().enumerate() {
        if matches!(ev.typ, EventType::RoundStart) {
            start = i;
        }
    }
    &events[start..]
}

fn round_state_view_for_infer(stv: &StateView) -> StateView {
    // IMPORTANT: training data (factory.rs) resets belief each round.
    // Runtime StateView includes match-long events, so we must slice to current round
    // for inference feature parity.
    let round_evs = current_round_events(&stv.events);
    let mut out = stv.clone();
    out.events = round_evs.to_vec();
    out
}

// IM3 signal: End-choice preference/avoidance for opponent plays.
// We replay board from current round events and compute:
// - played_mask28: opponent played tiles this round (28-bit mask)
// - prefer_new_open_values[v]: how often opponent chose an end that opens value v
// - avoid_new_open_values[v]: how often opponent could have opened v by choosing another legal end for same tile
fn extract_opp_endchoice_signals(round_events: &[EventView]) -> OppEndChoiceSignals {
    let mut b = board::Board::new();
    let mut played_mask: u32 = 0;
    let mut prefer = [0u8; 7];
    let mut avoid = [0u8; 7];

    let mut prev_was_opp_draw = false;

    for ev in round_events.iter() {
        // only plays affect board replay
        if ev.typ == EventType::Play {
            let (Some(t), Some(ei)) = (ev.tile, ev.end_idx) else {
                prev_was_opp_draw = false;
                continue;
            };
            let Some(end) = end_from_idx(ei) else {
                prev_was_opp_draw = false;
                continue;
            };

            // update signals only for opponent plays, and only if not forced immediately after opponent draw
            if ev.player == Some(Player::Opponent) {
                played_mask |= 1u32 << (t.id() as u32);

                if !prev_was_opp_draw && !b.is_empty() {
                    let ends = b.legal_ends_for_tile(t);
                    if ends.len() > 1 {
                        // chosen new value
                        let mut chosen_new: Option<u8> = None;
                        for &e in ends.iter() {
                            if e != end {
                                continue;
                            }
                            if let Some(es) = b.ends_raw()[e.idx()] {
                                let ov = es.open_value;
                                let nv = t.other_value(ov).unwrap_or(ov);
                                chosen_new = Some(nv);
                            }
                        }
                        if let Some(ch) = chosen_new {
                            if ch <= 6 {
                                prefer[ch as usize] = prefer[ch as usize].saturating_add(1);
                            }
                            // alternative ends => avoid
                            for &e in ends.iter() {
                                if e == end {
                                    continue;
                                }
                                if let Some(es) = b.ends_raw()[e.idx()] {
                                    let ov = es.open_value;
                                    let nv = t.other_value(ov).unwrap_or(ov);
                                    if nv != ch && nv <= 6 {
                                        avoid[nv as usize] = avoid[nv as usize].saturating_add(1);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // replay board move (for both players)
            let _ = b.play(t, end);
            prev_was_opp_draw = false;
            continue;
        }

        // draw/pass do not change board, but draw affects forced-play skip
        prev_was_opp_draw = (ev.player == Some(Player::Opponent) && ev.typ == EventType::Draw);
    }

    OppEndChoiceSignals {
        played_mask28: played_mask,
        prefer_new_open_values: prefer,
        avoid_new_open_values: avoid,
    }
}

fn build_inf_feat_235(
    feat193: &[f32; features::FEAT_DIM],
    played_mask28: u32,
    prefer: &[u8; 7],
    avoid: &[u8; 7],
) -> [f32; infer_model::INF_FEAT_DIM] {
    let mut out = [0f32; infer_model::INF_FEAT_DIM];
    out[0..features::FEAT_DIM].copy_from_slice(feat193);
    let mut off = features::FEAT_DIM;
    for i in 0..28 {
        out[off + i] = if (played_mask28 & (1u32 << (i as u32))) != 0 { 1.0 } else { 0.0 };
    }
    off += 28;
    for v in 0..7 {
        out[off + v] = (prefer[v].min(8) as f32) / 8.0;
    }
    off += 7;
    for v in 0..7 {
        out[off + v] = (avoid[v].min(8) as f32) / 8.0;
    }
    out
}

// ----------------------------
// Model cache (mtime-based)
// ----------------------------
struct ModelCache {
    path: String,
    mtime: Option<SystemTime>,
    model: Arc<mlp::MlpModel>,
}

static MODEL_CACHE: OnceLock<Mutex<Option<ModelCache>>> = OnceLock::new();

fn file_mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

/// Load model with cache.
/// P4: For JSON model_path, ALWAYS load JSON to preserve optional heads (spike/quantiles).
fn load_model_cached(model_path: &str) -> Result<Arc<mlp::MlpModel>, String> {
    let lock = MODEL_CACHE.get_or_init(|| Mutex::new(None));
    let mut g = lock.lock().unwrap();

    let p = Path::new(model_path);
    let mt = file_mtime(p);

    if let Some(c) = g.as_ref() {
        if c.path == model_path && c.mtime == mt {
            return Ok(Arc::clone(&c.model));
        }
    }

    let lower = model_path.to_ascii_lowercase();
    let m = if lower.ends_with(".bin") {
        mlp::MlpModel::load_weights_bin(p)?
    } else if lower.ends_with(".json") {
        mlp::MlpModel::load_model_json(p)?
    } else {
        if p.exists() {
            mlp::MlpModel::load_model_json(p)?
        } else {
            return Err(format!("model_path not found: {}", model_path));
        }
    };

    let arc = Arc::new(m);
    *g = Some(ModelCache {
        path: model_path.to_string(),
        mtime: mt,
        model: Arc::clone(&arc),
    });
    Ok(arc)
}

// parity with factory.rs
fn budget_sims(det: u32, think_ms: u32) -> u32 {
    let worlds = det.clamp(6, 28);
    let sims_per_world = (think_ms / 60).clamp(6, 26);
    let total = worlds.saturating_mul(sims_per_world);
    total.min(6000).max(100)
}

// Opening forced_best helper (boundary-only; not hot)
fn best_opening_tile(hand: &[tile::Tile]) -> Option<tile::Tile> {
    if hand.is_empty() {
        return None;
    }
    let mut doubles: Vec<tile::Tile> = hand.iter().copied().filter(|t| t.is_double()).collect();
    if !doubles.is_empty() {
        doubles.sort_by_key(|t| t.pips().0);
        return doubles.pop();
    }
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

// ----------------------------
// Build Board safely from Python state_dict snapshot (NO replay)
// ----------------------------
fn parse_board_from_state_dict(state_dict: &Bound<'_, PyDict>) -> PyResult<board::Board> {
    let board_any = dict_get_required(state_dict, "board")?;
    let b: &Bound<'_, PyDict> = board_any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("state.board must be dict"))?;

    // center_tile
    let center_tile: Option<tile::Tile> = match b.get_item("center_tile")? {
        Some(x) if !x.is_none() => {
            let s = x
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("board.center_tile must be str"))?;
            Some(tile::Tile::parse(&s).map_err(PyValueError::new_err)?)
        }
        _ => None,
    };

    if center_tile.is_none() {
        return Ok(board::Board::new());
    }

    // spinner_value
    let spinner_value: Option<u8> = match b.get_item("spinner_value")? {
        Some(x) if !x.is_none() => {
            let v = x.extract::<i64>().unwrap_or(-1);
            if (0..=6).contains(&v) {
                Some(v as u8)
            } else {
                None
            }
        }
        _ => None,
    };

    let spinner_sides_open: bool = match b.get_item("spinner_sides_open")? {
        Some(x) if !x.is_none() => x.extract::<bool>().unwrap_or(false),
        _ => false,
    };

    // ends
    let mut ends: [Option<board::EndState>; 4] = [None, None, None, None];
    if let Some(ends_any) = b.get_item("ends")? {
        if !ends_any.is_none() {
            let ed: &Bound<'_, PyDict> = ends_any
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("board.ends must be dict"))?;
            for (k, v) in ed.iter() {
                let ks = k.extract::<String>().unwrap_or_default();
                let idx = match ks.as_str() {
                    "right" => 0,
                    "left" => 1,
                    "up" => 2,
                    "down" => 3,
                    _ => continue,
                };
                let lst: &Bound<'_, PyList> = v
                    .downcast::<PyList>()
                    .map_err(|_| PyValueError::new_err("board.ends values must be list"))?;
                if lst.len() >= 1 {
                    let ov = lst.get_item(0)?.extract::<i64>().unwrap_or(-1);
                    let is_dbl = if lst.len() >= 2 {
                        lst.get_item(1)?.extract::<bool>().unwrap_or(false)
                    } else {
                        false
                    };
                    if (0..=6).contains(&ov) {
                        ends[idx] = Some(board::EndState {
                            open_value: ov as u8,
                            is_double_end: is_dbl,
                        });
                    }
                }
            }
        }
    }

    // arms full tiles
    let mut arms: [Vec<tile::Tile>; 4] = std::array::from_fn(|_| Vec::new());
    if let Some(arms_any) = b.get_item("arms")? {
        if !arms_any.is_none() {
            let ad: &Bound<'_, PyDict> = arms_any
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("board.arms must be dict"))?;
            for (k, v) in ad.iter() {
                let ks = k.extract::<String>().unwrap_or_default();
                let idx = match ks.as_str() {
                    "right" => 0,
                    "left" => 1,
                    "up" => 2,
                    "down" => 3,
                    _ => continue,
                };
                let lst: &Bound<'_, PyList> = v
                    .downcast::<PyList>()
                    .map_err(|_| PyValueError::new_err("board.arms values must be list"))?;
                for it in lst.iter() {
                    let s = it.extract::<String>().map_err(|_| PyValueError::new_err("arm tile must be str"))?;
                    arms[idx].push(tile::Tile::parse(&s).map_err(PyValueError::new_err)?);
                }
            }
        }
    }

    // played_tiles chronological
    let mut played: Vec<tile::Tile> = Vec::new();
    if let Some(pt_any) = b.get_item("played_tiles")? {
        if !pt_any.is_none() {
            let lst: &Bound<'_, PyList> = pt_any
                .downcast::<PyList>()
                .map_err(|_| PyValueError::new_err("board.played_tiles must be list"))?;
            for it in lst.iter() {
                let s = it.extract::<String>().map_err(|_| PyValueError::new_err("played_tiles must be str"))?;
                played.push(tile::Tile::parse(&s).map_err(PyValueError::new_err)?);
            }
        }
    }

    Ok(board::Board::from_snapshot_parts(
        center_tile,
        spinner_value,
        spinner_sides_open,
        ends,
        arms,
        played,
    ))
}

// ----------------------------
// Guard helpers (uniform opponent-hand sampling)
// ----------------------------

fn forbid_values_from_certain_pass(events: &[ismcts::BeliefEvent]) -> [bool; 7] {
    let mut forbid = [false; 7];
    for ev in events.iter() {
        if !matches!(ev.typ, ismcts::BeliefEventType::Pass) {
            continue;
        }
        if !matches!(ev.certainty, ismcts::BeliefCertainty::Certain) {
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

/// Uniform sample of opponent hand from hidden tiles (no belief weights).
/// Respects certain-pass forbids if feasible (hard-ish constraint).
fn determinize_opp_hand_uniform_only(st: &ismcts::InfoState, rng: &mut StdRng) -> Vec<tile::Tile> {
    let visible_mask: u32 = hash::hand_mask(&st.my_hand) | st.board.played_mask();
    let mut hidden: Vec<tile::Tile> = Vec::new();
    hidden.reserve(28);
    for id in 0u8..28u8 {
        if (visible_mask & (1u32 << (id as u32))) == 0 {
            hidden.push(tile::Tile(id));
        }
    }
    if hidden.is_empty() {
        return vec![];
    }

    let k = (st.opponent_tile_count.max(0) as usize).min(hidden.len());
    if k == 0 {
        return vec![];
    }

    let forbid = forbid_values_from_certain_pass(&st.events);

    let mut allowed: Vec<tile::Tile> = Vec::with_capacity(hidden.len());
    for &t in hidden.iter() {
        let (a, b) = t.pips();
        if forbid[a as usize] || forbid[b as usize] {
            continue;
        }
        allowed.push(t);
    }

    if allowed.len() >= k {
        allowed.shuffle(rng);
        allowed.truncate(k);
        return allowed;
    }

    hidden.shuffle(rng);
    hidden.truncate(k);
    hidden
}

// ----------------------------
// suggest_move_ismcts(state_dict -> best move dict)
// ----------------------------
#[derive(Clone)]
struct MoveOut {
    t: tile::Tile,
    e: board::End,
    visits: u32,

    // legacy info (kept for compatibility / eval)
    guard_used: bool,
    guard_risk: f32,
    immediate_win: bool,
}

#[derive(Clone)]
struct RootCandOut {
    t: tile::Tile,
    e: board::End,
    visits: u32,
    immediate_points: i32,
    spike_p: f32, // -1.0 when not available
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn suggest_move_ismcts(
    py: Python<'_>,
    state_dict: &Bound<'_, PyAny>,
    det: u32,
    think_ms: u32,
    seed: u64,
    model_path: Option<String>,
    // knobs (optional)
    opp_mix_greedy: Option<f32>,
    leaf_value_weight: Option<f32>,
    me_mix_greedy: Option<f32>,
    gift_penalty_weight: Option<f32>,
    pessimism_alpha_max: Option<f32>,
    // guard knobs
    enable_guard: Option<bool>,
    guard_top_k: Option<u32>,
    guard_worlds: Option<u32>,
    guard_close_threshold: Option<i32>,
    // optional forensic dump
    return_root_stats: Option<bool>,
    root_stats_top_k: Option<u32>,
) -> PyResult<PyObject> {
    let stv = parse_state_view(state_dict)?;
    if stv.round_over || !stv.current_turn_me {
        return Err(PyValueError::new_err("not my turn or round over"));
    }

    // Precompute root features/mask for spike head (cheap, deterministic)
    let feat_root = features::features_193(&stv);
    let mask_root = features::legal_mask_112(&stv);

    // IM3: optional inference tile probabilities (computed in boundary, applied only in determinize).
    // Toggle is via DOMINO_INFER=0/1 (default 0).
    let mut opp_infer_tile_p: [f32; 28] = [0.0; 28];
    let mut infer_dbg_round_events_len: usize = 0;
    let mut infer_dbg_played_popcnt: u32 = 0;
    let mut infer_dbg_pref_sum: u32 = 0;
    let mut infer_dbg_avoid_sum: u32 = 0;
    let mut infer_dbg_model_loaded: bool = false;

    let sd: &Bound<'_, PyDict> = state_dict
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("state must be dict"))?;

    let board_full = parse_board_from_state_dict(sd)?;

    // forced tile sanity: if forced tile not in hand, ignore
    let mut forced = match stv.forced_play_tile {
        Some(ft) if stv.my_hand.iter().any(|x| *x == ft) => Some(ft),
        _ => None,
    };

    // Opening forced_best emulation (only when board empty and no forced tile)
    if forced.is_none()
        && stv.opening_mode_forced_best
        && stv.started_from_beginning
        && board_full.is_empty()
    {
        forced = best_opening_tile(&stv.my_hand);
    }

    // Build belief events (opponent draw/pass only) + Stage1/2 inference signals from opponent plays
    let mut bev: Vec<ismcts::BeliefEvent> = Vec::new();
    let mut max_ply = 0i32;

    let played_mask_round: u32 = board_full.played_mask();
    let opp_sig: OppInferenceSignals = extract_opp_inference_signals(&stv.events, played_mask_round);

    // IM3: end-choice signals + inference model prediction (feature parity with INF1 training)
    if infer_model::infer_enabled() {
        let round_evs = current_round_events(&stv.events);
        let ecs = extract_opp_endchoice_signals(round_evs);

        // debug stats (no knobs; only emitted when inference enabled)
        infer_dbg_round_events_len = round_evs.len();
        infer_dbg_played_popcnt = ecs.played_mask28.count_ones();
        infer_dbg_pref_sum = ecs.prefer_new_open_values.iter().map(|&x| x as u32).sum();
        infer_dbg_avoid_sum = ecs.avoid_new_open_values.iter().map(|&x| x as u32).sum();

        // IMPORTANT: base features must be computed using current-round events only (parity with training)
        let stv_round = round_state_view_for_infer(&stv);
        let feat_infer_base = features::features_193(&stv_round);

        // build inf_feat and predict tile probabilities
        let inf_feat = build_inf_feat_235(&feat_infer_base, ecs.played_mask28, &ecs.prefer_new_open_values, &ecs.avoid_new_open_values);
        if let Some(m) = infer_model::load_infer_model_cached("inference_model.json") {
            infer_dbg_model_loaded = true;
            let mut p = m.predict_tile_probs(&inf_feat);
            // mask visible tiles (known info) to avoid meaningless weights
            let mut visible_mask: u32 = 0;
            for &t in stv.my_hand.iter() {
                visible_mask |= 1u32 << (t.id() as u32);
            }
            for &t in stv.board.played_tiles.iter() {
                visible_mask |= 1u32 << (t.id() as u32);
            }
            for i in 0..28 {
                if (visible_mask & (1u32 << (i as u32))) != 0 {
                    p[i] = 0.0;
                }
            }
            opp_infer_tile_p = p;
        }
    }

    for ev in stv.events.iter() {
        if ev.player != Some(Player::Opponent) {
            continue;
        }

        let typ = match ev.typ {
            EventType::Draw => ismcts::BeliefEventType::Draw,
            EventType::Pass => ismcts::BeliefEventType::Pass,
            _ => continue,
        };
        let certainty = match ev.certainty {
            Certainty::Certain => ismcts::BeliefCertainty::Certain,
            Certainty::Probable => ismcts::BeliefCertainty::Probable,
            Certainty::Possible => ismcts::BeliefCertainty::Possible,
        };

        let mut arr = [255u8; 4];
        let mut len: u8 = 0;
        for (i, v) in ev.open_ends.iter().copied().take(4).enumerate() {
            arr[i] = v;
            len += 1;
        }
        max_ply = max_ply.max(ev.ply);
        bev.push(ismcts::BeliefEvent {
            typ,
            ply: ev.ply,
            open_ends: arr,
            len,
            certainty,
        });
    }
    let ply_now = (max_ply + 1).max(0);

    let info = ismcts::InfoState {
        board: board_full,
        my_hand: stv.my_hand.clone(),
        opponent_tile_count: stv.opponent_tile_count,
        boneyard_count: stv.boneyard_count,
        forced_play_tile: forced,
        my_score: stv.my_score,
        opp_score: stv.opp_score,
        match_target: stv.match_target,
        current_turn_me: true,
        ply: ply_now,
        round_index: stv.round_index.max(1),
        events: bev,
        opp_played_tiles: opp_sig.played_tiles,
        opp_avoided_open_values: opp_sig.avoided_open_values,
        opp_infer_tile_p,
    };

    // P0: Root solver hook when boneyard=0 and opponent hand is fully inferable
    if info.boneyard_count == 0 {
        if let Some((t, e)) = ismcts::root_solve_best_move(&info) {
            let mut b2 = info.board.clone();
            let pts = b2.play(t, e).unwrap_or(0);
            let immediate_win = info.my_score + pts >= info.match_target;

            let d = PyDict::new(py);
            d.set_item("tile", t.to_str())?;
            d.set_item("end", e.as_str())?;
            d.set_item("visits", 0u32)?;
            d.set_item("guard_used", false)?;
            d.set_item("guard_risk", 0.0f32)?;
            d.set_item("immediate_win", immediate_win)?;
            // telemetry-friendly (optional fields; evaluate.py tolerates missing)
            d.set_item("model_has_spike", false)?;
            d.set_item("spike_used", false)?;
            d.set_item("spike_p_chosen", 0.0f32)?;
            d.set_item("spike_p_best_visit", 0.0f32)?;
            d.set_item("spike_safe_exists", false)?;
            d.set_item("chosen_by", "solver")?;
            // IM4.1: inference debug stats (no knobs; only meaningful when DOMINO_INFER=1)
            d.set_item("infer_enabled", infer_model::infer_enabled())?;
            if infer_model::infer_enabled() {
                d.set_item("infer_round_events_len", infer_dbg_round_events_len)?;
                d.set_item("infer_played_popcnt", infer_dbg_played_popcnt)?;
                d.set_item("infer_pref_sum", infer_dbg_pref_sum)?;
                d.set_item("infer_avoid_sum", infer_dbg_avoid_sum)?;
                d.set_item("infer_model_loaded", infer_dbg_model_loaded)?;
            }
            return Ok(d.into());
        }
    }

    // load model (cached)
    let mp = model_path.unwrap_or_else(|| "model.json".to_string());
    let model = load_model_cached(&mp).map_err(PyValueError::new_err)?;

    // params
    let params = ismcts::IsmctsParams {
        sims: budget_sims(det, think_ms),
        c_puct: 1.6,
        temperature: 1.0,
        max_plies: 160,
        opp_mix_greedy: opp_mix_greedy.unwrap_or(1.0).clamp(0.0, 1.0),
        leaf_value_weight: leaf_value_weight.unwrap_or(0.0).clamp(0.0, 1.0),
        me_mix_greedy: me_mix_greedy.unwrap_or(1.0).clamp(0.0, 1.0),
        gift_penalty_weight: gift_penalty_weight.unwrap_or(0.0).clamp(0.0, 1.0),
        pessimism_alpha_max: pessimism_alpha_max.unwrap_or(0.0).clamp(0.0, 1.0),
    };

    // guard controls (legacy fallback)
    let enable_guard = enable_guard.unwrap_or(true);
    let top_k = guard_top_k.unwrap_or(5).max(1) as usize;
    let worlds = guard_worlds.unwrap_or(12).max(1) as usize;
    let close_thr = guard_close_threshold.unwrap_or(20);
    let safe_eps: f32 = 1.0f32 / (worlds as f32);

    let want_root_stats: bool = return_root_stats.unwrap_or(false);
    let root_top_k: usize = root_stats_top_k.unwrap_or(12).max(1) as usize;

    // Heavy computation without GIL
    let out: (MoveOut, Option<(bool, bool, f32, f32, bool, String)>, Option<Vec<RootCandOut>>) = py
        .allow_threads(|| -> Result<(MoveOut, Option<(bool, bool, f32, f32, bool, String)>, Option<Vec<RootCandOut>>), String> {
            let visits = ismcts::ismcts_root_visits(&info, Some(model.as_ref()), params, seed);

            // legal moves list (tile,end,visits)
            let tiles: Vec<tile::Tile> = if let Some(ft) = info.forced_play_tile {
                vec![ft]
            } else {
                info.my_hand.clone()
            };

            let mut legal: Vec<(tile::Tile, board::End, u32)> = Vec::new();
            for t in tiles.iter().copied() {
                for e in info.board.legal_ends_for_tile(t) {
                    let aidx = features::encode_action(t, e.idx());
                    legal.push((t, e, visits[aidx]));
                }
            }
            if legal.is_empty() {
                return Err("no legal moves".into());
            }

            // sort by visits desc
            legal.sort_by(|a, b| b.2.cmp(&a.2));
            let best_by_vis = legal[0];

            // Optional: build root candidates list (top-K by visits).
            // Forensics only (NO influence on decision).
            let mut root_cands: Option<Vec<RootCandOut>> = None;
            if want_root_stats {
                let take_n = root_top_k.min(legal.len());
                let mut tmp: Vec<RootCandOut> = Vec::with_capacity(take_n);
                for (t, e, v) in legal.iter().copied().take(take_n) {
                    let mut b2 = info.board.clone();
                    let pts = b2.play(t, e).unwrap_or(0);
                    tmp.push(RootCandOut {
                        t,
                        e,
                        visits: v,
                        immediate_points: pts,
                        spike_p: -1.0,
                    });
                }
                root_cands = Some(tmp);
            }

            let mut best = best_by_vis;
            let mut guard_used = false;
            let mut chosen_risk: f32 = 0.0;
            let mut immediate_win = false;

            let my_to = (info.match_target - info.my_score).max(0);
            let op_to = (info.match_target - info.opp_score).max(0);
            let close = my_to.min(op_to);

            // -------------------------
            // P4-B1: Spike-based root decision (no sampling guard)
            // -------------------------
            if model.has_spike {
                // Immediate win shortcut (exact)
                for (t, e, v) in legal.iter().copied() {
                    let mut b2 = info.board.clone();
                    let pts = b2.play(t, e).unwrap_or(0);
                    if info.my_score + pts >= info.match_target {
                        best = (t, e, v);
                        chosen_risk = 0.0;
                        immediate_win = true;
                        let mv = MoveOut {
                            t: best.0,
                            e: best.1,
                            visits: best.2,
                            guard_used: false,
                            guard_risk: chosen_risk,
                            immediate_win,
                        };
                        let telem = Some((true,  // model_has_spike
                                          true,  // spike_used
                                          0.0,   // spike_p_chosen
                                          0.0,   // spike_p_best_visit
                                          true,  // spike_safe_exists
                                          "immediate_win".to_string(),
                        ));
                        return Ok((mv, telem, root_cands));
                    }
                }

                let (_pol, _v, spike) = model.predict_with_spike(&feat_root, &mask_root);

                let spike_of = |t: tile::Tile, e: board::End| -> f32 {
                    let a = features::encode_action(t, e.idx());
                    spike[a].clamp(0.0, 1.0)
                };

                // If forensic requested, fill spike_p for candidates we already captured.
                if want_root_stats {
                    if let Some(ref mut cs) = root_cands {
                        for c in cs.iter_mut() {
                            c.spike_p = spike_of(c.t, c.e);
                        }
                    }
                }

                let unsafe_thr: f32 = 0.5;
                let p_best_visit = spike_of(best_by_vis.0, best_by_vis.1);

                // safe exists?
                let mut safe_exists = false;
                for (t, e, _v) in legal.iter().copied() {
                    if spike_of(t, e) < unsafe_thr {
                        safe_exists = true;
                        break;
                    }
                }

                if p_best_visit < unsafe_thr {
                    best = best_by_vis;
                    chosen_risk = p_best_visit;
                    let mv = MoveOut {
                        t: best.0,
                        e: best.1,
                        visits: best.2,
                        guard_used: false,
                        guard_risk: chosen_risk,
                        immediate_win: false,
                    };
                    let telem = Some((true,
                                      true,
                                      chosen_risk,
                                      p_best_visit,
                                      safe_exists,
                                      "visits_ok".to_string(),
                    ));
                    return Ok((mv, telem, root_cands));
                }

                // Find best safe alternative (highest visits among p<thr)
                let mut best_safe: Option<(tile::Tile, board::End, u32, f32)> = None;
                for (t, e, v) in legal.iter().copied() {
                    let p = spike_of(t, e);
                    if p < unsafe_thr {
                        match best_safe {
                            None => best_safe = Some((t, e, v, p)),
                            Some((_bt, _be, bv, bp)) => {
                                if v > bv || (v == bv && p < bp) {
                                    best_safe = Some((t, e, v, p));
                                }
                            }
                        }
                    }
                }

                if let Some((t, e, v, p)) = best_safe {
                    best = (t, e, v);
                    chosen_risk = p;
                    let mv = MoveOut {
                        t: best.0,
                        e: best.1,
                        visits: best.2,
                        guard_used: false,
                        guard_risk: chosen_risk,
                        immediate_win: false,
                    };
                    let telem = Some((true,
                                      true,
                                      chosen_risk,
                                      p_best_visit,
                                      safe_exists,
                                      "spike_safe".to_string(),
                    ));
                    return Ok((mv, telem, root_cands));
                }

                // All unsafe: pick minimal spike (tie by visits)
                let mut best_min = legal[0];
                let mut best_p = spike_of(best_min.0, best_min.1);
                for (t, e, v) in legal.iter().copied() {
                    let p = spike_of(t, e);
                    if p < best_p - 1e-6 || ((p - best_p).abs() <= 1e-6 && v > best_min.2) {
                        best_min = (t, e, v);
                        best_p = p;
                    }
                }
                best = best_min;
                chosen_risk = best_p;

                let mv = MoveOut {
                    t: best.0,
                    e: best.1,
                    visits: best.2,
                    guard_used: false,
                    guard_risk: chosen_risk,
                    immediate_win: false,
                };
                let telem = Some((true,
                                  true,
                                  chosen_risk,
                                  p_best_visit,
                                  safe_exists,
                                  "spike_min".to_string(),
                ));
                return Ok((mv, telem, root_cands));
            }

            // -------------------------
            // Legacy sampling guard (fallback only; used when no spike head)
            // -------------------------
            if enable_guard && close <= close_thr {
                guard_used = true;

                let cands: Vec<(tile::Tile, board::End, u32)> = if legal.len() <= 28 {
                    legal.clone()
                } else {
                    legal.iter().take(top_k).copied().collect::<Vec<_>>()
                };

                // immediate win shortcut
                for (t, e, v) in cands.iter().copied() {
                    let mut b2 = info.board.clone();
                    let pts = b2.play(t, e).unwrap_or(0);
                    if info.my_score + pts >= info.match_target {
                        best = (t, e, v);
                        chosen_risk = 0.0;
                        immediate_win = true;
                        let mv = MoveOut {
                            t: best.0,
                            e: best.1,
                            visits: best.2,
                            guard_used,
                            guard_risk: chosen_risk,
                            immediate_win,
                        };
                        let telem = Some((model.has_spike,
                                          false,
                                          0.0,
                                          0.0,
                                          false,
                                          "guard_immediate_win".to_string(),
                        ));
                        return Ok((mv, telem, root_cands));
                    }
                }

                // If boneyard is empty, opponent hand can be exact if counts match.
                let opp_hand_exact: Option<Vec<tile::Tile>> = if info.boneyard_count == 0 && info.opponent_tile_count >= 0 {
                    let visible_mask: u32 = hash::hand_mask(&info.my_hand) | info.board.played_mask();
                    let mut opp: Vec<tile::Tile> = Vec::new();
                    for id in 0u8..28u8 {
                        if (visible_mask & (1u32 << (id as u32))) == 0 {
                            opp.push(tile::Tile(id));
                        }
                    }
                    if (opp.len() as i32) == info.opponent_tile_count { Some(opp) } else { None }
                } else {
                    None
                };

                let mut best_risk = 1.0f32;
                let mut best_cand = cands[0];

                for (t, e, v) in cands.iter().copied() {
                    let risk: f32 = if let Some(ref opp_exact) = opp_hand_exact {
                        let mut b2 = info.board.clone();
                        let _ = b2.play(t, e).unwrap_or(0);
                        if ismcts::opp_has_win_now_exists(&b2, opp_exact, info.opp_score, info.match_target) { 1.0 } else { 0.0 }
                    } else {
                        let mut hit = 0usize;
                        for i in 0..worlds {
                            let s2 = seed
                                ^ ((features::encode_action(t, e.idx()) as u64)
                                    .wrapping_mul(0x9E3779B97F4A7C15))
                                ^ ((i as u64).wrapping_mul(0xD1B54A32D192ED03));
                            let mut rng = StdRng::seed_from_u64(s2);

                            let opp_hand: Vec<tile::Tile> = if (i & 1) == 0 {
                                determinize_opp_hand_uniform_only(&info, &mut rng)
                            } else {
                                let (oh, _bone) = ismcts::determinize(&info, &mut rng);
                                oh
                            };

                            let mut b2 = info.board.clone();
                            let _ = b2.play(t, e).unwrap_or(0);

                            if ismcts::opp_has_win_now_exists(&b2, &opp_hand, info.opp_score, info.match_target) {
                                hit += 1;
                            }
                        }
                        (hit as f32) / (worlds as f32)
                    };

                    if risk <= safe_eps {
                        best = (t, e, v);
                        chosen_risk = risk;
                        let mv = MoveOut {
                            t: best.0,
                            e: best.1,
                            visits: best.2,
                            guard_used,
                            guard_risk: chosen_risk,
                            immediate_win: false,
                        };
                        let telem = Some((model.has_spike,
                                          false,
                                          0.0,
                                          0.0,
                                          false,
                                          "guard_safe".to_string(),
                        ));
                        return Ok((mv, telem, root_cands));
                    }

                    if risk < best_risk - 1e-6 || ((risk - best_risk).abs() <= 1e-6 && v > best_cand.2) {
                        best_risk = risk;
                        best_cand = (t, e, v);
                    }
                }

                best = best_cand;
                chosen_risk = best_risk;
            }

            let mv = MoveOut {
                t: best.0,
                e: best.1,
                visits: best.2,
                guard_used,
                guard_risk: chosen_risk,
                immediate_win,
            };
            let telem = Some((model.has_spike,
                              false,
                              0.0,
                              0.0,
                              false,
                              if guard_used { "guard_fallback" } else { "visits" }.to_string(),
            ));
            Ok((mv, telem, root_cands))
        })
        .map_err(PyValueError::new_err)?;

    // return dict under GIL
    let d = PyDict::new(py);
    d.set_item("tile", out.0.t.to_str())?;
    d.set_item("end", out.0.e.as_str())?;
    d.set_item("visits", out.0.visits)?;
    d.set_item("guard_used", out.0.guard_used)?;
    d.set_item("guard_risk", out.0.guard_risk)?;
    d.set_item("immediate_win", out.0.immediate_win)?;

    // Optional telemetry fields
    if let Some((model_has_spike, spike_used, p_chosen, p_best_visit, safe_exists, chosen_by)) = out.1 {
        d.set_item("model_has_spike", model_has_spike)?;
        d.set_item("spike_used", spike_used)?;
        d.set_item("spike_p_chosen", p_chosen)?;
        d.set_item("spike_p_best_visit", p_best_visit)?;
        d.set_item("spike_safe_exists", safe_exists)?;
        d.set_item("chosen_by", chosen_by)?;
    }

    // IM4.1: inference debug stats (no knobs; only meaningful when DOMINO_INFER=1)
    d.set_item("infer_enabled", infer_model::infer_enabled())?;
    if infer_model::infer_enabled() {
        d.set_item("infer_round_events_len", infer_dbg_round_events_len)?;
        d.set_item("infer_played_popcnt", infer_dbg_played_popcnt)?;
        d.set_item("infer_pref_sum", infer_dbg_pref_sum)?;
        d.set_item("infer_avoid_sum", infer_dbg_avoid_sum)?;
        d.set_item("infer_model_loaded", infer_dbg_model_loaded)?;
    }

    if let Some(cs) = out.2 {
        let lst = PyList::empty(py);
        for c in cs {
            let cd = PyDict::new(py);
            cd.set_item("tile", c.t.to_str())?;
            cd.set_item("end", c.e.as_str())?;
            cd.set_item("visits", c.visits)?;
            cd.set_item("immediate_points", c.immediate_points)?;
            if c.spike_p >= 0.0 {
                cd.set_item("spike_p", c.spike_p)?;
            } else {
                cd.set_item("spike_p", py.None())?;
            }
            lst.append(cd)?;
        }
        d.set_item("root_candidates", lst)?;
    }

    Ok(d.into())
}

// ----------------------------
// Existing boundary parser: state_dict -> StateView (features-compatible)
// ----------------------------
fn parse_state_view(state_any: &Bound<'_, PyAny>) -> PyResult<StateView> {
    let state: &Bound<'_, PyDict> = state_any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("state must be a dict"))?;

    let meta_any = dict_get_required(state, "meta")?;
    let meta: Bound<'_, PyDict> = meta_any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("state.meta must be dict"))?
        .clone();

    let board_any = dict_get_required(state, "board")?;
    let board: Bound<'_, PyDict> = board_any
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("state.board must be dict"))?
        .clone();

    // my_hand
    let my_hand_any = dict_get_required(state, "my_hand")?;
    let my_hand_list: Bound<'_, PyList> = my_hand_any
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("state.my_hand must be list"))?
        .clone();

    let mut my_hand: Vec<tile::Tile> = Vec::with_capacity(my_hand_list.len());
    for it in my_hand_list.iter() {
        let s = it
            .extract::<String>()
            .map_err(|_| PyValueError::new_err("my_hand must be list[str]"))?;
        my_hand.push(tile::Tile::parse(&s).map_err(PyValueError::new_err)?);
    }

    // forced_play_tile
    let forced_play_tile = match meta.get_item("forced_play_tile")? {
        Some(x) if !x.is_none() => {
            let s = x
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("forced_play_tile must be str or None"))?;
            Some(tile::Tile::parse(&s).map_err(PyValueError::new_err)?)
        }
        _ => None,
    };

    let current_turn = dict_get_required(&meta, "current_turn")?
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("meta.current_turn must be str"))?;
    let current_turn_me = current_turn == "me";

    let started_from_beginning = dict_get_required(&meta, "started_from_beginning")?
        .extract::<bool>()
        .map_err(|_| PyValueError::new_err("meta.started_from_beginning must be bool"))?;

    let opening_mode = dict_get_required(&meta, "opening_mode")?
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("meta.opening_mode must be str"))?;
    let opening_mode_forced_best = opening_mode == "forced_best";

    let round_over = dict_get_required(&meta, "round_over")?
        .extract::<bool>()
        .map_err(|_| PyValueError::new_err("meta.round_over must be bool"))?;

    let opponent_tile_count = dict_get_required(&meta, "opponent_tile_count")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("meta.opponent_tile_count must be int"))?;
    let boneyard_count = dict_get_required(&meta, "boneyard_count")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("meta.boneyard_count must be int"))?;
    let my_score = dict_get_required(&meta, "my_score")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("meta.my_score must be int"))?;
    let opp_score = dict_get_required(&meta, "opponent_score")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("meta.opponent_score must be int"))?;
    let match_target = dict_get_required(&meta, "match_target")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("meta.match_target must be int"))?;
    let round_index = dict_get_required(&meta, "round_index")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("meta.round_index must be int"))?;

    // board.center_tile
    let center_tile = match board.get_item("center_tile")? {
        Some(x) if !x.is_none() => {
            let s = x
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("board.center_tile must be str or None"))?;
            Some(tile::Tile::parse(&s).map_err(PyValueError::new_err)?)
        }
        _ => None,
    };

    let spinner_sides_open = dict_get_required(&board, "spinner_sides_open")?
        .extract::<bool>()
        .map_err(|_| PyValueError::new_err("board.spinner_sides_open must be bool"))?;

    let ends_sum = dict_get_required(&board, "ends_sum")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("board.ends_sum must be int"))?;

    let score_now = dict_get_required(&board, "current_score")?
        .extract::<i32>()
        .map_err(|_| PyValueError::new_err("board.current_score must be int"))?;

    // ends dict
    let mut ends_arr: [Option<(u8, bool)>; 4] = [None, None, None, None];
    if let Some(ends_any) = board.get_item("ends")? {
        if !ends_any.is_none() {
            let ends: &Bound<'_, PyDict> = ends_any
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("board.ends must be dict"))?;

            for (k, v) in ends.iter() {
                let ks = k.extract::<String>().map_err(|_| PyValueError::new_err("board.ends keys must be str"))?;
                let end_idx = match ks.as_str() {
                    "right" => 0,
                    "left" => 1,
                    "up" => 2,
                    "down" => 3,
                    _ => continue,
                };
                let vv: &Bound<'_, PyList> = v
                    .downcast::<PyList>()
                    .map_err(|_| PyValueError::new_err("board.ends values must be list"))?;

                if vv.len() >= 1 {
                    let open_val = vv.get_item(0)?.extract::<i64>().map_err(|_| PyValueError::new_err("end open_value must be int"))?;
                    let is_dbl = if vv.len() >= 2 { vv.get_item(1)?.extract::<bool>().unwrap_or(false) } else { false };
                    if (0..=6).contains(&open_val) {
                        ends_arr[end_idx] = Some((open_val as u8, is_dbl));
                    }
                }
            }
        }
    }

    // arms dict (lengths only)
    let mut arm_lens = [0usize; 4];
    if let Some(arms_any) = board.get_item("arms")? {
        if !arms_any.is_none() {
            let arms: &Bound<'_, PyDict> = arms_any
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("board.arms must be dict"))?;

            for (k, v) in arms.iter() {
                let ks = k.extract::<String>().map_err(|_| PyValueError::new_err("board.arms keys must be str"))?;
                let end_idx = match ks.as_str() {
                    "right" => 0,
                    "left" => 1,
                    "up" => 2,
                    "down" => 3,
                    _ => continue,
                };
                let lst: &Bound<'_, PyList> = v
                    .downcast::<PyList>()
                    .map_err(|_| PyValueError::new_err("board.arms values must be list"))?;
                arm_lens[end_idx] = lst.len();
            }
        }
    }

    // played_tiles
    let mut played_tiles: Vec<tile::Tile> = Vec::new();
    if let Some(pt_any) = board.get_item("played_tiles")? {
        if !pt_any.is_none() {
            let pt: &Bound<'_, PyList> = pt_any
                .downcast::<PyList>()
                .map_err(|_| PyValueError::new_err("board.played_tiles must be list"))?;
            played_tiles.reserve(pt.len());
            for it in pt.iter() {
                let s = it.extract::<String>().map_err(|_| PyValueError::new_err("played_tiles must be list[str]"))?;
                played_tiles.push(tile::Tile::parse(&s).map_err(PyValueError::new_err)?);
            }
        }
    }

    let board_view = BoardView {
        center_tile,
        spinner_sides_open,
        ends: ends_arr,
        arm_lens,
        ends_sum,
        score_now,
        played_tiles,
    };

    // events list
    let mut events: Vec<EventView> = Vec::new();
    if let Some(ev_any) = state.get_item("events")? {
        if !ev_any.is_none() {
            let ev_list: &Bound<'_, PyList> = ev_any
                .downcast::<PyList>()
                .map_err(|_| PyValueError::new_err("state.events must be list"))?;
            events.reserve(ev_list.len());

            for it in ev_list.iter() {
                let d: &Bound<'_, PyDict> = it
                    .downcast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("event must be dict"))?;

                let typ_s = dict_get_required(d, "type")?
                    .extract::<String>()
                    .map_err(|_| PyValueError::new_err("event.type must be str"))?;
                let ply = dict_get_required(d, "ply")?
                    .extract::<i32>()
                    .map_err(|_| PyValueError::new_err("event.ply must be int"))?;

                let player = match d.get_item("player")? {
                    Some(x) if !x.is_none() => {
                        let ps = x.extract::<String>().unwrap_or_default();
                        if ps == "me" {
                            Some(Player::Me)
                        } else if ps == "opponent" {
                            Some(Player::Opponent)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                let certainty = match d.get_item("certainty")? {
                    Some(x) if !x.is_none() => {
                        let cs = x.extract::<String>().unwrap_or_else(|_| "probable".to_string());
                        match cs.as_str() {
                            "certain" => Certainty::Certain,
                            "possible" => Certainty::Possible,
                            _ => Certainty::Probable,
                        }
                    }
                    _ => Certainty::Probable,
                };

                let open_ends: Vec<u8> = match d.get_item("open_ends")? {
                    Some(x) if !x.is_none() => {
                        let lst: &Bound<'_, PyList> = x
                            .downcast::<PyList>()
                            .map_err(|_| PyValueError::new_err("event.open_ends must be list"))?;
                        let mut out = Vec::with_capacity(lst.len());
                        for vv in lst.iter() {
                            let iv = vv.extract::<i64>().unwrap_or(-1);
                            if (0..=6).contains(&iv) {
                                out.push(iv as u8);
                            }
                        }
                        out
                    }
                    _ => vec![],
                };

                let et = match typ_s.as_str() {
                    "play" => EventType::Play,
                    "draw" => EventType::Draw,
                    "pass" => EventType::Pass,
                    "round_start" => EventType::RoundStart,
                    "match_start" => EventType::MatchStart,
                    "round_end" => EventType::RoundEnd,
                    _ => EventType::Other,
                };

                let tile_opt: Option<tile::Tile> = if et == EventType::Play {
                    match d.get_item("tile")? {
                        Some(x) if !x.is_none() => {
                            let ts = x.extract::<String>().unwrap_or_default();
                            tile::Tile::parse(&ts).ok()
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                let end_idx: Option<u8> = if et == EventType::Play {
                    match d.get_item("end")? {
                        Some(x) if !x.is_none() => {
                            let es = x.extract::<String>().unwrap_or_default();
                            let ei = match es.as_str() {
                                "right" => Some(0u8),
                                "left" => Some(1u8),
                                "up" => Some(2u8),
                                "down" => Some(3u8),
                                _ => None,
                            };
                            ei
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                let score_gained: i32 = match d.get_item("score_gained")? {
                    Some(x) if !x.is_none() => x.extract::<i32>().unwrap_or(0),
                    _ => 0,
                };

                events.push(EventView {
                    typ: et,
                    ply,
                    player,
                    open_ends,
                    certainty,
                    tile: tile_opt,
                    end_idx,
                    score_gained,
                });
            }
        }
    }

    Ok(StateView {
        my_hand,
        forced_play_tile,
        current_turn_me,
        started_from_beginning,
        opening_mode_forced_best,
        round_over,
        opponent_tile_count,
        boneyard_count,
        my_score,
        opp_score,
        match_target,
        round_index,
        board: board_view,
        events,
    })
}

// ---- serde_like Value -> PyObject ----
fn sval_to_pydict(py: Python<'_>, map: &std::collections::BTreeMap<String, board::serde_like::Value>) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    for (k, v) in map {
        d.set_item(k, sval_to_py(py, v)?)?;
    }
    Ok(d.into())
}
fn sval_to_py(py: Python<'_>, v: &board::serde_like::Value) -> PyResult<PyObject> {
    Ok(match v {
        board::serde_like::Value::Null => py.None(),
        board::serde_like::Value::Bool(b) => b.into_py(py),
        board::serde_like::Value::Int(i) => i.into_py(py),
        board::serde_like::Value::Str(s) => s.into_py(py),
        board::serde_like::Value::List(xs) => {
            let lst = PyList::empty(py);
            for x in xs {
                lst.append(sval_to_py(py, x)?)?;
            }
            lst.into()
        }
        board::serde_like::Value::Map(m) => {
            let d = PyDict::new(py);
            for (k, vv) in m {
                d.set_item(k, sval_to_py(py, vv)?)?;
            }
            d.into()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opp_avoided_uses_prev_event_open_ends_not_play_event_open_ends() {
        let t_55 = tile::Tile::parse("5-5").unwrap();
        let played_mask_round: u32 = 1u32 << (t_55.id() as u32);

        let events: Vec<EventView> = vec![
            EventView {
                typ: EventType::Play,
                ply: 1,
                player: Some(Player::Me),
                open_ends: vec![1, 5],
                certainty: Certainty::Certain,
                tile: None,
                end_idx: None,
                score_gained: 0,
            },
            EventView {
                typ: EventType::Play,
                ply: 2,
                player: Some(Player::Opponent),
                open_ends: vec![5, 6],
                certainty: Certainty::Certain,
                tile: Some(t_55),
                end_idx: None,
                score_gained: 0,
            },
        ];

        let sig = extract_opp_inference_signals(&events, played_mask_round);
        assert_eq!(sig.played_tiles.len(), 1);
        assert_eq!(sig.played_tiles[0], t_55);
        assert_eq!(sig.avoided_open_values[1], 1);
        assert_eq!(sig.avoided_open_values[5], 0);
    }

    #[test]
    fn opp_avoided_skips_forced_play_immediately_after_opp_draw() {
        let t_50 = tile::Tile::parse("5-0").unwrap();
        let played_mask_round: u32 = 1u32 << (t_50.id() as u32);

        let events: Vec<EventView> = vec![
            EventView {
                typ: EventType::Play,
                ply: 1,
                player: Some(Player::Me),
                open_ends: vec![1, 5],
                certainty: Certainty::Certain,
                tile: None,
                end_idx: None,
                score_gained: 0,
            },
            EventView {
                typ: EventType::Draw,
                ply: 2,
                player: Some(Player::Opponent),
                open_ends: vec![1, 5],
                certainty: Certainty::Certain,
                tile: None,
                end_idx: None,
                score_gained: 0,
            },
            EventView {
                typ: EventType::Play,
                ply: 3,
                player: Some(Player::Opponent),
                open_ends: vec![0, 1],
                certainty: Certainty::Certain,
                tile: Some(t_50),
                end_idx: None,
                score_gained: 0,
            },
        ];

        let sig = extract_opp_inference_signals(&events, played_mask_round);
        assert_eq!(sig.played_tiles.len(), 1);
        assert_eq!(sig.played_tiles[0], t_50);
        assert_eq!(sig.avoided_open_values[1], 0);
        assert_eq!(sig.avoided_open_values[5], 0);
    }

    #[test]
    fn current_round_events_slices_from_last_round_start() {
        let t_55 = tile::Tile::parse("5-5").unwrap();
        let events: Vec<EventView> = vec![
            EventView {
                typ: EventType::RoundStart,
                ply: 1,
                player: None,
                open_ends: vec![],
                certainty: Certainty::Certain,
                tile: None,
                end_idx: None,
                score_gained: 0,
            },
            EventView {
                typ: EventType::Play,
                ply: 2,
                player: Some(Player::Opponent),
                open_ends: vec![5],
                certainty: Certainty::Certain,
                tile: Some(t_55),
                end_idx: Some(0),
                score_gained: 0,
            },
            EventView {
                typ: EventType::RoundStart,
                ply: 10,
                player: None,
                open_ends: vec![],
                certainty: Certainty::Certain,
                tile: None,
                end_idx: None,
                score_gained: 0,
            },
        ];
        let sl = current_round_events(&events);
        assert_eq!(sl.len(), 1);
        assert!(matches!(sl[0].typ, EventType::RoundStart));
        assert_eq!(sl[0].ply, 10);
    }
}

// EXPORTS START
#[pymodule]
fn domino_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(generate_and_save, m)?)?;
    m.add_function(wrap_pyfunction!(state_apply_script, m)?)?;
    m.add_function(wrap_pyfunction!(features_from_state_dict, m)?)?;
    m.add_function(wrap_pyfunction!(suggest_move_ismcts, m)?)?;
    Ok(())
}
// EXPORTS END