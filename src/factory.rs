// FILE: src/factory.rs | version: 2026-01-17.stage2_5_wired
// CHANGELOG:
// - Stage-2.5: Wire runtime inference signals into Rust self-play generation:
//     * Track opponent played tiles (per player) each round.
//     * Track "avoided open values" negative evidence per player each round.
//     * Pass both into ismcts::InfoState so determinize() uses the same inference during PI generation.
//     * Do NOT update avoided counts on forced-play immediately after a draw (not a choice).
//
// RC5 (as before):
// - Add mode="endgame_mine" (record-only) to mine endgame states from normal selfplay (NO synthetic boards).
// - Add endgame filter knobs: endgame_close_threshold, endgame_boneyard_max, endgame_hand_max.
// - Backward compatible: default mode is "selfplay" (record everything).
//
// RC4 (as before):
// - Multi-core match generation (std threads) + batching + streaming writer.
// - PROGRESS lines for panel: "PROGRESS matches_done=.. total_matches=.. samples=.."
// - Pass leaf_value_weight to ISMCTS (value-in-leaf enabled).
// - Opening rules exactly as specified:
//   * Round 1 and after LOCKED: forced_best opener (compare best tile in each hand).
//   * After OUT: winner starts next round FREE (any tile).
// - Semantics preserved: PI = ISMCTS visits distribution (masked+normalized), Z = final match outcome ±1.
//
// PATCH (2026-06-06): Model Sync Fix
// ---------------------------------
// Fix stale weights.bin vs model.json mtime selection.
//
// P4-A1 (2026-01-14):
// - Replace quantiles[K] with spike[action] targets (binary win-now-exists per action).
// - Write (feat, pi, spike[112], z, mask) into shards (see shards.rs).
// - For now, spike is computed from true opponent hand (no determinization) during self-play.
//
// IM1 (2026-01-17): Inference sidecar (INF1) dataset generation
// - Generate INF1 sidecar files alongside DSH2 shards
// - Add inference-specific features and labels to Sample struct
// - Add end-choice preference/avoidance signals

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::board::{Board, End};
use crate::features::{self, ACTION_SIZE, FEAT_DIM};
use crate::inf_shards::{InfShardWriter, INF_FEAT_DIM, INF_LABEL_SIZE};
use crate::ismcts::{self, BeliefCertainty, BeliefEvent, BeliefEventType, InfoState, IsmctsParams};
use crate::mlp::MlpModel;
use crate::shards::{Codec, ShardWriter, RECORD_SIZE};
use crate::tile::Tile;

// -----------------------------
// Config + Manifest
// -----------------------------
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactoryConfig {
    pub matches: u32,

    // record mode
    // - "selfplay" (default): record every me-turn as training sample.
    // - "endgame_mine": record only when state is inside endgame window (close/boneyard/hands small).
    pub mode: Option<String>,

    pub det: u32,
    pub think_ms: u32,
    pub temperature: f32,
    pub max_moves_per_round: u32,
    pub max_rounds: u32,
    pub match_target: i32,
    pub seed: Option<u64>,

    pub model_path: String,
    pub codec: String,
    pub zstd_level: Option<i32>,
    pub shard_max_samples: Option<u32>,

    // Optional quality/perf knobs
    pub threads: Option<u32>,
    pub opp_mix_greedy: Option<f32>,
    pub leaf_value_weight: Option<f32>,

    // Strategic knobs
    pub me_mix_greedy: Option<f32>,
    pub gift_penalty_weight: Option<f32>,
    pub pessimism_alpha_max: Option<f32>,

    // endgame_mine knobs; ignored in selfplay mode
    pub endgame_close_threshold: Option<i32>, // default 25
    pub endgame_boneyard_max: Option<i32>,    // default 2
    pub endgame_hand_max: Option<i32>,        // default 6
}

#[derive(Clone, Debug, Serialize)]
pub struct Manifest {
    pub run_id: String,
    pub ruleset_id: String,
    pub feat_dim: u32,
    pub action_size: u32,
    pub record_size: u32,

    pub matches: u32,
    pub samples: u64,

    pub codec: String,
    pub model_runtime: String,
    pub config: serde_json::Value,

    pub shards: Vec<crate::shards::ShardInfo>,

    // IM1: inference sidecar (INF1). Optional and does NOT affect DSH2 tooling.
    pub infer_feat_dim: Option<u32>,
    pub infer_label_size: Option<u32>,
    pub infer_record_size: Option<u32>,
    pub infer_samples: Option<u64>,
    pub infer_codec: Option<String>,
    pub infer_shards: Option<Vec<crate::shards::ShardInfo>>,
}

// -----------------------------
// Endgame mining filter (record-only)
// -----------------------------
#[derive(Copy, Clone, Debug)]
struct EndgameFilter {
    close_thr: i32,  // min distance-to-target between players
    bone_max: i32,   // boneyard count cap
    hand_max: usize, // both hands <= this
}

impl EndgameFilter {
    fn should_record(
        &self,
        match_target: i32,
        my_score: i32,
        opp_score: i32,
        my_hand_len: usize,
        opp_hand_len: usize,
        bone_cnt: i32,
    ) -> bool {
        let my_to = (match_target - my_score).max(0);
        let op_to = (match_target - opp_score).max(0);
        let close = op_to;

        close <= self.close_thr
            && bone_cnt <= self.bone_max
            && my_hand_len <= self.hand_max
            && opp_hand_len <= self.hand_max
    }
}

pub fn generate_and_save(config_path: &str, output_dir: &str) -> Result<String, String> {
    let cfg_txt = fs::read_to_string(config_path).map_err(|e| format!("read config: {e}"))?;
    let cfg: FactoryConfig =
        serde_json::from_str(&cfg_txt).map_err(|e| format!("parse config json: {e}"))?;

    let out_dir = PathBuf::from(output_dir);
    fs::create_dir_all(&out_dir).map_err(|e| format!("create out_dir: {e}"))?;

    let base_seed = cfg.seed.unwrap_or_else(default_seed);
    let run_id = make_run_id(Some(base_seed));

    let codec = Codec::from_str(cfg.codec.as_str());
    let zstd_level = cfg.zstd_level.unwrap_or(3);
    let shard_max_samples = cfg.shard_max_samples.unwrap_or(50_000).max(1);

    // Load model ONCE
    let (model, model_runtime) = load_model_once(&cfg.model_path)?;
    let model = Arc::new(model);

    // Convert det/think_ms to sims budget (base sims)
    let sims = budget_sims(cfg.det, cfg.think_ms);
    let opp_mix_greedy = cfg.opp_mix_greedy.unwrap_or(1.0).clamp(0.0, 1.0);
    let leaf_value_weight = cfg.leaf_value_weight.unwrap_or(0.0).clamp(0.0, 1.0);

    let me_mix_greedy = cfg.me_mix_greedy.unwrap_or(1.0).clamp(0.0, 1.0);
    let gift_penalty_weight = cfg.gift_penalty_weight.unwrap_or(0.0).clamp(0.0, 1.0);
    let pessimism_alpha_max = cfg.pessimism_alpha_max.unwrap_or(0.0).clamp(0.0, 1.0);

    let params = IsmctsParams {
        sims,
        c_puct: 1.6,
        temperature: cfg.temperature,
        max_plies: 160,
        opp_mix_greedy,
        leaf_value_weight,
        me_mix_greedy,
        gift_penalty_weight,
        pessimism_alpha_max,
    };

    // Mode: selfplay vs endgame_mine (record-only)
    let mode = cfg.mode.clone().unwrap_or_else(|| "selfplay".to_string());
    let end_filter: Option<EndgameFilter> = if mode.trim().eq_ignore_ascii_case("endgame_mine") {
        Some(EndgameFilter {
            close_thr: cfg.endgame_close_threshold.unwrap_or(25).max(5),
            bone_max: cfg.endgame_boneyard_max.unwrap_or(2).max(0),
            hand_max: (cfg.endgame_hand_max.unwrap_or(6).max(1) as usize),
        })
    } else {
        None
    };

    let mut writer = ShardWriter::new(&out_dir, &run_id, codec, zstd_level)?;
    writer.start_new_shard()?;

    // IM1: inference sidecar writer (INF1)
    let mut inf_writer = InfShardWriter::new(&out_dir, &run_id, codec, zstd_level)?;
    inf_writer.start_new_shard()?;

    // -----------------------------
    // Parallel match generation + streaming writer + PROGRESS
    // -----------------------------
    use std::sync::mpsc;
    use std::thread;

    fn mix_seed(base: u64, i: u64) -> u64 {
        // splitmix64-style mix (deterministic)
        let mut x = base ^ i.wrapping_mul(0x9E3779B97F4A7C15);
        x = x.wrapping_add(0xD1B54A32D192ED03);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    let matches_total: usize = cfg.matches.max(1) as usize;

    let auto_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let cfg_threads = cfg.threads.unwrap_or(0) as usize;
    let threads = if cfg_threads == 0 { auto_threads } else { cfg_threads };
    let threads = threads.max(1).min(matches_total);

    // Report progress every N matches (avoid log spam)
    let report_every: usize = if matches_total >= 5_000 {
        250
    } else if matches_total >= 1_000 {
        100
    } else {
        25
    };

    println!(
        "PROGRESS matches_done={} total_matches={} samples={}",
        0,
        matches_total,
        writer.total_samples()
    );
    let _ = std::io::stdout().flush();

    // batching
    const BATCH_MATCHES: usize = 4;
    const BATCH_SAMPLES_SOFT: usize = 4096;

    // Send (matches_in_batch, samples)
    let (tx, rx) =
        mpsc::sync_channel::<Result<(usize, Vec<Sample>), String>>(threads.saturating_mul(4).max(8));
    let mut handles = Vec::with_capacity(threads);

    for tid in 0..threads {
        let txc = tx.clone();
        let modelc = Arc::clone(&model);
        let paramsc = params;
        let temperature = cfg.temperature;
        let max_moves_per_round = cfg.max_moves_per_round;
        let match_target = cfg.match_target;
        let max_rounds = cfg.max_rounds;
        let end_filter_c = end_filter; // Copy

        handles.push(thread::spawn(move || {
            let mut batch: Vec<Sample> = Vec::new();
            let mut batch_matches_count: usize = 0;

            for mi in (tid..matches_total).step_by(threads) {
                let seed = mix_seed(base_seed, mi as u64);
                let mut r = StdRng::seed_from_u64(seed);

                let res = selfplay_one_match(
                    &mut r,
                    modelc.as_ref(),
                    paramsc,
                    temperature,
                    max_moves_per_round,
                    match_target,
                    max_rounds,
                    end_filter_c,
                );

                match res {
                    Ok(samples) => {
                        batch.extend(samples);
                        batch_matches_count += 1;

                        if batch_matches_count >= BATCH_MATCHES || batch.len() >= BATCH_SAMPLES_SOFT {
                            if txc
                                .send(Ok((batch_matches_count, std::mem::take(&mut batch))))
                                .is_err()
                            {
                                return;
                            }
                            batch_matches_count = 0;
                        }
                    }
                    Err(e) => {
                        let _ = txc.send(Err(e));
                        return;
                    }
                }
            }

            if !batch.is_empty() {
                let _ = txc.send(Ok((batch_matches_count, batch)));
            }
        }));
    }
    drop(tx);

    let mut matches_done: usize = 0;
    let mut next_report: usize = report_every;
    let mut first_err: Option<String> = None;

    for msg in rx {
        match msg {
            Ok((batch_matches, samples)) => {
                if first_err.is_some() {
                    continue;
                }
                for s in samples {
                    if writer.total_samples() > 0 && (writer.total_samples() as u32) % shard_max_samples == 0 {
                        writer.start_new_shard()?;
                        inf_writer.start_new_shard()?;
                    }
                    // P4-A1: write feat, pi, spike[112], z, mask
                    writer.write_sample(&s.feat, &s.pi, &s.spike, s.z, &s.mask)?;

                    // IM1: write inference sidecar record
                    let (inf_feat, inf_label) = s.build_inf_record();
                    inf_writer.write_sample(
                        &inf_feat,
                        &inf_label,
                        s.inf_opp_cnt,
                        s.inf_bone_cnt,
                    )?;
                }

                matches_done = matches_done.saturating_add(batch_matches);
                if matches_done >= next_report {
                    println!(
                        "PROGRESS matches_done={} total_matches={} samples={}",
                        matches_done,
                        matches_total,
                        writer.total_samples()
                    );
                    let _ = std::io::stdout().flush();

                    while next_report <= matches_done {
                        next_report = next_report.saturating_add(report_every);
                    }
                }
            }
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }

    for h in handles {
        let _ = h.join();
    }
    if let Some(e) = first_err {
        return Err(e);
    }

    if matches_done != matches_total {
        println!(
            "PROGRESS matches_done={} total_matches={} samples={}",
            matches_total,
            matches_total,
            writer.total_samples()
        );
        let _ = std::io::stdout().flush();
    }

    let shards = writer.finish_all()?;
    let infer_shards = inf_writer.finish_all()?;
    let samples_total: u64 = shards.iter().map(|x| x.samples as u64).sum();
    let infer_samples_total: u64 = infer_shards.iter().map(|x| x.samples as u64).sum();

    let manifest = Manifest {
        run_id: run_id.clone(),
        ruleset_id: crate::board::RULESET_ID.to_string(),
        feat_dim: FEAT_DIM as u32,
        action_size: ACTION_SIZE as u32,
        record_size: RECORD_SIZE as u32,
        matches: cfg.matches,
        samples: samples_total,
        codec: codec.as_str().to_string(),
        model_runtime,
        config: serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null),
        shards,

        infer_feat_dim: Some(INF_FEAT_DIM as u32),
        infer_label_size: Some(INF_LABEL_SIZE as u32),
        infer_record_size: Some(crate::inf_shards::INF_RECORD_SIZE as u32),
        infer_samples: Some(infer_samples_total),
        infer_codec: Some(codec.as_str().to_string()),
        infer_shards: Some(infer_shards),
    };

    let manifest_path = out_dir.join(format!("{run_id}.manifest.json"));
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap())
        .map_err(|e| format!("write manifest: {e}"))?;

    Ok(manifest_path.to_string_lossy().to_string())
}

// -----------------------------
// Utilities
// -----------------------------
fn budget_sims(det: u32, think_ms: u32) -> u32 {
    let worlds = det.clamp(6, 28);
    let sims_per_world = (think_ms / 60).clamp(6, 26);
    let total = worlds.saturating_mul(sims_per_world);
    total.min(6000).max(100)
}

fn default_seed() -> u64 {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    t ^ 0xA5A5_1234_55AA_900Du64
}

fn make_run_id(seed: Option<u64>) -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let s = seed.unwrap_or_else(default_seed);
    format!("run_{t}_{:08x}", (s as u32))
}

fn load_model_once(model_path: &str) -> Result<(MlpModel, String), String> {
    let p = Path::new(model_path);

    fn file_mtime(path: &Path) -> Option<SystemTime> {
        fs::metadata(path).ok().and_then(|m| m.modified().ok())
    }
    fn is_newer_or_equal(a: &Path, b: &Path) -> bool {
        match (file_mtime(a), file_mtime(b)) {
            (Some(ta), Some(tb)) => ta >= tb,
            (Some(_), None) => true,
            _ => false,
        }
    }

    if p.exists() && p.to_string_lossy().to_lowercase().ends_with(".bin") {
        let m = MlpModel::load_weights_bin(p)?;
        return Ok((m, format!("weights_bin:{model_path}")));
    }

    if p.exists() && p.to_string_lossy().to_lowercase().ends_with(".json") {
        // P4: always load JSON to preserve extra heads (spike/quantiles).
        let m = MlpModel::load_model_json(p)?;
        return Ok((m, format!("json:{model_path}")));
    }

    // Fallback: choose the newest between model.json and model.weights.bin.
    let bin = Path::new("model.weights.bin");
    let json = Path::new("model.json");

    if bin.exists() && (!json.exists() || is_newer_or_equal(bin, json)) {
        let m = MlpModel::load_weights_bin(bin)?;
        return Ok((m, "weights_bin:model.weights.bin".into()));
    }
    if json.exists() {
        let m = MlpModel::load_model_json(json)?;
        let _ = m.save_weights_bin("model.weights.bin");
        return Ok((m, "json_refresh_bin:model.weights.bin".into()));
    }

    Err("no model found (expected model.weights.bin or model.json)".into())
}

// -----------------------------
// Self-play core (match-to-target)
// -----------------------------
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum RoundStartMode {
    ForcedBest,
    FreeWinnerStarts,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum RoundEndKind {
    Out,
    Locked,
    Other,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PlayerId {
    A,
    B,
}

fn other(p: PlayerId) -> PlayerId {
    match p {
        PlayerId::A => PlayerId::B,
        PlayerId::B => PlayerId::A,
    }
}

#[inline]
fn tiles_mask28(tiles: &[Tile]) -> u32 {
    let mut m: u32 = 0;
    for &t in tiles.iter() {
        m |= 1u32 << (t.id() as u32);
    }
    m
}

#[inline]
fn mask28_to_i8(m: u32) -> [i8; 28] {
    let mut out = [0i8; 28];
    for i in 0..28 {
        if (m & (1u32 << (i as u32))) != 0 {
            out[i] = 1;
        }
    }
    out
}

/// IM1 signal: End-choice preference/avoidance for a played tile when it had >1 legal end.
/// - prefer[new_open_value_of_chosen_end]++
/// - avoid[new_open_value_of_other_ends]++
fn update_endchoice_signal(board: &Board, t: Tile, chosen_end: End, prefer: &mut [u8; 7], avoid: &mut [u8; 7]) {
    if board.is_empty() {
        return;
    }
    let ends = board.legal_ends_for_tile(t);
    if ends.len() <= 1 {
        return;
    }

    let mut chosen_new: Option<u8> = None;
    for &e in ends.iter() {
        let Some(es) = board.ends_raw()[e.idx()] else { continue; };
        let open_val = es.open_value;
        let new_val = t.other_value(open_val).unwrap_or(open_val);
        if e == chosen_end {
            chosen_new = Some(new_val);
            break;
        }
    }
    let Some(ch) = chosen_new else { return; };
    if ch <= 6 {
        prefer[ch as usize] = prefer[ch as usize].saturating_add(1);
    }

    for &e in ends.iter() {
        if e == chosen_end {
            continue;
        }
        let Some(es) = board.ends_raw()[e.idx()] else { continue; };
        let open_val = es.open_value;
        let new_val = t.other_value(open_val).unwrap_or(open_val);
        if new_val == ch {
            continue;
        }
        if new_val <= 6 {
            avoid[new_val as usize] = avoid[new_val as usize].saturating_add(1);
        }
    }
}

struct Sample {
    feat: [f32; FEAT_DIM],
    pi: [f32; ACTION_SIZE],
    z: f32,
    spike: [i8; ACTION_SIZE],
    mask: [i8; ACTION_SIZE],

    // IM1 inference sidecar extras (kept compact; inf_feat is built at write time)
    inf_opp_played_mask: u32,          // 28-bit mask
    inf_opp_endpref: [u8; 7],          // counts
    inf_opp_endavoid: [u8; 7],         // counts
    inf_label_opp_hand_mask: u32,      // 28-bit truth mask
    inf_opp_cnt: u8,
    inf_bone_cnt: u8,
}

impl Sample {
    fn build_inf_record(&self) -> ([f32; INF_FEAT_DIM], [i8; INF_LABEL_SIZE]) {
        let mut out = [0f32; INF_FEAT_DIM];
        // 0..193 base feat
        out[0..FEAT_DIM].copy_from_slice(&self.feat);

        // 193..221 opp_played_mask28
        let mut off = FEAT_DIM;
        for i in 0..28 {
            out[off + i] = if (self.inf_opp_played_mask & (1u32 << (i as u32))) != 0 { 1.0 } else { 0.0 };
        }
        off += 28;

        // normalize counts (cap at 8) to keep range stable
        for v in 0..7 {
            out[off + v] = (self.inf_opp_endpref[v].min(8) as f32) / 8.0;
        }
        off += 7;
        for v in 0..7 {
            out[off + v] = (self.inf_opp_endavoid[v].min(8) as f32) / 8.0;
        }
        let label = mask28_to_i8(self.inf_label_opp_hand_mask);
        (out, label)
    }
}

fn spike_win_now_vec(
    board: &Board,
    legal: &[(Tile, End)],
    opp_hand_truth: &[Tile],
    opp_score: i32,
    target: i32,
) -> [i8; ACTION_SIZE] {
    let mut out = [0i8; ACTION_SIZE];
    for &(t, e) in legal.iter() {
        let mut b2 = board.clone();
        if b2.play(t, e).is_err() {
            continue;
        }
        let win_now = ismcts::opp_has_win_now_exists(&b2, opp_hand_truth, opp_score, target);
        let a = features::encode_action(t, e.idx());
        out[a] = if win_now { 1 } else { 0 };
    }
    out
}

fn deal(rng: &mut StdRng) -> (Vec<Tile>, Vec<Tile>, Vec<Tile>) {
    let mut tiles: Vec<Tile> = (0u8..28u8).map(Tile).collect();
    tiles.shuffle(rng);
    (tiles[0..7].to_vec(), tiles[7..14].to_vec(), tiles[14..].to_vec())
}

fn best_opening_tile(hand: &[Tile]) -> Tile {
    let mut doubles: Vec<Tile> = hand.iter().copied().filter(|t| t.is_double()).collect();
    if !doubles.is_empty() {
        doubles.sort_by_key(|t| t.pips().0);
        return *doubles.last().unwrap();
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
    best
}

fn opening_key(t: Tile) -> (i32, i32, i32) {
    let (a, b) = t.pips();
    let dbl = if t.is_double() { 1 } else { 0 };
    (dbl, (a as i32 + b as i32), a as i32)
}

fn apply_play(
    board: &mut Board,
    hand: &mut Vec<Tile>,
    score: &mut i32,
    t: Tile,
    e: End,
) -> Result<i32, String> {
    let pts = board.play(t, e)?;
    if let Some(i) = hand.iter().position(|x| *x == t) {
        hand.swap_remove(i);
    }
    *score += pts;
    Ok(pts)
}

fn round_to_nearest_5(x: i32) -> i32 {
    ((x as f64) / 5.0).round() as i32 * 5
}

fn award_out(winner_score: &mut i32, loser_hand: &[Tile]) {
    let pips: i32 = loser_hand.iter().map(|t| t.pip_sum() as i32).sum();
    *winner_score += round_to_nearest_5(pips);
}

// Margin-aware Z (continuous in [-1,+1])
fn margin_z(my_score: i32, opp_score: i32, target: i32) -> f32 {
    let t = target.max(1) as f32;
    let diff = my_score - opp_score;
    if diff == 0 {
        return 0.0;
    }
    let sign = if diff > 0 { 1.0f32 } else { -1.0f32 };
    let margin = diff.abs() as f32;
    let m = (margin / t).clamp(0.0, 1.0);

    let mut strength = 0.10 + 0.90 * m;

    if sign < 0.0 && opp_score >= target && my_score >= (target - 20) && margin <= 15.0 {
        strength = strength.max(0.60);
    }

    (sign * strength).clamp(-1.0, 1.0)
}

fn locked_award(
    score_a: &mut i32,
    score_b: &mut i32,
    hand_a: &[Tile],
    hand_b: &[Tile],
) -> Option<PlayerId> {
    let pa: i32 = hand_a.iter().map(|t| t.pip_sum() as i32).sum();
    let pb: i32 = hand_b.iter().map(|t| t.pip_sum() as i32).sum();
    let diff = (pa - pb).abs();
    let pts = round_to_nearest_5(diff);
    if pts <= 0 {
        return None;
    }
    if pa < pb {
        *score_a += pts;
        return Some(PlayerId::A);
    }
    if pb < pa {
        *score_b += pts;
        return Some(PlayerId::B);
    }
    None
}

fn open_ends_pack(b: &Board) -> (u8, [u8; 4]) {
    let v = b.open_end_values();
    let mut arr = [255u8; 4];
    let mut len = 0u8;
    for (i, x) in v.iter().take(4).enumerate() {
        arr[i] = *x;
        len += 1;
    }
    (len, arr)
}

fn legal_moves(board: &Board, hand: &Vec<Tile>, forced: Option<Tile>) -> Vec<(Tile, End)> {
    let tiles: Vec<Tile> = if let Some(ft) = forced { vec![ft] } else { hand.clone() };
    let mut out = Vec::new();
    for t in tiles {
        for e in board.legal_ends_for_tile(t) {
            out.push((t, e));
        }
    }
    out
}

fn visits_to_pi_and_mask(
    visits: &[u32; ACTION_SIZE],
    mask: &[i8; ACTION_SIZE],
    temperature: f32,
) -> [f32; ACTION_SIZE] {
    let mut out = [0f32; ACTION_SIZE];
    let mut sum = 0f64;
    for i in 0..ACTION_SIZE {
        if mask[i] == 0 {
            continue;
        }
        sum += visits[i] as f64;
    }
    if sum <= 1e-12 {
        let legal = mask.iter().filter(|&&m| m != 0).count().max(1) as f32;
        for i in 0..ACTION_SIZE {
            if mask[i] != 0 {
                out[i] = 1.0 / legal;
            }
        }
        return out;
    }
    let t = temperature.max(1e-6) as f64;
    let pow = 1.0 / t;

    let mut sump = 0f64;
    for i in 0..ACTION_SIZE {
        if mask[i] == 0 {
            continue;
        }
        let v = (visits[i] as f64).powf(pow);
        out[i] = v as f32;
        sump += v;
    }
    let denom = (sump + 1e-12) as f32;
    for i in 0..ACTION_SIZE {
        if mask[i] != 0 {
            out[i] /= denom;
        } else {
            out[i] = 0.0;
        }
    }
    out
}

fn sample_move_from_pi(
    rng: &mut StdRng,
    pi: &[f32; ACTION_SIZE],
    legal: &[(Tile, End)],
    p_argmax: f32,
) -> (Tile, End) {
    if legal.is_empty() {
        return (Tile(0), End::Right);
    }
    if rng.gen::<f32>() < p_argmax {
        let mut best = legal[0];
        let mut bestp = -1.0f32;
        for &(t, e) in legal.iter() {
            let a = features::encode_action(t, e.idx());
            let p = pi[a];
            if p > bestp {
                bestp = p;
                best = (t, e);
            }
        }
        return best;
    }

    let mut sum_legal = 0.0f32;
    for &(t, e) in legal.iter() {
        let a = features::encode_action(t, e.idx());
        sum_legal += pi[a].max(0.0);
    }
    if sum_legal <= 1e-12 {
        return legal[legal.len() - 1];
    }
    let mut r = rng.gen::<f32>() * sum_legal;
    for &(t, e) in legal.iter() {
        let a = features::encode_action(t, e.idx());
        r -= pi[a].max(0.0);
        if r <= 0.0 {
            return (t, e);
        }
    }
    legal[legal.len() - 1]
}

fn build_info_state(
    board: &Board,
    my_hand: &Vec<Tile>,
    opp_cnt: i32,
    bone_cnt: i32,
    my_score: i32,
    opp_score: i32,
    match_target: i32,
    events: &Vec<BeliefEvent>,
    forced: Option<Tile>,
    ply: i32,
    round_index: i32,
    opp_played_tiles: &Vec<Tile>,
    opp_avoided_open_values: &[u8; 7],
) -> InfoState {
    InfoState {
        board: board.clone(),
        my_hand: my_hand.clone(),
        opponent_tile_count: opp_cnt,
        boneyard_count: bone_cnt,
        forced_play_tile: forced,
        my_score,
        opp_score,
        match_target,
        current_turn_me: true,
        ply,
        round_index,
        events: events.clone(),
        opp_played_tiles: opp_played_tiles.clone(),
        opp_avoided_open_values: *opp_avoided_open_values,
        opp_infer_tile_p: [0.0f32; 28],
    }
}

fn feat_and_mask_from_info(st: &InfoState) -> ([f32; FEAT_DIM], [i8; ACTION_SIZE]) {
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
        let mut open_ends = Vec::with_capacity(ev.len as usize);
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
        current_turn_me: true,
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

    let feat = features::features_193(&sv);
    let mask = features::legal_mask_112(&sv);
    (feat, mask)
}

#[derive(Clone)]
struct PendingRec {
    feat: [f32; FEAT_DIM],
    pi: [f32; ACTION_SIZE],
    spike: [i8; ACTION_SIZE],
    mask: [i8; ACTION_SIZE],

    inf_opp_played_mask: u32,
    inf_opp_endpref: [u8; 7],
    inf_opp_endavoid: [u8; 7],
    inf_label_opp_hand_mask: u32,
    inf_opp_cnt: u8,
    inf_bone_cnt: u8,
}

// -----------------------------
// Self-play match
// -----------------------------
fn selfplay_one_match(
    rng: &mut StdRng,
    model: &MlpModel,
    ismcts_params: IsmctsParams,
    temperature: f32,
    max_moves_per_round: u32,
    match_target: i32,
    max_rounds: u32,
    end_filter: Option<EndgameFilter>,
) -> Result<Vec<Sample>, String> {
    let mut score_a = 0i32;
    let mut score_b = 0i32;

    let mut ev_a: Vec<BeliefEvent> = Vec::new();
    let mut ev_b: Vec<BeliefEvent> = Vec::new();
    let mut ply_counter: i32 = 0;

    let mut pend_a: Vec<PendingRec> = Vec::new();
    let mut pend_b: Vec<PendingRec> = Vec::new();

    let mut prev_end_kind: Option<RoundEndKind> = None;
    let mut prev_out_winner: Option<PlayerId> = None;

    let mut rounds = 0u32;
    while score_a < match_target && score_b < match_target && rounds < max_rounds {
        rounds += 1;

        // Belief evidence must NOT carry across rounds
        ev_a.clear();
        ev_b.clear();
        ply_counter = 0;

        // Stage-2.5: reset inference trackers each round
        let mut played_a: Vec<Tile> = Vec::new();
        let mut played_b: Vec<Tile> = Vec::new();
        let mut avoided_a: [u8; 7] = [0u8; 7];
        let mut avoided_b: [u8; 7] = [0u8; 7];
        let mut played_mask_a: u32 = 0;
        let mut played_mask_b: u32 = 0;
        let mut endpref_a: [u8; 7] = [0u8; 7];
        let mut endpref_b: [u8; 7] = [0u8; 7];
        let mut endavoid_a: [u8; 7] = [0u8; 7];
        let mut endavoid_b: [u8; 7] = [0u8; 7];

        let (mut hand_a, mut hand_b, mut boneyard) = deal(rng);

        let mut board = Board::new();
        let mut forced_a: Option<Tile> = None;
        let mut forced_b: Option<Tile> = None;

        let (start_mode, starter) = if rounds == 1 {
            let a_best = best_opening_tile(&hand_a);
            let b_best = best_opening_tile(&hand_b);
            let a_k = opening_key(a_best);
            let b_k = opening_key(b_best);
            let starter = if b_k > a_k { PlayerId::B } else { PlayerId::A };
            (RoundStartMode::ForcedBest, starter)
        } else {
            match prev_end_kind {
                Some(RoundEndKind::Out) => (RoundStartMode::FreeWinnerStarts, prev_out_winner.unwrap_or(PlayerId::A)),
                _ => {
                    let a_best = best_opening_tile(&hand_a);
                    let b_best = best_opening_tile(&hand_b);
                    let a_k = opening_key(a_best);
                    let b_k = opening_key(b_best);
                    let starter = if b_k > a_k { PlayerId::B } else { PlayerId::A };
                    (RoundStartMode::ForcedBest, starter)
                }
            }
        };

        let mut turn = starter;

        if start_mode == RoundStartMode::ForcedBest {
            let t = match starter {
                PlayerId::A => best_opening_tile(&hand_a),
                PlayerId::B => best_opening_tile(&hand_b),
            };
            match starter {
                PlayerId::A => {
                    let _ = apply_play(&mut board, &mut hand_a, &mut score_a, t, End::Right)?;
                    played_a.push(t);
                    played_mask_a |= 1u32 << (t.id() as u32);
                }
                PlayerId::B => {
                    let _ = apply_play(&mut board, &mut hand_b, &mut score_b, t, End::Right)?;
                    played_b.push(t);
                    played_mask_b |= 1u32 << (t.id() as u32);
                }
            }
            ply_counter += 1;
            if score_a >= match_target || score_b >= match_target {
                break;
            }
            turn = other(starter);
        }

        let mut ended_kind = RoundEndKind::Other;
        let mut out_winner: Option<PlayerId> = None;

        for _ply in 0..max_moves_per_round {
            if boneyard.is_empty() {
                let a_can = !legal_moves(&board, &hand_a, forced_a).is_empty();
                let b_can = !legal_moves(&board, &hand_b, forced_b).is_empty();
                if !a_can && !b_can {
                    let _w = locked_award(&mut score_a, &mut score_b, &hand_a, &hand_b);
                    ended_kind = RoundEndKind::Locked;
                    out_winner = None;
                    break;
                }
            }

            if score_a >= match_target || score_b >= match_target {
                break;
            }

            match turn {
                PlayerId::A => {
                    let (ended, kind, outw) = player_turn(
                        rng,
                        model,
                        rounds as i32,
                        ismcts_params,
                        temperature,
                        match_target,
                        &mut board,
                        &mut hand_a,
                        &mut hand_b,
                        &mut boneyard,
                        &mut score_a,
                        &mut score_b,
                        &mut forced_a,
                        &mut ev_a,
                        &mut ev_b,
                        &mut ply_counter,
                        &mut pend_a,
                        &mut played_a,
                        &mut played_mask_a,
                        &played_b,
                        &played_mask_b,
                        &mut avoided_a,
                        &avoided_b,
                        &mut endpref_a,
                        &mut endavoid_a,
                        &endpref_b,
                        &endavoid_b,
                        PlayerId::A,
                        end_filter,
                    )?;
                    if ended {
                        ended_kind = kind;
                        out_winner = outw;
                        break;
                    }
                    turn = PlayerId::B;
                }
                PlayerId::B => {
                    let (ended, kind, outw) = player_turn(
                        rng,
                        model,
                        rounds as i32,
                        ismcts_params,
                        temperature,
                        match_target,
                        &mut board,
                        &mut hand_b,
                        &mut hand_a,
                        &mut boneyard,
                        &mut score_b,
                        &mut score_a,
                        &mut forced_b,
                        &mut ev_b,
                        &mut ev_a,
                        &mut ply_counter,
                        &mut pend_b,
                        &mut played_b,
                        &mut played_mask_b,
                        &played_a,
                        &played_mask_a,
                        &mut avoided_b,
                        &avoided_a,
                        &mut endpref_b,
                        &mut endavoid_b,
                        &endpref_a,
                        &endavoid_a,
                        PlayerId::B,
                        end_filter,
                    )?;
                    if ended {
                        ended_kind = kind;
                        out_winner = outw;
                        break;
                    }
                    turn = PlayerId::A;
                }
            }
        }

        prev_end_kind = Some(ended_kind);
        prev_out_winner = if ended_kind == RoundEndKind::Out { out_winner } else { None };

        if score_a >= match_target || score_b >= match_target {
            break;
        }
    }

    let z_a: f32 = margin_z(score_a, score_b, match_target);
    let z_b: f32 = -z_a;

    let mut out: Vec<Sample> = Vec::with_capacity(pend_a.len() + pend_b.len());
    for rec in pend_a {
        out.push(Sample {
            feat: rec.feat,
            pi: rec.pi,
            spike: rec.spike,
            z: z_a,
            mask: rec.mask,
            inf_opp_played_mask: rec.inf_opp_played_mask,
            inf_opp_endpref: rec.inf_opp_endpref,
            inf_opp_endavoid: rec.inf_opp_endavoid,
            inf_label_opp_hand_mask: rec.inf_label_opp_hand_mask,
            inf_opp_cnt: rec.inf_opp_cnt,
            inf_bone_cnt: rec.inf_bone_cnt,
        });
    }
    for rec in pend_b {
        out.push(Sample {
            feat: rec.feat,
            pi: rec.pi,
            spike: rec.spike,
            z: z_b,
            mask: rec.mask,
            inf_opp_played_mask: rec.inf_opp_played_mask,
            inf_opp_endpref: rec.inf_opp_endpref,
            inf_opp_endavoid: rec.inf_opp_endavoid,
            inf_label_opp_hand_mask: rec.inf_label_opp_hand_mask,
            inf_opp_cnt: rec.inf_opp_cnt,
            inf_bone_cnt: rec.inf_bone_cnt,
        });
    }
    Ok(out)
}

fn player_turn(
    rng: &mut StdRng,
    model: &MlpModel,
    round_index: i32,
    ismcts_params: IsmctsParams,
    temperature: f32,
    match_target: i32,
    board: &mut Board,
    my_hand: &mut Vec<Tile>,
    opp_hand: &mut Vec<Tile>,
    boneyard: &mut Vec<Tile>,
    my_score: &mut i32,
    opp_score: &mut i32,
    forced: &mut Option<Tile>,
    my_belief_events: &mut Vec<BeliefEvent>,
    opp_belief_events: &mut Vec<BeliefEvent>,
    ply_counter: &mut i32,
    pending: &mut Vec<PendingRec>,
    my_played_tiles: &mut Vec<Tile>,
    my_played_mask: &mut u32,
    opp_played_tiles: &Vec<Tile>,
    opp_played_mask: &u32,
    my_avoided_open_values: &mut [u8; 7],
    opp_avoided_open_values: &[u8; 7],
    my_endpref: &mut [u8; 7],
    my_endavoid: &mut [u8; 7],
    opp_endpref: &[u8; 7],
    opp_endavoid: &[u8; 7],
    me_id: PlayerId,
    end_filter: Option<EndgameFilter>,
) -> Result<(bool, RoundEndKind, Option<PlayerId>), String> {
    loop {
        let legal = legal_moves(board, my_hand, *forced);
        if !legal.is_empty() {
            break;
        }
        if boneyard.is_empty() {
            let (len, arr) = open_ends_pack(board);
            *ply_counter += 1;
            opp_belief_events.push(BeliefEvent {
                typ: BeliefEventType::Pass,
                ply: *ply_counter,
                open_ends: arr,
                len,
                certainty: BeliefCertainty::Certain,
            });
            return Ok((false, RoundEndKind::Other, None));
        }

        let drawn = boneyard.pop().unwrap();
        my_hand.push(drawn);

        let (len, arr) = open_ends_pack(board);
        *ply_counter += 1;
        opp_belief_events.push(BeliefEvent {
            typ: BeliefEventType::Draw,
            ply: *ply_counter,
            open_ends: arr,
            len,
            certainty: BeliefCertainty::Certain,
        });

        if !board.legal_ends_for_tile(drawn).is_empty() {
            *forced = Some(drawn);
            break;
        }
    }

    let opp_cnt = opp_hand.len() as i32;
    let bone_cnt = boneyard.len() as i32;
    let forced_before_move = forced.is_some();

    let info = build_info_state(
        board,
        my_hand,
        opp_cnt,
        bone_cnt,
        *my_score,
        *opp_score,
        match_target,
        my_belief_events,
        *forced,
        *ply_counter,
        round_index,
        opp_played_tiles,
        opp_avoided_open_values,
    );

    let legal = legal_moves(board, my_hand, *forced);
    let mut mask = [0i8; ACTION_SIZE];
    for &(t, e) in legal.iter() {
        mask[features::encode_action(t, e.idx())] = 1;
    }

    let seed = rng.gen::<u64>() ^ ((me_id == PlayerId::A) as u64);
    let visits = ismcts::ismcts_root_visits(&info, Some(model), ismcts_params, seed);

    let (feat, mask2) = feat_and_mask_from_info(&info);
    debug_assert!(legal
        .iter()
        .all(|&(t, e)| mask2[features::encode_action(t, e.idx())] == 1));

    let pi = visits_to_pi_and_mask(&visits, &mask2, temperature);

    // Spike targets (truth): per-legal-action, does opponent have win-now after this move?
    let spike = spike_win_now_vec(&info.board, &legal, opp_hand.as_slice(), *opp_score, match_target);

    // --- ENDGAME MINING: record-only filter (does NOT affect gameplay) ---
    let record = if let Some(f) = end_filter {
        f.should_record(
            match_target,
            *my_score,
            *opp_score,
            my_hand.len(),
            opp_hand.len(),
            bone_cnt,
        )
    } else {
        true
    };

    if record {
        let inf_label_opp = tiles_mask28(opp_hand.as_slice());
        pending.push(PendingRec {
            feat,
            pi,
            spike,
            mask: mask2,
            inf_opp_played_mask: *opp_played_mask,
            inf_opp_endpref: *opp_endpref,
            inf_opp_endavoid: *opp_endavoid,
            inf_label_opp_hand_mask: inf_label_opp,
            inf_opp_cnt: (opp_cnt.max(0).min(28) as u8),
            inf_bone_cnt: (bone_cnt.max(0).min(28) as u8),
        });
    }

    let (t, e) = sample_move_from_pi(rng, &pi, &legal, 0.05);

    // IM1: end-choice signal update (only if this was a real choice, not forced after draw)
    if !forced_before_move {
        update_endchoice_signal(board, t, e, my_endpref, my_endavoid);
    }

    // Stage-2.5: update "avoided open values" only if this was a real choice (not forced-play after draw)
    if !forced_before_move {
        let opens = board.open_end_values();
        let (a, b) = t.pips();
        for v in opens {
            if v <= 6 && v != a && v != b {
                my_avoided_open_values[v as usize] =
                    my_avoided_open_values[v as usize].saturating_add(1);
            }
        }
    }

    let _pts = apply_play(board, my_hand, my_score, t, e)?;
    my_played_tiles.push(t);
    *my_played_mask |= 1u32 << (t.id() as u32);

    *ply_counter += 1;
    if *forced == Some(t) {
        *forced = None;
    }

    if *my_score >= match_target || *opp_score >= match_target {
        return Ok((true, RoundEndKind::Other, None));
    }

    if my_hand.is_empty() {
        award_out(my_score, opp_hand);
        if *my_score >= match_target {
            return Ok((true, RoundEndKind::Other, None));
        }
        return Ok((true, RoundEndKind::Out, Some(me_id)));
    }

    Ok((false, RoundEndKind::Other, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::End;

    #[test]
    fn endchoice_signal_prefers_chosen_new_open_value_and_avoids_alternative() {
        // Board with two different open values (6 and 1), tile 6-1 can go on both ends.
        let mut b = Board::new();
        let t61 = Tile::parse("6-1").unwrap();
        let _ = b.play(t61, End::Right).unwrap(); // first tile, opens right=6 left=1

        // Now consider playing the SAME tile 6-1 again is illegal; use a different tile with same pips.
        // We'll use tile 6-0? No; we need a tile that matches both ends 6 and 1 => tile 6-1 (unique).
        // Instead: we simulate the signal update on the pre-move board using a hypothetical legal tile 6-1.
        // The function only uses ends/open values and tile pips, so this is a pure signal unit test.
        let mut pref = [0u8; 7];
        let mut avoid = [0u8; 7];
        update_endchoice_signal(&b, t61, End::Right, &mut pref, &mut avoid);

        // chosen_end=Right means we matched open=6, so new open value becomes 1 => prefer[1]++
        // alternative_end=Left matches open=1, so new open value would become 6 => avoid[6]++
        assert_eq!(pref[1], 1);
        assert_eq!(avoid[6], 1);
    }
}