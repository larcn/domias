// FILE: src/board.rs | version: 2026-01-11.rc3
// CHANGELOG:
// - RC3: Add Board::from_snapshot_parts(...) to build Board directly from snapshot state
//        WITHOUT replay. This avoids spinner timeline bugs (first-double-to-spinner depends on play order).
// - RC2: Add read-only accessors required for hot-path hashing/search (played_mask, played_order, ends, arms).
// - RC2: NO rule/semantics changes to core play/score.

use std::collections::BTreeMap;

use crate::tile::Tile;

pub const RULESET_ID: &str = "fives_house_v3_target_immediate";

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum End {
    Right = 0,
    Left = 1,
    Up = 2,
    Down = 3,
}

impl End {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn as_str(self) -> &'static str {
        match self {
            End::Right => "right",
            End::Left => "left",
            End::Up => "up",
            End::Down => "down",
        }
    }
    pub fn from_str(s: &str) -> Option<End> {
        match s {
            "right" => Some(End::Right),
            "left" => Some(End::Left),
            "up" => Some(End::Up),
            "down" => Some(End::Down),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EndState {
    pub open_value: u8,
    pub is_double_end: bool,
}

#[derive(Clone, Debug)]
pub struct Board {
    pub center_tile: Option<Tile>,
    pub spinner_value: Option<u8>,
    pub spinner_sides_open: bool,

    ends: [Option<EndState>; 4],
    arms: [Vec<Tile>; 4],

    played_mask: u32, // 28-bit
    played_order: Vec<Tile>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self {
            center_tile: None,
            spinner_value: None,
            spinner_sides_open: false,
            ends: [None, None, None, None],
            arms: std::array::from_fn(|_| Vec::new()),
            played_mask: 0,
            played_order: Vec::new(),
        }
    }

    /// Build board directly from snapshot-like parts WITHOUT replay.
    /// This is required when external code (Python engine) is authoritative for the state,
    /// because replay order can change spinner promotion behavior.
    pub fn from_snapshot_parts(
        center_tile: Option<Tile>,
        spinner_value: Option<u8>,
        spinner_sides_open: bool,
        ends: [Option<EndState>; 4],
        arms: [Vec<Tile>; 4],
        played_order: Vec<Tile>,
    ) -> Self {
        if center_tile.is_none() {
            // Defensive: an "empty" board should be empty regardless of other fields.
            return Board::new();
        }

        let mut b = Board::new();
        b.center_tile = center_tile;
        b.spinner_value = spinner_value;
        b.spinner_sides_open = spinner_sides_open;
        b.ends = ends;
        b.arms = arms;
        b.played_order = played_order;

        b.played_mask = 0;
        for t in b.played_order.iter().copied() {
            b.played_mask |= 1u32 << (t.id() as u32);
        }
        b
    }

    // -------------------------
    // Read-only accessors (hot-path friendly)
    // -------------------------
    #[inline]
    pub fn played_mask(&self) -> u32 {
        self.played_mask
    }
    #[inline]
    pub fn played_order(&self) -> &[Tile] {
        &self.played_order
    }
    #[inline]
    pub fn ends_raw(&self) -> &[Option<EndState>; 4] {
        &self.ends
    }
    #[inline]
    pub fn arms_raw(&self) -> &[Vec<Tile>; 4] {
        &self.arms
    }

    // -------------------------
    // Basic helpers
    // -------------------------
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.center_tile.is_none()
    }

    #[inline]
    fn arm_started(&self, e: End) -> bool {
        !self.arms[e.idx()].is_empty()
    }

    #[inline]
    fn has_played(&self, t: Tile) -> bool {
        (self.played_mask & (1u32 << (t.id() as u32))) != 0
    }

    #[inline]
    fn add_played(&mut self, t: Tile) {
        self.played_mask |= 1u32 << (t.id() as u32);
        self.played_order.push(t);
    }

    pub fn open_end_values(&self) -> Vec<u8> {
        let mut v: Vec<u8> = self
            .ends
            .iter()
            .filter_map(|e| e.map(|x| x.open_value))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn legal_ends_for_tile(&self, t: Tile) -> Vec<End> {
        if self.is_empty() {
            return vec![End::Right];
        }
        let mut out = Vec::new();
        for e in [End::Right, End::Left, End::Up, End::Down] {
            if let Some(es) = self.ends[e.idx()] {
                if t.has(es.open_value) {
                    out.push(e);
                }
            }
        }
        out
    }

    // -------------------------
    // Scoring
    // -------------------------
    pub fn ends_sum(&self) -> i32 {
        if self.is_empty() || self.center_tile.is_none() {
            return 0;
        }

        if let (Some(sv), Some(ct)) = (self.spinner_value, self.center_tile) {
            if ct.is_double() {
                let sv = sv as i32;
                let r = self.arm_started(End::Right);
                let l = self.arm_started(End::Left);
                let u = self.arm_started(End::Up);
                let d = self.arm_started(End::Down);
                let any = r || l || u || d;

                if !any {
                    return sv * 2;
                }

                if (r && !l) || (l && !r) {
                    let started_end = if r { End::Right } else { End::Left };
                    if let Some(es) = self.ends[started_end.idx()] {
                        let val = es.open_value as i32;
                        return sv * 2 + if es.is_double_end { val * 2 } else { val };
                    }
                    return sv * 2;
                }

                let mut total: i32 = 0;
                for e in [End::Right, End::Left, End::Up, End::Down] {
                    if matches!(e, End::Up | End::Down) && !self.arm_started(e) {
                        continue;
                    }
                    if let Some(es) = self.ends[e.idx()] {
                        let val = es.open_value as i32;
                        total += if es.is_double_end { val * 2 } else { val };
                    }
                }
                return total;
            }
        }

        let mut total: i32 = 0;
        for e in [End::Right, End::Left, End::Up, End::Down] {
            if let Some(es) = self.ends[e.idx()] {
                let val = es.open_value as i32;
                total += if es.is_double_end { val * 2 } else { val };
            }
        }
        total
    }

    pub fn score_now(&self) -> i32 {
        let s = self.ends_sum();
        if s != 0 && (s % 5 == 0) {
            s
        } else {
            0
        }
    }

    // -------------------------
    // Core play
    // -------------------------
    pub fn play(&mut self, t: Tile, end: End) -> Result<i32, String> {
        if self.has_played(t) {
            return Err(format!("tile already played: {}", t.to_str()));
        }

        if self.is_empty() {
            self.play_first(t);
            return Ok(self.score_now());
        }

        let es = self
            .ends[end.idx()]
            .ok_or_else(|| format!("end not open: {}", end.as_str()))?;

        if !t.has(es.open_value) {
            return Err(format!(
                "illegal: {} cannot go on {}({})",
                t.to_str(),
                end.as_str(),
                es.open_value
            ));
        }

        let new_val = t.other_value(es.open_value)?;

        self.add_played(t);
        self.arms[end.idx()].push(t);
        self.ends[end.idx()] = Some(EndState {
            open_value: new_val,
            is_double_end: t.is_double(),
        });

        if self.spinner_value.is_none() && t.is_double() {
            self.promote_first_double_to_spinner(t, end);
        }

        self.check_open_spinner_sides();
        Ok(self.score_now())
    }

    fn play_first(&mut self, t: Tile) {
        self.center_tile = Some(t);
        self.add_played(t);
        self.arms = std::array::from_fn(|_| Vec::new());

        let (a, _b) = t.pips();
        if t.is_double() {
            self.spinner_value = Some(a);
            self.spinner_sides_open = false;
            self.ends = [None, None, None, None];
            self.ends[End::Right.idx()] = Some(EndState {
                open_value: a,
                is_double_end: false,
            });
            self.ends[End::Left.idx()] = Some(EndState {
                open_value: a,
                is_double_end: false,
            });
        } else {
            let (hi, lo) = t.pips();
            self.spinner_value = None;
            self.spinner_sides_open = false;
            self.ends = [None, None, None, None];
            self.ends[End::Right.idx()] = Some(EndState {
                open_value: hi,
                is_double_end: false,
            });
            self.ends[End::Left.idx()] = Some(EndState {
                open_value: lo,
                is_double_end: false,
            });
        }
    }

    fn check_open_spinner_sides(&mut self) {
        if self.spinner_value.is_none() || self.spinner_sides_open {
            return;
        }
        if self.arm_started(End::Right) && self.arm_started(End::Left) {
            let sv = self.spinner_value.unwrap();
            self.spinner_sides_open = true;
            self.ends[End::Up.idx()] = Some(EndState {
                open_value: sv,
                is_double_end: false,
            });
            self.ends[End::Down.idx()] = Some(EndState {
                open_value: sv,
                is_double_end: false,
            });
        }
    }

    fn promote_first_double_to_spinner(&mut self, spinner_tile: Tile, played_end: End) {
        if self.center_tile.is_none() {
            return;
        }
        if self.spinner_value.is_some() {
            return;
        }
        if !spinner_tile.is_double() {
            return;
        }
        if !matches!(played_end, End::Right | End::Left) {
            return;
        }

        let old_order = self.played_order.clone();
        let ct0 = match self.center_tile {
            Some(x) => x,
            None => return,
        };

        let arms0 = self.arms.clone();
        let path = &arms0[played_end.idx()];
        if path.is_empty() || *path.last().unwrap() != spinner_tile {
            return;
        }

        let opp_end = if played_end == End::Right {
            End::Left
        } else {
            End::Right
        };
        let chain_arm = opp_end;

        let path_wo = &path[..path.len().saturating_sub(1)];
        let opp_arm_tiles = &arms0[opp_end.idx()];

        let mut chain_tiles: Vec<Tile> = Vec::with_capacity(path_wo.len() + 1 + opp_arm_tiles.len());
        for &t in path_wo.iter().rev() {
            chain_tiles.push(t);
        }
        chain_tiles.push(ct0);
        for &t in opp_arm_tiles.iter() {
            chain_tiles.push(t);
        }

        let mut b2 = Board::new();
        b2.play_first(spinner_tile);
        for t in chain_tiles {
            let _ = b2.play(t, chain_arm);
        }

        b2.spinner_sides_open = false;
        b2.ends[End::Up.idx()] = None;
        b2.ends[End::Down.idx()] = None;

        self.center_tile = b2.center_tile;
        self.spinner_value = b2.spinner_value;
        self.spinner_sides_open = b2.spinner_sides_open;
        self.ends = b2.ends;
        self.arms = b2.arms;

        self.played_order = old_order;
        self.played_mask = 0;
        for t in self.played_order.iter().copied() {
            self.played_mask |= 1u32 << (t.id() as u32);
        }
    }

    // -------------------------
    // Snapshot boundary
    // -------------------------
    pub fn snapshot(&self) -> BTreeMap<String, serde_like::Value> {
        use serde_like::Value;

        let mut out = BTreeMap::<String, Value>::new();
        out.insert("ruleset".into(), Value::Str(RULESET_ID.into()));
        out.insert(
            "center_tile".into(),
            match self.center_tile {
                Some(t) => Value::Str(t.to_str()),
                None => Value::Null,
            },
        );
        out.insert(
            "spinner_value".into(),
            match self.spinner_value {
                Some(v) => Value::Int(v as i64),
                None => Value::Null,
            },
        );
        out.insert("spinner_sides_open".into(), Value::Bool(self.spinner_sides_open));

        let mut ends = BTreeMap::<String, Value>::new();
        for e in [End::Right, End::Left, End::Up, End::Down] {
            if let Some(es) = self.ends[e.idx()] {
                ends.insert(
                    e.as_str().into(),
                    Value::List(vec![Value::Int(es.open_value as i64), Value::Bool(es.is_double_end)]),
                );
            }
        }
        out.insert("ends".into(), Value::Map(ends));

        let mut arms = BTreeMap::<String, Value>::new();
        for e in [End::Right, End::Left, End::Up, End::Down] {
            let list = self.arms[e.idx()]
                .iter()
                .map(|t| Value::Str(t.to_str()))
                .collect();
            arms.insert(e.as_str().into(), Value::List(list));
        }
        out.insert("arms".into(), Value::Map(arms));

        out.insert(
            "played_tiles".into(),
            Value::List(self.played_order.iter().map(|t| Value::Str(t.to_str())).collect()),
        );

        out.insert("ends_sum".into(), Value::Int(self.ends_sum() as i64));
        out.insert("current_score".into(), Value::Int(self.score_now() as i64));
        out.insert("is_empty".into(), Value::Bool(self.is_empty()));
        out
    }
}

pub mod serde_like {
    use std::collections::BTreeMap;

    #[derive(Clone, Debug)]
    pub enum Value {
        Null,
        Bool(bool),
        Int(i64),
        Str(String),
        List(Vec<Value>),
        Map(BTreeMap<String, Value>),
    }
}