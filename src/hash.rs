// FILE: src/hash.rs | version: 2026-01-03.rc2
// CHANGELOG:
// - RC2: Fix invalid hex literal seed (was non-compiling).
// - RC2: Keep hot-path safe hashing (NO JSON, NO Strings).

use crate::board::{Board, End};
use crate::tile::Tile;

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
fn mix(h: &mut u64, x: u64) {
    *h = splitmix64(*h ^ x);
}

#[inline]
pub fn hand_mask(hand: &[Tile]) -> u32 {
    let mut m: u32 = 0;
    for t in hand.iter().copied() {
        m |= 1u32 << (t.id() as u32);
    }
    m
}

#[inline]
pub fn board_hash64(b: &Board) -> u64 {
    // pure numeric seed
    let mut h: u64 = 0xD011_005E_EE0D_1234u64;

    mix(&mut h, b.played_mask() as u64);
    mix(&mut h, match b.center_tile { Some(t) => t.id() as u64, None => 255 });
    mix(&mut h, match b.spinner_value { Some(v) => v as u64, None => 255 });
    mix(&mut h, if b.spinner_sides_open { 1 } else { 0 });

    for e in [End::Right, End::Left, End::Up, End::Down] {
        if let Some(es) = b.ends_raw()[e.idx()] {
            let x = (es.open_value as u64)
                | ((es.is_double_end as u64) << 7)
                | ((e.idx() as u64) << 8);
            mix(&mut h, x);
        } else {
            mix(&mut h, 0xFF00u64 | (e.idx() as u64));
        }
    }

    for e in [End::Right, End::Left, End::Up, End::Down] {
        mix(&mut h, 0xA5A5_0000u64 | (e.idx() as u64));
        for t in b.arms_raw()[e.idx()].iter().copied() {
            mix(&mut h, t.id() as u64);
        }
        mix(&mut h, 0x5A5A_0000u64 | (b.arms_raw()[e.idx()].len() as u64));
    }

    h
}

#[inline]
pub fn state_hash64(
    board_h: u64,
    my_hand_mask: u32,
    opp_cnt: i32,
    bone_cnt: i32,
    forced: Option<Tile>,
    turn_me: bool,
) -> u64 {
    let mut h = board_h;
    mix(&mut h, my_hand_mask as u64);
    mix(&mut h, (opp_cnt as i64 as u64) ^ ((bone_cnt as i64 as u64) << 32));
    mix(&mut h, match forced { Some(t) => t.id() as u64, None => 255 });
    mix(&mut h, if turn_me { 1 } else { 2 });
    h
}