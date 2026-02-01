// FILE: src/tile.rs | version: 2026-01-03.rc2
// CHANGELOG:
// - RC2: Fix Rust compile error in Tile::parse (removed invalid "(s or '')" pattern).
// - RC2: Keep parsing boundary-only and Python-compatible normalization (hi>=lo).

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Tile(pub u8); // 0..27

impl Tile {
    #[inline]
    pub fn id(self) -> u8 {
        self.0
    }

    #[inline]
    pub fn is_double(self) -> bool {
        let (a, b) = self.pips();
        a == b
    }

    #[inline]
    pub fn pip_sum(self) -> u8 {
        let (a, b) = self.pips();
        a + b
    }

    #[inline]
    pub fn has(self, v: u8) -> bool {
        let (a, b) = self.pips();
        a == v || b == v
    }

    #[inline]
    pub fn other_value(self, v: u8) -> Result<u8, String> {
        let (a, b) = self.pips();
        if a == v {
            return Ok(b);
        }
        if b == v {
            return Ok(a);
        }
        Err(format!("tile {} does not contain {}", self.to_str(), v))
    }

    #[inline]
    pub fn pips(self) -> (u8, u8) {
        let id = self.0 as usize;
        (TILE_HI[id], TILE_LO[id])
    }

    pub fn to_str(self) -> String {
        let (a, b) = self.pips();
        format!("{}-{}", a, b)
    }

    pub fn from_pips(hi: u8, lo: u8) -> Result<Tile, String> {
        let (a, b) = norm_pips(hi, lo)?;
        Ok(Tile(pips_to_id(a, b)))
    }

    /// Boundary-only parser (NOT for hot paths).
    /// Accepts: "6-5", "65", "5|6", "[6-5]" ... and normalizes hi>=lo.
    pub fn parse(s: &str) -> Result<Tile, String> {
        let mut t = s.trim().to_string();
        t = t.replace('[', "")
            .replace(']', "")
            .replace(' ', "")
            .replace('|', "-")
            .replace(',', "-");

        if let Some(idx) = t.find('-') {
            let (a, bpart) = t.split_at(idx);
            let b = &bpart[1..];
            let hi: u8 = a.parse().map_err(|_| format!("bad tile: {s}"))?;
            let lo: u8 = b.parse().map_err(|_| format!("bad tile: {s}"))?;
            return Tile::from_pips(hi, lo);
        }

        if t.len() == 2 && t.as_bytes().iter().all(|c| c.is_ascii_digit()) {
            let bytes = t.as_bytes();
            let hi = (bytes[0] - b'0') as u8;
            let lo = (bytes[1] - b'0') as u8;
            return Tile::from_pips(hi, lo);
        }

        Err(format!("cannot parse tile: {s}"))
    }
}

#[inline]
fn norm_pips(a: u8, b: u8) -> Result<(u8, u8), String> {
    if a > 6 || b > 6 {
        return Err(format!("tile out of range: {}-{}", a, b));
    }
    Ok(if a >= b { (a, b) } else { (b, a) })
}

#[inline]
fn pips_to_id(hi: u8, lo: u8) -> u8 {
    let start = (hi as u16 * (hi as u16 + 1)) / 2;
    (start + lo as u16) as u8
}

const TILE_HI: [u8; 28] = [
    0,
    1, 1,
    2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6,
];

const TILE_LO: [u8; 28] = [
    0,
    0, 1,
    0, 1, 2,
    0, 1, 2, 3,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 5,
    0, 1, 2, 3, 4, 5, 6,
];