// FILE: src/mlp.rs | version: 2026-01-14.p4a3 (+Spike head)
// CHANGELOG:
// - RC2: Fix model.json key names to match your Python trainer (W1/Wp/Wv uppercase).
// - RC2: Keep JSON load boundary-only (once per run). Runtime format remains weights.bin v1.
// - P3-A (2026-xx-xx):
//   * Extend MlpModel with optional quantile head (Wq, bq) of size QUANTILES_K.
//   * load_model_json reads optional "Wq" and "bq" from model.json.
//   * New method predict_quantile(...) returns (policy, quantiles[K], mean_value).
// - P4-A3:
//   * Add optional spike head (Ws, bs) of size ACTION_SIZE.
//   * load_model_json reads optional "Ws" and "bs" from model.json.
//   * New method predict_with_spike(...) returns (policy, value, spike[ACTION_SIZE]).

use std::fs;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use crate::features::{ACTION_SIZE, FEAT_DIM};

pub const QUANTILES_K: usize = 11;

#[derive(Clone, Debug)]
pub struct MlpModel {
    pub hidden: usize,
    pub quantiles: usize, // 0 => no quantile head; QUANTILES_K => full head
    pub has_spike: bool,

    // W1: [hidden, FEAT_DIM]
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,

    // Wp: [ACTION_SIZE, hidden]
    pub wp: Vec<f32>,
    pub bp: Vec<f32>,

    // Wv: [hidden] (single row)
    pub wv: Vec<f32>,
    pub bv: f32,

    // Optional quantile head:
    // Wq: [quantiles, hidden] (row-major), bq: [quantiles]
    pub wq: Vec<f32>,
    pub bq: Vec<f32>,

    // Optional spike head:
    // Ws: [ACTION_SIZE, hidden] (row-major), bs: [ACTION_SIZE]
    pub ws: Vec<f32>,
    pub bs: Vec<f32>,
}

impl MlpModel {
    pub fn new_empty(hidden: usize) -> Self {
        Self {
            hidden,
            quantiles: 0,
            has_spike: false,
            w1: vec![0.0; hidden * FEAT_DIM],
            b1: vec![0.0; hidden],
            wp: vec![0.0; ACTION_SIZE * hidden],
            bp: vec![0.0; ACTION_SIZE],
            wv: vec![0.0; hidden],
            bv: 0.0,
            wq: Vec::new(),
            bq: Vec::new(),
            ws: Vec::new(),
            bs: Vec::new(),
        }
    }

    // ----------------------------
    // Loading
    // ----------------------------
    pub fn load_weights_bin(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let mut f = fs::File::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| format!("read {:?}: {e}", path))?;
        Self::from_weights_bin_bytes(&buf)
    }

    pub fn from_weights_bin_bytes(buf: &[u8]) -> Result<Self, String> {
        // Binary format v1:
        // magic "DMLP"(4), u32 ver(1), u32 feat_dim, u32 action_size, u32 hidden,
        // then f32 blocks: W1, b1, Wp, bp, Wv, bv
        if buf.len() < 20 {
            return Err("weights.bin too small".into());
        }
        if &buf[0..4] != b"DMLP" {
            return Err("bad weights.bin magic (expected DMLP)".into());
        }
        let ver = u32_le(&buf[4..8])?;
        if ver != 1 {
            return Err(format!("unsupported weights.bin version: {ver}"));
        }
        let feat = u32_le(&buf[8..12])? as usize;
        let act = u32_le(&buf[12..16])? as usize;
        let hidden = u32_le(&buf[16..20])? as usize;

        if feat != FEAT_DIM {
            return Err(format!("feat_dim mismatch: bin={feat} code={FEAT_DIM}"));
        }
        if act != ACTION_SIZE {
            return Err(format!("action_size mismatch: bin={act} code={ACTION_SIZE}"));
        }
        if hidden == 0 || hidden > 8192 {
            return Err(format!("invalid hidden={hidden}"));
        }

        let mut model = MlpModel::new_empty(hidden);
        let mut off = 20usize;

        off = read_f32_block(buf, off, &mut model.w1)?;
        off = read_f32_block(buf, off, &mut model.b1)?;
        off = read_f32_block(buf, off, &mut model.wp)?;
        off = read_f32_block(buf, off, &mut model.bp)?;
        off = read_f32_block(buf, off, &mut model.wv)?;

        if off + 4 > buf.len() {
            return Err("weights.bin truncated at bv".into());
        }
        model.bv = f32_le(&buf[off..off + 4])?;

        // Binary v1 does not contain quantile head
        model.quantiles = 0;
        model.wq.clear();
        model.bq.clear();
        // Binary v1 does not contain spike head
        model.has_spike = false;
        model.ws.clear();
        model.bs.clear();

        Ok(model)
    }

    pub fn load_model_json(path: impl AsRef<Path>) -> Result<Self, String> {
        // Boundary-only: parse model.json written by your Python trainer (train.py / ai.py).
        let path = path.as_ref();
        let raw = fs::read_to_string(path).map_err(|e| format!("read {:?}: {e}", path))?;
        let d: JsonModel = serde_json::from_str(&raw).map_err(|e| format!("json parse: {e}"))?;

        if d.r#type != "mlp_pv_v1" {
            return Err("incompatible model type".into());
        }
        if d.feat_dim as usize != FEAT_DIM {
            return Err(format!("feat_dim mismatch json={} code={}", d.feat_dim, FEAT_DIM));
        }
        if d.action_size as usize != ACTION_SIZE {
            return Err(format!(
                "action_size mismatch json={} code={}",
                d.action_size, ACTION_SIZE
            ));
        }

        let hidden = d.hidden as usize;

        let w1 = flatten2(&d.W1, hidden, FEAT_DIM, "W1")?;
        let b1 = flatten1(&d.b1, hidden, "b1")?;
        let wp = flatten2(&d.Wp, ACTION_SIZE, hidden, "Wp")?;
        let bp = flatten1(&d.bp, ACTION_SIZE, "bp")?;

        // Wv is [[...]] shape (1, hidden) in your JSON
        let wv_mat = flatten2(&d.Wv, 1, hidden, "Wv")?;
        let wv = wv_mat;

        if d.bv.len() != 1 {
            return Err("bv shape mismatch".into());
        }
        let bv = d.bv[0];

        // Optional quantile head
        let (wq, bq, quantiles) = if let (Some(wq_mat), Some(bq_vec)) = (d.Wq.as_ref(), d.bq.as_ref()) {
            let q = bq_vec.len();
            if q == 0 {
                (Vec::new(), Vec::new(), 0usize)
            } else {
                if q != QUANTILES_K {
                    return Err(format!("quantiles mismatch json={} code={}", q, QUANTILES_K));
                }
                let wq_flat = flatten2(wq_mat, q, hidden, "Wq")?;
                (wq_flat, bq_vec.clone(), q)
            }
        } else {
            (Vec::new(), Vec::new(), 0usize)
        };

        // Optional spike head
        let (ws, bs, has_spike) = if let (Some(ws_mat), Some(bs_vec)) = (d.Ws.as_ref(), d.bs.as_ref()) {
            if bs_vec.len() != ACTION_SIZE {
                return Err(format!("bs shape mismatch: got {} expected {}", bs_vec.len(), ACTION_SIZE));
            }
            let ws_flat = flatten2(ws_mat, ACTION_SIZE, hidden, "Ws")?;
            (ws_flat, bs_vec.clone(), true)
        } else {
            (Vec::new(), Vec::new(), false)
        };

        Ok(Self {
            hidden,
            quantiles,
            has_spike,
            w1,
            b1,
            wp,
            bp,
            wv,
            bv,
            wq,
            bq,
            ws,
            bs,
        })
    }

    /// Write binary weights v1 (preferred runtime format).
    /// NOTE: quantile head is NOT stored in v1; only scalar value head.
    pub fn save_weights_bin(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"DMLP");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(FEAT_DIM as u32).to_le_bytes());
        out.extend_from_slice(&(ACTION_SIZE as u32).to_le_bytes());
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());

        write_f32s(&mut out, &self.w1);
        write_f32s(&mut out, &self.b1);
        write_f32s(&mut out, &self.wp);
        write_f32s(&mut out, &self.bp);
        write_f32s(&mut out, &self.wv);
        out.extend_from_slice(&self.bv.to_le_bytes());

        fs::write(path, out).map_err(|e| format!("write {:?}: {e}", path))
    }

    // ----------------------------
    // Inference
    // ----------------------------
    pub fn predict(
        &self,
        feat: &[f32; FEAT_DIM],
        mask: &[i8; ACTION_SIZE],
    ) -> ([f32; ACTION_SIZE], f32) {
        let mut h = vec![0f32; self.hidden];

        // z1 = W1*x + b1, h = relu(z1)
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            let row_off = i * FEAT_DIM;
            for j in 0..FEAT_DIM {
                sum += self.w1[row_off + j] * feat[j];
            }
            h[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // logits = Wp*h + bp, masked
        let mut logits = vec![0f32; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            let mut s = self.bp[a];
            let row_off = a * self.hidden;
            for i in 0..self.hidden {
                s += self.wp[row_off + i] * h[i];
            }
            if mask[a] == 0 {
                s = -1.0e9;
            }
            logits[a] = s;
        }

        // stable softmax
        let mut maxv = logits[0];
        for &v in logits.iter() {
            if v > maxv {
                maxv = v;
            }
        }
        let mut expsum: f64 = 0.0;
        let mut exps = vec![0f64; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            let e = ((logits[a] as f64) - (maxv as f64)).exp();
            exps[a] = e;
            expsum += e;
        }
        let denom = (expsum + 1e-12) as f32;

        let mut policy = [0f32; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            policy[a] = (exps[a] as f32) / denom;
        }

        // value = tanh(h·wv + bv)
        let mut vr = self.bv;
        for i in 0..self.hidden {
            vr += self.wv[i] * h[i];
        }
        let value = vr.tanh();

        (policy, value)
    }

    /// Value-only inference (NO policy head, NO softmax) using caller-provided scratch buffer.
    /// This is the safe way to enable leaf-value inside MCTS without destroying throughput.
    pub fn predict_value_only(&self, feat: &[f32; FEAT_DIM], h: &mut [f32]) -> f32 {
        debug_assert_eq!(h.len(), self.hidden);

        // hidden = relu(W1*x + b1)
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            let row_off = i * FEAT_DIM;
            for j in 0..FEAT_DIM {
                sum += self.w1[row_off + j] * feat[j];
            }
            h[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // value = tanh(h·wv + bv)
        let mut vr = self.bv;
        for i in 0..self.hidden {
            vr += self.wv[i] * h[i];
        }
        vr.tanh()
    }

    /// Policy + quantile inference:
    ///  - policy: [ACTION_SIZE]
    ///  - quantiles: Vec<f32> of length QUANTILES_K
    ///  - mean_value: scalar in [-1,+1] (average of quantiles or scalar value fallback)
    pub fn predict_quantile(
        &self,
        feat: &[f32; FEAT_DIM],
        mask: &[i8; ACTION_SIZE],
    ) -> ([f32; ACTION_SIZE], Vec<f32>, f32) {
        let mut h = vec![0f32; self.hidden];

        // hidden
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            let row_off = i * FEAT_DIM;
            for j in 0..FEAT_DIM {
                sum += self.w1[row_off + j] * feat[j];
            }
            h[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // policy (same as predict)
        let mut logits = vec![0f32; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            let mut s = self.bp[a];
            let row_off = a * self.hidden;
            for i in 0..self.hidden {
                s += self.wp[row_off + i] * h[i];
            }
            if mask[a] == 0 {
                s = -1.0e9;
            }
            logits[a] = s;
        }

        let mut maxv = logits[0];
        for &v in logits.iter() {
            if v > maxv {
                maxv = v;
            }
        }
        let mut expsum: f64 = 0.0;
        let mut exps = vec![0f64; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            let e = ((logits[a] as f64) - (maxv as f64)).exp();
            exps[a] = e;
            expsum += e;
        }
        let denom = (expsum + 1e-12) as f32;

        let mut policy = [0f32; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            policy[a] = (exps[a] as f32) / denom;
        }

        // quantiles
        let mut q_vec = vec![0.0f32; QUANTILES_K];
        let mut mean: f32;

        if self.quantiles == QUANTILES_K && self.wq.len() == QUANTILES_K * self.hidden && self.bq.len() == QUANTILES_K {
            // use learned quantile head
            for k in 0..QUANTILES_K {
                let mut s = self.bq[k];
                let row_off = k * self.hidden;
                for i in 0..self.hidden {
                    s += self.wq[row_off + i] * h[i];
                }
                q_vec[k] = s;
            }
            let sum_q: f32 = q_vec.iter().copied().sum();
            mean = (sum_q / (QUANTILES_K as f32)).clamp(-1.0, 1.0);
        } else {
            // fallback: compute scalar value and replicate it
            let mut vr = self.bv;
            for i in 0..self.hidden {
                vr += self.wv[i] * h[i];
            }
            let v = vr.tanh();
            mean = v;
            for k in 0..QUANTILES_K {
                q_vec[k] = v;
            }
        }

        (policy, q_vec, mean)
    }

    /// Policy + value + spike inference:
    /// spike[a] in [0,1] where 1 means "high risk" (interpreted by caller).
    pub fn predict_with_spike(
        &self,
        feat: &[f32; FEAT_DIM],
        mask: &[i8; ACTION_SIZE],
    ) -> ([f32; ACTION_SIZE], f32, [f32; ACTION_SIZE]) {
        // hidden
        let mut h = vec![0f32; self.hidden];
        for i in 0..self.hidden {
            let mut sum = self.b1[i];
            let row_off = i * FEAT_DIM;
            for j in 0..FEAT_DIM {
                sum += self.w1[row_off + j] * feat[j];
            }
            h[i] = if sum > 0.0 { sum } else { 0.0 };
        }

        // policy logits
        let mut logits = vec![0f32; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            let mut s = self.bp[a];
            let row_off = a * self.hidden;
            for i in 0..self.hidden {
                s += self.wp[row_off + i] * h[i];
            }
            if mask[a] == 0 {
                s = -1.0e9;
            }
            logits[a] = s;
        }

        // stable softmax
        let mut maxv = logits[0];
        for &v in logits.iter() {
            if v > maxv { maxv = v; }
        }
        let mut expsum: f64 = 0.0;
        let mut exps = vec![0f64; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            let e = ((logits[a] as f64) - (maxv as f64)).exp();
            exps[a] = e;
            expsum += e;
        }
        let denom = (expsum + 1e-12) as f32;
        let mut policy = [0f32; ACTION_SIZE];
        for a in 0..ACTION_SIZE {
            policy[a] = (exps[a] as f32) / denom;
        }

        // scalar value
        let mut vr = self.bv;
        for i in 0..self.hidden {
            vr += self.wv[i] * h[i];
        }
        let value = vr.tanh();

        // spike
        let mut spike = [0f32; ACTION_SIZE];
        if self.has_spike && self.ws.len() == ACTION_SIZE * self.hidden && self.bs.len() == ACTION_SIZE {
            for a in 0..ACTION_SIZE {
                let mut s = self.bs[a];
                let row_off = a * self.hidden;
                for i in 0..self.hidden {
                    s += self.ws[row_off + i] * h[i];
                }
                // sigmoid
                spike[a] = 1.0 / (1.0 + (-s).exp());
            }
        } else {
            // no spike head => all zeros
        }

        (policy, value, spike)
    }
}

// ----------------------------
// JSON schema (matches your Python model.json keys)
// ----------------------------
#[allow(non_snake_case)]
#[derive(Deserialize)]
struct JsonModel {
    #[serde(rename = "type")]
    r#type: String,
    feat_dim: u32,
    action_size: u32,
    hidden: u32,

    #[serde(rename = "W1")]
    W1: Vec<Vec<f32>>,
    b1: Vec<f32>,

    #[serde(rename = "Wp")]
    Wp: Vec<Vec<f32>>,
    bp: Vec<f32>,

    #[serde(rename = "Wv")]
    Wv: Vec<Vec<f32>>,
    bv: Vec<f32>,

    // Optional quantile head
    #[serde(rename = "Wq")]
    Wq: Option<Vec<Vec<f32>>>,
    bq: Option<Vec<f32>>,

    // Optional spike head
    #[serde(rename = "Ws")]
    Ws: Option<Vec<Vec<f32>>>,
    #[serde(rename = "bs")]
    bs: Option<Vec<f32>>,
}

fn flatten2(v: &[Vec<f32>], rows: usize, cols: usize, name: &str) -> Result<Vec<f32>, String> {
    if v.len() != rows {
        return Err(format!("{name} rows mismatch"));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        if v[r].len() != cols {
            return Err(format!("{name} cols mismatch at r={r}"));
        }
        out.extend_from_slice(&v[r]);
    }
    Ok(out)
}
fn flatten1(v: &[f32], n: usize, name: &str) -> Result<Vec<f32>, String> {
    if v.len() != n {
        return Err(format!("{name} len mismatch: got {} expected {}", v.len(), n));
    }
    Ok(v.to_vec())
}

fn u32_le(b: &[u8]) -> Result<u32, String> {
    if b.len() < 4 {
        return Err("u32_le: truncated".into());
    }
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn f32_le(b: &[u8]) -> Result<f32, String> {
    if b.len() < 4 {
        return Err("f32_le: truncated".into());
    }
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn read_f32_block(buf: &[u8], mut off: usize, out: &mut [f32]) -> Result<usize, String> {
    let need = out.len() * 4;
    if off + need > buf.len() {
        return Err("weights.bin truncated".into());
    }
    for i in 0..out.len() {
        out[i] = f32_le(&buf[off..off + 4])?;
        off += 4;
    }
    Ok(off)
}
fn write_f32s(dst: &mut Vec<u8>, xs: &[f32]) {
    for &x in xs {
        dst.extend_from_slice(&x.to_le_bytes());
    }
}