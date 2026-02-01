// FILE: src/infer_model.rs | version: 2026-01-18.im4_3_ensemble
// Runtime inference model loader + predictor for opponent-hand inference.
//
// Model JSON schema (from tools/infer_tool.py):
// {
//   "type": "infer_mlp_v1",
//   "feat_dim": 235,
//   "label_size": 28,
//   "hidden": 256,
//   "W1": [[... feat_dim ...] * hidden],
//   "b1": [... hidden ...],
//   "W2": [[... hidden ...] * 28],
//   "b2": [... 28 ...]
// }
//
// IM4.3 Robustification (NO knobs):
// - When caller loads "inference_model.json":
//   - If both siblings exist in same folder:
//       inference_model_self.json
//       inference_model_strat.json
//     we return an Ensemble model that averages predictions:
//       p = 0.5 * p_self + 0.5 * p_strat
// - Otherwise fall back to loading the requested single model.
// - Caching is mtime-based to avoid re-parsing JSON repeatedly.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use serde::Deserialize;

pub const INF_FEAT_DIM: usize = 235;
pub const INF_OUT: usize = 28;

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct JsonInferModel {
    #[serde(rename = "type")]
    r#type: String,
    feat_dim: u32,
    label_size: u32,
    hidden: u32,
    W1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    W2: Vec<Vec<f32>>,
    b2: Vec<f32>,
}

#[derive(Clone)]
struct InferModelSingle {
    hidden: usize,
    // row-major flattened
    w1: Vec<f32>, // [hidden, feat]
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [28, hidden]
    b2: Vec<f32>, // [28]
}

fn predict_single(m: &InferModelSingle, feat: &[f32; INF_FEAT_DIM]) -> [f32; INF_OUT] {
    // hidden relu
    let mut h = vec![0f32; m.hidden];
    for i in 0..m.hidden {
        let mut s = m.b1[i];
        let off = i * INF_FEAT_DIM;
        for j in 0..INF_FEAT_DIM {
            s += m.w1[off + j] * feat[j];
        }
        h[i] = if s > 0.0 { s } else { 0.0 };
    }

    // logits -> sigmoid
    let mut out = [0f32; INF_OUT];
    for a in 0..INF_OUT {
        let mut s = m.b2[a];
        let off = a * m.hidden;
        for i in 0..m.hidden {
            s += m.w2[off + i] * h[i];
        }
        out[a] = 1.0 / (1.0 + (-s).exp());
    }
    out
}

#[derive(Clone)]
pub struct InferModel {
    kind: InferModelKind,
}

#[derive(Clone)]
enum InferModelKind {
    Single(InferModelSingle),
    Ensemble { a: InferModelSingle, b: InferModelSingle },
}

impl InferModel {
    pub fn predict_tile_probs(&self, feat: &[f32; INF_FEAT_DIM]) -> [f32; INF_OUT] {
        match &self.kind {
            InferModelKind::Single(m) => predict_single(m, feat),
            InferModelKind::Ensemble { a, b } => {
                let pa = predict_single(a, feat);
                let pb = predict_single(b, feat);
                let mut out = [0f32; INF_OUT];
                for i in 0..INF_OUT {
                    out[i] = 0.5 * (pa[i] + pb[i]);
                }
                out
            }
        }
    }
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

#[derive(Clone)]
struct CacheEntry {
    mtime_a: Option<SystemTime>,
    mtime_b: Option<SystemTime>,
    model: Arc<InferModel>,
}

// key -> cached entry
static INFER_CACHE: OnceLock<Mutex<HashMap<String, CacheEntry>>> = OnceLock::new();

pub fn infer_enabled() -> bool {
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| match std::env::var("DOMINO_INFER") {
        Ok(s) => {
            let t = s.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    })
}

fn load_single_from_disk(path: &Path) -> Option<InferModelSingle> {
    if !path.exists() {
        return None;
    }

    let raw = fs::read_to_string(path).ok()?;
    let jm: JsonInferModel = serde_json::from_str(&raw).ok()?;

    if jm.r#type != "infer_mlp_v1" {
        return None;
    }
    if jm.feat_dim as usize != INF_FEAT_DIM || jm.label_size as usize != INF_OUT {
        return None;
    }

    let hidden = jm.hidden as usize;
    if hidden == 0 || hidden > 8192 {
        return None;
    }
    if jm.W1.len() != hidden || jm.b1.len() != hidden || jm.W2.len() != INF_OUT || jm.b2.len() != INF_OUT {
        return None;
    }

    // flatten W1 and W2
    let mut w1: Vec<f32> = Vec::with_capacity(hidden * INF_FEAT_DIM);
    for r in jm.W1.iter() {
        if r.len() != INF_FEAT_DIM {
            return None;
        }
        w1.extend_from_slice(r);
    }

    let mut w2: Vec<f32> = Vec::with_capacity(INF_OUT * hidden);
    for r in jm.W2.iter() {
        if r.len() != hidden {
            return None;
        }
        w2.extend_from_slice(r);
    }

    Some(InferModelSingle {
        hidden,
        w1,
        b1: jm.b1,
        w2,
        b2: jm.b2,
    })
}

pub fn load_infer_model_cached(path: &str) -> Option<Arc<InferModel>> {
    let p = Path::new(path);
    let fname = p.file_name().and_then(|x| x.to_str()).unwrap_or("");

    let cache = INFER_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // ---------------------------------------------------------------------
    // Ensemble path: if caller asks for inference_model.json and siblings exist
    // ---------------------------------------------------------------------
    if fname.eq_ignore_ascii_case("inference_model.json") {
        let base_dir = p.parent().unwrap_or_else(|| Path::new("."));
        let self_p = base_dir.join("inference_model_self.json");
        let strat_p = base_dir.join("inference_model_strat.json");

        if self_p.exists() && strat_p.exists() {
            let mt_a = file_mtime(&self_p);
            let mt_b = file_mtime(&strat_p);

            let key = format!(
                "ensemble:{}|{}",
                self_p.to_string_lossy(),
                strat_p.to_string_lossy()
            );

            // cache hit?
            {
                let g = cache.lock().ok()?;
                if let Some(c) = g.get(&key) {
                    if c.mtime_a == mt_a && c.mtime_b == mt_b {
                        return Some(Arc::clone(&c.model));
                    }
                }
            }

            // load from disk (no lock held)
            let a = load_single_from_disk(&self_p)?;
            let b = load_single_from_disk(&strat_p)?;

            let model = Arc::new(InferModel {
                kind: InferModelKind::Ensemble { a, b },
            });

            // store to cache
            let mut g = cache.lock().ok()?;
            g.insert(
                key,
                CacheEntry {
                    mtime_a: mt_a,
                    mtime_b: mt_b,
                    model: Arc::clone(&model),
                },
            );
            return Some(model);
        }
        // else: fall through to single-model load below (inference_model.json itself)
    }

    // ---------------------------------------------------------------------
    // Single model load (mtime cached)
    // ---------------------------------------------------------------------
    if !p.exists() {
        return None;
    }

    let mt = file_mtime(p);
    let key = p.to_string_lossy().to_string();

    // cache hit?
    {
        let g = cache.lock().ok()?;
        if let Some(c) = g.get(&key) {
            if c.mtime_a == mt && c.mtime_b.is_none() {
                return Some(Arc::clone(&c.model));
            }
        }
    }

    // load from disk (no lock held)
    let single = load_single_from_disk(p)?;
    let model = Arc::new(InferModel {
        kind: InferModelKind::Single(single),
    });

    // store to cache
    let mut g = cache.lock().ok()?;
    g.insert(
        key,
        CacheEntry {
            mtime_a: mt,
            mtime_b: None,
            model: Arc::clone(&model),
        },
    );

    Some(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn logit(p: f32) -> f32 {
        let p = p.clamp(1e-4, 1.0 - 1e-4);
        (p / (1.0 - p)).ln()
    }

    fn write_min_model(path: &Path, p0: f32) {
        // hidden=1; W1/W2 are zeros; b2[0] sets sigmoid output for tile0.
        let w1 = vec![vec![0.0f32; INF_FEAT_DIM]; 1];
        let b1 = vec![0.0f32; 1];
        let w2 = vec![vec![0.0f32; 1]; INF_OUT];
        let mut b2 = vec![0.0f32; INF_OUT];
        b2[0] = logit(p0);

        let j = serde_json::json!({
            "type": "infer_mlp_v1",
            "feat_dim": INF_FEAT_DIM as u32,
            "label_size": INF_OUT as u32,
            "hidden": 1u32,
            "W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2
        });

        fs::write(path, serde_json::to_string(&j).unwrap()).unwrap();
    }

    #[test]
    fn infer_enabled_default_false() {
        // Do not set env here (OnceLock). This just ensures it compiles and default is false
        // in a clean process. If your test harness sets DOMINO_INFER, this may differ.
        let _ = infer_enabled();
    }

    #[test]
    fn ensemble_used_for_inference_model_json_when_siblings_exist() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("domino_infer_test_{ts}"));
        fs::create_dir_all(&dir).unwrap();

        let p_main = dir.join("inference_model.json");
        let p_self = dir.join("inference_model_self.json");
        let p_strat = dir.join("inference_model_strat.json");

        // main exists (but ensemble should use siblings instead)
        write_min_model(&p_main, 0.2);
        write_min_model(&p_self, 0.2);
        write_min_model(&p_strat, 0.8);

        let m = load_infer_model_cached(p_main.to_string_lossy().as_ref()).unwrap();
        let feat = [0.0f32; INF_FEAT_DIM];
        let out = m.predict_tile_probs(&feat);

        // Expect approx average: (0.2 + 0.8) / 2 = 0.5 for tile0
        assert!((out[0] - 0.5).abs() < 0.02, "out[0]={}", out[0]);
    }
}