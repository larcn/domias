// FILE: src/shards.rs | version: 2026-01-14.p4a1 (Spike targets v2)
// CHANGELOG:
// - RC3: Add Serialize, Deserialize to ShardInfo for serde compatibility.
// - RC2: Fix invalid Rust pattern "(s or '')" by implementing safe Codec::from_str.
// - RC2: Remove unnecessary/unsafe transmute; use zstd Encoder<'static, BufWriter<File>> directly.
// - RC2: Keep record format stable: header + repeated fixed-size records (feat, pi, z, mask).
// - P4-A1: Replace q[11] with spike[action] targets (win-now-exists risk, per action).
//   Record layout now (DSH2):
//     feat : FEAT_DIM * f32
//     pi   : ACTION_SIZE * f32
//     spike: ACTION_SIZE * i8     // NEW (0/1), per action
//     z    : f32
//     mask : ACTION_SIZE * i8

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Serialize, Deserialize};
use zstd::stream::write::Encoder;

use crate::features::{ACTION_SIZE, FEAT_DIM};

pub const SHARD_MAGIC: [u8; 4] = *b"DSH2";
pub const SHARD_HEADER_LEN: usize = 16;

/// Record layout (little-endian):
/// - feat: FEAT_DIM * f32
/// - pi  : ACTION_SIZE * f32
/// - spike: ACTION_SIZE * i8
/// - z   : f32
/// - mask: ACTION_SIZE * i8
pub const RECORD_SIZE: usize =
    (FEAT_DIM * 4) +
    (ACTION_SIZE * 4) +
    (ACTION_SIZE * 1) +
    4 +
    (ACTION_SIZE * 1);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Codec {
    Raw,
    Zstd,
}

impl Codec {
    pub fn from_str(s: &str) -> Codec {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "zstd" | "zst" | "z" => Codec::Zstd,
            _ => Codec::Raw,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Codec::Raw => "raw",
            Codec::Zstd => "zstd",
        }
    }

    /// File extension for a shard payload.
    pub fn ext(self) -> &'static str {
        match self {
            Codec::Raw => "dsh",
            Codec::Zstd => "dsh.zst",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    pub filename: String,
    pub codec: String,
    pub samples: u32,
    pub bytes_on_disk: u64,
}

pub struct ShardWriter {
    out_dir: PathBuf,
    run_id: String,
    codec: Codec,
    zstd_level: i32,

    shard_index: u32,
    samples_in_shard: u32,
    total_samples: u64,

    sink: Option<ShardSink>,
    shards: Vec<ShardInfo>,
}

struct ShardSink {
    path: PathBuf,
    inner: SinkInner,
}

enum SinkInner {
    Raw(BufWriter<File>),
    Zstd(Encoder<'static, BufWriter<File>>),
}

impl ShardWriter {
    pub fn new(
        out_dir: impl AsRef<Path>,
        run_id: &str,
        codec: Codec,
        zstd_level: i32,
    ) -> Result<Self, String> {
        let out_dir = out_dir.as_ref().to_path_buf();
        fs::create_dir_all(&out_dir).map_err(|e| format!("create_dir_all {:?}: {e}", out_dir))?;

        Ok(Self {
            out_dir,
            run_id: run_id.to_string(),
            codec,
            zstd_level,
            shard_index: 0,
            samples_in_shard: 0,
            total_samples: 0,
            sink: None,
            shards: Vec::new(),
        })
    }

    pub fn shards(&self) -> &[ShardInfo] {
        &self.shards
    }

    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    pub fn samples_in_current_shard(&self) -> u32 {
        self.samples_in_shard
    }

    pub fn start_new_shard(&mut self) -> Result<(), String> {
        // finish any open shard first
        self.finish_shard()?;

        let fname = format!(
            "{}.shard_{:05}.{}",
            self.run_id,
            self.shard_index,
            self.codec.ext()
        );
        let path = self.out_dir.join(&fname);

        let file = File::create(&path).map_err(|e| format!("create {:?}: {e}", path))?;
        let bw = BufWriter::new(file);

        let inner = match self.codec {
            Codec::Raw => SinkInner::Raw(bw),
            Codec::Zstd => {
                let enc = Encoder::new(bw, self.zstd_level)
                    .map_err(|e| format!("zstd encoder init (level={}): {e}", self.zstd_level))?;
                SinkInner::Zstd(enc)
            }
        };

        let mut sink = ShardSink { path, inner };

        // header is written INSIDE the (possibly compressed) stream:
        // magic(4), feat_dim(u32), action_size(u32), record_size(u32)
        let mut header = [0u8; SHARD_HEADER_LEN];
        header[0..4].copy_from_slice(&SHARD_MAGIC);
        header[4..8].copy_from_slice(&(FEAT_DIM as u32).to_le_bytes());
        header[8..12].copy_from_slice(&(ACTION_SIZE as u32).to_le_bytes());
        header[12..16].copy_from_slice(&(RECORD_SIZE as u32).to_le_bytes());
        sink_write_all(&mut sink.inner, &header)?;

        self.sink = Some(sink);
        self.samples_in_shard = 0;
        Ok(())
    }

    pub fn write_sample(
        &mut self,
        feat: &[f32; FEAT_DIM],
        pi: &[f32; ACTION_SIZE],
        spike: &[i8; ACTION_SIZE],
        z: f32,
        mask: &[i8; ACTION_SIZE],
    ) -> Result<(), String> {
        if self.sink.is_none() {
            self.start_new_shard()?;
        }
        let sink = self.sink.as_mut().unwrap();

        // record bytes
        sink_write_all(&mut sink.inner, as_bytes_f32(feat))?;
        sink_write_all(&mut sink.inner, as_bytes_f32(pi))?;
        sink_write_all(&mut sink.inner, as_bytes_i8(spike))?;
        sink_write_all(&mut sink.inner, &z.to_le_bytes())?;
        sink_write_all(&mut sink.inner, as_bytes_i8(mask))?;

        self.samples_in_shard += 1;
        self.total_samples += 1;
        Ok(())
    }

    pub fn finish_shard(&mut self) -> Result<(), String> {
        let Some(sink) = self.sink.take() else {
            return Ok(());
        };

        let samples = self.samples_in_shard;

        // finalize stream
        match sink.inner {
            SinkInner::Raw(mut w) => {
                w.flush().map_err(|e| format!("flush raw shard: {e}"))?;
            }
            SinkInner::Zstd(enc) => {
                let mut w = enc.finish().map_err(|e| format!("zstd finish: {e}"))?;
                w.flush().map_err(|e| format!("flush zstd shard: {e}"))?;
            }
        }

        let meta = fs::metadata(&sink.path).map_err(|e| format!("metadata {:?}: {e}", sink.path))?;
        let bytes_on_disk = meta.len();

        let filename = sink
            .path
            .file_name()
            .ok_or_else(|| "shard filename missing".to_string())?
            .to_string_lossy()
            .to_string();

        self.shards.push(ShardInfo {
            filename,
            codec: self.codec.as_str().to_string(),
            samples,
            bytes_on_disk,
        });

        self.shard_index += 1;
        self.samples_in_shard = 0;
        Ok(())
    }

    pub fn finish_all(mut self) -> Result<Vec<ShardInfo>, String> {
        self.finish_shard()?;
        Ok(self.shards)
    }
}

fn sink_write_all(inner: &mut SinkInner, bytes: &[u8]) -> Result<(), String> {
    match inner {
        SinkInner::Raw(w) => w.write_all(bytes).map_err(|e| format!("write raw: {e}")),
        SinkInner::Zstd(w) => w.write_all(bytes).map_err(|e| format!("write zstd: {e}")),
    }
}

fn as_bytes_f32(x: &[f32]) -> &[u8] {
    // Safe for POD f32 slice; no ownership escape; used only for writing.
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn as_bytes_i8(x: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len()) }
}