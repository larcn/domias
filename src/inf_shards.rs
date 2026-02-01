// FILE: src/inf_shards.rs | version: 2026-01-17.im1_inf1
// Sidecar shard format for Opponent Hand Inference training (INF1).
//
// This does NOT modify DSH2. It writes separate files alongside normal shards.
//
// INF1 record layout (little-endian):
//   inf_feat : INF_FEAT_DIM * f32
//   label    : 28 * i8          (opp hand truth mask, 0/1)
//   opp_cnt  : u8
//   bone_cnt : u8
//
// Header (16 bytes):
//   magic(4)="INF1"
//   feat_dim(u32)=INF_FEAT_DIM
//   label_size(u32)=28
//   record_size(u32)=INF_RECORD_SIZE

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use zstd::stream::write::Encoder;

use crate::shards::{Codec, ShardInfo};

pub const INF_MAGIC: [u8; 4] = *b"INF1";
pub const INF_HEADER_LEN: usize = 16;

pub const INF_LABEL_SIZE: usize = 28;

// MVP inference feature dim:
//  - base features_193
//  - opp_played_mask28
//  - endchoice_prefer_values7
//  - endchoice_avoid_values7
pub const INF_FEAT_DIM: usize = 193 + 28 + 7 + 7; // 235

pub const INF_RECORD_SIZE: usize = (INF_FEAT_DIM * 4) + (INF_LABEL_SIZE * 1) + 2;

pub struct InfShardWriter {
    out_dir: PathBuf,
    run_id: String,
    codec: Codec,
    zstd_level: i32,

    shard_index: u32,
    samples_in_shard: u32,
    total_samples: u64,

    sink: Option<InfSink>,
    shards: Vec<ShardInfo>,
}

struct InfSink {
    path: PathBuf,
    inner: InfSinkInner,
}

enum InfSinkInner {
    Raw(BufWriter<File>),
    Zstd(Encoder<'static, BufWriter<File>>),
}

impl InfShardWriter {
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

    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    pub fn start_new_shard(&mut self) -> Result<(), String> {
        self.finish_shard()?;

        let fname = format!(
            "{}.infer_{:05}.{}",
            self.run_id,
            self.shard_index,
            self.ext()
        );
        let path = self.out_dir.join(&fname);

        let file = File::create(&path).map_err(|e| format!("create {:?}: {e}", path))?;
        let bw = BufWriter::new(file);

        let inner = match self.codec {
            Codec::Raw => InfSinkInner::Raw(bw),
            Codec::Zstd => {
                let enc = Encoder::new(bw, self.zstd_level)
                    .map_err(|e| format!("zstd encoder init (level={}): {e}", self.zstd_level))?;
                InfSinkInner::Zstd(enc)
            }
        };

        let mut sink = InfSink { path, inner };

        // header inside stream
        let mut header = [0u8; INF_HEADER_LEN];
        header[0..4].copy_from_slice(&INF_MAGIC);
        header[4..8].copy_from_slice(&(INF_FEAT_DIM as u32).to_le_bytes());
        header[8..12].copy_from_slice(&(INF_LABEL_SIZE as u32).to_le_bytes());
        header[12..16].copy_from_slice(&(INF_RECORD_SIZE as u32).to_le_bytes());
        inf_write_all(&mut sink.inner, &header)?;

        self.sink = Some(sink);
        self.samples_in_shard = 0;
        Ok(())
    }

    pub fn write_sample(
        &mut self,
        inf_feat: &[f32; INF_FEAT_DIM],
        label_opp_hand: &[i8; INF_LABEL_SIZE],
        opp_cnt: u8,
        bone_cnt: u8,
    ) -> Result<(), String> {
        if self.sink.is_none() {
            self.start_new_shard()?;
        }
        let sink = self.sink.as_mut().unwrap();

        inf_write_all(&mut sink.inner, as_bytes_f32(inf_feat))?;
        inf_write_all(&mut sink.inner, as_bytes_i8(label_opp_hand))?;
        inf_write_all(&mut sink.inner, &[opp_cnt])?;
        inf_write_all(&mut sink.inner, &[bone_cnt])?;

        self.samples_in_shard += 1;
        self.total_samples += 1;
        Ok(())
    }

    pub fn finish_shard(&mut self) -> Result<(), String> {
        let Some(sink) = self.sink.take() else { return Ok(()); };

        let samples = self.samples_in_shard;

        match sink.inner {
            InfSinkInner::Raw(mut w) => {
                w.flush().map_err(|e| format!("flush raw inf shard: {e}"))?;
            }
            InfSinkInner::Zstd(enc) => {
                let mut w = enc.finish().map_err(|e| format!("zstd finish inf: {e}"))?;
                w.flush().map_err(|e| format!("flush zstd inf: {e}"))?;
            }
        }

        let meta = fs::metadata(&sink.path).map_err(|e| format!("metadata {:?}: {e}", sink.path))?;
        let bytes_on_disk = meta.len();

        let filename = sink
            .path
            .file_name()
            .ok_or_else(|| "inf shard filename missing".to_string())?
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

    fn ext(&self) -> &'static str {
        match self.codec {
            Codec::Raw => "inf",
            Codec::Zstd => "inf.zst",
        }
    }
}

fn inf_write_all(inner: &mut InfSinkInner, bytes: &[u8]) -> Result<(), String> {
    match inner {
        InfSinkInner::Raw(w) => w.write_all(bytes).map_err(|e| format!("write raw inf: {e}")),
        InfSinkInner::Zstd(w) => w.write_all(bytes).map_err(|e| format!("write zstd inf: {e}")),
    }
}

fn as_bytes_f32(x: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}
fn as_bytes_i8(x: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inf_record_size_matches_layout() {
        let expect = (INF_FEAT_DIM * 4) + (INF_LABEL_SIZE * 1) + 2;
        assert_eq!(INF_RECORD_SIZE, expect);
        assert_eq!(INF_FEAT_DIM, 235);
        assert_eq!(INF_LABEL_SIZE, 28);
    }
}