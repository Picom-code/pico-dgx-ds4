//! Vision (docs/vision.md): the image encoder's client. The encoder (tools/vision/vision_encoder.py, the
//! checkpoint's own ViT + aligner) turns each image record of a request into its span's rows: bf16
//! [rows][5120] for `[IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]`. The prompt's
//! placeholder ids are expanded into that many image ids, and the rows go to the engine with the SUBMIT
//! (protocol FLAG_IMAGES). Text requests never come here.
//!
//! Wire format (TCP, loopback): request u32 len + JSON {"images": [record, ...]}; reply u32 len + JSON
//! {"ok", "error", "images": [{"rows", "hash", "grid", "ms", "bytes"}]}, then u64 len + the rows.

use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// `<｜deepseek_image｜>`: the placeholder the chat encoding writes, and the id of every span position.
pub const IMAGE_TOKEN_ID: u32 = 129264;
pub const IMAGE_DIM: usize = 5120;
pub const MAX_IMAGES: usize = 100;
pub const MAX_IMAGE_ROWS: usize = 1100;

#[derive(Debug, Clone)]
pub struct Encoded {
    /// rows per image, in prompt order
    pub rows: Vec<u32>,
    pub hashes: Vec<u64>,
    /// every image's rows, bf16 little-endian, in order
    pub data: Vec<u8>,
    pub ms: f64,
}

pub async fn encode(addr: &str, records: &[Value], timeout: Duration) -> Result<Encoded, String> {
    if records.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    let t0 = Instant::now();
    let fut = async {
        let mut s = TcpStream::connect(addr).await.map_err(|e| format!("the image encoder is not reachable: {e}"))?;
        let body = json!({ "images": records }).to_string().into_bytes();
        s.write_all(&(body.len() as u32).to_le_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(&body).await.map_err(|e| e.to_string())?;
        let mut n4 = [0u8; 4];
        s.read_exact(&mut n4).await.map_err(|e| format!("image encoder: {e}"))?;
        let mut h = vec![0u8; u32::from_le_bytes(n4) as usize];
        s.read_exact(&mut h).await.map_err(|e| format!("image encoder: {e}"))?;
        let mut n8 = [0u8; 8];
        s.read_exact(&mut n8).await.map_err(|e| format!("image encoder: {e}"))?;
        let n = u64::from_le_bytes(n8) as usize;
        if n > (MAX_IMAGES * MAX_IMAGE_ROWS * IMAGE_DIM * 2) {
            return Err("image encoder: reply too large".to_string());
        }
        let mut data = vec![0u8; n];
        s.read_exact(&mut data).await.map_err(|e| format!("image encoder: {e}"))?;
        let h: Value = serde_json::from_slice(&h).map_err(|e| format!("image encoder: {e}"))?;
        Ok::<_, String>((h, data))
    };
    let (h, data) = tokio::time::timeout(timeout, fut).await.map_err(|_| "the image encoder timed out".to_string())??;
    if h.get("ok").and_then(Value::as_bool) != Some(true) {
        let e = h.get("error").and_then(Value::as_str).unwrap_or("unknown error");
        return Err(format!("cannot read the image: {e}"));
    }
    let imgs = h.get("images").and_then(Value::as_array).ok_or("image encoder: no images in its reply")?;
    if imgs.len() != records.len() {
        return Err("image encoder: wrong number of images".into());
    }
    let mut rows = Vec::new();
    let mut hashes = Vec::new();
    for i in imgs {
        let r = i.get("rows").and_then(Value::as_u64).ok_or("image encoder: rows")?;
        if !(3..=MAX_IMAGE_ROWS as u64).contains(&r) {
            return Err("image encoder: bad span".into());
        }
        rows.push(r as u32);
        hashes.push(i.get("hash").and_then(Value::as_u64).unwrap_or(0));
    }
    let total: usize = rows.iter().map(|&r| r as usize).sum();
    if data.len() != total * IMAGE_DIM * 2 {
        return Err("image encoder: the rows do not match the spans".into());
    }
    Ok(Encoded { rows, hashes, data, ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// The prompt's ids with each placeholder expanded into its image's span, and the spans' starts.
pub fn expand(ids: &[u32], rows: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let n = ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    if n != rows.len() {
        return Err(format!("the prompt has {n} image placeholders for {} images", rows.len()));
    }
    let extra: usize = rows.iter().map(|&r| r as usize - 1).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut starts = Vec::with_capacity(rows.len());
    let mut k = 0;
    for &t in ids {
        if t == IMAGE_TOKEN_ID {
            starts.push(out.len() as u32);
            out.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(rows[k] as usize));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, starts))
}

/// The SUBMIT's image part (after prompt[]): u32 n, {u32 start, u32 rows, u64 hash} per image, the rows.
pub fn payload(starts: &[u32], e: &Encoded) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + 16 * starts.len() + e.data.len());
    v.extend_from_slice(&(starts.len() as u32).to_le_bytes());
    for i in 0..starts.len() {
        v.extend_from_slice(&starts[i].to_le_bytes());
        v.extend_from_slice(&e.rows[i].to_le_bytes());
        v.extend_from_slice(&e.hashes[i].to_le_bytes());
    }
    v.extend_from_slice(&e.data);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_placeholders() {
        let (ids, st) = expand(&[0, 5, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID], &[3, 4]).unwrap();
        assert_eq!(ids.len(), 3 + 3 + 4);
        assert_eq!(st, vec![2, 6]);
        assert!(ids[2..5].iter().all(|&t| t == IMAGE_TOKEN_ID) && ids[5] == 6);
        assert!(expand(&[IMAGE_TOKEN_ID], &[3, 3]).is_err());
    }
}
