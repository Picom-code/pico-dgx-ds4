//! Vision (docs/vision.md): the image encoder's client. The encoder (tools/vision/vision_encoder.py, the
//! checkpoint's own ViT + aligner) turns each image record of a request into its span's rows: bf16
//! [rows][5120] for `[IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]`. The prompt's
//! placeholder ids are expanded into that many image ids, and the rows go to the engine with the SUBMIT
//! (protocol FLAG_IMAGES). Text requests never come here.
//!
//! Wire format (TCP, loopback): request u32 len + JSON {"images": [record, ...]}; reply u32 len + JSON
//! {"ok", "error", "images": [{"rows", "hash", "grid", "ms", "bytes"}]}, then u64 len + the rows.
//! The request also carries "max_rows", the cap on the rows of all its images together, so that an
//! encoder that knows it stops early; the reply's JSON is checked against the cap here before any of
//! the rows are read.

use serde_json::Value;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// `<｜deepseek_image｜>`: the placeholder the chat encoding writes, and the id of every span position.
pub const IMAGE_TOKEN_ID: u32 = 129264;
pub const IMAGE_DIM: usize = 5120;
/// The encoder's JSON reply for images is a few hundred bytes per image.
const MAX_HEADER: usize = 16 << 20;
pub const MAX_IMAGES: usize = 100;
pub const MAX_IMAGE_ROWS: usize = 1100;
/// The default cap on the rows (image tokens) of all the images of one request (--max-image-rows).
/// Memory follows rows, not the number of images: a row is 10 KiB here (bf16 x 5120) and again in the
/// engine, so 100 full-size images (994 rows each) are ~1 GB per copy. 40,000 rows is what 100 images
/// of ~380 rows, or 40 full-size ones, take.
pub const DEFAULT_MAX_IMAGE_ROWS: usize = 40_000;
/// What the engine's frame limit has room for; --max-image-rows cannot be raised past it.
pub const MAX_TOTAL_IMAGE_ROWS: usize = MAX_IMAGES * MAX_IMAGE_ROWS;

#[derive(Debug, Clone)]
pub struct Encoded {
    /// rows per image, in prompt order
    pub rows: Vec<u32>,
    pub hashes: Vec<u64>,
    /// The SUBMIT's image part, as `payload` completes it: room for the span table (u32 n and 16 bytes
    /// per image), then every image's rows, bf16 little-endian, in order. The rows are read from the
    /// encoder straight into their place here, so the front-end holds them once, not once as the reply
    /// and again as the SUBMIT's part.
    pub data: Vec<u8>,
    pub ms: f64,
}

/// A loopback encoder RPC whose answer is its JSON header: video expansion (video.rs), which returns
/// frame records and no binary data. Image rows go through `encode`, which checks the header against
/// the row cap before it reads them.
pub(crate) async fn exchange(addr: &str, body: &Value, timeout: Duration, max_data: usize) -> Result<(Value, Vec<u8>), String> {
    let body = body.to_string().into_bytes();
    if body.len() > 256 << 20 {
        return Err("media request exceeds 256 MiB".into());
    }
    let fut = async {
        let mut s = TcpStream::connect(addr).await.map_err(|e| format!("the image encoder is not reachable: {e}"))?;
        s.write_all(&(body.len() as u32).to_le_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(&body).await.map_err(|e| e.to_string())?;
        let mut n4 = [0u8; 4];
        s.read_exact(&mut n4).await.map_err(|e| format!("image encoder: {e}"))?;
        let header_len = u32::from_le_bytes(n4) as usize;
        if header_len > 128 << 20 {
            return Err("image encoder: JSON reply too large".into());
        }
        let mut h = vec![0u8; header_len];
        s.read_exact(&mut h).await.map_err(|e| format!("image encoder: {e}"))?;
        let mut n8 = [0u8; 8];
        s.read_exact(&mut n8).await.map_err(|e| format!("image encoder: {e}"))?;
        let n = u64::from_le_bytes(n8) as usize;
        if n > max_data {
            return Err("image encoder: reply too large".to_string());
        }
        let mut data = vec![0u8; n];
        s.read_exact(&mut data).await.map_err(|e| format!("image encoder: {e}"))?;
        let h: Value = serde_json::from_slice(&h).map_err(|e| format!("image encoder: {e}"))?;
        Ok::<_, String>((h, data))
    };
    tokio::time::timeout(timeout, fut).await.map_err(|_| "the image encoder timed out".to_string())?
}

/// Whether an error of `encode` is the request's fault (answered 400) rather than the encoder's (503).
pub fn is_request_error(e: &str) -> bool {
    e.starts_with("cannot read") || e.starts_with("at most ")
}

fn too_many_rows(max_rows: usize, total: usize) -> String {
    format!("at most {max_rows} image tokens per request (--max-image-rows); the images of this request take {total}")
}

/// The rows and hash of each image from the encoder's reply header, checked against the request and the
/// cap on its rows in total. Nothing of the rows themselves has been read or allocated at this point.
fn spans(h: &Value, n_records: usize, max_rows: usize) -> Result<(Vec<u32>, Vec<u64>), String> {
    if h.get("ok").and_then(Value::as_bool) != Some(true) {
        let e = h.get("error").and_then(Value::as_str).unwrap_or("unknown error");
        // an encoder that knows the cap stops at the image that crosses it
        if h.get("code").and_then(Value::as_str) == Some("too_many_rows") && e.starts_with("at most ") {
            return Err(e.to_string());
        }
        return Err(format!("cannot read the image: {e}"));
    }
    let imgs = h.get("images").and_then(Value::as_array).ok_or("image encoder: no images in its reply")?;
    if imgs.len() != n_records {
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
    if total > max_rows {
        return Err(too_many_rows(max_rows, total));
    }
    Ok((rows, hashes))
}

/// The images' rows from the encoder. `max_rows` caps the rows of all the images together: over it the
/// request is refused (an error that `is_request_error`) before any row is read from the encoder.
pub async fn encode(addr: &str, records: &[Value], timeout: Duration, max_rows: usize) -> Result<Encoded, String> {
    if records.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    let max_rows = max_rows.min(MAX_TOTAL_IMAGE_ROWS);
    let t0 = Instant::now();
    let fut = async {
        let mut s = TcpStream::connect(addr).await.map_err(|e| format!("the image encoder is not reachable: {e}"))?;
        // serialised from the records where they are (no copy of them first), and freed once sent:
        // inline images are megabytes each, and the wait for the encoder is the long part
        let body = EncodeRequest { images: records, max_rows }.to_bytes()?;
        s.write_all(&(body.len() as u32).to_le_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(&body).await.map_err(|e| e.to_string())?;
        drop(body);
        let mut n4 = [0u8; 4];
        s.read_exact(&mut n4).await.map_err(|e| format!("image encoder: {e}"))?;
        let header_len = u32::from_le_bytes(n4) as usize;
        if header_len > MAX_HEADER {
            return Err("image encoder: JSON reply too large".to_string());
        }
        let mut h = vec![0u8; header_len];
        s.read_exact(&mut h).await.map_err(|e| format!("image encoder: {e}"))?;
        let h: Value = serde_json::from_slice(&h).map_err(|e| format!("image encoder: {e}"))?;
        // the spans, and the cap on their rows, before the rows: a refused request reads and allocates none
        let (rows, hashes) = spans(&h, records.len(), max_rows)?;
        let total: usize = rows.iter().map(|&r| r as usize).sum();
        let mut n8 = [0u8; 8];
        s.read_exact(&mut n8).await.map_err(|e| format!("image encoder: {e}"))?;
        if u64::from_le_bytes(n8) != (total * IMAGE_DIM * 2) as u64 {
            return Err("image encoder: the rows do not match the spans".to_string());
        }
        let table = span_table_len(rows.len());
        let mut data = vec![0u8; table + total * IMAGE_DIM * 2];
        s.read_exact(&mut data[table..]).await.map_err(|e| format!("image encoder: {e}"))?;
        Ok::<_, String>((rows, hashes, data))
    };
    let (rows, hashes, data) = tokio::time::timeout(timeout, fut).await.map_err(|_| "the image encoder timed out".to_string())??;
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

/// The bytes of the SUBMIT's span table for n images: u32 n, then {u32 start, u32 rows, u64 hash} each.
fn span_table_len(n: usize) -> usize {
    4 + 16 * n
}

/// The SUBMIT's image part (after prompt[]): u32 n, {u32 start, u32 rows, u64 hash} per image, the rows.
/// The rows are already in place in `e.data`; this writes the table in front of them and hands the
/// buffer over, without copying the rows.
pub fn payload(starts: &[u32], e: Encoded) -> Vec<u8> {
    assert_eq!(starts.len(), e.rows.len(), "one span per image");
    let mut v = e.data;
    let table = &mut v[..span_table_len(starts.len())];
    table[..4].copy_from_slice(&(starts.len() as u32).to_le_bytes());
    for i in 0..starts.len() {
        let at = 4 + 16 * i;
        table[at..at + 4].copy_from_slice(&starts[i].to_le_bytes());
        table[at + 4..at + 8].copy_from_slice(&e.rows[i].to_le_bytes());
        table[at + 8..at + 16].copy_from_slice(&e.hashes[i].to_le_bytes());
    }
    v
}

/// The encoder's request, serialised from the image records in place.
#[derive(serde::Serialize)]
struct EncodeRequest<'a> {
    images: &'a [Value],
    max_rows: usize,
}

impl EncodeRequest<'_> {
    /// The request's JSON in a buffer of exactly its size: measured first, so that megabytes of inline
    /// images are not also held in the slack of a buffer grown by doubling.
    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0 += b.len();
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut n = Count(0);
        serde_json::to_writer(&mut n, self).map_err(|e| e.to_string())?;
        let mut v = Vec::with_capacity(n.0);
        serde_json::to_writer(&mut v, self).map_err(|e| e.to_string())?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expands_placeholders() {
        let (ids, st) = expand(&[0, 5, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID], &[3, 4]).unwrap();
        assert_eq!(ids.len(), 3 + 3 + 4);
        assert_eq!(st, vec![2, 6]);
        assert!(ids[2..5].iter().all(|&t| t == IMAGE_TOKEN_ID) && ids[5] == 6);
        assert!(expand(&[IMAGE_TOKEN_ID], &[3, 3]).is_err());
    }

    /// A stand-in for tools/vision/vision_encoder.py: answers one request with `reply`, then the rows
    /// of its images (zeros) when `send_rows`; otherwise it sends nothing more and keeps the connection
    /// open. Returns its address and, when it is done, the request it was sent.
    pub(crate) async fn fake_encoder(reply: Value, send_rows: bool) -> (String, tokio::task::JoinHandle<Value>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let h = tokio::spawn(async move {
            let (mut s, _) = l.accept().await.unwrap();
            let mut n4 = [0u8; 4];
            s.read_exact(&mut n4).await.unwrap();
            let mut body = vec![0u8; u32::from_le_bytes(n4) as usize];
            s.read_exact(&mut body).await.unwrap();
            let hdr = reply.to_string().into_bytes();
            s.write_all(&(hdr.len() as u32).to_le_bytes()).await.unwrap();
            s.write_all(&hdr).await.unwrap();
            if send_rows {
                let rows: u64 = reply["images"].as_array().map_or(0, |a| a.iter().map(|i| i["rows"].as_u64().unwrap()).sum());
                let n = rows as usize * IMAGE_DIM * 2;
                s.write_all(&(n as u64).to_le_bytes()).await.unwrap();
                let _ = s.write_all(&row_bytes(n)).await;
            } else {
                // the client must answer from the header alone: nothing else ever comes
                let _ = s.read(&mut [0u8; 1]).await;
            }
            serde_json::from_slice(&body).unwrap()
        });
        (addr, h)
    }

    /// What the stand-in encoder sends as rows: a pattern, so that a test sees where each byte lands.
    fn row_bytes(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[tokio::test]
    async fn rows_land_in_the_submit_part_without_a_copy() {
        let rows = [3u32, 5, 4];
        let (addr, _enc) = fake_encoder(reply_with_rows(&rows), true).await;
        let e = encode(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        let n = 12 * IMAGE_DIM * 2;
        assert_eq!(e.data.len(), 4 + 16 * 3 + n, "room for the span table, then the rows");
        let (at, hashes) = (e.data.as_ptr(), e.hashes.clone());
        let part = payload(&[7, 20, 40], e);
        assert_eq!(part.as_ptr(), at, "the same buffer: the rows are not copied");
        // the layout the engine reads (docs/protocol.md): u32 n, {u32 start, u32 rows, u64 hash} each, the rows
        let mut want = 3u32.to_le_bytes().to_vec();
        for (i, (start, r)) in [7u32, 20, 40].iter().zip(rows).enumerate() {
            want.extend_from_slice(&start.to_le_bytes());
            want.extend_from_slice(&r.to_le_bytes());
            want.extend_from_slice(&hashes[i].to_le_bytes());
        }
        want.extend_from_slice(&row_bytes(n));
        assert!(part == want, "span table and rows as before");
        // and the request the encoder gets is the same JSON as before, plus the cap
        let (addr, enc) = fake_encoder(reply_with_rows(&[3]), true).await;
        let recs = vec![json!({"url": "data:image/png;base64,QUJD", "detail": "high"})];
        encode(&addr, &recs, T, 123).await.unwrap();
        assert_eq!(enc.await.unwrap(), json!({"images": recs, "max_rows": 123}));
        let bytes = EncodeRequest { images: &recs, max_rows: 123 }.to_bytes().unwrap();
        assert_eq!(bytes, json!({"images": recs, "max_rows": 123}).to_string().into_bytes());
        assert_eq!(bytes.capacity(), bytes.len(), "sized once");
    }

    pub(crate) fn reply_with_rows(rows: &[u32]) -> Value {
        json!({"ok": true, "images": rows.iter().enumerate().map(|(i, r)| json!({"rows": r, "hash": i})).collect::<Vec<_>>()})
    }

    fn records(n: usize) -> Vec<Value> {
        vec![json!({"url": "data:image/png;base64,AAAA"}); n]
    }

    const T: Duration = Duration::from_secs(20);

    #[tokio::test]
    async fn hundred_small_images_pass() {
        // 100 images of 383 rows (a 768 x 768 picture) are 38,300 rows: under the default cap
        let rows = vec![383u32; 100];
        assert!(rows.iter().sum::<u32>() as usize <= DEFAULT_MAX_IMAGE_ROWS);
        let (h, _) = spans(&reply_with_rows(&rows), 100, DEFAULT_MAX_IMAGE_ROWS).unwrap();
        assert_eq!(h, rows);
        // and end to end with the stand-in encoder (short spans, so the test moves 3 MB rather than 392)
        let (addr, enc) = fake_encoder(reply_with_rows(&[3; 100]), true).await;
        let e = encode(&addr, &records(100), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        assert_eq!((e.rows.len(), e.data.len()), (100, 4 + 16 * 100 + 300 * IMAGE_DIM * 2));
        let sent = enc.await.unwrap();
        assert_eq!(sent["max_rows"], DEFAULT_MAX_IMAGE_ROWS, "the encoder is told the cap");
        assert_eq!(sent["images"].as_array().unwrap().len(), 100);
    }

    #[tokio::test]
    async fn rows_over_the_cap_are_refused_before_any_row_is_read() {
        // 41 full-size images (994 rows each) are 40,754 rows: over the default cap of 40,000
        let (addr, _enc) = fake_encoder(reply_with_rows(&[994; 41]), false).await;
        let t0 = Instant::now();
        let e = encode(&addr, &records(41), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "at most 40000 image tokens per request (--max-image-rows); the images of this request take 40754");
        assert!(is_request_error(&e), "answered 400, not 503");
        assert!(t0.elapsed() < Duration::from_secs(5), "refused from the reply's header, without waiting for rows");
        // 40 of them are exactly 39,760: allowed
        assert!(spans(&reply_with_rows(&[994; 40]), 40, DEFAULT_MAX_IMAGE_ROWS).is_ok());
    }

    #[tokio::test]
    async fn the_cap_is_a_setting() {
        // exactly at the cap passes, one row over does not
        assert!(spans(&reply_with_rows(&[3, 4, 5]), 3, 12).is_ok());
        assert!(spans(&reply_with_rows(&[3, 4, 5]), 3, 11).unwrap_err().starts_with("at most 11 image tokens"));
        // raised, 100 full-size images fit; the engine's frame limit is the ceiling whatever is asked
        assert!(spans(&reply_with_rows(&[994; 100]), 100, MAX_TOTAL_IMAGE_ROWS).is_ok());
        let (addr, enc) = fake_encoder(reply_with_rows(&[3]), true).await;
        encode(&addr, &records(1), T, usize::MAX).await.unwrap();
        assert_eq!(enc.await.unwrap()["max_rows"], MAX_TOTAL_IMAGE_ROWS);
        assert!(MAX_TOTAL_IMAGE_ROWS * IMAGE_DIM * 2 + (8 << 20) <= crate::protocol::MAX_FRAME);
    }

    #[tokio::test]
    async fn errors_are_sorted_into_the_requests_and_the_encoders() {
        // an encoder that knows the cap stops early and says so: the request's fault
        let stop = json!({"ok": false, "code": "too_many_rows", "error": "at most 40000 image tokens per request; the first 41 of 100 images already take 40754"});
        let (addr, _enc) = fake_encoder(stop, false).await;
        let e = encode(&addr, &records(100), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.starts_with("at most 40000 image tokens") && is_request_error(&e), "{e}");
        // an unreadable image: the request's fault, as before
        let (addr, _enc) = fake_encoder(json!({"ok": false, "error": "cannot identify image file"}), false).await;
        let e = encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "cannot read the image: cannot identify image file");
        assert!(is_request_error(&e));
        // more than 100 images never reaches the encoder
        let e = encode("127.0.0.1:1", &records(101), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "at most 100 images per request");
        // a broken or absent encoder is not the request's fault
        let (addr, _enc) = fake_encoder(reply_with_rows(&[3, 3]), true).await;
        let e = encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(!is_request_error(&e), "{e}");
        let (addr, _enc) = fake_encoder(reply_with_rows(&[2000]), true).await;
        assert!(!is_request_error(&encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err()));
        let gone = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().to_string();
        let e = encode(&gone, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.starts_with("the image encoder is not reachable") && !is_request_error(&e), "{e}");
    }
}
