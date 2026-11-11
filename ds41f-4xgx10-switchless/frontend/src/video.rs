//! Video content parts become timestamped image parts before the normal chat encoding.
use crate::vision::{exchange, MAX_IMAGES};
use serde_json::{json, Value};
use std::time::Duration;

fn is_video(p: &Value) -> bool {
    matches!(p.get("type").and_then(Value::as_str), Some("video_url" | "video" | "input_video"))
}

fn invalid(msg: &str) -> String {
    format!("cannot read the video: {msg}")
}

fn record(p: &Value) -> Result<Value, String> {
    let source = p.get("video_url").or_else(|| p.get("video")).unwrap_or(p);
    let mut r = match source {
        Value::String(s) if !s.is_empty() => json!({"url": s}),
        Value::Object(_) => source.clone(),
        _ => return Err(invalid("video_url must contain a URL or a source object")),
    };
    if !["url", "data", "source"].iter().any(|k| r.get(*k).is_some()) {
        return Err(invalid("video part needs a URL or base64 data"));
    }
    if let Some(n) = r.get("max_frames").or_else(|| p.get("max_frames")).cloned() {
        if !n.as_u64().map_or(false, |n| (1..=MAX_IMAGES as u64).contains(&n)) {
            return Err(invalid("max_frames must be an integer between 1 and 100"));
        }
        r["max_frames"] = n;
    }
    Ok(r)
}

fn collect(content: &Value, videos: &mut Vec<Value>, images: &mut usize) -> Result<(), String> {
    if let Some(parts) = content.as_array() {
        for p in parts {
            if is_video(p) {
                videos.push(record(p)?);
            } else if matches!(p.get("type").and_then(Value::as_str), Some("image" | "image_url")) {
                *images += 1;
            } else if p.get("type").and_then(Value::as_str) == Some("tool_result") {
                collect(&p["content"], videos, images)?;
            }
        }
    }
    Ok(())
}

fn replacements(reply: &Value, count: usize, budget: usize) -> Result<Vec<Vec<Value>>, String> {
    if reply.get("ok").and_then(Value::as_bool) != Some(true) {
        return Err(invalid(reply.get("error").and_then(Value::as_str).unwrap_or("decoder failed")));
    }
    let clips = reply.get("videos").and_then(Value::as_array).ok_or("video encoder: missing videos")?;
    if clips.len() != count {
        return Err("video encoder: wrong number of videos".into());
    }
    let mut total = 0;
    let mut out = Vec::new();
    for (i, clip) in clips.iter().enumerate() {
        let duration = clip["duration"].as_f64().filter(|v| v.is_finite() && *v > 0.0).ok_or("video encoder: invalid duration")?;
        let frames = clip["frames"].as_array().filter(|f| !f.is_empty()).ok_or("video encoder: no frames")?;
        total += frames.len();
        if total > budget {
            return Err("video encoder: too many frames".into());
        }
        let mut parts = vec![json!({"type": "text", "text": format!("[Video {}: duration {duration:.3} seconds]", i + 1)})];
        let mut previous = -1.0;
        for (j, f) in frames.iter().enumerate() {
            let timestamp = f["timestamp"].as_f64().filter(|v| v.is_finite() && *v >= 0.0 && *v >= previous && *v <= duration).ok_or("video encoder: invalid timestamp")?;
            previous = timestamp;
            let url = f["url"].as_str().filter(|u| u.starts_with("data:image/")).ok_or("video encoder: invalid frame")?;
            parts.push(json!({"type": "text", "text": format!("[Video {}, frame {} at {timestamp:.3} seconds]", i + 1, j + 1)}));
            parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
        }
        parts.push(json!({"type": "text", "text": format!("[End of video {}]", i + 1)}));
        out.push(parts);
    }
    Ok(out)
}

fn replace(content: &mut Value, parts: &mut std::vec::IntoIter<Vec<Value>>) {
    if let Some(blocks) = content.as_array_mut() {
        let old = std::mem::take(blocks);
        for mut p in old {
            if is_video(&p) {
                blocks.extend(parts.next().expect("one replacement per collected video"));
            } else {
                if p.get("type").and_then(Value::as_str) == Some("tool_result") {
                    replace(&mut p["content"], parts);
                }
                blocks.push(p);
            }
        }
    }
}

pub async fn expand(req: &mut Value, addr: Option<&str>) -> Result<(), String> {
    let Some(messages) = req.get("messages").and_then(Value::as_array) else { return Ok(()) };
    let (mut videos, mut images) = (Vec::new(), 0);
    for m in messages {
        if !m.is_object() {
            return Err(invalid("messages must be objects"));
        }
        collect(&m["content"], &mut videos, &mut images)?;
    }
    if videos.is_empty() {
        return Ok(());
    }
    let addr = addr.ok_or_else(|| invalid("video input requires a configured vision encoder (--vision-addr)"))?;
    let budget = MAX_IMAGES.saturating_sub(images);
    if videos.len() > budget {
        return Err(invalid("at most 100 images and video frames combined per request"));
    }
    let timeout = Duration::from_secs(90 + 80 * videos.len() as u64);
    let (reply, _) = exchange(addr, &json!({"videos": videos, "frame_budget": budget}), timeout, 0).await?;
    let mut parts = replacements(&reply, videos.len(), budget)?.into_iter();
    for m in req["messages"].as_array_mut().unwrap() {
        replace(&mut m["content"], &mut parts);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_and_limits() {
        assert_eq!(record(&json!({"type":"video_url","video_url":"https://example.com/a.mp4"})).unwrap()["url"], "https://example.com/a.mp4");
        assert_eq!(record(&json!({"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA","max_frames":3}})).unwrap()["max_frames"], 3);
        for n in [json!(0), json!(101), json!(true), json!(1.5)] {
            assert!(record(&json!({"type":"video_url","video_url":{"url":"x","max_frames":n}})).is_err());
        }
        assert!(record(&json!({"type":"video_url"})).is_err());
    }

    #[test]
    fn timestamped_frames_keep_content_order() {
        let reply = json!({"ok":true,"videos":[{"duration":2.0,"frames":[{"timestamp":0.0,"url":"data:image/jpeg;base64,AA"},{"timestamp":1.0,"url":"data:image/jpeg;base64,BB"}]}]});
        let mut parts = replacements(&reply, 1, 2).unwrap().into_iter();
        let mut content = json!([{"type":"text","text":"before"},{"type":"video_url","video_url":"x"},{"type":"text","text":"after"}]);
        replace(&mut content, &mut parts);
        assert_eq!(content[0]["text"], "before");
        assert!(content[2]["text"].as_str().unwrap().contains("0.000 seconds"));
        assert!(content[4]["text"].as_str().unwrap().contains("1.000 seconds"));
        assert_eq!(content.as_array().unwrap().last().unwrap()["text"], "after");
        assert!(replacements(&reply, 1, 1).is_err());
    }

    #[tokio::test]
    async fn text_unchanged_and_video_requires_encoder() {
        let mut r = json!({"messages":[{"role":"user","content":"hello"}]});
        let old = r.clone();
        expand(&mut r, None).await.unwrap();
        assert_eq!(r, old);
        r["messages"][0]["content"] = json!([{"type":"video_url","video_url":"https://example.com/a.mp4"}]);
        assert!(expand(&mut r, None).await.unwrap_err().contains("--vision-addr"));
    }
}
