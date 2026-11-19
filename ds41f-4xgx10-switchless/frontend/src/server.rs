//! OpenAI-compatible HTTP server: /v1/models, /v1/chat/completions (streaming and not), /health.
//!
//! Requests are rendered with the publisher's chat encoding the way the kit's SGLang does it (tools
//! hosted on a leading system message, effort tiers from chat_template_kwargs or reasoning_effort),
//! matched to a session for prefix reuse, sent to rank 0, and the streamed ids are decoded, checked
//! for stop strings and loops, and parsed into reasoning, content and tool calls.

use crate::encoding::{api_tool_name, encode_messages_with_tail, Delta, EncodeOptions, Encoding, StreamParser, ThinkingMode};
use crate::engine::{EngineClient, Event as EngineEvent};
use crate::gate::{Gate, Pass, Refused, RETRY_AFTER_S};
use crate::guard::{LoopGuard, StopMatcher, LINE_WINDOW_TOKENS};
use crate::protocol::{DoneReason, Submit, FLAG_KEEP, FLAG_SEED};
use crate::pyjson::truthy;
use crate::session::SessionTable;
use crate::tokenizer::{StreamDecoder, Tokenizer};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_name: String,
    pub api_key: Option<String>,
    /// Default and cap for max_tokens (the kit's DSV41_MAX_NEW_TOKENS).
    pub max_tokens_cap: u32,
    /// Reasoning effort budget when the request names none.
    pub default_effort: i64,
    /// The chat encoding: the weights' own encoding.py (V4.1, or 1017's).
    pub encoding: Encoding,
    /// Enable the experimental tail renderer for unmarked 1017 requests.
    pub allow_tail_cache_render: bool,
    pub default_thinking: bool,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub loop_guard: Option<LoopGuard>,
    pub tokenize_threads: usize,
    /// Write each request, its prompt and its ids here, and at its end the output's ids (debugging and
    /// bench traces; holds conversation text).
    pub dump_dir: Option<std::path::PathBuf>,
    /// How long a request waits while the engine is down (the ring lent, or a restart) before it is
    /// answered 503 with Retry-After.
    pub hold: Duration,
    /// vision: the image encoder's address (host:port); None: image input is refused as before
    pub vision_addr: Option<String>,
    /// vision: the most rows (image tokens) the images of one request may take together
    /// (--max-image-rows); memory on rank 0's box follows rows, not the number of images
    pub max_image_rows: usize,
    /// The largest request body taken, in bytes (--max-body-mb); a larger one is answered 413.
    pub max_body_bytes: usize,
    /// vision: the most rows the image requests being answered may take together
    /// (--max-image-rows-in-flight); a request that does not fit is answered 429 with Retry-After
    pub max_image_rows_in_flight: usize,
}

/// The default body limit: 64 MiB, which holds a 1M-token prompt several times over. Image and video
/// requests may need more (--max-body-mb); see docs/vision.md for what a large body costs in memory.
pub const DEFAULT_MAX_BODY_BYTES: usize = 64 << 20;

pub struct App {
    pub cfg: ServerConfig,
    pub tok: Arc<Tokenizer>,
    pub engine: EngineClient,
    pub gate: Arc<Gate>,
    /// vision: the rows held by the image requests in flight (text requests never touch it)
    pub image_rows: Arc<crate::vision::RowsInFlight>,
    sessions: Mutex<SessionTable>,
    think_end_id: Option<u32>,
    counter: AtomicU64,
}

impl App {
    pub fn new(cfg: ServerConfig, tok: Arc<Tokenizer>, engine: EngineClient) -> Arc<App> {
        let think_end_id = tok.token_id("</think>");
        let gate = Gate::new(cfg.hold);
        let image_rows = crate::vision::RowsInFlight::new(cfg.max_image_rows_in_flight);
        Arc::new(App { cfg, tok, engine, gate, image_rows, sessions: Mutex::new(SessionTable::new(256, 16 << 20)), think_end_id, counter: AtomicU64::new(0) })
    }

    /// The session table an earlier run saved (session.rs `load`), for the next turns of its
    /// conversations: what happened, for the log.
    pub fn load_sessions(&self, path: &std::path::Path, max_age_s: u64) -> String {
        match self.sessions.lock().unwrap().load(path, &self.tok, max_age_s) {
            Ok(n) => format!("session table: {n} sessions from {}", path.display()),
            Err(e) if e.is_empty() => format!("session table: none saved at {}", path.display()),
            Err(e) => format!("session table: none taken ({e})"),
        }
    }

    /// The session table for the next start (session.rs `save`): what happened, for the log.
    pub fn save_sessions(&self, path: &std::path::Path, max_age_s: u64) -> String {
        match self.sessions.lock().unwrap().save(path, max_age_s) {
            Ok(n) => format!("session table: {n} sessions saved to {}", path.display()),
            Err(e) => format!("session table: NOT saved: {e}"),
        }
    }

    fn random_hex(&self, n: usize) -> String {
        // splitmix64 over time and a counter: ids only need to be unique
        let mut x = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
            ^ self.counter.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E3779B97F4A7C15);
        let mut s = String::new();
        while s.len() < n {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            s.push_str(&format!("{z:016x}"));
        }
        s.truncate(n);
        s
    }
}

pub fn router(app: Arc<App>) -> Router {
    let max_body = app.cfg.max_body_bytes;
    let routes = Router::new().route("/health", get(health)).route("/v1/models", get(models)).route("/v1/chat/completions", post(chat));
    body_limit(routes, max_body).with_state(app)
}

/// Bodies over `max` bytes are answered 413 before a handler runs.
fn body_limit<S: Clone + Send + Sync + 'static>(routes: Router<S>, max: usize) -> Router<S> {
    routes.layer(axum::extract::DefaultBodyLimit::max(max))
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

fn error(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    ApiError(status, kind.to_string(), msg.into()).response()
}

/// A request's failure: its status, error type and message. The answer, or once a stream has started
/// (a stream that waited at the gate), its error event.
struct ApiError(StatusCode, String, String);

impl ApiError {
    fn json(&self) -> Value {
        json!({"error": {"message": self.2, "type": self.1, "param": null, "code": self.0.as_u16()}})
    }
    fn response(&self) -> Response {
        (self.0, [("content-type", "application/json")], self.json().to_string()).into_response()
    }
}

/// Refused at the gate: 503, with Retry-After as an answer.
fn refused(r: Refused) -> ApiError {
    let msg = match r {
        Refused::Held(t) => format!("the engine has been down for the {} s this request waited (the ring is lent to a development job, or the engine restarts); try again later", t.as_secs()),
        Refused::Closing => "the server is shutting down".into(),
    };
    ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), msg)
}

/// What a request refused for the image rows in flight is told to wait before it tries again: image
/// requests take seconds (about half a second per full-size image to encode, then the prefill).
const IMAGE_RETRY_AFTER_S: u64 = 5;

/// The limit on image work in flight (--max-image-rows-in-flight). The cap on one request's rows
/// (--max-image-rows) bounds one request; several at once multiply it, on a box where running out of
/// memory hangs rank 0. So a request with images takes its share of a budget of rows before anything
/// is done for it (before the encoder, the read of the rows and the engine), and gives it back when
/// it ends: answered, failed or abandoned by its client. What it will take is not known until the
/// encoder has answered, so it first holds the most its images can take (1100 rows each, at most the
/// request cap), and from the encoder's answer on what they do take. A clip's frames are image parts
/// by the time a request gets here (video.rs), so they are counted like any other image; the decoding
/// before that is one clip at a time in the encoder.
///
/// A request that does not fit is refused at once: 429 with Retry-After. It is not queued, because a
/// waiting request is its body held in memory (megabytes of inline images each) for as long as the
/// ones ahead of it take, and that queue would need a bound and a time limit of its own; the clients'
/// SDKs retry a 429 after Retry-After by themselves. A text request has no images and is never
/// counted, delayed or refused here.
pub(crate) fn hold_image_rows(cfg: &ServerConfig, budget: &Arc<crate::vision::RowsInFlight>, p: &mut Prepared) -> Result<(), Response> {
    if p.images.is_empty() {
        return Ok(());
    }
    let rows = crate::vision::rows_estimate(p.images.len(), cfg.max_image_rows);
    match budget.reserve(rows) {
        Ok(held) => {
            p.rows_held = Some(held);
            Ok(())
        }
        Err(in_flight) => {
            eprintln!("image rows in flight: {in_flight} of {}; a request with {} images and video frames (up to {rows} rows) is answered 429", budget.max(), p.images.len());
            let msg = format!(
                "too much image work in flight: the image requests being answered hold {in_flight} of {} image tokens (--max-image-rows-in-flight) and this request's {} images and video frames may take up to {rows}; try again in a few seconds",
                budget.max(),
                p.images.len()
            );
            let mut resp = error(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", msg);
            resp.headers_mut().insert("retry-after", IMAGE_RETRY_AFTER_S.into());
            Err(resp)
        }
    }
}

/// A stream that waits for the engine carries an SSE comment this often, so that a client's idle
/// timeout sees bytes (and so does a long prefill's wait for its first token).
const KEEP_ALIVE: Duration = Duration::from_secs(5);

fn bad_request(msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

fn check_auth(app: &App, headers: &HeaderMap) -> Result<(), Response> {
    let Some(key) = &app.cfg.api_key else { return Ok(()) };
    let got = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .map(str::trim)
        .unwrap_or("");
    // constant-time comparison
    let ok = got.len() == key.len() && got.bytes().zip(key.bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0;
    if ok {
        Ok(())
    } else {
        Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "invalid API key"))
    }
}

async fn health(State(app): State<Arc<App>>) -> Response {
    if app.gate.held() {
        let msg = format!("the ring is lent to a development job: requests wait up to {} s\n", app.gate.hold_limit().as_secs());
        (StatusCode::SERVICE_UNAVAILABLE, msg).into_response()
    } else if app.engine.info().is_some() {
        (StatusCode::OK, "ok\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "engine not connected\n").into_response()
    }
}

async fn models(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&app, &headers) {
        return r;
    }
    let max_len = app.engine.info().map(|i| i.max_context);
    let body = json!({"object": "list", "data": [{
        "id": app.cfg.model_name, "object": "model", "created": 0, "owned_by": "dsgb10",
        "root": app.cfg.model_name, "max_model_len": max_len}]});
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

// ------------------------------------------------------------------ request → prompt

/// What the request asks for, with the prompt rendered.
pub struct Prepared {
    pub prompt: String,
    pub mode: ThinkingMode,
    pub tools: bool,
    pub stream: bool,
    pub include_usage: bool,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: Option<u64>,
    pub stops: Vec<String>,
    /// besides EOS, unless ignore_eos
    pub stop_ids: Vec<u32>,
    pub ignore_eos: bool,
    /// vision: the image records in prompt order (encoding.py's), one per placeholder; empty for text
    pub images: Vec<Value>,
    /// vision: this request's share of the image rows in flight (`hold_image_rows`), given back when
    /// the request ends, however it ends; None for text
    pub rows_held: Option<crate::vision::RowsHeld>,
}

fn num_f64(v: &Value) -> Option<f64> {
    v.as_number().and_then(|n| n.to_string().parse::<f64>().ok())
}

fn num_i64(v: &Value) -> Option<i64> {
    let s = v.as_number()?.to_string();
    if s.contains(['.', 'e', 'E']) {
        let f: f64 = s.parse().ok()?;
        return (f.fract() == 0.0 && f.abs() < 9e15).then_some(f as i64);
    }
    s.parse().ok()
}

/// API effort names mapped to the publisher's 1-100 prompt hint. Zero selects chat mode; None is the
/// server's default effort (--effort, 75). These hints guide reasoning; they do not impose a
/// reasoning-token limit.
///
/// A name this server already took keeps the value it had, so a client that sends it gets the same
/// prompt as before: low 50, high and xhigh 75, max 100 (the publisher's tiers), and medium the
/// default effort. `none`, `minimal` and `ultra` are the new names.
const EFFORT_LEVELS: &[(&str, Option<i64>)] = &[
    ("none", Some(0)), ("minimal", Some(1)), ("low", Some(50)), ("medium", None),
    ("high", Some(75)), ("xhigh", Some(75)), ("ultra", Some(95)), ("max", Some(100)),
];

fn invalid_effort(v: &Value) -> String {
    format!("invalid reasoning_effort {v}; allowed values: none, minimal, low, medium, high, xhigh, ultra, max, or a number from 1 to 100")
}

/// A request's own `thinking` field, the on/off switch other OpenAI-compatible servers take:
/// `{"type": "enabled"}` or `{"type": "disabled"}` (other keys, such as budget_tokens, are ignored), or
/// a plain boolean. `auto` and `adaptive` leave the choice to the server default. Anything else is an
/// error rather than a switch that is silently ignored.
fn thinking_switch(v: &Value) -> Result<Option<bool>, String> {
    let kind = match v {
        Value::Bool(b) => return Ok(Some(*b)),
        Value::Object(o) => match o.get("type") {
            None | Some(Value::Null) => return Ok(None),
            Some(t) => t,
        },
        other => other,
    };
    match kind.as_str() {
        Some("enabled") => Ok(Some(true)),
        Some("disabled") => Ok(Some(false)),
        Some("auto" | "adaptive") => Ok(None),
        _ => Err(format!("invalid thinking type {kind}; allowed values: enabled, disabled, auto")),
    }
}

/// The effort a request names, wherever it names it (the top-level field or an alias): one of the
/// names above (case and surrounding spaces do not matter, as before), an integer 1-100 as it is, an
/// OpenAI fraction in [0, 1] times 100, or such a number in a string ("40"). Anything else is an error:
/// a typo no longer falls back to the default silently.
pub fn effort_budget(v: Option<&Value>, default: i64) -> Result<i64, String> {
    match v {
        None | Some(Value::Null) => Ok(default),
        Some(Value::String(s)) => {
            let t = s.trim().to_ascii_lowercase();
            if let Some((_, budget)) = EFFORT_LEVELS.iter().find(|(name, _)| *name == t) {
                return Ok(budget.unwrap_or(default));
            }
            // a number in a string is read the way it always was, as a float: "40" is 40, "0.5" is 50
            // and "1" is the fraction 1.0, so 100
            match t.parse::<f64>() {
                Ok(f) if f.is_finite() && f != 0.0 => effort_budget(Some(&json!(f)), default).map_err(|_| invalid_effort(v.unwrap())),
                _ => Err(invalid_effort(v.unwrap())),
            }
        }
        Some(n @ Value::Number(_)) => {
            let s = n.as_number().unwrap().to_string();
            let f = num_f64(n).unwrap_or(f64::NAN);
            if !s.contains(['.', 'e', 'E']) && (1.0..=100.0).contains(&f) {
                Ok(f as i64)
            } else if s.contains(['.', 'e', 'E']) && (0.0..=1.0).contains(&f) {
                Ok(((f * 100.0).round() as i64).clamp(1, 100))
            } else if (1.0..=100.0).contains(&f) {
                Ok(f.round() as i64)
            } else {
                Err(invalid_effort(n))
            }
        }
        Some(v) => Err(invalid_effort(v)),
    }
}

/// The tool payload the kit's SGLang renders: the function's name, description, parameters and
/// strict, in that order, and only those the client sent.
fn tool_payload(t: &Value) -> Result<Value, String> {
    let f = t.get("function").and_then(Value::as_object).ok_or("each tool needs a function object")?;
    if !f.get("name").map_or(false, Value::is_string) {
        return Err("each tool function needs a name".into());
    }
    let mut out = Map::new();
    for k in ["name", "description", "parameters", "strict"] {
        if let Some(v) = f.get(k) {
            if !v.is_null() {
                out.insert(k.into(), v.clone());
            }
        }
    }
    let mut tool = json!({"type": "function", "function": Value::Object(out)});
    if let Some(ns) = t.get("namespace").filter(|v| !v.is_null()) {
        tool["namespace"] = ns.clone(); // 1017's encoding qualifies the name with it; V4.1's ignores it
    }
    Ok(tool)
}

/// A tool call for the API, as 1017's tool_calls_to_openai_format makes it: a `namespace::name` the
/// model wrote is the bare name, with the namespace beside the function (api_tool_name).
fn tool_call_json(enc: Encoding, index: Option<usize>, id_hex: String, name: &str, args: String) -> Value {
    let (ns, name) = api_tool_name(enc, name);
    let mut call = json!({"id": format!("call_{id_hex}"), "type": "function", "function": {"name": name, "arguments": args}});
    if let Some(i) = index {
        call["index"] = json!(i);
    }
    if let Some(ns) = ns {
        call["namespace"] = json!(ns);
    }
    call
}

/// Messages for the encoder: OpenAI fields only, developer → system, null content → "".
fn convert_messages(msgs: &[Value], vision: bool) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let o = m.as_object().ok_or(format!("messages[{i}] is not an object"))?;
        let role = o.get("role").and_then(Value::as_str).ok_or(format!("messages[{i}] has no role"))?;
        let role = match role {
            "developer" => "system",
            r @ ("system" | "user" | "assistant" | "tool") => r,
            r => return Err(format!("messages[{i}]: unsupported role {r:?}")),
        };
        let mut n = Map::new();
        n.insert("role".into(), json!(role));
        let content = match o.get("content") {
            None | Some(Value::Null) => json!(""),
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p.get("type").and_then(Value::as_str) {
                        Some("image_url" | "image") if vision => {}
                        Some("input_image" | "video_url" | "input_audio" | "audio_url") if vision => {
                            return Err("audio and video input are not supported; images go in image_url parts".into())
                        }
                        Some("image_url" | "image" | "input_image" | "video_url" | "input_audio" | "audio_url") => {
                            return Err("image, audio and video input are not supported".into())
                        }
                        _ => {}
                    }
                }
                Value::Array(parts.clone())
            }
            Some(c) => c.clone(),
        };
        n.insert("content".into(), content);
        for k in ["name", "tool_calls", "tool_call_id", "reasoning_content"] {
            if let Some(v) = o.get(k) {
                if !v.is_null() {
                    n.insert(k.into(), v.clone());
                }
            }
        }
        if !n.contains_key("reasoning_content") {
            if let Some(Value::String(r)) = o.get("reasoning") {
                n.insert("reasoning_content".into(), json!(r));
            }
        }
        out.push(Value::Object(n));
    }
    Ok(out)
}

pub fn prepare(cfg: &ServerConfig, req: &Value) -> Result<Prepared, String> {
    let r = req.as_object().ok_or("the request body must be a JSON object")?;
    let msgs = r.get("messages").and_then(Value::as_array).ok_or("messages must be a list")?;
    if msgs.is_empty() {
        return Err("messages must not be empty".into());
    }
    if let Some(n) = r.get("n").filter(|v| !v.is_null()) {
        if num_i64(n) != Some(1) {
            return Err("only n = 1 is supported".into());
        }
    }
    let mut messages = convert_messages(msgs, cfg.vision_addr.is_some())?;
    messages.iter_mut().for_each(crate::encoding::neutralize_image_placeholder);

    // tools, hosted on a leading system message
    let tool_choice_none = r.get("tool_choice").and_then(Value::as_str) == Some("none");
    let tools: Vec<Value> = match r.get("tools") {
        Some(Value::Array(ts)) if !ts.is_empty() && !tool_choice_none => ts.iter().map(tool_payload).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    if !tools.is_empty() {
        if messages[0]["role"] != "system" {
            messages.insert(0, json!({"role": "system", "content": ""}));
        }
        messages[0].as_object_mut().unwrap().insert("tools".into(), Value::Array(tools.clone()));
    }

    // Thinking and effort. A request that sends what it sent before is answered as before:
    //   - on: `chat_template_kwargs.thinking` / `.enable_thinking`, a top-level `reasoning_effort`,
    //     `reasoning.effort` or `reasoning.enabled`; otherwise the server default (--thinking). An
    //     effort inside chat_template_kwargs only says how much: alone it does not switch thinking on.
    //   - how much: the effort in chat_template_kwargs, else the top-level one, else `reasoning`'s.
    // Two things are new. The request's own `thinking` field is a switch too. And every explicit
    // off-switch wins: any of the switches set to off, or any effort field saying `none`, gives chat
    // mode whatever else the request carries, so a request never thinks against an off-switch it sent.
    // Every effort field is validated, also one that another field shadows.
    let ctk = r.get("chat_template_kwargs").and_then(Value::as_object);
    let reasoning = r.get("reasoning").and_then(Value::as_object);
    let given = |v: Option<&'_ Value>| v.filter(|v| !v.is_null()).cloned();
    let top_effort = given(r.get("reasoning_effort"));
    let template_effort = given(ctk.and_then(|c| c.get("reasoning_effort")));
    let nested_effort = given(reasoning.and_then(|c| c.get("effort")));
    let nested_reasoning_effort = given(reasoning.and_then(|c| c.get("reasoning_effort")));
    let mut effort_off = false;
    for effort in [&top_effort, &template_effort, &nested_effort, &nested_reasoning_effort].into_iter().flatten() {
        effort_off |= effort_budget(Some(effort), cfg.default_effort)? == 0;
    }
    let switch = |v: Option<&'_ Value>| v.filter(|v| !v.is_null()).map(|v| truthy(Some(v)));
    let switches = [
        r.get("thinking").filter(|v| !v.is_null()).map(thinking_switch).transpose()?.flatten(),
        switch(ctk.and_then(|c| c.get("thinking"))),
        switch(ctk.and_then(|c| c.get("enable_thinking"))),
        switch(reasoning.and_then(|c| c.get("enabled"))),
    ];
    let off = effort_off || switches.contains(&Some(false));
    let asked = switches.contains(&Some(true)) || top_effort.is_some() || nested_effort.is_some();
    let thinking = !off && (asked || cfg.default_thinking);
    // `reasoning.effort`, when the key is there at all, hides `reasoning.reasoning_effort` (as before)
    let nested = given(reasoning.and_then(|c| c.get("effort").or_else(|| c.get("reasoning_effort"))));
    let effort = effort_budget(template_effort.as_ref().or(top_effort.as_ref()).or(nested.as_ref()), cfg.default_effort)?;
    // The encoder accepts 1-100 even in chat mode, where it omits the effort prefix.
    let budget = json!(if effort == 0 { cfg.default_effort } else { effort });
    let mode = if thinking { ThinkingMode::Thinking } else { ThinkingMode::Chat };
    let mut o = EncodeOptions::new(mode);
    o.reasoning_effort = Some(&budget);
    o.encoding = cfg.encoding;
    let tail_cache_render = cfg.allow_tail_cache_render && !r.contains_key("cache_render_format");
    let (prompt, images) = encode_messages_with_tail(&messages, &o, tail_cache_render)
        .map_err(|e| format!("cannot encode messages: {e}"))?;
    if images.len() > crate::vision::MAX_IMAGES {
        return Err(format!("at most {} images per request", crate::vision::MAX_IMAGES));
    }

    // sampling and limits
    let max_tokens = match r.get("max_completion_tokens").filter(|v| !v.is_null()).or(r.get("max_tokens").filter(|v| !v.is_null())) {
        None => cfg.max_tokens_cap,
        Some(v) => match num_i64(v) {
            Some(n) if n >= 1 => (n as u64).min(cfg.max_tokens_cap as u64) as u32,
            _ => return Err("max_tokens must be a positive integer".into()),
        },
    };
    let fnum = |k: &str| r.get(k).filter(|v| !v.is_null()).map(|v| num_f64(v).ok_or(format!("{k} must be a number"))).transpose();
    let mut temperature = fnum("temperature")?.map_or(cfg.default_temperature, |t| t as f32);
    let mut top_p = fnum("top_p")?.map_or(cfg.default_top_p, |t| t as f32);
    if !(0.0..=100.0).contains(&temperature) || !(0.0..=1.0).contains(&top_p) {
        return Err("temperature must be >= 0 and top_p in [0, 1]".into());
    }
    if top_p == 0.0 {
        temperature = 0.0;
        top_p = 1.0;
    }
    let top_k = fnum("top_k")?.map_or(0, |k| if k > 0.0 { k as u32 } else { 0 });
    let seed = r.get("seed").filter(|v| !v.is_null()).map(|v| num_i64(v).map(|s| s as u64).ok_or("seed must be an integer")).transpose()?;
    let stops = match r.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().map(|s| s.as_str().map(str::to_string).ok_or("stop must be strings")).collect::<Result<_, _>>()?,
        _ => return Err("stop must be a string or a list of strings".into()),
    };
    let stop_ids: Vec<u32> = match r.get("stop_token_ids") {
        Some(Value::Array(a)) => a.iter().filter_map(|v| num_i64(v).map(|x| x as u32)).collect(),
        _ => Vec::new(),
    };
    let ignore_eos = truthy(r.get("ignore_eos"));
    let stream = truthy(r.get("stream"));
    let include_usage = truthy(r.get("stream_options").and_then(|s| s.get("include_usage")));
    Ok(Prepared {
        prompt,
        mode,
        tools: !tools.is_empty(),
        stream,
        include_usage,
        max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        stops,
        stop_ids,
        ignore_eos,
        images,
        rows_held: None,
    })
}

// ------------------------------------------------------------------ generation

#[derive(Clone, Copy, PartialEq, Debug)]
enum Finish {
    Stop,
    Length,
    ToolCalls,
}

impl Finish {
    fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
        }
    }
}

struct Usage {
    prompt: usize,
    cached: u32,
    completion: usize,
    reasoning: usize,
}

impl Usage {
    fn json(&self) -> Value {
        json!({"prompt_tokens": self.prompt, "completion_tokens": self.completion,
               "total_tokens": self.prompt + self.completion,
               "prompt_tokens_details": {"cached_tokens": self.cached},
               "completion_tokens_details": {"reasoning_tokens": self.reasoning}})
    }
}

/// One request in flight on the engine: decodes, stops, guards and parses its token stream.
struct Gen {
    app: Arc<App>,
    /// counts the request in flight at the gate until it ends (or its client goes away)
    _pass: Pass,
    req_id: u64,
    session_id: u64,
    prompt_text: String,
    prompt_ids: Vec<u32>,
    rx: mpsc::UnboundedReceiver<EngineEvent>,
    stop_ids: Vec<u32>,
    stopper: StopMatcher,
    parser: StreamParser,
    dec: StreamDecoder,
    /// every id the engine sent (the session holds these)
    ids: Vec<u32>,
    /// ids that count as output (up to a stop string, a loop or a stop id)
    kept: usize,
    gen_bytes: Vec<u8>,
    gen_ends: Vec<usize>,
    stopped: Option<Finish>,
    done: bool,
    t0: Instant,
    first_token: Option<Instant>,
}

impl Gen {
    /// Feed engine events until something can be sent. Returns the deltas and, at the end, the
    /// finish reason and usage.
    async fn next(&mut self, out: &mut Vec<Delta>) -> Option<Result<(Finish, Usage), String>> {
        loop {
            let ev = match self.rx.recv().await {
                Some(ev) => ev,
                None => return Some(Err("engine stream ended".into())),
            };
            match ev {
                EngineEvent::Tokens(t) => {
                    self.first_token.get_or_insert_with(Instant::now);
                    for &id in &t.ids {
                        self.ids.push(id);
                        if self.stopped.is_some() {
                            continue;
                        }
                        self.kept = self.ids.len();
                        let tb = self.app.tok.token_bytes(id);
                        self.gen_bytes.extend_from_slice(tb);
                        self.gen_ends.push(self.gen_bytes.len());
                        if self.stop_ids.contains(&id) {
                            // a stop id ends the output; its text is not part of the answer
                            self.flush(out);
                            self.stopped = Some(Finish::Stop);
                            continue;
                        }
                        let mut text = String::new();
                        self.dec.push(&self.app.tok, id, &mut text);
                        let (pass, hit) = self.stopper.push(&text);
                        self.parser.push(&pass, out);
                        if hit {
                            self.stopped = Some(Finish::Stop);
                            self.app.engine.cancel(self.req_id);
                        }
                    }
                    if self.stopped.is_none() {
                        if let Some(g) = &self.app.cfg.loop_guard {
                            let out_ids = &self.ids[..self.kept];
                            let from = out_ids.len().saturating_sub(LINE_WINDOW_TOKENS);
                            let start = if from == 0 { 0 } else { self.gen_ends[from - 1] };
                            let tail = String::from_utf8_lossy(&self.gen_bytes[start..]);
                            if let Some(keep) = g.check(out_ids, &tail) {
                                eprintln!("req {}: repetition loop after {} tokens; keeping {keep}", self.req_id, out_ids.len());
                                self.stopped = Some(Finish::Stop);
                                self.app.engine.cancel(self.req_id);
                            }
                        }
                    }
                    if !out.is_empty() {
                        return None;
                    }
                }
                EngineEvent::Done(d) => {
                    self.done = true;
                    if d.reason == DoneReason::Error {
                        self.app.sessions.lock().unwrap().abandon(self.session_id);
                        self.app.engine.release(self.session_id);
                        return Some(Err(if d.error.is_empty() { "engine error".into() } else { d.error }));
                    }
                    if self.stopped.is_none() {
                        self.flush(out);
                    }
                    self.parser.finish(out);
                    let finish = if self.parser.tool_calls_seen() > 0 {
                        Finish::ToolCalls
                    } else if let Some(f) = self.stopped {
                        f
                    } else {
                        match d.reason {
                            DoneReason::Stop | DoneReason::Cancelled => Finish::Stop,
                            _ => Finish::Length,
                        }
                    };
                    let kept = &self.ids[..self.kept];
                    let reasoning = match self.app.think_end_id {
                        Some(te) if self.parser_was_thinking() => kept.iter().position(|&x| x == te).unwrap_or(kept.len()),
                        _ => 0,
                    };
                    self.app.sessions.lock().unwrap().finish(self.session_id, &self.prompt_text, &self.prompt_ids, &self.ids, &self.app.tok);
                    let usage = Usage { prompt: self.prompt_ids.len(), cached: d.cached_tokens, completion: self.kept, reasoning };
                    let total = self.t0.elapsed().as_secs_f64();
                    let ttft = self.first_token.map_or(total, |t| (t - self.t0).as_secs_f64());
                    if let Some(dir) = &self.app.cfg.dump_dir {
                        // the output's ids: the first `kept` count as output, the rest came after a stop
                        let rec = json!({"req_id": self.req_id, "ids": self.ids, "kept": self.kept, "finish": finish.as_str(),
                            "usage": usage.json(), "ttft_s": ttft, "total_s": total});
                        if let Err(e) = std::fs::write(dir.join(format!("{:06}.out.json", self.req_id)), rec.to_string()) {
                            eprintln!("dump: {e}");
                        }
                    }
                    eprintln!(
                        "req {} session {}: prompt {} (cached {}), completion {}, {}; ttft {:.3} s, {:.1} tok/s",
                        self.req_id,
                        self.session_id,
                        usage.prompt,
                        usage.cached,
                        usage.completion,
                        finish.as_str(),
                        ttft,
                        if total > ttft { usage.completion as f64 / (total - ttft) } else { 0.0 }
                    );
                    return Some(Ok((finish, usage)));
                }
            }
        }
    }

    /// Pass on the text still held by the decoder and the stop matcher.
    fn flush(&mut self, out: &mut Vec<Delta>) {
        let mut rest = String::new();
        self.dec.finish(&mut rest);
        let (pass, hit) = self.stopper.push(&rest);
        self.parser.push(&pass, out);
        if !hit {
            let tail = self.stopper.finish();
            self.parser.push(&tail, out);
        }
    }

    fn parser_was_thinking(&self) -> bool {
        self.prompt_text.ends_with(crate::encoding::THINK_START)
    }
}

impl Drop for Gen {
    fn drop(&mut self) {
        if !self.done {
            // The client went away mid-request. A cancelled session keeps what it processed, of
            // which what we received is a prefix: still good for the conversation's next turn.
            self.app.engine.cancel(self.req_id);
            self.app.sessions.lock().unwrap().finish(self.session_id, &self.prompt_text, &self.prompt_ids, &self.ids, &self.app.tok);
        }
    }
}

async fn start(app: &Arc<App>, p: &Prepared, pass: Pass) -> Result<Gen, ApiError> {
    let unavailable = |msg: String| ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), msg);
    let Some(info) = app.engine.info() else {
        return Err(unavailable("engine not connected".into()));
    };
    if !p.images.is_empty() {
        return start_images(app, p, pass, info).await;
    }
    let (plan, released) = {
        let app2 = app.clone();
        let text = p.prompt.clone();
        tokio::task::spawn_blocking(move || {
            let mut t = app2.sessions.lock().unwrap();
            t.plan(&text, &app2.tok, app2.cfg.tokenize_threads)
        })
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "server_error".into(), e.to_string()))?
    };
    for s in released {
        app.engine.release(s);
    }
    let ctx = info.max_context as usize;
    if plan.ids.len() >= ctx {
        app.sessions.lock().unwrap().abandon(plan.session_id);
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "invalid_request_error".into(),
            format!("the prompt is {} tokens; the context window is {ctx}", plan.ids.len()),
        ));
    }
    let max_new = (p.max_tokens as usize).min(ctx - plan.ids.len()) as u32;
    let mut stop_ids = p.stop_ids.clone();
    if !p.ignore_eos {
        stop_ids.push(info.eos_id);
    }
    let submit = Submit {
        req_id: 0,
        session_id: plan.session_id,
        base_session_id: plan.base_session_id,
        seed: p.seed.unwrap_or(0),
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        max_new_tokens: max_new,
        flags: FLAG_KEEP | if p.seed.is_some() { FLAG_SEED } else { 0 },
        n_top_logprobs: 0,
        prefix_hint: plan.reused as u32,
        stop_ids: stop_ids.clone(),
        prompt: plan.ids.clone(),
        images: None,
    };
    let (req_id, rx) = match app.engine.submit(submit) {
        Ok(x) => x,
        Err(e) => {
            app.sessions.lock().unwrap().abandon(plan.session_id);
            return Err(unavailable(e));
        }
    };
    // where it starts (counts only); rank 0's "serve: req N restart:" line says what the engine found
    eprintln!("req {req_id} plan: {}", plan.describe(p.prompt.len()));
    Ok(Gen {
        app: app.clone(),
        _pass: pass,
        req_id,
        session_id: plan.session_id,
        prompt_text: p.prompt.clone(),
        prompt_ids: plan.ids,
        rx,
        stop_ids,
        stopper: StopMatcher::new(p.stops.clone()),
        parser: StreamParser::with_tools(p.mode, p.tools),
        dec: StreamDecoder::default(),
        ids: Vec::new(),
        kept: 0,
        gen_bytes: Vec::new(),
        gen_ends: Vec::new(),
        stopped: None,
        done: false,
        t0: Instant::now(),
        first_token: None,
    })
}

/// A request with images (docs/vision.md): its images through the encoder, the placeholders expanded into
/// their spans, and a session of its own that the engine frees at the end (no KEEP) and the text table
/// never sees, so text requests' prefix reuse is untouched.
async fn start_images(app: &Arc<App>, p: &Prepared, pass: Pass, info: crate::protocol::HelloEngine) -> Result<Gen, ApiError> {
    let bad = |msg: String| ApiError(StatusCode::BAD_REQUEST, "invalid_request_error".into(), msg);
    let Some(addr) = app.cfg.vision_addr.clone() else {
        return Err(bad("image, audio and video input are not supported".into()));
    };
    // Each URL fetch can take 20 s; allow a full 100-image request to finish encoding.
    let image_timeout = Duration::from_secs(90 + 20 * p.images.len().saturating_sub(1) as u64);
    let enc = crate::vision::encode(&addr, &p.images, image_timeout, app.cfg.max_image_rows).await.map_err(|e| {
        if crate::vision::is_request_error(&e) { bad(e) } else { ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), e) }
    })?;
    let ids = {
        let (app2, text) = (app.clone(), p.prompt.clone());
        tokio::task::spawn_blocking(move || app2.tok.encode_parallel(&text, app2.cfg.tokenize_threads))
            .await
            .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, "server_error".into(), e.to_string()))?
    };
    let (ids, starts) = crate::vision::expand(&ids, &enc.rows).map_err(bad)?;
    let (n_images, n_rows, encoder_ms) = (enc.rows.len(), enc.rows.iter().sum::<u32>(), enc.ms);
    // the encoder has said what the images really take: hold that, not the estimate
    if let Some(held) = &p.rows_held {
        held.shrink_to(n_rows as usize);
    }
    let ctx = info.max_context as usize;
    if ids.len() >= ctx {
        return Err(bad(format!("the prompt is {} tokens with its images; the context window is {ctx}", ids.len())));
    }
    let max_new = (p.max_tokens as usize).min(ctx - ids.len()) as u32;
    let mut stop_ids = p.stop_ids.clone();
    if !p.ignore_eos {
        stop_ids.push(info.eos_id);
    }
    let session_id = (1u64 << 63) | u64::from_str_radix(&app.random_hex(15), 16).unwrap_or(1);
    let submit = Submit {
        req_id: 0,
        session_id,
        base_session_id: 0,
        seed: p.seed.unwrap_or(0),
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        max_new_tokens: max_new,
        flags: crate::protocol::FLAG_IMAGES | if p.seed.is_some() { FLAG_SEED } else { 0 },
        n_top_logprobs: 0,
        prefix_hint: 0,
        stop_ids: stop_ids.clone(),
        prompt: ids.clone(),
        images: Some(crate::vision::payload(&starts, enc)), // the encoder's buffer itself, not a copy
    };
    let (req_id, rx) = app.engine.submit(submit).map_err(|e| ApiError(StatusCode::SERVICE_UNAVAILABLE, "server_error".into(), e))?;
    eprintln!(
        "req {req_id} images: {n_images} ({n_rows} rows), encoder {encoder_ms:.0} ms; prompt {} ids; session {session_id} (not kept)",
        ids.len()
    );
    Ok(Gen {
        app: app.clone(),
        _pass: pass,
        req_id,
        session_id,
        prompt_text: p.prompt.clone(),
        prompt_ids: ids,
        rx,
        stop_ids,
        stopper: StopMatcher::new(p.stops.clone()),
        parser: StreamParser::with_tools(p.mode, p.tools),
        dec: StreamDecoder::default(),
        ids: Vec::new(),
        kept: 0,
        gen_bytes: Vec::new(),
        gen_ends: Vec::new(),
        stopped: None,
        done: false,
        t0: Instant::now(),
        first_token: None,
    })
}

async fn chat(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = check_auth(&app, &headers) {
        return r;
    }
    let mut req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("invalid JSON: {e}")),
    };
    // A request with inline images is megabytes: each copy is freed when its last reader is done,
    // rather than held until the answer ends. The body's bytes are not needed once parsed.
    drop(body);
    if let Err(e) = crate::video::expand(&mut req, app.cfg.vision_addr.as_deref()).await {
        return if e.starts_with("cannot read") { bad_request(e) } else { error(StatusCode::SERVICE_UNAVAILABLE, "server_error", e) };
    }
    let mut p = match prepare(&app.cfg, &req) {
        Ok(p) => p,
        Err(e) => return bad_request(e),
    };
    if let Err(refusal) = hold_image_rows(&app.cfg, &app.image_rows, &mut p) {
        return refusal;
    }
    // The parsed request is only read again by --dump-dir's record.
    let req = if app.cfg.dump_dir.is_some() { req } else { Value::Null };
    // While the engine is down (the ring lent to a job, or a restart) a request waits at the gate. A
    // stream that has to wait starts at once and carries comments meanwhile, so that its client's
    // timeout sees bytes; refused, it ends with an error event. Anything else waits here, refused with
    // 503 and Retry-After.
    if p.stream && !app.gate.open(app.engine.info().is_some()) {
        return stream(app, p, req, None);
    }
    let pass = match app.gate.enter(&app.engine).await {
        Ok(pass) => pass,
        Err(r) => {
            let mut resp = refused(r).response();
            resp.headers_mut().insert("retry-after", RETRY_AFTER_S.into());
            return resp;
        }
    };
    if p.stream {
        return stream(app, p, req, Some(pass));
    }
    let mut gen = match start(&app, &p, pass).await {
        Ok(g) => g,
        Err(e) => return e.response(),
    };
    dump(&app, &req, &p, &gen);
    let id = format!("chatcmpl-{}", app.random_hex(24));
    let created = now_secs();
    let model = app.cfg.model_name.clone();
    // non-streaming
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    let mut out = Vec::new();
    let (finish, usage) = loop {
        let end = gen.next(&mut out).await;
        for d in out.drain(..) {
            match d {
                Delta::Reasoning(s) => reasoning.push_str(&s),
                Delta::Content(s) => content.push_str(&s),
                Delta::ToolCall(_, name, args) => calls.push(tool_call_json(app.cfg.encoding, None, app.random_hex(24), &name, args)),
            }
        }
        match end {
            None => continue,
            Some(Ok(x)) => break x,
            Some(Err(e)) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
        }
    };
    let mut message = json!({"role": "assistant",
        "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
    if p.mode == ThinkingMode::Thinking {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    let body = json!({"id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": null, "finish_reason": finish.as_str()}],
        "usage": usage.json()});
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

/// The request's record, with --dump-dir.
fn dump(app: &App, req: &Value, p: &Prepared, gen: &Gen) {
    if let Some(dir) = &app.cfg.dump_dir {
        let rec = json!({"request": req, "prompt": p.prompt, "ids": gen.prompt_ids, "session": gen.session_id, "req_id": gen.req_id});
        if let Err(e) = std::fs::write(dir.join(format!("{:06}.json", gen.req_id)), rec.to_string()) {
            eprintln!("dump: {e}");
        }
    }
}

/// A streamed answer. Without a pass it waits at the gate first: the stream starts at once with a
/// comment saying why, and a refusal (the hold limit, a shutdown) is its error event. Any stream gets
/// a comment after KEEP_ALIVE without an event (waiting at the gate, or for a long prefill).
fn stream(app: Arc<App>, p: Prepared, req: Value, pass: Option<Pass>) -> Response {
    let (tx, rx) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let last = |e: ApiError| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(Event::default().data(e.json().to_string())).await;
                let _ = tx.send(Event::default().data("[DONE]")).await;
            }
        };
        let pass = match pass {
            Some(pass) => pass,
            None => {
                let why = if app.gate.held() { "the ring is lent to a development job" } else { "the engine is not connected" };
                let note = format!("waiting for the engine ({why}), up to {} s", app.gate.hold_limit().as_secs());
                if tx.send(Event::default().comment(note)).await.is_err() {
                    return;
                }
                tokio::select! {
                    r = app.gate.enter(&app.engine) => match r {
                        Ok(pass) => pass,
                        Err(r) => return last(refused(r)).await,
                    },
                    _ = tx.closed() => return, // the client went away: counted out at the gate
                }
            }
        };
        let mut gen = match start(&app, &p, pass).await {
            Ok(g) => g,
            Err(e) => return last(e).await,
        };
        dump(&app, &req, &p, &gen);
        let id = format!("chatcmpl-{}", app.random_hex(24));
        let created = now_secs();
        let model = app.cfg.model_name.clone();
        let chunk = |delta: Value, finish: Option<&str>| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish}]})
        };
        let send = |v: Value| tx.send(Event::default().data(v.to_string()));
        if send(chunk(json!({"role": "assistant", "content": ""}), None)).await.is_err() {
            return;
        }
        let mut out = Vec::new();
        loop {
            let end = gen.next(&mut out).await;
            for d in out.drain(..) {
                let delta = match d {
                    Delta::Reasoning(s) => json!({"reasoning_content": s}),
                    Delta::Content(s) => json!({"content": s}),
                    Delta::ToolCall(i, name, args) => {
                        json!({"tool_calls": [tool_call_json(app.cfg.encoding, Some(i), app.random_hex(24), &name, args)]})
                    }
                };
                if send(chunk(delta, None)).await.is_err() {
                    return; // client gone; dropping gen cancels the request
                }
            }
            match end {
                None => continue,
                Some(Ok((finish, usage))) => {
                    let _ = send(chunk(json!({}), Some(finish.as_str()))).await;
                    if p.include_usage {
                        let _ = send(json!({"id": id, "object": "chat.completion.chunk", "created": created,
                                            "model": model, "choices": [], "usage": usage.json()}))
                        .await;
                    }
                    break;
                }
                Some(Err(e)) => {
                    let _ = send(json!({"error": {"message": e, "type": "server_error", "param": null, "code": 500}})).await;
                    break;
                }
            }
        }
        let _ = tx.send(Event::default().data("[DONE]")).await;
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let stream = futures_util::StreamExt::map(stream, Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(KEEP_ALIVE)).into_response()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn cfg() -> ServerConfig {
        ServerConfig {
            model_name: "m".into(),
            api_key: None,
            max_tokens_cap: 32768,
            default_effort: 75,
            encoding: Encoding::V41,
            allow_tail_cache_render: false,
            default_thinking: false,
            default_temperature: 1.0,
            default_top_p: 0.95,
            loop_guard: None,
            tokenize_threads: 1,
            dump_dir: None,
            hold: Duration::from_secs(600),
            vision_addr: None,
            max_image_rows: crate::vision::DEFAULT_MAX_IMAGE_ROWS,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_image_rows_in_flight: crate::vision::DEFAULT_MAX_IMAGE_ROWS_IN_FLIGHT,
        }
    }

    #[test]
    fn efforts() {
        for &(name, budget) in EFFORT_LEVELS {
            assert_eq!(effort_budget(Some(&json!(name)), 75).unwrap(), budget.unwrap_or(75));
        }
        for (v, want) in [(json!(30), 30), (json!(0.5), 50), (json!(80.0), 80)] {
            assert_eq!(effort_budget(Some(&v), 75).unwrap(), want);
        }
        for v in [json!("banana"), json!("4o"), json!(""), json!("0"), json!("101"), json!(true), json!(0), json!(101), json!(-5), json!(1.5e3)] {
            let e = effort_budget(Some(&v), 75).expect_err(&v.to_string());
            assert!(e.contains(&format!("invalid reasoning_effort {v};")), "the message quotes what was sent: {e}");
        }
    }

    /// What this server answered before the validation, for every input it took: the same budget now.
    #[test]
    fn efforts_keep_their_earlier_values() {
        for (v, want) in [
            (json!("low"), 50),
            (json!("medium"), 75),
            (json!("high"), 75),
            (json!("xhigh"), 75),
            (json!("max"), 100),
            (json!("HIGH"), 75),
            (json!(" Max "), 100),
            (json!(40), 40),
            (json!("40"), 40),
            (json!(1), 1),
            (json!(100), 100),
            (json!(0.5), 50),
            (json!(80.0), 80),
            (json!("0.5"), 50),
            (json!("1"), 100),
            (json!(1.0), 100),
        ] {
            assert_eq!(effort_budget(Some(&v), 75).unwrap(), want, "{v}");
        }
        // medium is the server's default effort (--effort), as before
        assert_eq!(effort_budget(Some(&json!("medium")), 60).unwrap(), 60);
        assert_eq!(effort_budget(None, 60).unwrap(), 60);
        // the new names
        assert_eq!(effort_budget(Some(&json!("none")), 75).unwrap(), 0);
        assert_eq!(effort_budget(Some(&json!("minimal")), 75).unwrap(), 1);
        assert_eq!(effort_budget(Some(&json!("ultra")), 75).unwrap(), 95);
        // through a request, at the top level and through the aliases, in the prompt the model sees
        for encoding in [Encoding::V41, Encoding::V41_1017] {
            let cfg = ServerConfig { encoding, ..cfg() };
            for (v, want) in [(json!("low"), 50), (json!("medium"), 75), (json!("xhigh"), 75), (json!(40), 40), (json!("40"), 40), (json!(0.3), 30)] {
                for field in ["top", "template", "reasoning"] {
                    let mut r = request();
                    match field {
                        "top" => r["reasoning_effort"] = v.clone(),
                        "template" => r["chat_template_kwargs"] = json!({"thinking": true, "reasoning_effort": v}),
                        _ => r["reasoning"] = json!({"effort": v}),
                    }
                    let p = prepare(&cfg, &r).unwrap_or_else(|e| panic!("{field} {v}: {e}"));
                    assert_eq!(p.mode, ThinkingMode::Thinking, "{field} {v}");
                    assert!(p.prompt.contains(&format!("Reasoning Effort: {want} ")), "{field} {v}");
                }
            }
        }
    }

    fn request() -> Value {
        json!({"messages": [{"role": "user", "content": "hi"}]})
    }

    #[test]
    fn tail_cache_keeps_history_and_moves_only_the_active_effort() {
        use crate::encoding::{ASSISTANT_SP, BOS, EOS, SYSTEM_SP, THINK_END, THINK_START, USER_SP};

        let stock = ServerConfig { encoding: Encoding::V41_1017, ..cfg() };
        let tail = ServerConfig { allow_tail_cache_render: true, ..stock.clone() };
        let mut req = json!({
            "reasoning_effort": "low",
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "first answer", "reasoning_content": "replayed reasoning"},
                {"role": "user", "content": "active user"}
            ]
        });
        let effort_hint = |budget| {
            format!("Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
        };
        let original = format!(
            "{BOS}{SYSTEM_SP}{}instructions{USER_SP}first user{ASSISTANT_SP}{THINK_END}\
             first answer{EOS}{USER_SP}active user{ASSISTANT_SP}{THINK_START}",
            effort_hint(50),
        );
        assert_eq!(prepare(&stock, &req).unwrap().prompt, original);

        let prefix = format!(
            "{BOS}{SYSTEM_SP}instructions{USER_SP}first user{ASSISTANT_SP}{THINK_START}\
             replayed reasoning{THINK_END}first answer{EOS}{USER_SP}active user"
        );
        for (tier, budget) in [("low", 50), ("max", 100)] {
            req["reasoning_effort"] = json!(tier);
            assert_eq!(prepare(&tail, &req).unwrap().prompt, format!(
                "{prefix}{SYSTEM_SP}{}{ASSISTANT_SP}{THINK_START}", effort_hint(budget),
            ));
        }
        req["thinking"] = json!({"type": "disabled"});
        assert_eq!(prepare(&tail, &req).unwrap().prompt, format!("{prefix}{ASSISTANT_SP}{THINK_END}"));
        req["messages"].as_array_mut().unwrap().extend([
            json!({"role": "assistant", "content": "active answer"}),
            json!({"role": "user", "content": "third user"}),
        ]);
        assert_eq!(prepare(&tail, &req).unwrap().prompt, format!(
            "{prefix}{ASSISTANT_SP}{THINK_END}active answer{EOS}{USER_SP}third user{ASSISTANT_SP}{THINK_END}"
        ));
    }

    #[test]
    fn tail_cache_uses_stock_for_other_encodings_markers_and_unsupported_history() {
        let stock = ServerConfig { encoding: Encoding::V41_1017, ..cfg() };
        let tail = ServerConfig { allow_tail_cache_render: true, ..stock.clone() };
        let mut req = json!({
            "reasoning_effort": "high",
            "messages": [
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "first answer", "reasoning_content": "replayed reasoning"},
                {"role": "user", "content": "active user"}
            ]
        });
        let v41_tail = ServerConfig { allow_tail_cache_render: true, ..cfg() };
        assert_eq!(prepare(&v41_tail, &req).unwrap().prompt, prepare(&cfg(), &req).unwrap().prompt);

        let original = prepare(&stock, &req).unwrap().prompt;
        for marker in [json!("deepseek-turn-v1"), json!("unknown"), Value::Null, json!(false)] {
            let mut marked = req.clone();
            marked["cache_render_format"] = marker;
            assert_eq!(prepare(&tail, &marked).unwrap().prompt, original);
        }

        req["messages"][1]["reasoning_content"] = json!({"unsupported": true});
        assert_eq!(prepare(&tail, &req).unwrap().prompt, prepare(&stock, &req).unwrap().prompt);
        req["messages"].as_array_mut().unwrap().pop();
        assert_eq!(prepare(&tail, &req).unwrap().prompt, prepare(&stock, &req).unwrap().prompt);
    }

    #[test]
    fn tail_cache_applies_default_effort_after_sorted_tool_results() {
        use crate::encoding::{ASSISTANT_SP, SYSTEM_SP, THINK_START, USER_SP};

        let stock = ServerConfig { encoding: Encoding::V41_1017, default_thinking: true, default_effort: 62, ..cfg() };
        let tail = ServerConfig { allow_tail_cache_render: true, ..stock.clone() };
        let mut req = json!({
            "tools": [{"type": "function", "function": {"name": "demo::lookup", "parameters": {"type": "object"}}}],
            "messages": [
                {"role": "user", "content": "lookup"},
                {"role": "assistant", "reasoning_content": "lookup rationale", "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "demo::lookup", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "demo::lookup", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_b", "content": "second result"},
                {"role": "tool", "tool_call_id": "call_a", "content": "first result"}
            ]
        });
        let rendered = prepare(&tail, &req).unwrap();
        assert_eq!(rendered.mode, ThinkingMode::Thinking);
        assert!(rendered.prompt.contains("lookup rationale</think>"));
        assert_eq!(rendered.prompt.matches("invoke name=\"demo::lookup\"").count(), 2);
        assert!(rendered.prompt.contains(&format!(
            "{USER_SP}<tool_result>first result</tool_result>\n\n<tool_result>second result</tool_result>\
             {SYSTEM_SP}Reasoning Effort: 62 "
        )));
        assert!(rendered.prompt.ends_with(&format!("\n\n{ASSISTANT_SP}{THINK_START}")));
        assert_eq!(rendered.prompt.matches("Reasoning Effort:").count(), 1);

        req["messages"][2]["tool_call_id"] = json!("orphaned");
        assert_eq!(prepare(&tail, &req).unwrap().prompt, prepare(&stock, &req).unwrap().prompt);
    }

    #[test]
    fn tail_cache_keeps_stock_images_and_placeholder_order() {
        use crate::encoding::{ASSISTANT_SP, IMAGE_PLACEHOLDER, SYSTEM_SP, THINK_START, USER_SP};

        let stock = ServerConfig { encoding: Encoding::V41_1017, vision_addr: Some("127.0.0.1:1".into()), ..cfg() };
        let tail = ServerConfig { allow_tail_cache_render: true, ..stock.clone() };
        let req = json!({
            "reasoning_effort": "high",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "first image"},
                    {"type": "image_url", "image_url": {"url": "synthetic-first"}}
                ]},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": [
                    {"type": "text", "text": "second image"},
                    {"type": "image_url", "image_url": {"url": "synthetic-second"}}
                ]}
            ]
        });
        let rendered = prepare(&tail, &req).unwrap();
        assert_eq!(rendered.images, prepare(&stock, &req).unwrap().images);
        assert_eq!(rendered.images, [
            json!({"type": "image", "url": "synthetic-first"}),
            json!({"type": "image", "url": "synthetic-second"}),
        ]);
        let first_slot = format!("{USER_SP}first image\n\n{IMAGE_PLACEHOLDER}{ASSISTANT_SP}");
        let second_slot = format!("{USER_SP}second image\n\n{IMAGE_PLACEHOLDER}{SYSTEM_SP}Reasoning Effort: 75 ");
        assert!(rendered.prompt.find(&first_slot).unwrap() < rendered.prompt.find(&second_slot).unwrap());
        assert!(rendered.prompt.ends_with(&format!("\n\n{ASSISTANT_SP}{THINK_START}")));
    }

    /// A server started with --thinking.
    fn cfg_thinking() -> ServerConfig {
        ServerConfig { default_thinking: true, ..cfg() }
    }

    #[test]
    fn thinking_is_off_unless_asked() {
        for encoding in [Encoding::V41, Encoding::V41_1017] {
            // the server default: a request with no thinking field is answered in chat mode
            let cfg = ServerConfig { encoding, ..cfg() };
            let p = prepare(&cfg, &request()).unwrap();
            assert_eq!(p.mode, ThinkingMode::Chat);
            assert!(!p.prompt.contains("Reasoning Effort:") && !p.prompt.ends_with("<think>"));
            // a request can still ask for it
            let mut r = request();
            r["reasoning_effort"] = json!("high");
            assert_eq!(prepare(&cfg, &r).unwrap().mode, ThinkingMode::Thinking);
            let mut r = request();
            r["chat_template_kwargs"] = json!({"thinking": true});
            assert_eq!(prepare(&cfg, &r).unwrap().mode, ThinkingMode::Thinking);
            // --thinking makes it the server-wide default, and a request can still switch it off
            let cfg = ServerConfig { encoding, ..cfg_thinking() };
            assert_eq!(prepare(&cfg, &request()).unwrap().mode, ThinkingMode::Thinking);
            let mut r = request();
            r["reasoning_effort"] = json!("none");
            assert_eq!(prepare(&cfg, &r).unwrap().mode, ThinkingMode::Chat);
        }
    }

    #[test]
    fn default_thinking_and_named_efforts() {
        for encoding in [Encoding::V41, Encoding::V41_1017] {
            let mut cfg = cfg_thinking();
            cfg.encoding = encoding;
            let p = prepare(&cfg, &request()).unwrap();
            assert_eq!(p.mode, ThinkingMode::Thinking);
            assert!(p.prompt.contains("Reasoning Effort: 75 "));
            assert!(p.prompt.ends_with("<think>"));
            for &(name, budget) in EFFORT_LEVELS {
                let mut r = request();
                r["reasoning_effort"] = json!(name);
                let p = prepare(&cfg, &r).unwrap();
                if name == "none" {
                    assert_eq!(p.mode, ThinkingMode::Chat);
                    assert!(!p.prompt.contains("Reasoning Effort:"));
                    assert!(!p.prompt.ends_with("<think>"));
                } else {
                    let budget = budget.unwrap_or(cfg.default_effort);
                    assert_eq!(p.mode, ThinkingMode::Thinking);
                    assert!(p.prompt.contains(&format!("Reasoning Effort: {budget} ")), "{name}");
                }
            }
        }
    }

    #[test]
    fn effort_validation_and_precedence() {
        for bad in [json!("banana"), json!("hihg"), json!(""), json!(true), json!(0), json!(101), json!([]), json!({})] {
            let mut r = request();
            r["reasoning_effort"] = bad;
            r["chat_template_kwargs"] = json!({"thinking": false});
            let error = prepare(&cfg(), &r).err().expect("invalid API effort must fail even with thinking off");
            assert!(error.contains("allowed values: none, minimal, low, medium, high, xhigh, ultra, max"));
        }
        let mut r = request();
        r["reasoning_effort"] = json!("none");
        r["chat_template_kwargs"] = json!({"thinking": true, "reasoning_effort": "max"});
        assert_eq!(prepare(&cfg(), &r).unwrap().mode, ThinkingMode::Chat);
        // an explicit off-switch wins over an effort, as it always did
        r["reasoning_effort"] = json!("minimal");
        r["chat_template_kwargs"] = json!({"thinking": false});
        assert_eq!(prepare(&cfg(), &r).unwrap().mode, ThinkingMode::Chat);
        assert_eq!(prepare(&cfg_thinking(), &r).unwrap().mode, ThinkingMode::Chat);
        r["chat_template_kwargs"] = json!({"reasoning_effort": "banana"});
        assert!(prepare(&cfg(), &r).is_err());
    }

    #[test]
    fn thinking_field_is_honoured() {
        let with = |cfg: &ServerConfig, extra: Value| {
            let mut r = request();
            r.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            prepare(cfg, &r)
        };
        for cfg in [cfg(), cfg_thinking()] {
            // disabled switches thinking off whatever the server default, and whatever else asks for it
            for off in [
                json!({"thinking": {"type": "disabled"}}),
                json!({"thinking": false}),
                json!({"thinking": {"type": "disabled"}, "reasoning_effort": "high"}),
                json!({"thinking": {"type": "disabled"}, "chat_template_kwargs": {"thinking": true}}),
                json!({"thinking": {"type": "disabled"}, "reasoning": {"enabled": true, "effort": "max"}}),
                // and so does every other off-switch, against the request's own "enabled"
                json!({"thinking": {"type": "enabled"}, "reasoning_effort": "none"}),
                json!({"thinking": {"type": "enabled"}, "chat_template_kwargs": {"thinking": false}}),
                json!({"thinking": true, "chat_template_kwargs": {"enable_thinking": false}}),
                json!({"thinking": true, "reasoning": {"enabled": false}}),
            ] {
                let p = with(&cfg, off.clone()).unwrap();
                assert_eq!(p.mode, ThinkingMode::Chat, "{off}");
                assert!(!p.prompt.contains("Reasoning Effort:") && !p.prompt.ends_with("<think>"), "{off}");
            }
            // enabled switches it on, at the default effort or the one the request names
            for (on, budget) in [
                (json!({"thinking": {"type": "enabled"}}), 75),
                (json!({"thinking": true}), 75),
                (json!({"thinking": {"type": "enabled", "budget_tokens": 2048}}), 75),
                (json!({"thinking": {"type": "enabled"}, "reasoning_effort": "max"}), 100),
                (json!({"thinking": {"type": "enabled"}, "chat_template_kwargs": {"reasoning_effort": 40}}), 40),
            ] {
                let p = with(&cfg, on.clone()).unwrap();
                assert_eq!(p.mode, ThinkingMode::Thinking, "{on}");
                assert!(p.prompt.contains(&format!("Reasoning Effort: {budget} ")) && p.prompt.ends_with("<think>"), "{on}");
            }
            // no opinion: the server default
            for auto in [json!({"thinking": {"type": "auto"}}), json!({"thinking": {"type": "adaptive"}}), json!({"thinking": null}), json!({"thinking": {}})] {
                let want = if cfg.default_thinking { ThinkingMode::Thinking } else { ThinkingMode::Chat };
                assert_eq!(with(&cfg, auto.clone()).unwrap().mode, want, "{auto}");
            }
            // a typo fails loudly instead of being ignored
            for bad in [json!({"thinking": {"type": "banana"}}), json!({"thinking": "sometimes"}), json!({"thinking": 1}), json!({"thinking": {"type": true}})] {
                let e = with(&cfg, bad.clone()).err().unwrap_or_else(|| panic!("{bad} must fail"));
                assert!(e.contains("allowed values: enabled, disabled, auto"), "{e}");
            }
        }
    }

    /// main's rule before this branch (f211791), copied as it was: whether a request thinks and the
    /// effort written into its prompt. It never refused a value, and it did not read `thinking`.
    fn main_rule(cfg: &ServerConfig, req: &Value) -> (bool, i64) {
        fn effort_budget(v: Option<&Value>, default: i64) -> i64 {
            match v {
                Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                    "low" => 50,
                    "high" | "xhigh" => 75,
                    "max" => 100,
                    t => t.parse::<f64>().ok().map_or(default, |f| effort_budget(Some(&json!(f)), default)),
                },
                Some(n @ Value::Number(_)) => {
                    let s = n.as_number().unwrap().to_string();
                    let f = num_f64(n).unwrap_or(f64::NAN);
                    if !s.contains(['.', 'e', 'E']) {
                        let i = f as i64;
                        if (1..=100).contains(&i) {
                            i
                        } else {
                            default
                        }
                    } else if (0.0..=1.0).contains(&f) {
                        ((f * 100.0).round() as i64).clamp(1, 100)
                    } else if (1.0..=100.0).contains(&f) {
                        f.round() as i64
                    } else {
                        default
                    }
                }
                _ => default,
            }
        }
        let r = req.as_object().unwrap();
        let ctk = r.get("chat_template_kwargs").and_then(Value::as_object);
        let reasoning = r.get("reasoning").and_then(Value::as_object);
        let top_effort = r.get("reasoning_effort").filter(|v| !v.is_null());
        let thinking = if let Some(v) = ctk.and_then(|c| c.get("thinking")).filter(|v| !v.is_null()) {
            truthy(Some(v))
        } else if let Some(v) = ctk.and_then(|c| c.get("enable_thinking")).filter(|v| !v.is_null()) {
            truthy(Some(v))
        } else if let Some(e) = top_effort.or_else(|| reasoning.and_then(|x| x.get("effort")).filter(|v| !v.is_null())) {
            e.as_str() != Some("none")
        } else if let Some(v) = reasoning.and_then(|x| x.get("enabled")).filter(|v| !v.is_null()) {
            truthy(Some(v))
        } else {
            cfg.default_thinking
        };
        let effort_src = ctk
            .and_then(|c| c.get("reasoning_effort"))
            .filter(|v| !v.is_null())
            .or(top_effort)
            .or_else(|| reasoning.and_then(|x| x.get("effort").or_else(|| x.get("reasoning_effort"))).filter(|v| !v.is_null()));
        (thinking, effort_budget(effort_src, cfg.default_effort))
    }

    /// Every combination of the fields that say whether and how much to think, each with the values main
    /// took, under each server default: the answer is main's (same mode, same effort in the prompt),
    /// with the two rules that are new and nothing else:
    ///   1. an explicit off-switch always wins (main let some on-switches win over some off-switches);
    ///   2. the request's own `thinking` field, which main did not read, is a switch.
    /// In particular a request that main answered without thinking is still answered without thinking,
    /// unless it carries `thinking` enabled, and a request without an off-switch and without the
    /// `thinking` field is answered exactly as on main.
    #[test]
    fn thinking_and_effort_as_on_main_for_every_combination() {
        let field = [None, Some(json!({"type": "enabled"})), Some(json!({"type": "disabled"})), Some(json!(true)), Some(json!(false)), Some(json!({"type": "auto"}))];
        let flags = [None, Some(true), Some(false)];
        let template_efforts = [None, Some(json!(40)), Some(json!("none")), Some(json!("high"))];
        let top_efforts = [
            None, Some(Value::Null), Some(json!("none")), Some(json!("low")), Some(json!("medium")), Some(json!("high")), Some(json!("xhigh")),
            Some(json!("max")), Some(json!("HIGH")), Some(json!(" Max ")), Some(json!(40)), Some(json!("40")), Some(json!(0.5)), Some(json!(1)),
            Some(json!(100)), Some(json!(80.0)), Some(json!("1")),
        ];
        let reasonings = [
            None, Some(json!({"enabled": true})), Some(json!({"enabled": false})), Some(json!({"effort": "high"})), Some(json!({"effort": "none"})),
            Some(json!({"reasoning_effort": "low"})), Some(json!({"reasoning_effort": "none"})), Some(json!({"effort": null, "reasoning_effort": 30})),
            Some(json!({"enabled": true, "effort": "max"})), Some(json!({"enabled": false, "effort": "max"})),
        ];
        let is_none = |v: &Value| v.as_str() == Some("none");
        let (mut cases, mut same_as_main, mut off_won, mut field_on, mut main_chat) = (0, 0, 0, 0, 0);
        for (default_thinking, default_effort) in [(false, 75), (true, 75), (false, 60), (true, 60)] {
            let cfg = ServerConfig { default_thinking, default_effort, ..cfg() };
            for f in &field {
                for t in flags {
                    for e in flags {
                        for te in &template_efforts {
                            for top in &top_efforts {
                                for re in &reasonings {
                                    let mut r = request();
                                    let mut ctk = serde_json::Map::new();
                                    if let Some(t) = t {
                                        ctk.insert("thinking".into(), json!(t));
                                    }
                                    if let Some(e) = e {
                                        ctk.insert("enable_thinking".into(), json!(e));
                                    }
                                    if let Some(te) = te {
                                        ctk.insert("reasoning_effort".into(), te.clone());
                                    }
                                    if !ctk.is_empty() {
                                        r["chat_template_kwargs"] = Value::Object(ctk);
                                    }
                                    if let Some(top) = top {
                                        r["reasoning_effort"] = top.clone();
                                    }
                                    if let Some(re) = re {
                                        r["reasoning"] = re.clone();
                                    }
                                    // what main answered: it saw everything but the `thinking` field
                                    let (main_thinks, main_effort) = main_rule(&cfg, &r);
                                    if let Some(f) = f {
                                        r["thinking"] = f.clone();
                                    }
                                    // the request's explicit off-switches, read off the request itself
                                    let off = matches!(f, Some(v) if *v == json!(false) || v["type"] == "disabled")
                                        || t == Some(false)
                                        || e == Some(false)
                                        || te.as_ref().is_some_and(is_none)
                                        || top.as_ref().is_some_and(is_none)
                                        || re.as_ref().is_some_and(|re| re["enabled"] == json!(false) || is_none(&re["effort"]) || is_none(&re["reasoning_effort"]));
                                    let on_by_field = matches!(f, Some(v) if *v == json!(true) || v["type"] == "enabled");
                                    let want = !off && (main_thinks || on_by_field);
                                    let p = prepare(&cfg, &r).unwrap_or_else(|e| panic!("main took {r}; now: {e}"));
                                    assert_eq!(p.mode == ThinkingMode::Thinking, want, "{r} (server default {default_thinking}); main thinking: {main_thinks}");
                                    if want {
                                        assert!(p.prompt.contains(&format!("Reasoning Effort: {main_effort} ")) && p.prompt.ends_with("<think>"), "{r}: main's effort is {main_effort}");
                                    } else {
                                        assert!(!p.prompt.contains("Reasoning Effort:") && !p.prompt.ends_with("<think>"), "{r}");
                                    }
                                    // the guarantees, spelled out
                                    if !main_thinks && !on_by_field {
                                        assert_eq!(p.mode, ThinkingMode::Chat, "main answered {r} without thinking");
                                        main_chat += 1;
                                    }
                                    if want == main_thinks {
                                        same_as_main += 1;
                                    } else if main_thinks {
                                        assert!(off, "{r}: thinking on main, chat now, with no off-switch in the request");
                                        off_won += 1;
                                    } else {
                                        assert!(on_by_field && !off, "{r}: chat on main, thinking now, without `thinking` enabled");
                                        field_on += 1;
                                    }
                                    if f.is_none() && !off {
                                        assert_eq!(want, main_thinks, "{r}: no off-switch and no `thinking` field");
                                    }
                                    cases += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 4 * 6 * 3 * 3 * 4 * 17 * 10);
        assert_eq!(cases, same_as_main + off_won + field_on);
        assert!(same_as_main > cases / 2 && off_won > 0 && field_on > 0 && main_chat > 0, "{cases} {same_as_main} {off_won} {field_on} {main_chat}");
    }

    /// The intended differences from main for values, apart from the two rules above: a value that is
    /// not an effort returns 400 wherever it is sent (main fell back to the default silently), and the
    /// new names have their own value (main read `minimal` and `ultra` as unknown: the default, 75).
    #[test]
    fn what_differs_from_main_is_a_400_or_a_new_name() {
        let cfg = cfg();
        for bad in [json!("banana"), json!("hihg"), json!(""), json!("0"), json!("101"), json!(0), json!(101), json!(-5), json!(true), json!([]), json!({})] {
            for place in ["top", "template", "reasoning.effort", "reasoning.reasoning_effort"] {
                // also next to an off-switch, and where another field's effort is the one used
                for extra in [json!({}), json!({"thinking": false}), json!({"thinking": true, "reasoning_effort": "max"})] {
                    let mut r = request();
                    r["chat_template_kwargs"] = extra.clone();
                    match place {
                        "top" => r["reasoning_effort"] = bad.clone(),
                        "template" => r["chat_template_kwargs"]["reasoning_effort"] = bad.clone(),
                        "reasoning.effort" => r["reasoning"] = json!({"effort": bad}),
                        _ => r["reasoning"] = json!({"reasoning_effort": bad}),
                    }
                    if place == "template" && extra.get("reasoning_effort").is_some() {
                        continue;
                    }
                    let _ = main_rule(&cfg, &r); // main answered it
                    let e = prepare(&cfg, &r).err().unwrap_or_else(|| panic!("{r} must be refused"));
                    assert!(e.starts_with("invalid reasoning_effort ") && e.contains("allowed values: none, minimal, low, medium, high, xhigh, ultra, max"), "{e}");
                }
            }
        }
        for (name, now) in [("minimal", 1), ("ultra", 95)] {
            let mut r = request();
            r["reasoning_effort"] = json!(name);
            assert_eq!(main_rule(&cfg, &r), (true, 75));
            let p = prepare(&cfg, &r).unwrap();
            assert!(p.mode == ThinkingMode::Thinking && p.prompt.contains(&format!("Reasoning Effort: {now} ")), "{name}");
        }
        // `none` switches off in any case and with spaces (main: only the exact lower-case word)
        let mut r = request();
        r["reasoning_effort"] = json!(" None ");
        assert_eq!(main_rule(&cfg_thinking(), &r).0, true);
        assert_eq!(prepare(&cfg_thinking(), &r).unwrap().mode, ThinkingMode::Chat);
    }

    #[test]
    fn thinking_aliases_remain_supported() {
        for alias in [
            json!({"chat_template_kwargs": {"thinking": false}}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
            json!({"chat_template_kwargs": {"thinking": true, "reasoning_effort": "none"}}),
            json!({"reasoning": {"enabled": false}}),
            json!({"reasoning": {"effort": "none"}}),
            json!({"reasoning": {"reasoning_effort": "none"}}),
        ] {
            let mut r = request();
            r.as_object_mut().unwrap().extend(alias.as_object().unwrap().clone());
            assert_eq!(prepare(&cfg_thinking(), &r).unwrap().mode, ThinkingMode::Chat, "{r}");
        }
        let mut r = request();
        r["reasoning_effort"] = Value::Null;
        assert_eq!(prepare(&cfg_thinking(), &r).unwrap().mode, ThinkingMode::Thinking);
        let cfg = cfg();
        assert_eq!(prepare(&cfg, &request()).unwrap().mode, ThinkingMode::Chat);
        r["reasoning"] = json!({"effort": "low"});
        assert_eq!(prepare(&cfg, &r).unwrap().mode, ThinkingMode::Thinking);
    }

    #[test]
    fn image_count_boundary() {
        let mut cfg = cfg();
        cfg.vision_addr = Some("127.0.0.1:18950".into());
        let image = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}});
        let mut r = json!({"messages": [{"role": "user", "content": "hi"}]});
        r["messages"][0]["content"] = json!(vec![image.clone(); 100]);
        assert_eq!(prepare(&cfg, &r).unwrap().images.len(), 100);
        r["messages"][0]["content"] = json!(vec![image; 101]);
        assert_eq!(prepare(&cfg, &r).err().unwrap(), "at most 100 images per request");
    }

    #[test]
    fn prepare_like_the_kit() {
        let req: Value = serde_json::from_str(
            r#"{"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "tools": [{"type": "function", "function": {"parameters": {"type": "object"}, "name": "f", "extra": 1}}],
                "chat_template_kwargs": {"thinking": true, "reasoning_effort": "max"}, "max_tokens": 100000, "stop": "x"}"#,
        )
        .unwrap();
        let p = prepare(&cfg(), &req).unwrap();
        assert!(p.prompt.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 100 (range 1-100"));
        assert!(p.prompt.contains("be brief\n\n## Tools"));
        assert!(p.prompt.contains("{\"name\": \"f\", \"parameters\": {\"type\": \"object\"}}"));
        assert!(p.prompt.ends_with("<｜User｜>hi<｜Assistant｜><think>"));
        assert_eq!((p.max_tokens, p.tools, p.mode), (32768, true, ThinkingMode::Thinking));
        assert_eq!(p.stops, vec!["x".to_string()]);
    }

    fn image_request(n: usize) -> Value {
        let image = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}});
        json!({"messages": [{"role": "user", "content": vec![image; n]}]})
    }

    #[test]
    fn image_rows_in_flight_are_limited() {
        use crate::vision::{RowsInFlight, DEFAULT_MAX_IMAGE_ROWS_IN_FLIGHT, MAX_IMAGE_ROWS};
        let mut cfg = cfg();
        cfg.vision_addr = Some("127.0.0.1:18950".into());
        assert_eq!(cfg.max_image_rows_in_flight, 80_000, "the default: two requests at the default request cap");
        let budget = RowsInFlight::new(DEFAULT_MAX_IMAGE_ROWS_IN_FLIGHT);
        // a request with images holds the most they can take: 1100 rows each, at most the request cap
        let mut one = prepare(&cfg, &image_request(1)).unwrap();
        hold_image_rows(&cfg, &budget, &mut one).map_err(|_| ()).unwrap();
        assert_eq!(budget.in_flight(), MAX_IMAGE_ROWS);
        let mut a = prepare(&cfg, &image_request(100)).unwrap();
        hold_image_rows(&cfg, &budget, &mut a).map_err(|_| ()).unwrap();
        assert_eq!(budget.in_flight(), MAX_IMAGE_ROWS + 40_000);
        drop(one);
        // two full requests fit; the third is refused at once: 429, Retry-After, and it holds nothing
        let mut b = prepare(&cfg, &image_request(40)).unwrap();
        hold_image_rows(&cfg, &budget, &mut b).map_err(|_| ()).unwrap();
        assert_eq!(budget.in_flight(), 80_000);
        for n in [1, 40, 100] {
            let mut c = prepare(&cfg, &image_request(n)).unwrap();
            let refusal = hold_image_rows(&cfg, &budget, &mut c).unwrap_err();
            assert_eq!(refusal.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(refusal.headers()["retry-after"], "5");
            assert!(c.rows_held.is_none());
        }
        assert_eq!(budget.in_flight(), 80_000);
        // a text request is never counted or refused, however full the budget is
        let mut text = prepare(&cfg, &json!({"messages": [{"role": "user", "content": "hi"}]})).unwrap();
        assert!(hold_image_rows(&cfg, &budget, &mut text).is_ok() && text.rows_held.is_none());
        assert_eq!(budget.in_flight(), 80_000);
        // once the encoder has answered, a request holds what its images take, and others fit again
        a.rows_held.as_ref().unwrap().shrink_to(38_300);
        assert_eq!(budget.in_flight(), 78_300);
        let mut d = prepare(&cfg, &image_request(1)).unwrap();
        hold_image_rows(&cfg, &budget, &mut d).map_err(|_| ()).unwrap();
        assert_eq!(budget.in_flight(), 79_400);
        // a request that ends (answered, failed, or its client gone: the request is dropped) gives its rows back
        drop(a);
        drop(b);
        drop(d);
        assert_eq!(budget.in_flight(), 0);
    }

    #[tokio::test]
    async fn the_refusal_for_rows_in_flight_reads_as_an_api_error() {
        let mut cfg = cfg();
        cfg.vision_addr = Some("127.0.0.1:18950".into());
        let budget = crate::vision::RowsInFlight::new(40_000);
        let mut a = prepare(&cfg, &image_request(40)).unwrap();
        hold_image_rows(&cfg, &budget, &mut a).map_err(|_| ()).unwrap();
        let mut b = prepare(&cfg, &image_request(3)).unwrap();
        let refusal = hold_image_rows(&cfg, &budget, &mut b).unwrap_err();
        let body = axum::body::to_bytes(refusal.into_body(), 1 << 20).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], 429);
        assert_eq!(v["error"]["type"], "rate_limit_error");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("hold 40000 of 40000 image tokens (--max-image-rows-in-flight)") && msg.contains("3 images and video frames may take up to 3300"), "{msg}");
    }

    /// POST a body of `len` bytes to a route that takes its body the way the chat route does, behind
    /// the body limit `max`: the HTTP status.
    async fn post_status(max: usize, len: usize) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let routes: Router = body_limit(Router::new().route("/v1/chat/completions", post(|body: Bytes| async move { body.len().to_string() })), max);
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(l, routes).await.unwrap() });
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        let head = format!("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {len}\r\nconnection: close\r\n\r\n");
        s.write_all(head.as_bytes()).await.unwrap();
        let _ = s.write_all(&vec![b' '; len]).await; // the server may answer 413 and close before it is all sent
        let mut reply = Vec::new();
        let _ = s.read_to_end(&mut reply).await;
        server.abort();
        let reply = String::from_utf8_lossy(&reply);
        reply.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or_else(|| panic!("no status in {reply:?}"))
    }

    #[tokio::test]
    async fn body_over_the_limit_is_413() {
        assert_eq!(DEFAULT_MAX_BODY_BYTES, 64 << 20, "the default stays at 64 MiB; --max-body-mb raises it");
        assert_eq!(cfg().max_body_bytes, DEFAULT_MAX_BODY_BYTES);
        // a small limit, so the test moves kilobytes: at the limit passes, one byte over is refused
        assert_eq!(post_status(4096, 4096).await, 200);
        assert_eq!(post_status(4096, 4097).await, 413);
        assert_eq!(post_status(4096, 1 << 20).await, 413);
        // the limit follows the setting
        assert_eq!(post_status(2 << 20, 1 << 20).await, 200);
    }
}
