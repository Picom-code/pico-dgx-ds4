//! Port of the publisher's DeepSeek-V4.1 chat encoding (`encoding/encoding.py`), text only.
//! A derivative of that MIT-licensed code (Copyright (c) 2023 DeepSeek, `ref/LICENSE-deepseek`; the
//! root `NOTICE`).
//!
//! Messages stay as JSON values (OpenAI chat format) and every function mirrors its Python
//! counterpart, including the quirks: Python truthiness (`content or ""`), `str()` of non-string
//! values formatted into templates, `json.dumps`/`json.loads` behaviour for tool schemas and
//! arguments (pyjson.rs), and an error wherever the Python raises. `dsfront enccheck` compares it
//! byte for byte with encoding.py over cases from tools/gen_encoding_cases.py.
//!
//! The parse side has the reference's strict parser and `StreamParser`, an incremental version for
//! streamed output that falls back to plain content where the strict one would raise.

use crate::pyjson::{dumps as to_json, py_str, truthy, write_escaped};
use serde_json::{json, Map, Value};

pub const BOS: &str = "<｜begin▁of▁sentence｜>";
pub const EOS: &str = "<｜end▁of▁sentence｜>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const DSML: &str = "｜DSML｜";
pub const USER_SP: &str = "<｜User｜>";
pub const ASSISTANT_SP: &str = "<｜Assistant｜>";
pub const LATEST_REMINDER_SP: &str = "<｜latest_reminder｜>";
pub const SYSTEM_SP: &str = "<｜System｜>";
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

const TASK_TOKENS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

/// encoding.py's REASONING_EFFORT_MAPPINGS and default.
pub const EFFORT_NAMES: [(&str, i64); 4] = [("low", 25), ("high", 50), ("xhigh", 75), ("max", 100)];
/// DeepSeek-V4.1-Flash-1017's encoding.py: no "xhigh", and each name one step up.
pub const EFFORT_NAMES_1017: [(&str, i64); 3] = [("low", 50), ("high", 75), ("max", 100)];
pub const DEFAULT_EFFORT_NAME: &str = "high";

/// Which publisher's encoding.py the port follows. Both are the V4.1 format; 1017's adds tool
/// namespaces (`namespace::name`), keeps the reasoning within a turn across tool calls (a user message
/// of tool results alone does not start a turn), keeps a tool result's own content blocks, merges tool
/// messages before images are collected, and renames the effort tiers (EFFORT_NAMES_1017).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Encoding {
    /// `deepseek-ai/DeepSeek-V4.1-Flash` @ fb2764a5 (ref/encoding), and its uncensored variant.
    #[default]
    V41,
    /// `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ 1bae0e10 (ref/encoding-1017).
    V41_1017,
}

impl Encoding {
    pub fn parse(s: &str) -> Result<Self, EncodeError> {
        match s {
            "v4.1" | "v41" => Ok(Encoding::V41),
            "1017" | "v4.1-1017" => Ok(Encoding::V41_1017),
            _ => Err(err(format!("unknown encoding `{s}`: v4.1 or 1017"))),
        }
    }

    fn effort_names(self) -> &'static [(&'static str, i64)] {
        match self {
            Encoding::V41 => &EFFORT_NAMES,
            Encoding::V41_1017 => &EFFORT_NAMES_1017,
        }
    }
}

const TOOL_CALLS_BLOCK: &str = " calls";
const TOOL_CALL_TAG: &str = " invoke";
const TOOL_PARAM_TAG: &str = " parameter";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    pub fn parse(s: &str) -> Result<Self, EncodeError> {
        match s {
            "chat" => Ok(ThinkingMode::Chat),
            "thinking" => Ok(ThinkingMode::Thinking),
            _ => Err(err(format!("Invalid thinking_mode `{s}`"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodeError(pub String);

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn err(s: impl Into<String>) -> EncodeError {
    EncodeError(s.into())
}

pub struct EncodeOptions<'a> {
    pub thinking_mode: ThinkingMode,
    pub context: Option<&'a [Value]>,
    pub drop_thinking: bool,
    pub add_default_bos_token: bool,
    /// An int in [1, 100] or one of the encoding's effort names; None (or JSON null) means the default.
    pub reasoning_effort: Option<&'a Value>,
    pub encoding: Encoding,
}

impl<'a> EncodeOptions<'a> {
    pub fn new(thinking_mode: ThinkingMode) -> Self {
        EncodeOptions {
            thinking_mode,
            context: None,
            drop_thinking: true,
            add_default_bos_token: true,
            reasoning_effort: None,
            encoding: Encoding::V41,
        }
    }
}

// ------------------------------------------------------------------ small helpers

fn get<'v>(m: &'v Value, k: &str) -> Option<&'v Value> {
    m.as_object().and_then(|o| o.get(k))
}

fn get_str<'v>(m: &'v Value, k: &str) -> Option<&'v str> {
    get(m, k).and_then(Value::as_str)
}

fn role(m: &Value) -> Option<&str> {
    get_str(m, "role")
}

/// Python `x or ""` followed by formatting into a template.
fn or_empty(v: Option<&Value>) -> String {
    if truthy(v) {
        py_str(v.unwrap())
    } else {
        String::new()
    }
}

fn obj_mut(m: &mut Value) -> &mut Map<String, Value> {
    m.as_object_mut().expect("message is an object")
}

fn is_task_set(m: &Value) -> bool {
    !matches!(get(m, "task"), None | Some(Value::Null))
}

// ------------------------------------------------------------------ OpenAI-format conversions

/// Python None: absent or JSON null.
fn py_none(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null))
}

/// Python `"::" in x` for a namespace: a substring of a str, a key of a dict, an item of a list.
fn py_contains_sep(v: &Value) -> Result<bool, EncodeError> {
    match v {
        Value::String(s) => Ok(s.contains("::")),
        Value::Object(o) => Ok(o.contains_key("::")),
        Value::Array(a) => Ok(a.iter().any(|x| x.as_str() == Some("::"))),
        _ => Err(err("TypeError: argument of type is not iterable")),
    }
}

/// encoding.py (1017) _split_tool_name: `namespace::name` into its parts, checking an explicit namespace
/// against the prefix. The namespace comes back as given (None, or any value) unless the name carries one.
fn split_tool_name(name: Option<&Value>, namespace: Option<&Value>) -> Result<(Option<Value>, String), EncodeError> {
    let name = match name {
        None => return Err(err("KeyError: 'name'")),
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return Err(err("AttributeError: tool name has no attribute 'partition'")),
    };
    let mut namespace = if py_none(namespace) { None } else { namespace.cloned() };
    let mut name = name.to_string();
    if let Some((prefix, bare)) = name.split_once("::") {
        if !(namespace.is_none() || namespace.as_ref().and_then(Value::as_str) == Some(prefix)) {
            return Err(err(format!("Conflicting tool namespaces: {} != {prefix}", py_str(namespace.as_ref().unwrap()))));
        }
        let (p, b) = (prefix.to_string(), bare.to_string());
        namespace = Some(Value::String(p));
        name = b;
    }
    if name.contains("::") {
        return Err(err(format!("Tool name must not contain '::': {name}")));
    }
    if let Some(ns) = &namespace {
        if py_contains_sep(ns)? {
            return Err(err(format!("Tool namespace must not contain '::': {}", py_str(ns))));
        }
    }
    Ok((namespace, name))
}

/// encoding.py (1017) _tool_name_for_encoding: the name as the prompt writes it, `namespace::name` or
/// `name`; a namespace object stands for its "name".
fn tool_name_for_encoding(tool: &Value) -> Result<String, EncodeError> {
    let ns = match get(tool, "namespace") {
        Some(Value::Object(o)) => Some(o.get("name").ok_or_else(|| err("KeyError: 'name'"))?),
        other => other,
    };
    let (ns, name) = split_tool_name(get(tool, "name"), ns)?;
    Ok(match ns {
        None => name,
        Some(ns) => format!("{}::{name}", py_str(&ns)),
    })
}

/// The name of a tool call the model wrote, for the API. 1017's decode_dsml_to_arguments splits
/// `namespace::name`, and its tool_calls_to_openai_format puts the namespace beside the function; a name
/// with a second `::`, which it refuses, is reported as written. V4.1 reports every name as written.
pub fn api_tool_name(enc: Encoding, name: &str) -> (Option<&str>, &str) {
    if enc == Encoding::V41_1017 {
        if let Some((ns, bare)) = name.split_once("::") {
            if !bare.contains("::") {
                return (Some(ns), bare);
            }
        }
    }
    (None, name)
}

fn tools_from_openai_format(tools: &Value, enc: Encoding) -> Result<Vec<Value>, EncodeError> {
    let arr = tools.as_array().ok_or_else(|| err("tools must be a list"))?;
    if enc == Encoding::V41 {
        return arr.iter().map(|t| get(t, "function").cloned().ok_or_else(|| err("KeyError: 'function'"))).collect();
    }
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let t = t.as_object().ok_or_else(|| err("AttributeError: tool is not an object"))?;
        let mut f = match t.get("function") {
            None => return Err(err("KeyError: 'function'")),
            Some(Value::Object(f)) => f.clone(),
            Some(_) => return Err(err("TypeError: tool function is not a dict")),
        };
        if !py_none(t.get("namespace")) {
            f.insert("namespace".into(), t["namespace"].clone());
        }
        let name = tool_name_for_encoding(&Value::Object(f.clone()))?;
        f.insert("name".into(), Value::String(name));
        if let Some(Value::Object(ns)) = f.shift_remove("namespace") {
            if truthy(ns.get("description")) {
                let nd = ns["description"].as_str().ok_or_else(|| err("TypeError: namespace description is not a str"))?;
                let fd = match f.get("description") {
                    d if !truthy(d) => "",
                    Some(Value::String(s)) => s.as_str(),
                    Some(_) => return Err(err("TypeError: tool description is not a str")),
                    None => unreachable!(),
                };
                let d = format!("{nd}\n{fd}");
                f.insert("description".into(), Value::String(d));
            }
        }
        out.push(Value::Object(f));
    }
    Ok(out)
}

/// (name, arguments) per call. Under 1017 the name is the one the prompt writes, namespace included.
fn tool_calls_from_openai_format(tcs: &Value, enc: Encoding) -> Result<Vec<(Value, Value)>, EncodeError> {
    let arr = tcs.as_array().ok_or_else(|| err("tool_calls must be a list"))?;
    arr.iter()
        .map(|tc| {
            let f = get(tc, "function").ok_or_else(|| err("KeyError: 'function'"))?;
            if enc == Encoding::V41 {
                let name = get(f, "name").cloned().ok_or_else(|| err("KeyError: 'name'"))?;
                let args = get(f, "arguments").cloned().ok_or_else(|| err("KeyError: 'arguments'"))?;
                return Ok((name, args));
            }
            if !f.is_object() {
                return Err(err("TypeError: tool call function is not a dict"));
            }
            // `tool_call.get("namespace") or function.get("namespace")`
            let ns = get(tc, "namespace").filter(|v| truthy(Some(v))).or_else(|| get(f, "namespace"));
            let (ns, name) = split_tool_name(get(f, "name"), ns)?;
            let args = get(f, "arguments").cloned().ok_or_else(|| err("KeyError: 'arguments'"))?;
            let mut call = json!({"name": name});
            if let Some(ns) = ns {
                call["namespace"] = ns;
            }
            Ok((Value::String(tool_name_for_encoding(&call)?), args))
        })
        .collect()
}

pub fn tool_calls_to_openai_format(calls: &[(String, String)]) -> Vec<Value> {
    calls
        .iter()
        .map(|(name, args)| json!({"type": "function", "function": {"name": name, "arguments": args}}))
        .collect()
}

/// encoding.py decode_dsml_to_arguments: rebuild a JSON arguments string from DSML parameters.
fn decode_dsml_to_arguments(args: &[(String, String, String)]) -> String {
    let parts: Vec<String> = args
        .iter()
        .map(|(k, v, is_str)| {
            let value = if is_str == "true" { to_json(&Value::String(v.clone())) } else { v.clone() };
            format!("{}: {}", to_json(&Value::String(k.clone())), value)
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

// ------------------------------------------------------------------ preprocessing

/// encoding.py merge_tool_messages. Under 1017 a tool message's content blocks are its result when
/// its content is not a string, and a user message's list content is its blocks.
pub fn merge_tool_messages(messages: &[Value], enc: Encoding) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();
    for msg in messages {
        let mut msg = msg.clone();
        match role(&msg) {
            Some("tool") => {
                let mut content = get(&msg, "content").cloned().unwrap_or(json!(""));
                if enc == Encoding::V41_1017
                    && truthy(get(&msg, "content_blocks"))
                    && !matches!(get(&msg, "content"), Some(Value::String(_)))
                {
                    content = get(&msg, "content_blocks").unwrap().clone();
                }
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": get(&msg, "tool_call_id").cloned().unwrap_or(json!("")),
                    "content": content,
                });
                let last_is_merged_user = merged
                    .last()
                    .map_or(false, |m| role(m) == Some("user") && get(m, "content_blocks").is_some());
                if last_is_merged_user {
                    if let Some(Value::Array(b)) = obj_mut(merged.last_mut().unwrap()).get_mut("content_blocks") {
                        b.push(block);
                    }
                } else {
                    merged.push(json!({"role": "user", "content_blocks": [block]}));
                }
            }
            Some("user") => {
                let blocks = match get(&msg, "content_blocks") {
                    None | Some(Value::Null) => match get(&msg, "content").cloned().unwrap_or(json!("")) {
                        c @ Value::Array(_) if enc == Encoding::V41_1017 => c,
                        c => json!([{"type": "text", "text": c}]),
                    },
                    Some(b) => b.clone(),
                };
                let extend = merged.last().map_or(false, |m| {
                    role(m) == Some("user") && get(m, "content_blocks").is_some() && !is_task_set(m)
                });
                if extend {
                    if let (Some(Value::Array(dst)), Value::Array(src)) =
                        (obj_mut(merged.last_mut().unwrap()).get_mut("content_blocks"), blocks)
                    {
                        dst.extend(src);
                    }
                } else {
                    obj_mut(&mut msg).insert("content_blocks".into(), blocks);
                    merged.push(msg);
                }
            }
            _ => merged.push(msg),
        }
    }
    merged
}

/// encoding.py sort_tool_results_by_call_order (in place).
pub fn sort_tool_results_by_call_order(messages: &mut [Value]) {
    let mut order: Vec<(String, usize)> = Vec::new();
    for msg in messages.iter_mut() {
        let r = role(msg).map(str::to_string);
        if r.as_deref() == Some("assistant") && truthy(get(msg, "tool_calls")) {
            order.clear();
            if let Some(tcs) = get(msg, "tool_calls").and_then(Value::as_array) {
                for (idx, tc) in tcs.iter().enumerate() {
                    let id = get(tc, "id");
                    let id = if truthy(id) {
                        id.cloned()
                    } else {
                        Some(get(tc, "function").and_then(|f| get(f, "id")).cloned().unwrap_or(json!("")))
                    };
                    if let Some(id) = id {
                        if truthy(Some(&id)) {
                            let key = to_json(&id);
                            if let Some(e) = order.iter_mut().find(|(k, _)| *k == key) {
                                e.1 = idx;
                            } else {
                                order.push((key, idx));
                            }
                        }
                    }
                }
            }
        } else if r.as_deref() == Some("user") && truthy(get(msg, "content_blocks")) {
            let blocks = match get(msg, "content_blocks").and_then(Value::as_array) {
                Some(b) => b.clone(),
                None => continue,
            };
            let is_tr = |b: &Value| get_str(b, "type") == Some("tool_result");
            let mut tool_blocks: Vec<Value> = blocks.iter().filter(|b| is_tr(b)).cloned().collect();
            if tool_blocks.len() > 1 && !order.is_empty() {
                let key_of = |b: &Value| {
                    let id = to_json(get(b, "tool_use_id").unwrap_or(&json!("")));
                    order.iter().find(|(k, _)| *k == id).map_or(0, |(_, i)| *i)
                };
                tool_blocks.sort_by_key(|b| key_of(b)); // stable, like Python's sorted
                let mut it = tool_blocks.into_iter();
                let new_blocks: Vec<Value> =
                    blocks.into_iter().map(|b| if is_tr(&b) { it.next().unwrap() } else { b }).collect();
                obj_mut(msg).insert("content_blocks".into(), Value::Array(new_blocks));
            }
        }
    }
}

fn is_image_block(b: &Value) -> bool {
    matches!(get_str(b, "type"), Some("image") | Some("image_url"))
}

fn extract_image(b: &Value) -> Result<Value, EncodeError> {
    let mut rec = Map::new();
    rec.insert("type".into(), json!("image"));
    if get_str(b, "type") == Some("image_url") {
        let url = match get(b, "image_url") {
            Some(Value::String(s)) => json!(s),
            Some(v) if truthy(Some(v)) => get(v, "url").cloned().unwrap_or(json!("")),
            _ => json!(""),
        };
        rec.insert("url".into(), url);
    } else {
        for k in ["source", "url", "data"] {
            if let Some(v) = get(b, k) {
                rec.insert(k.into(), v.clone());
            }
        }
    }
    if !["source", "url", "data"].iter().any(|k| truthy(rec.get(*k))) {
        return Err(err("Image block does not contain a valid source"));
    }
    Ok(Value::Object(rec))
}

fn process_image_blocks(blocks: &[Value]) -> Result<(Vec<Value>, Vec<Value>), EncodeError> {
    let mut new_blocks = Vec::new();
    let mut images = Vec::new();
    for block in blocks {
        if !block.is_object() {
            new_blocks.push(block.clone());
            continue;
        }
        if is_image_block(block) {
            new_blocks.push(json!({"type": "text", "text": IMAGE_PLACEHOLDER}));
            images.push(extract_image(block)?);
        } else if get_str(block, "type") == Some("tool_result") && get(block, "content").map_or(false, Value::is_array) {
            let mut b = block.clone();
            let inner = get(block, "content").unwrap().as_array().unwrap().clone();
            let (nb, nested) = process_image_blocks(&inner)?;
            obj_mut(&mut b).insert("content".into(), Value::Array(nb));
            new_blocks.push(b);
            images.extend(nested);
        } else if get_str(block, "type") == Some("text") {
            let text = or_empty(get(block, "text"));
            if text.contains(IMAGE_PLACEHOLDER) {
                return Err(err("Text block contains the image placeholder. Images should be separate content blocks."));
            }
            new_blocks.push(block.clone());
        } else {
            new_blocks.push(block.clone());
        }
    }
    Ok((new_blocks, images))
}

/// The image placeholder typed as ordinary text (a doc, a log, an error message quoted back) is not an
/// image slot. Rewrite it with ASCII bars in every string of a request, so it tokenises as plain text and
/// the request is served; the reference encoder rejects such requests. Image parts are unaffected: their
/// placeholder is inserted later, by process_image_blocks.
pub fn neutralize_image_placeholder(v: &mut Value) {
    match v {
        Value::String(s) => {
            if s.contains(IMAGE_PLACEHOLDER) {
                *s = s.replace(IMAGE_PLACEHOLDER, "<|deepseek_image|>");
            }
        }
        Value::Array(a) => a.iter_mut().for_each(neutralize_image_placeholder),
        Value::Object(o) => o.values_mut().for_each(neutralize_image_placeholder),
        _ => {}
    }
}

fn validate_no_image_sp_tokens(msg: &Value) -> Result<(), EncodeError> {
    if let Some(Value::String(c)) = get(msg, "content") {
        if c.contains(IMAGE_PLACEHOLDER) {
            return Err(err("Message content contains the image special token."));
        }
    }
    if let Some(Value::String(c)) = get(msg, "reasoning_content") {
        if c.contains(IMAGE_PLACEHOLDER) {
            return Err(err("reasoning_content contains the image special token"));
        }
    }
    Ok(())
}

/// encoding.py process_image_messages: list content becomes content_blocks, images become placeholders.
pub fn process_image_messages(messages: &[Value]) -> Result<(Vec<Value>, Vec<Value>), EncodeError> {
    let mut processed = Vec::new();
    let mut images = Vec::new();
    for msg in messages {
        let mut msg = msg.clone();
        if !msg.is_object() {
            return Err(err("message must be an object"));
        }
        validate_no_image_sp_tokens(&msg)?;
        if get(&msg, "content").map_or(false, Value::is_array) && get(&msg, "content_blocks").is_none() {
            let c = obj_mut(&mut msg).shift_remove("content").unwrap();
            obj_mut(&mut msg).insert("content_blocks".into(), c);
        }
        if truthy(get(&msg, "content_blocks")) {
            let blocks = get(&msg, "content_blocks").unwrap().as_array().cloned().ok_or_else(|| err("content_blocks must be a list"))?;
            let (nb, imgs) = process_image_blocks(&blocks)?;
            images.extend(imgs);
            let needs_content = !matches!(get(&msg, "content"), Some(Value::String(_)));
            if needs_content {
                let mut texts = Vec::new();
                for b in &nb {
                    if b.is_object() && get_str(b, "type") == Some("text") {
                        match get(b, "text") {
                            None => texts.push(String::new()),
                            Some(Value::String(s)) => texts.push(s.clone()),
                            Some(_) => return Err(err("TypeError: text block text must be a string")),
                        }
                    }
                }
                obj_mut(&mut msg).insert("content".into(), json!(texts.join("\n\n")));
            }
            obj_mut(&mut msg).insert("content_blocks".into(), Value::Array(nb));
        }
        processed.push(msg);
    }
    Ok((processed, images))
}

// ------------------------------------------------------------------ rendering

fn render_reasoning_effort(index: usize, mode: ThinkingMode, effort: Option<&Value>, enc: Encoding) -> Result<String, EncodeError> {
    let names = enc.effort_names();
    let budget = match effort {
        None | Some(Value::Null) => names.iter().find(|(n, _)| *n == DEFAULT_EFFORT_NAME).unwrap().1,
        Some(Value::String(s)) => names
            .iter()
            .find(|(n, _)| n == s)
            .map(|(_, b)| *b)
            .ok_or_else(|| err(format!("Invalid reasoning effort for deepseek_v41: {s}")))?,
        Some(Value::Number(n)) => {
            let t = n.to_string();
            match (t.contains(['.', 'e', 'E']), t.parse::<i64>()) {
                (false, Ok(v)) if (1..=100).contains(&v) => v,
                _ => return Err(err(format!("Invalid reasoning effort for deepseek_v41: {t}"))),
            }
        }
        Some(v) => return Err(err(format!("Invalid reasoning effort for deepseek_v41: {v}"))),
    };
    if index == 0 && mode == ThinkingMode::Thinking {
        Ok(format!(
            "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
        ))
    } else {
        Ok(String::new())
    }
}

fn render_tools(tools: &[Value]) -> String {
    let schemas: Vec<String> = tools.iter().map(to_json).collect();
    let d = DSML;
    let b = TOOL_CALLS_BLOCK;
    let c = TOOL_CALL_TAG;
    let p = TOOL_PARAM_TAG;
    format!(
        "## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{d}{b}>\" block like the following:

<{d}{b}>
<{d}{c} name=\"$TOOL_NAME\">
<{d}{p} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{d}{p}>
...
</{d}{c}>
<{d}{c} name=\"$TOOL_NAME2\">
...
</{d}{c}>
</{d}{b}>

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by {THINK_START}), you MUST output your complete reasoning inside {THINK_START}...{THINK_END} BEFORE any tool calls or final response.

Otherwise, output directly after {THINK_END} with tool calls or final response.

### Available Tool Schemas

{}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
",
        schemas.join("\n")
    )
}

/// json.loads, or None where Python raises.
fn py_json_loads(s: &str) -> Option<Value> {
    crate::pyjson::loads(s)
}

/// encoding.py encode_arguments_to_dsml.
fn encode_arguments_to_dsml(arguments: &Value) -> String {
    let mut args = arguments.clone();
    if !args.is_object() {
        for _ in 0..2 {
            match &args {
                Value::String(s) => match py_json_loads(s) {
                    Some(v) => args = v,
                    None => break,
                },
                _ => break,
            }
        }
        if !args.is_object() {
            args = json!({"arguments": arguments.clone()});
        }
    }
    let mut out = Vec::new();
    for (k, v) in args.as_object().unwrap() {
        let (is_str, value) = match v {
            Value::String(s) => ("true", s.clone()),
            other => ("false", to_json(other)),
        };
        out.push(format!("<{DSML}{TOOL_PARAM_TAG} name=\"{k}\" string=\"{is_str}\">{value}</{DSML}{TOOL_PARAM_TAG}>"));
    }
    out.join("\n")
}

/// The tool-call block of an assistant message, from (name, arguments) pairs in the OpenAI form
/// (arguments a JSON string or an object): `\n\n<｜DSML｜ calls>...</｜DSML｜ calls>`.
pub fn render_tool_calls(calls: &[(Value, Value)]) -> String {
    let list: Vec<String> = calls
        .iter()
        .map(|(name, args)| {
            format!(
                "<{DSML}{TOOL_CALL_TAG} name=\"{}\">\n{}\n</{DSML}{TOOL_CALL_TAG}>",
                py_str(name),
                encode_arguments_to_dsml(args)
            )
        })
        .collect();
    format!("\n\n<{DSML}{TOOL_CALLS_BLOCK}>\n{}\n</{DSML}{TOOL_CALLS_BLOCK}>", list.join("\n"))
}

/// encoding.py find_last_user_index (mid-conversation system messages count as user). Under 1017 a
/// user message whose blocks hold no text or image, tool results alone, does not start a turn.
pub fn find_last_user_index(messages: &[Value], enc: Encoding) -> Result<Option<usize>, EncodeError> {
    for i in (0..messages.len()).rev() {
        let r = role(&messages[i]);
        if !(r == Some("user") || (r == Some("system") && i > 0)) {
            continue;
        }
        let blocks = get(&messages[i], "content_blocks");
        if enc == Encoding::V41_1017 && truthy(blocks) {
            let blocks = blocks.unwrap().as_array().ok_or_else(|| err("AttributeError: content block is not an object"))?;
            let mut starts_turn = false;
            for b in blocks {
                if !b.is_object() {
                    return Err(err("AttributeError: content block is not an object"));
                }
                if matches!(get_str(b, "type"), Some("text") | Some("image") | Some("image_url")) {
                    starts_turn = true;
                    break;
                }
            }
            if !starts_turn {
                continue;
            }
        }
        return Ok(Some(i));
    }
    Ok(None)
}

fn render_message(
    index: usize,
    messages: &[Value],
    mode: ThinkingMode,
    drop_thinking: bool,
    effort: Option<&Value>,
    enc: Encoding,
) -> Result<String, EncodeError> {
    let msg = &messages[index];
    let last_user = find_last_user_index(messages, enc)?;
    let r = role(msg);
    let content = get(msg, "content");
    let tools = if truthy(get(msg, "tools")) { Some(tools_from_openai_format(get(msg, "tools").unwrap(), enc)?) } else { None };
    let tool_calls = if truthy(get(msg, "tool_calls")) {
        Some(tool_calls_from_openai_format(get(msg, "tool_calls").unwrap(), enc)?)
    } else {
        None
    };
    let wo_eos = truthy(get(msg, "wo_eos"));

    let effort_prompt = render_reasoning_effort(index, mode, effort, enc)?;
    let mut prompt = String::new();
    if index == 0 && (!effort_prompt.is_empty() || r == Some("system")) {
        prompt.push_str(SYSTEM_SP);
    }
    prompt.push_str(&effort_prompt);

    match r {
        Some("system") => {
            if index > 0 {
                prompt.push_str(SYSTEM_SP);
            }
            prompt.push_str(&or_empty(content));
            if let Some(t) = &tools {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(t));
            }
            if truthy(get(msg, "response_format")) {
                prompt.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                prompt.push_str(&to_json(get(msg, "response_format").unwrap()));
            }
        }
        Some("user") => {
            prompt.push_str(USER_SP);
            if truthy(get(msg, "content_blocks")) {
                let blocks = get(msg, "content_blocks").unwrap().as_array().ok_or_else(|| err("content_blocks must be a list"))?;
                let mut parts = Vec::new();
                for b in blocks {
                    if !b.is_object() {
                        return Err(err("AttributeError: content block is not an object"));
                    }
                    match get_str(b, "type") {
                        Some("text") => parts.push(match get(b, "text") {
                            None => String::new(),
                            Some(Value::String(s)) => s.clone(),
                            Some(_) => return Err(err("TypeError: text must be a string")),
                        }),
                        Some("tool_result") => {
                            let tc = get(b, "content").cloned().unwrap_or(json!(""));
                            let text = match &tc {
                                Value::Array(items) => {
                                    let mut tp = Vec::new();
                                    for it in items {
                                        if !it.is_object() {
                                            return Err(err("AttributeError: tool result block is not an object"));
                                        }
                                        if get_str(it, "type") == Some("text") {
                                            tp.push(match get(it, "text") {
                                                None => String::new(),
                                                Some(Value::String(s)) => s.clone(),
                                                Some(_) => return Err(err("TypeError: text must be a string")),
                                            });
                                        } else {
                                            tp.push(format!("[Unsupported {}]", py_str(get(it, "type").unwrap_or(&Value::Null))));
                                        }
                                    }
                                    tp.join("\n\n")
                                }
                                other => py_str(other),
                            };
                            parts.push(format!("<tool_result>{text}</tool_result>"));
                        }
                        _ => parts.push(format!("[Unsupported {}]", py_str(get(b, "type").unwrap_or(&Value::Null)))),
                    }
                }
                prompt.push_str(&parts.join("\n\n"));
            } else {
                // `prompt += content or ""`: concatenation, so only a string will do.
                match content {
                    Some(Value::String(s)) => prompt.push_str(s),
                    c if !truthy(c) => {}
                    _ => return Err(err("TypeError: user content must be a string")),
                }
            }
        }
        Some("latest_reminder") => {
            prompt.push_str(LATEST_REMINDER_SP);
            prompt.push_str(&py_str(content.unwrap_or(&Value::Null)));
        }
        Some("tool") => {
            return Err(err("deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()"))
        }
        Some("assistant") => {
            let tc_content = match &tool_calls {
                Some(tcs) => render_tool_calls(tcs),
                None => String::new(),
            };
            let summary = or_empty(content);
            let rc = or_empty(get(msg, "reasoning_content"));
            let prev_has_task = index >= 1 && is_task_set(&messages[index - 1]);
            let mut thinking_part = String::new();
            if mode == ThinkingMode::Thinking && !prev_has_task {
                let after_last_user = match last_user {
                    Some(lu) => index > lu,
                    None => true,
                };
                if !drop_thinking || after_last_user {
                    thinking_part = format!("{rc}{THINK_END}");
                }
            }
            prompt.push_str(&thinking_part);
            prompt.push_str(&summary);
            prompt.push_str(&tc_content);
            if !wo_eos {
                prompt.push_str(EOS);
            }
        }
        other => return Err(err(format!("Unknown role: {}", other.map_or("None".to_string(), |s| s.to_string())))),
    }

    // Transition tokens, depending on what follows.
    if index + 1 < messages.len() && !matches!(role(&messages[index + 1]), Some("assistant") | Some("latest_reminder")) {
        return Ok(prompt);
    }
    if is_task_set(msg) {
        let task = get(msg, "task").unwrap();
        let t = task.as_str().and_then(|t| TASK_TOKENS.iter().find(|(n, _)| *n == t));
        let (name, tok) = t.ok_or_else(|| err(format!("Invalid task: '{}'", py_str(task))))?;
        if *name != "action" {
            prompt.push_str(tok);
        } else {
            prompt.push_str(ASSISTANT_SP);
            prompt.push_str(if mode != ThinkingMode::Thinking { THINK_END } else { THINK_START });
            prompt.push_str(tok);
        }
    } else if r == Some("user") || (r == Some("system") && index > 0) {
        prompt.push_str(ASSISTANT_SP);
        let at_or_after_last_user = match last_user {
            Some(lu) => index >= lu,
            None => true,
        };
        if !drop_thinking && mode == ThinkingMode::Thinking {
            prompt.push_str(THINK_START);
        } else if drop_thinking && mode == ThinkingMode::Thinking && at_or_after_last_user {
            prompt.push_str(THINK_START);
        } else {
            prompt.push_str(THINK_END);
        }
    }
    Ok(prompt)
}

/// encoding.py _drop_thinking_messages.
fn drop_thinking_messages(messages: &[Value], enc: Encoding) -> Result<Vec<Value>, EncodeError> {
    let last_user = find_last_user_index(messages, enc)?;
    let keep = ["user", "system", "tool", "latest_reminder", "direct_search_results"];
    let mut out = Vec::new();
    for (idx, msg) in messages.iter().enumerate() {
        let r = role(msg);
        let late = match last_user {
            Some(lu) => idx >= lu,
            None => true,
        };
        if r.map_or(false, |r| keep.contains(&r)) || late {
            out.push(msg.clone());
        } else if r == Some("assistant") {
            let mut m = msg.clone();
            obj_mut(&mut m).shift_remove("reasoning_content");
            out.push(m);
        }
    }
    Ok(out)
}

/// encoding.py _encode_messages_text. V4.1's merges the tool messages and sorts the results here;
/// 1017's gets them merged and sorted (encode_messages).
fn encode_messages_text(messages: &[Value], o: &EncodeOptions, context: &[Value]) -> Result<String, EncodeError> {
    let enc = o.encoding;
    let (messages, context): (Vec<Value>, Vec<Value>) = if enc == Encoding::V41 {
        let merged = merge_tool_messages(messages, enc);
        let mut all: Vec<Value> = context.iter().cloned().chain(merged).collect();
        sort_tool_results_by_call_order(&mut all);
        let messages: Vec<Value> = all.split_off(context.len());
        let context: Vec<Value> = if context.is_empty() {
            Vec::new()
        } else {
            let mut c = merge_tool_messages(context, enc);
            sort_tool_results_by_call_order(&mut c);
            c
        };
        (messages, context)
    } else {
        (messages.to_vec(), context.to_vec())
    };
    let mut full: Vec<Value> = context.iter().cloned().chain(messages.iter().cloned()).collect();

    let mut prompt = if o.add_default_bos_token && context.is_empty() { BOS.to_string() } else { String::new() };
    let effective_drop = o.drop_thinking && !full.iter().any(|m| truthy(get(m, "tools")));
    // Signed, as in Python: with odd roles in the context the count can come out negative (nothing
    // is rendered then).
    let (num_to_render, context_len): (isize, isize) = if o.thinking_mode == ThinkingMode::Thinking && effective_drop {
        full = drop_thinking_messages(&full, enc)?;
        let n = full.len() as isize - drop_thinking_messages(&context, enc)?.len() as isize;
        (n, full.len() as isize - n)
    } else {
        (messages.len() as isize, context.len() as isize)
    };
    for idx in 0..num_to_render.max(0) {
        let i = idx + context_len;
        if i < 0 || i as usize >= full.len() {
            return Err(err("IndexError: message index out of range"));
        }
        prompt.push_str(&render_message(i as usize, &full, o.thinking_mode, effective_drop, o.reasoning_effort, enc)?);
    }
    Ok(prompt)
}

/// encoding.py encode_messages (text): the prompt string, plus the image records it found.
pub fn encode_messages(messages: &[Value], o: &EncodeOptions) -> Result<(String, Vec<Value>), EncodeError> {
    if o.encoding == Encoding::V41_1017 {
        // validated before the merge, which can drop a tool message's own fields; the images are
        // collected once the tool results are in their final order
        let ctx = o.context.unwrap_or(&[]);
        for m in ctx.iter().chain(messages) {
            validate_no_image_sp_tokens(m)?;
        }
        let context = merge_tool_messages(ctx, o.encoding);
        let mut all: Vec<Value> = context.iter().cloned().chain(merge_tool_messages(messages, o.encoding)).collect();
        sort_tool_results_by_call_order(&mut all); // in place, so the context comes out sorted too
        let merged = all.split_off(context.len());
        let context = if all.is_empty() { Vec::new() } else { process_image_messages(&all)?.0 };
        let (processed, images) = process_image_messages(&merged)?;
        let prompt = encode_messages_text(&processed, o, &context)?;
        return Ok((prompt, images));
    }
    let context = match o.context {
        Some(c) if !c.is_empty() => process_image_messages(c)?.0,
        _ => Vec::new(),
    };
    let (processed, images) = process_image_messages(messages)?;
    let prompt = encode_messages_text(&processed, o, &context)?;
    Ok((prompt, images))
}

// ------------------------------------------------------------------ parsing model output

/// The earliest occurrence of any stop string at or after `index`:
/// (new index, text before it, matched stop).
fn read_until_stop<'a>(index: usize, text: &'a str, stops: &[&'a str]) -> (usize, &'a str, Option<&'a str>) {
    let mut best: Option<(usize, &str)> = None;
    for s in stops {
        if let Some(p) = text[index..].find(s) {
            let p = index + p;
            if best.map_or(true, |(bp, _)| p < bp) {
                best = Some((p, s));
            }
        }
    }
    match best {
        Some((p, s)) => (p + s.len(), &text[index..p], Some(s)),
        None => (text.len(), &text[index..], None),
    }
}

/// Python `re` `\s` on str patterns: str.isspace().
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

/// `^\s*name="(.*?)">\n$` with DOTALL.
fn match_tool_name(s: &str) -> Option<String> {
    let t = s.trim_start_matches(py_isspace);
    let rest = t.strip_prefix("name=\"")?;
    let mut from = 0;
    while let Some(p) = rest[from..].find("\">\n") {
        let end = from + p + 3;
        if end == rest.len() || (end + 1 == rest.len() && rest.ends_with('\n')) {
            return Some(rest[..from + p].to_string());
        }
        from += p + 1;
    }
    None
}

/// `^ name="(.*?)" string="(true|false)">` of match_param's pattern: the name, the flag and where
/// the value starts. The lazy name ends at the first `" string="` that a flag and `">` follow, which
/// the text up to there decides: what it finds in a parameter that is still arriving stays found.
fn match_param_head(s: &str) -> Option<(&str, &'static str, usize)> {
    const NAME: &str = " name=\"";
    const STRING: &str = "\" string=\"";
    let rest = s.strip_prefix(NAME)?;
    let mut from = 0;
    while let Some(p) = rest[from..].find(STRING) {
        let name_end = from + p;
        let after = &rest[name_end + STRING.len()..];
        for flag in ["true", "false"] {
            if after.strip_prefix(flag).is_some_and(|x| x.starts_with("\">")) {
                return Some((&rest[..name_end], flag, NAME.len() + name_end + STRING.len() + flag.len() + 2));
            }
        }
        from = name_end + 1;
    }
    None
}

/// `^ name="(.*?)" string="(true|false)">(.*?)<$` with DOTALL.
fn match_param(s: &str) -> Option<(String, String, String)> {
    let (name, flag, at) = match_param_head(s)?;
    // The lazy value runs up to a '<' that ends the string (or precedes a final newline). Whether
    // there is one does not depend on where the name ended, so no later `" string="` is tried.
    let v = &s[at..];
    let value = v.strip_suffix('<').or_else(|| v.strip_suffix("<\n"))?;
    Some((name.to_string(), flag.to_string(), value.to_string()))
}

/// encoding.py parse_tool_calls: (new index, last stop, [(name, arguments json)]).
fn parse_tool_calls(mut index: usize, text: &str) -> Result<(usize, Option<String>, Vec<(String, String)>), EncodeError> {
    let calls_end = format!("</{DSML}{TOOL_CALLS_BLOCK}>");
    let call_start = format!("<{DSML}{TOOL_CALL_TAG}");
    let call_end = format!("</{DSML}{TOOL_CALL_TAG}");
    let param_start = format!("<{DSML}{TOOL_PARAM_TAG}");
    let param_end = format!("/{DSML}{TOOL_PARAM_TAG}");
    let mut calls = Vec::new();
    let mut stop: Option<String> = None;
    while index < text.len() {
        let (i, between, s) = read_until_stop(index, text, &[&call_start, &calls_end]);
        index = i;
        if between != ">\n" {
            return Err(err(format!("Tool call format error: expected '>\\n' but got '{between}'")));
        }
        stop = s.map(str::to_string);
        if s == Some(calls_end.as_str()) {
            break;
        }
        if s.is_none() {
            return Err(err("Missing special token in tool calls"));
        }
        let (i, name_content, s) = read_until_stop(index, text, &[&param_start, &call_end]);
        index = i;
        stop = s.map(str::to_string);
        let name = match_tool_name(name_content).ok_or_else(|| err(format!("Tool name format error: '{name_content}'")))?;
        let mut args: Vec<(String, String, String)> = Vec::new();
        while stop.as_deref() == Some(param_start.as_str()) {
            let (i, pc, _) = read_until_stop(index, text, &[&param_end]);
            index = i;
            let (pname, flag, value) = match_param(pc).ok_or_else(|| err(format!("Parameter format error: '{pc}'")))?;
            if args.iter().any(|(k, _, _)| *k == pname) {
                return Err(err(format!("Duplicate parameter name: '{pname}'")));
            }
            args.push((pname, value, flag));
            let (i, c, s) = read_until_stop(index, text, &[&param_start, &call_end]);
            index = i;
            stop = s.map(str::to_string);
            if c != ">\n" {
                return Err(err(format!("Parameter format error: expected '>\\n' but got '{c}'")));
            }
        }
        calls.push((name, decode_dsml_to_arguments(&args)));
    }
    Ok((index, stop, calls))
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedMessage {
    pub content: String,
    pub reasoning_content: String,
    /// (name as the model wrote it, `namespace::name` under 1017 too (api_tool_name splits it),
    /// arguments as a JSON string)
    pub tool_calls: Vec<(String, String)>,
}

impl ParsedMessage {
    /// A StreamParser's deltas, added in order, are the message: a tool call sent in pieces is put
    /// together (its index is its place among the calls).
    pub fn add(&mut self, d: Delta) {
        match d {
            Delta::Reasoning(s) => self.reasoning_content.push_str(&s),
            Delta::Content(s) => self.content.push_str(&s),
            Delta::ToolCall(_, name, args) => self.tool_calls.push((name, args)),
            Delta::ToolCallStart(_, name) => self.tool_calls.push((name, String::new())),
            Delta::ToolArgs(i, piece) => {
                if let Some((_, args)) = self.tool_calls.get_mut(i) {
                    args.push_str(&piece);
                }
            }
        }
    }
}

/// encoding.py parse_message_from_completion_text: the whole completion, ending with EOS. 1017's
/// refuses a tool name it cannot split (a second `::`).
pub fn parse_message_from_completion_text(text: &str, mode: ThinkingMode, enc: Encoding) -> Result<ParsedMessage, EncodeError> {
    let tc_start = format!("\n\n<{DSML}{TOOL_CALLS_BLOCK}");
    let mut index = 0;
    let mut reasoning = String::new();
    if mode == ThinkingMode::Thinking {
        let (i, delta, stop) = read_until_stop(index, text, &[THINK_END, &tc_start]);
        index = i;
        reasoning = delta.to_string();
        if stop != Some(THINK_END) {
            return Err(err("Invalid thinking format: missing </think>"));
        }
    }
    let (i, delta, stop) = read_until_stop(index, text, &[EOS, &tc_start]);
    index = i;
    let summary = delta.to_string();
    let mut stop = stop.map(str::to_string);
    let mut calls = Vec::new();
    if stop.as_deref() == Some(tc_start.as_str()) {
        let (i, s, c) = parse_tool_calls(index, text)?;
        index = i;
        calls = c;
        if enc == Encoding::V41_1017 {
            for (name, _) in &calls {
                split_tool_name(Some(&Value::String(name.clone())), None)?;
            }
        }
        let _ = s;
        let (i, rest, s) = read_until_stop(index, text, &[EOS]);
        index = i;
        stop = s.map(str::to_string);
        if !rest.is_empty() {
            return Err(err("Unexpected content after tool calls"));
        }
    } else if stop.as_deref() != Some(EOS) {
        return Err(err("Invalid format: missing EOS token"));
    }
    if index != text.len() || !(stop.is_none() || stop.as_deref() == Some(EOS)) {
        return Err(err("Unexpected content at end"));
    }
    for sp in [BOS, EOS, THINK_START, THINK_END, DSML] {
        if summary.contains(sp) || reasoning.contains(sp) {
            return Err(err(format!("Unexpected special token '{sp}' in content")));
        }
    }
    Ok(ParsedMessage { content: summary, reasoning_content: reasoning, tool_calls: calls })
}

// ------------------------------------------------------------------ streaming parser

/// What the model has said so far, split the way the API reports it.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A complete tool call: (index, name, arguments JSON).
    ToolCall(usize, String, String),
    /// A tool call whose arguments follow in pieces (StreamParser::stream_args): (index, name).
    ToolCallStart(usize, String),
    /// The next piece of that call's arguments JSON: (index, text). A call's pieces, in order, are
    /// the arguments its ToolCall would have carried.
    ToolArgs(usize, String),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Reasoning,
    Content,
    /// Inside the tool-call block, in the state parse_tool_calls would be in.
    ExpectCall,
    Name,
    Param,
    AfterParam,
    /// After the block: only EOS may follow.
    AfterCalls,
    /// Malformed tool calls: everything else is plain content.
    Raw,
    Done,
}

const CALLS_END: &str = "</｜DSML｜ calls>";
const CALL_START: &str = "<｜DSML｜ invoke";
const CALL_END: &str = "</｜DSML｜ invoke";
const PARAM_START: &str = "<｜DSML｜ parameter";
const PARAM_END: &str = "/｜DSML｜ parameter";
/// What a value cannot run into: PARAM_END, where its parameter is matched, and PARAM_END with the '<'
/// (and the newline it may precede) that match_param takes as the value's end.
const VALUE_STOPS: [&str; 3] = [PARAM_END, "</｜DSML｜ parameter", "<\n/｜DSML｜ parameter"];
const TC_START: &str = "\n\n<｜DSML｜ calls";

/// Incremental version of parse_message_from_completion_text for streamed output. Feed decoded text
/// as it arrives: reasoning and content come out as soon as they cannot be the start of a marker,
/// each tool call once its invoke block has parsed (with the same rules as parse_tool_calls).
/// Where the reference parser would raise on malformed tool calls, the rest of the output is passed
/// on as plain content instead.
///
/// With `stream_args` a tool call is not held until its end: its name goes out when it has parsed
/// (ToolCallStart) and its arguments as they arrive (ToolArgs), a value's text as soon as it cannot
/// be the start of the value's end. Reasoning and content are what they are without it, piece for
/// piece, and so is every call that parses. A call that does not (malformed, or cut off) is still
/// passed on as plain content, after the start and the pieces already sent, and does not count in
/// tool_calls_seen: its client is left with a call whose arguments stop where it did. They lack at
/// least the brace that closes them, so they are not complete JSON, unless a value the model wrote
/// with string="false" (passed on as written, here as in a ToolCall) is not JSON and closes them.
pub struct StreamParser {
    phase: Phase,
    /// Text not yet consumed. In the tool-call phases it starts at the current invoke (or at the `>`
    /// that closes the previous one).
    buf: String,
    /// Parse position in `buf` and the offset from which the next stop search may start.
    pos: usize,
    scan: usize,
    calls: usize,
    name: String,
    args: Vec<(String, String, String)>,
    tools: bool,
    stream_args: bool,
    /// stream_args: how much of the parameter being read has gone out. None before its name, then
    /// the bytes of its value.
    value_sent: Option<usize>,
    pub saw_eos: bool,
}

impl StreamParser {
    pub fn new(mode: ThinkingMode) -> Self {
        Self::with_tools(mode, true)
    }

    /// `tools = false`: DSML in the output is plain content (the request offered no tools).
    pub fn with_tools(mode: ThinkingMode, tools: bool) -> Self {
        StreamParser {
            tools,
            phase: if mode == ThinkingMode::Thinking {
                Phase::Reasoning
            } else if tools {
                Phase::Content
            } else {
                Phase::Raw
            },
            buf: String::new(),
            pos: 0,
            scan: 0,
            calls: 0,
            name: String::new(),
            args: Vec::new(),
            stream_args: false,
            value_sent: None,
            saw_eos: false,
        }
    }

    /// Tool calls in pieces (ToolCallStart, ToolArgs) instead of whole (ToolCall).
    pub fn stream_args(mut self, on: bool) -> Self {
        self.stream_args = on;
        self
    }

    pub fn tool_calls_seen(&self) -> usize {
        self.calls
    }

    pub fn push(&mut self, text: &str, out: &mut Vec<Delta>) {
        if self.phase == Phase::Done {
            return;
        }
        self.buf.push_str(text);
        self.advance(out, false);
    }

    /// End of generation: flush everything that is left.
    pub fn finish(&mut self, out: &mut Vec<Delta>) {
        self.advance(out, true);
        let rest = std::mem::take(&mut self.buf);
        match self.phase {
            Phase::Reasoning if !rest.is_empty() => out.push(Delta::Reasoning(rest)),
            Phase::Content | Phase::Raw if !rest.is_empty() => out.push(Delta::Content(rest)),
            // Cut off right after a complete call (or the block marker): parse_tool_calls accepts that.
            Phase::ExpectCall if self.pos >= rest.len() => {}
            Phase::ExpectCall | Phase::Name | Phase::Param | Phase::AfterParam => {
                // Cut off inside the block (max_tokens or a stop): show the unfinished call as text.
                let raw = self.unparsed(&rest);
                if !raw.trim().is_empty() && !CALLS_END.starts_with(raw.trim_start_matches(['>', '\n'])) {
                    out.push(Delta::Content(raw));
                }
            }
            _ => {}
        }
        self.phase = Phase::Done;
    }

    /// The raw text of the block from the current invoke on, for the fallback.
    fn unparsed(&self, rest: &str) -> String {
        if self.calls == 0 {
            format!("{TC_START}{rest}")
        } else {
            rest.strip_prefix('>').unwrap_or(rest).to_string()
        }
    }

    /// Length of the longest suffix of `s` that is a proper prefix of one of `markers`.
    fn holdback(s: &str, markers: &[&str]) -> usize {
        let mut best = 0;
        for m in markers {
            let max = m.len().saturating_sub(1).min(s.len());
            for k in (1..=max).rev() {
                if s.is_char_boundary(s.len() - k) && m.starts_with(&s[s.len() - k..]) {
                    best = best.max(k);
                    break;
                }
            }
        }
        best
    }

    /// _read_until_stop on the buffer from `pos`, for text that is still arriving: the earliest
    /// stop, or None if none has arrived yet. None of the stop sets here has one stop inside another,
    /// so an occurrence found now stays the earliest whatever follows.
    fn find_stop(&mut self, stops: &[&'static str]) -> Option<(usize, &'static str)> {
        let from = self.scan.max(self.pos);
        let mut best: Option<(usize, &'static str)> = None;
        for s in stops {
            if let Some(p) = self.buf[from..].find(s) {
                let p = from + p;
                if best.map_or(true, |(bp, _)| p < bp) {
                    best = Some((p, s));
                }
            }
        }
        if best.is_none() {
            let longest = stops.iter().map(|s| s.len()).max().unwrap_or(1);
            let mut scan = self.buf.len().saturating_sub(longest - 1).max(self.pos);
            while !self.buf.is_char_boundary(scan) {
                scan -= 1;
            }
            self.scan = scan;
        }
        best
    }

    /// Consume up to `end` (exclusive) of the buffer and reset the parse position.
    fn consume(&mut self, end: usize) {
        self.buf.drain(..end);
        self.pos = 0;
        self.scan = 0;
    }

    /// Text between the parse position and a stop must be exactly ">\n". While the stop has not
    /// arrived, false once that can no longer happen.
    fn gap_can_be_ok(&self, stops: &[&str]) -> bool {
        let gap = &self.buf[self.pos..];
        let n = gap.len().min(2);
        if gap.as_bytes()[..n] != b">\n"[..n] {
            return false;
        }
        n < 2 || stops.iter().any(|s| s.starts_with(&gap[2..]) || gap[2..].starts_with(s))
    }

    fn fail(&mut self, out: &mut Vec<Delta>) {
        let raw = self.unparsed(&self.buf);
        self.buf.clear();
        self.pos = 0;
        self.scan = 0;
        self.phase = Phase::Raw;
        out.push(Delta::Content(raw));
    }

    fn emit_call(&mut self, out: &mut Vec<Delta>) {
        if self.stream_args {
            // the name and the parameters have gone out: what is left closes the object
            out.push(Delta::ToolArgs(self.calls, if self.args.is_empty() { "{}" } else { "}" }.into()));
            self.name.clear();
        } else {
            let args = decode_dsml_to_arguments(&self.args);
            out.push(Delta::ToolCall(self.calls, std::mem::take(&mut self.name), args));
        }
        self.args.clear();
        self.calls += 1;
    }

    /// stream_args: the next piece of the arguments JSON, as decode_dsml_to_arguments writes it, for
    /// the parameter being read, of which `value` has arrived (all of it if `done`).
    fn args_piece(&self, name: &str, flag: &str, value: &str, done: bool) -> String {
        let is_str = flag == "true";
        let mut piece = String::new();
        let sent = match self.value_sent {
            Some(n) => n,
            None => {
                piece.push_str(if self.args.is_empty() { "{" } else { ", " });
                piece.push_str(&to_json(&Value::String(name.to_string())));
                piece.push_str(": ");
                if is_str {
                    piece.push('"');
                }
                0
            }
        };
        if is_str {
            write_escaped(&mut piece, &value[sent..]);
        } else {
            piece.push_str(&value[sent..]);
        }
        if done && is_str {
            piece.push('"');
        }
        piece
    }

    /// stream_args: send what has arrived of the parameter at `pos`, whose end has not: its name once
    /// match_param_head has it, then its value up to what could be the start of one of VALUE_STOPS.
    /// That is all a value's end can take, so what goes out is a prefix of the value match_param will
    /// find, and of a parameter that will not match, nothing past where it stops.
    fn send_arriving(&mut self, out: &mut Vec<Delta>) {
        let Some((name, flag, at)) = match_param_head(&self.buf[self.pos..]) else {
            return;
        };
        if self.args.iter().any(|(k, _, _)| k == name) {
            return; // a repeated name: the call fails at this parameter's end
        }
        let value = &self.buf[self.pos + at..];
        let value = &value[..value.len() - Self::holdback(value, &VALUE_STOPS)];
        let (piece, sent) = (self.args_piece(name, flag, value, false), value.len());
        self.send_args(piece, out);
        self.value_sent = Some(sent);
    }

    fn send_args(&self, piece: String, out: &mut Vec<Delta>) {
        if !piece.is_empty() {
            out.push(Delta::ToolArgs(self.calls, piece));
        }
    }

    fn advance(&mut self, out: &mut Vec<Delta>, at_end: bool) {
        loop {
            match self.phase {
                Phase::Reasoning => {
                    let end = self.buf.find(THINK_END);
                    let eos = self.buf.find(EOS);
                    if let Some(e) = eos.filter(|&e| end.map_or(true, |t| e < t)) {
                        if e > 0 {
                            out.push(Delta::Reasoning(self.buf[..e].to_string()));
                        }
                        self.buf.clear();
                        self.saw_eos = true;
                        self.phase = Phase::Done;
                        return;
                    }
                    if let Some(p) = end {
                        if p > 0 {
                            out.push(Delta::Reasoning(self.buf[..p].to_string()));
                        }
                        self.consume(p + THINK_END.len());
                        self.phase = if self.tools { Phase::Content } else { Phase::Raw };
                        continue;
                    }
                    let keep = if at_end { 0 } else { Self::holdback(&self.buf, &[THINK_END, EOS]) };
                    let n = self.buf.len() - keep;
                    if n > 0 {
                        out.push(Delta::Reasoning(self.buf[..n].to_string()));
                        self.consume(n);
                    }
                    return;
                }
                Phase::Content | Phase::Raw => {
                    let eos = self.buf.find(EOS);
                    let tc = if self.phase == Phase::Content { self.buf.find(TC_START) } else { None };
                    if let Some(e) = eos.filter(|&e| tc.map_or(true, |t| e < t)) {
                        if e > 0 {
                            out.push(Delta::Content(self.buf[..e].to_string()));
                        }
                        self.buf.clear();
                        self.saw_eos = true;
                        self.phase = Phase::Done;
                        return;
                    }
                    if let Some(t) = tc {
                        if t > 0 {
                            out.push(Delta::Content(self.buf[..t].to_string()));
                        }
                        self.consume(t + TC_START.len());
                        self.phase = Phase::ExpectCall;
                        continue;
                    }
                    let markers: &[&str] = if self.phase == Phase::Content { &[EOS, TC_START] } else { &[EOS] };
                    let keep = if at_end { 0 } else { Self::holdback(&self.buf, markers) };
                    let n = self.buf.len() - keep;
                    if n > 0 {
                        out.push(Delta::Content(self.buf[..n].to_string()));
                        self.consume(n);
                    }
                    return;
                }
                Phase::ExpectCall => {
                    let stops = [CALL_START, CALLS_END];
                    match self.find_stop(&stops) {
                        Some((p, s)) => {
                            if &self.buf[self.pos..p] != ">\n" {
                                self.fail(out);
                                continue;
                            }
                            if s == CALLS_END {
                                self.consume(p + s.len());
                                self.phase = Phase::AfterCalls;
                            } else {
                                // keep the invoke text in the buffer until the call is complete
                                self.pos = p + s.len();
                                self.scan = self.pos;
                                self.phase = Phase::Name;
                            }
                        }
                        None => {
                            if !self.gap_can_be_ok(&stops) {
                                self.fail(out);
                                continue;
                            }
                            return;
                        }
                    }
                }
                Phase::Name => match self.find_stop(&[PARAM_START, CALL_END]) {
                    Some((p, s)) => {
                        let Some(name) = match_tool_name(&self.buf[self.pos..p]) else {
                            self.fail(out);
                            continue;
                        };
                        self.name = name;
                        if self.stream_args {
                            out.push(Delta::ToolCallStart(self.calls, self.name.clone()));
                        }
                        if s == PARAM_START {
                            self.pos = p + s.len();
                            self.scan = self.pos;
                            self.phase = Phase::Param;
                        } else {
                            self.emit_call(out);
                            self.consume(p + s.len());
                            self.phase = Phase::ExpectCall;
                        }
                    }
                    None => return,
                },
                Phase::Param => match self.find_stop(&[PARAM_END]) {
                    Some((p, s)) => {
                        let Some((pname, flag, value)) = match_param(&self.buf[self.pos..p]) else {
                            self.fail(out);
                            continue;
                        };
                        if self.args.iter().any(|(k, _, _)| *k == pname) {
                            self.fail(out);
                            continue;
                        }
                        if self.stream_args {
                            self.send_args(self.args_piece(&pname, &flag, &value, true), out);
                            self.value_sent = None;
                        }
                        self.args.push((pname, value, flag));
                        self.pos = p + s.len();
                        self.scan = self.pos;
                        self.phase = Phase::AfterParam;
                    }
                    None => {
                        if self.stream_args {
                            self.send_arriving(out);
                        }
                        return;
                    }
                },
                Phase::AfterParam => {
                    let stops = [PARAM_START, CALL_END];
                    match self.find_stop(&stops) {
                        Some((p, s)) => {
                            if &self.buf[self.pos..p] != ">\n" {
                                self.fail(out);
                                continue;
                            }
                            if s == PARAM_START {
                                self.pos = p + s.len();
                                self.scan = self.pos;
                                self.phase = Phase::Param;
                            } else {
                                self.emit_call(out);
                                self.consume(p + s.len());
                                self.phase = Phase::ExpectCall;
                            }
                        }
                        None => {
                            if !self.gap_can_be_ok(&stops) {
                                self.fail(out);
                                continue;
                            }
                            return;
                        }
                    }
                }
                Phase::AfterCalls => {
                    if let Some(e) = self.buf.find(EOS) {
                        let before = self.buf[..e].to_string();
                        if !before.trim().is_empty() {
                            out.push(Delta::Content(before));
                        }
                        self.buf.clear();
                        self.saw_eos = true;
                        self.phase = Phase::Done;
                        return;
                    }
                    let keep = if at_end { 0 } else { Self::holdback(&self.buf, &[EOS]) };
                    let n = self.buf.len() - keep;
                    if n > 0 && !self.buf[..n].trim().is_empty() {
                        // text after the block, which the reference rejects: pass it on
                        out.push(Delta::Content(self.buf[..n].to_string()));
                    }
                    if n > 0 {
                        self.consume(n);
                    }
                    return;
                }
                Phase::Done => {
                    self.buf.clear();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool loop with no tools declared: 1017 keeps the turn's reasoning across the tool result,
    /// V4.1 drops it (the byte-exact check is enccheck against each encoding.py).
    #[test]
    fn reasoning_across_tool_calls_1017() {
        let msgs: Vec<Value> = serde_json::from_str(
            r#"[{"role":"user","content":"q"},
                {"role":"assistant","reasoning_content":"first thought","tool_calls":[{"id":"a","function":{"name":"f","arguments":"{}"}}]},
                {"role":"tool","tool_call_id":"a","content":"r"}]"#,
        )
        .unwrap();
        let mut o = EncodeOptions::new(ThinkingMode::Thinking);
        let (v41, _) = encode_messages(&msgs, &o).unwrap();
        o.encoding = Encoding::V41_1017;
        let (e1017, _) = encode_messages(&msgs, &o).unwrap();
        assert!(!v41.contains("first thought"));
        assert!(e1017.contains("first thought</think>"));
        assert!(e1017.ends_with("<tool_result>r</tool_result><｜Assistant｜><think>"));
    }

    #[test]
    fn tool_namespaces_1017() {
        let msgs: Vec<Value> = serde_json::from_str(
            r#"[{"role":"system","content":"","tools":[{"type":"function","namespace":{"name":"web","description":"Web tools."},
                  "function":{"name":"search","description":"Search","parameters":{}}}]},
                {"role":"user","content":"q"},
                {"role":"assistant","content":"","tool_calls":[{"id":"a","namespace":"web","function":{"name":"search","arguments":"{}"}}]}]"#,
        )
        .unwrap();
        let mut o = EncodeOptions::new(ThinkingMode::Chat);
        o.encoding = Encoding::V41_1017;
        let (p, _) = encode_messages(&msgs, &o).unwrap();
        assert!(p.contains(r#""name": "web::search", "description": "Web tools.\nSearch""#));
        assert!(p.contains(r#"<｜DSML｜ invoke name="web::search">"#));
        let bad: Vec<Value> = serde_json::from_str(
            r#"[{"role":"user","content":"q"},{"role":"assistant","tool_calls":[{"namespace":"x","function":{"name":"web::search","arguments":"{}"}}]}]"#,
        )
        .unwrap();
        assert!(encode_messages(&bad, &o).is_err()); // conflicting namespaces
        assert_eq!(api_tool_name(Encoding::V41_1017, "web::search"), (Some("web"), "search"));
        assert_eq!(api_tool_name(Encoding::V41_1017, "a::b::c"), (None, "a::b::c"));
        assert_eq!(api_tool_name(Encoding::V41, "web::search"), (None, "web::search"));
    }

    #[test]
    fn effort_names_1017() {
        let msgs: Vec<Value> = serde_json::from_str(r#"[{"role":"user","content":"q"}]"#).unwrap();
        let mut o = EncodeOptions::new(ThinkingMode::Thinking);
        o.encoding = Encoding::V41_1017;
        assert!(encode_messages(&msgs, &o).unwrap().0.contains("Reasoning Effort: 75 "));
        let x = json!("xhigh");
        o.reasoning_effort = Some(&x);
        assert!(encode_messages(&msgs, &o).is_err());
    }

    #[test]
    fn readme_example() {
        let msgs: Vec<Value> = serde_json::from_str(
            r#"[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is 2+2?"}]"#,
        )
        .unwrap();
        let effort = json!(75);
        let mut o = EncodeOptions::new(ThinkingMode::Thinking);
        o.reasoning_effort = Some(&effort);
        let (p, _) = encode_messages(&msgs, &o).unwrap();
        assert_eq!(
            p,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nYou are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
        );
        let parsed =
            parse_message_from_completion_text("Simple arithmetic.</think>2 + 2 = 4.<｜end▁of▁sentence｜>", ThinkingMode::Thinking, Encoding::V41)
                .unwrap();
        assert_eq!(parsed.reasoning_content, "Simple arithmetic.");
        assert_eq!(parsed.content, "2 + 2 = 4.");
    }

    #[test]
    fn stream_parser_tool_calls() {
        let text = "Let me look.</think>Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"ls\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
        let whole = parse_message_from_completion_text(text, ThinkingMode::Thinking, Encoding::V41).unwrap();
        // feed in small pieces
        let (out, p) = stream(text, 3, ThinkingMode::Thinking, false);
        let m = message(out);
        assert_eq!(m, whole);
        assert_eq!(m.tool_calls[0].1, "{\"path\": \"a.txt\", \"n\": 3}");
        assert!(p.saw_eos);
    }

    /// A StreamParser's deltas for `text`, fed `n` characters at a time, and the parser at the end.
    fn stream(text: &str, n: usize, mode: ThinkingMode, stream_args: bool) -> (Vec<Delta>, StreamParser) {
        let mut p = StreamParser::new(mode).stream_args(stream_args);
        let mut out = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for ch in chars.chunks(n) {
            p.push(&ch.iter().collect::<String>(), &mut out);
        }
        p.finish(&mut out);
        (out, p)
    }

    fn message(deltas: Vec<Delta>) -> ParsedMessage {
        let mut m = ParsedMessage::default();
        deltas.into_iter().for_each(|d| m.add(d));
        m
    }

    /// The reasoning and content among the deltas, piece for piece.
    fn text_of(deltas: &[Delta]) -> Vec<Delta> {
        deltas.iter().filter(|d| matches!(d, Delta::Reasoning(_) | Delta::Content(_))).cloned().collect()
    }

    #[test]
    fn param_pattern() {
        let m = |s: &str| match_param(s).map(|(name, flag, value)| format!("{name}|{flag}|{value}"));
        assert_eq!(m(" name=\"a\" string=\"true\">x<"), Some("a|true|x".into()));
        assert_eq!(m(" name=\"a\" string=\"false\">1<\n"), Some("a|false|1".into()));
        assert_eq!(m(" name=\"a\" string=\"true\">x<y<"), Some("a|true|x<y".into()));
        assert_eq!(m(" name=\"a\" string=\"true\"><"), Some("a|true|".into()));
        // the lazy name ends at the first `" string="` that a flag follows
        assert_eq!(m(" name=\"a\" string=\"no\" string=\"true\">x<"), Some("a\" string=\"no|true|x".into()));
        assert_eq!(m(" name=\"a\" string=\"true\">b\" string=\"false\">x<"), Some("a|true|b\" string=\"false\">x".into()));
        for bad in [" name=\"a\" string=\"true\">x", " name=\"a\" string=\"true\">x<\n\n", "name=\"a\" string=\"true\">x<", " name=\"a\" string=\"tru"] {
            assert_eq!(m(bad), None, "{bad:?}");
        }
    }

    /// stream_args: a call's pieces are the call however the text arrives, the reasoning and content
    /// are what they are without it, and a value goes out before its end has come.
    #[test]
    fn stream_parser_tool_calls_in_pieces() {
        let text = concat!(
            "Let me look.</think>Writing.\n\n<｜DSML｜ calls>\n",
            "<｜DSML｜ invoke name=\"write\">\n",
            "<｜DSML｜ parameter name=\"path\" string=\"true\">a \"b\"\\c.txt</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"text\" string=\"true\">if a < b && c </ d {\n\t\"é\"\n}\n</｜DSML｜ parameter>\n",
            "<｜DSML｜ parameter name=\"opts\" string=\"false\">{\"n\": 3, \"tags\": [\"<x>\"]}</｜DSML｜ parameter>\n",
            "</｜DSML｜ invoke>\n",
            "<｜DSML｜ invoke name=\"now\">\n",
            "</｜DSML｜ invoke>\n",
            "</｜DSML｜ calls><｜end▁of▁sentence｜>"
        );
        let whole = parse_message_from_completion_text(text, ThinkingMode::Thinking, Encoding::V41).unwrap();
        let write = r#"{"path": "a \"b\"\\c.txt", "text": "if a < b && c </ d {\n\t\"é\"\n}\n", "opts": {"n": 3, "tags": ["<x>"]}}"#;
        assert_eq!(whole.tool_calls, [("write".to_string(), write.to_string()), ("now".to_string(), "{}".to_string())]);
        for n in 1..=40 {
            let (out, p) = stream(text, n, ThinkingMode::Thinking, true);
            assert!(!out.iter().any(|d| matches!(d, Delta::ToolCall(..))), "{n} at a time: a whole call");
            assert_eq!(text_of(&out), text_of(&stream(text, n, ThinkingMode::Thinking, false).0), "{n} at a time");
            assert_eq!(message(out), whole, "{n} at a time");
            assert_eq!(p.tool_calls_seen(), 2);
            assert!(p.saw_eos);
        }
        // Up to the middle of a value: the call's name and that much of its arguments are out, and
        // the call does not count yet.
        let mut p = StreamParser::new(ThinkingMode::Thinking).stream_args(true);
        let mut out = Vec::new();
        let feed = |p: &mut StreamParser, out: &mut Vec<Delta>, s: &str| s.chars().for_each(|c| p.push(c.encode_utf8(&mut [0; 4]), out));
        let at = text.find("&& c").unwrap() + "&& c".len();
        feed(&mut p, &mut out, &text[..at]);
        assert_eq!(out.iter().filter(|d| **d == Delta::ToolCallStart(0, "write".into())).count(), 1);
        let args = |out: &[Delta]| message(out.to_vec()).tool_calls[0].1.clone();
        assert_eq!(args(&out), r#"{"path": "a \"b\"\\c.txt", "text": "if a < b && c"#);
        assert_eq!(p.tool_calls_seen(), 0);
        // what could be the start of the value's end waits for what follows it
        feed(&mut p, &mut out, " </");
        assert_eq!(args(&out), r#"{"path": "a \"b\"\\c.txt", "text": "if a < b && c "#);
        feed(&mut p, &mut out, " d");
        assert_eq!(args(&out), r#"{"path": "a \"b\"\\c.txt", "text": "if a < b && c </ d"#);
    }

    /// stream_args: a call that does not parse is passed on as content exactly as without it. What
    /// was sent of it stays unfinished, and it is not counted.
    #[test]
    fn stream_parser_in_pieces_malformed() {
        const END: &str = "</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>";
        let call = |name: &str, rest: &str| format!("<｜DSML｜ invoke name=\"{name}\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.txt{rest}");
        let good = call("stat", "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n");
        // each with what is sent of the call's arguments: up to where it stops being one, however
        // the text arrives
        let bad = [
            ("cut off in a value", String::new(), r#"{"path": "a.txt"#),
            ("cut off in a value's end", "</｜DSML｜ param".to_string(), r#"{"path": "a.txt"#),
            ("a value's end without its '<'", format!("/｜DSML｜ parameter>\n{END}"), r#"{"path": "a.txt"#),
            (
                "a repeated name",
                format!("</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"path\" string=\"true\">b</｜DSML｜ parameter>\n{END}"),
                r#"{"path": "a.txt""#,
            ),
            (
                "text between parameters",
                format!("</｜DSML｜ parameter>\nand <｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter>\n{END}"),
                r#"{"path": "a.txt""#,
            ),
        ];
        for (case, rest, sent) in &bad {
            for (before, seen) in [("", 0), (good.as_str(), 1)] {
                let text = format!("Reading.\n\n<｜DSML｜ calls>\n{before}{}", call("read", rest));
                for n in 1..=12 {
                    let what = format!("{case}, after {seen} good, {n} at a time");
                    let (whole, pw) = stream(&text, n, ThinkingMode::Chat, false);
                    let (pieces, pp) = stream(&text, n, ThinkingMode::Chat, true);
                    assert_eq!(text_of(&pieces), text_of(&whole), "{what}");
                    assert_eq!((pw.tool_calls_seen(), pp.tool_calls_seen()), (seen, seen), "{what}");
                    let (whole, pieces) = (message(whole), message(pieces));
                    assert_eq!(pieces.tool_calls[..seen], whole.tool_calls[..], "{what}");
                    let (name, args) = &pieces.tool_calls[seen];
                    assert_eq!((name.as_str(), args.as_str()), ("read", *sent), "{what}");
                    assert!(crate::pyjson::loads(args).is_none(), "{what}: {args} is complete JSON");
                }
            }
        }
    }
}
