//! chat.rs - Gemma 4's chat format, from the chat template the model ships with: its turns, the empty thinking
//! block that opens the model's turn when it isn't thinking, its own tool format, and the reader that keeps a
//! call out of the reply's text.

use crate::chat::format::text_of;
use serde_json::Value;

/// The markers Gemma 4's format writes, broken in user and tool text whether or not the vocabulary keeps them as
/// control tokens (Tokenizer::defuse).
pub const MARKERS: &[&str] = &[
    "<|turn>",
    "<turn|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|channel>",
    "<channel|>",
    "<|\"|>",
    "<|think|>",
];

/// How the template opens the model's turn with thinking off (its default): an empty thinking block, so the model
/// writes its answer straight away. A next turn's prompt has the model's reply where the block was.
const OPEN: &str = "<|turn>model\n";
pub const NO_THOUGHT: &str = "<|channel>thought\n<channel|>";

/// The messages in Gemma 4's turn format, with the model's turn opened.
pub fn prompt_of(msgs: &[Value]) -> String {
    let mut s = String::new();
    for m in msgs {
        let role = m["role"].as_str().unwrap_or("user");
        let role = if role == "assistant" { "model" } else { role };
        let content = text_of(m);
        s += &format!("<|turn>{role}\n{content}<turn|>\n");
    }
    s + OPEN + NO_THOUGHT
}

// ---- tools, in Gemma 4's own format. Its chat template is the reference for every string here.

/// The quote Gemma 4 puts round a string.
const Q: &str = "<|\"|>";
const CALL: &str = "<|tool_call>";
const CALL_END: &str = "<tool_call|>";
const RESPONSE: &str = "<|tool_response>";

/// A JSON value as Gemma 4 writes an argument: a string inside its quote token, an object with its keys in order
/// (inside the quote token too when `keys` is set), a list in square brackets, anything else as JSON has it.
fn arg(v: &Value, keys: bool) -> String {
    match v {
        Value::String(s) => format!("{Q}{s}{Q}"),
        Value::Object(m) => {
            let mut ks: Vec<&String> = m.keys().collect();
            ks.sort();
            let parts: Vec<String> = ks
                .iter()
                .map(|k| {
                    if keys { format!("{Q}{k}{Q}:{}", arg(&m[*k], keys)) } else { format!("{k}:{}", arg(&m[*k], keys)) }
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(|x| arg(x, keys)).collect::<Vec<_>>().join(",")),
        other => other.to_string(),
    }
}

fn upper(v: &Value) -> String {
    v.as_str().unwrap_or("").to_uppercase()
}

/// A list of names, each in the quote token.
fn names(v: &Value) -> String {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str()).map(|s| format!("{Q}{s}{Q}")).collect::<Vec<_>>().join(","))
        .unwrap_or_default()
}

pub fn filled(v: &Value) -> bool {
    match v {
        Value::Object(m) => !m.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    }
}

/// A schema's properties as a declaration lists them: each with its description, what a string may be, what a list
/// holds, an object's own properties, and its type last.
fn params(props: &Value) -> String {
    // the template skips a property that is named like one of a schema's own keys
    const OWN: [&str; 5] = ["description", "type", "properties", "required", "nullable"];
    let Some(m) = props.as_object() else { return String::new() };
    let mut ks: Vec<&String> = m.keys().filter(|k| !OWN.contains(&k.as_str())).collect();
    ks.sort();
    let mut out = Vec::new();
    for k in ks {
        let v = &m[k];
        let ty = upper(&v["type"]);
        let mut f: Vec<String> = Vec::new();
        if let Some(d) = v["description"].as_str().filter(|d| !d.is_empty()) {
            f.push(format!("description:{Q}{d}{Q}"));
        }
        if ty == "STRING" {
            if filled(&v["enum"]) {
                f.push(format!("enum:{}", arg(&v["enum"], true)));
            }
        } else if ty == "ARRAY"
            && let Some(items) = v["items"].as_object().filter(|i| !i.is_empty())
        {
            let mut iks: Vec<&String> = items.keys().filter(|k| !items[*k].is_null()).collect();
            iks.sort();
            let parts: Vec<String> = iks
                .iter()
                .map(|ik| {
                    let iv = &items[*ik];
                    match ik.as_str() {
                        "properties" => format!("properties:{{{}}}", params(iv)),
                        "required" => format!("required:[{}]", names(iv)),
                        "type" if iv.is_string() => format!("type:{Q}{}{Q}", upper(iv)),
                        "type" => format!(
                            "type:[{}]",
                            iv.as_array()
                                .map(|a| a.iter().map(|t| format!("{Q}{}{Q}", upper(t))).collect::<Vec<_>>().join(","))
                                .unwrap_or_default()
                        ),
                        _ => format!("{ik}:{}", arg(iv, true)),
                    }
                })
                .collect();
            f.push(format!("items:{{{}}}", parts.join(",")));
        }
        if v["nullable"].as_bool() == Some(true) {
            f.push("nullable:true".into());
        }
        if ty == "OBJECT" {
            let inner = if v["properties"].is_object() { &v["properties"] } else { v };
            f.push(format!("properties:{{{}}}", params(inner)));
            if filled(&v["required"]) {
                f.push(format!("required:[{}]", names(&v["required"])));
            }
        }
        f.push(format!("type:{Q}{ty}{Q}"));
        out.push(format!("{k}:{{{}}}", f.join(",")));
    }
    out.join(",")
}

/// One tool as the system turn declares it.
fn declaration(tool: &Value) -> String {
    let f = if tool["function"].is_object() { &tool["function"] } else { tool };
    let mut s = format!(
        "declaration:{}{{description:{Q}{}{Q}",
        f["name"].as_str().unwrap_or(""),
        f["description"].as_str().unwrap_or("")
    );
    let p = &f["parameters"];
    if filled(p) {
        s += ",parameters:{";
        if filled(&p["properties"]) {
            s += &format!("properties:{{{}}},", params(&p["properties"]));
        }
        if filled(&p["required"]) {
            s += &format!("required:[{}],", names(&p["required"]));
        }
        if let Some(t) = p["type"].as_str() {
            s += &format!("type:{Q}{}{Q}", t.to_uppercase());
        }
        s += "}";
    }
    s + "}"
}

/// A model turn's text without its thinking (what sits between <|channel> and <channel|>).
fn no_thinking(text: &str) -> String {
    let mut out = String::new();
    for part in text.split("<channel|>") {
        out += part.split("<|channel>").next().unwrap_or("");
    }
    out.trim().to_string()
}

/// A request's arguments as an object, whether it sent them as JSON text (OpenAI) or as an object.
fn args_of(f: &Value) -> Value {
    match &f["arguments"] {
        Value::String(a) => serde_json::from_str(a).unwrap_or(Value::Null),
        v => v.clone(),
    }
}

/// The messages and the tools in Gemma 4's turn format. The tools are declared in the system turn. A model turn that
/// called tools carries its calls and then their results, and stays open while the model has not yet written the
/// text that follows them.
pub fn prompt_with_tools(msgs: &[Value], tools: &[Value]) -> String {
    let mut s = String::new();
    let lead = msgs.first().is_some_and(|m| matches!(m["role"].as_str(), Some("system" | "developer")));
    let rest = if lead { &msgs[1..] } else { msgs };
    if lead || !tools.is_empty() {
        s += "<|turn>system\n";
        if lead {
            s += text_of(&msgs[0]).trim();
        }
        for t in tools {
            s += &format!("<|tool>{}<tool|>", declaration(t));
        }
        s += "<turn|>\n";
    }
    // whether the last turn written is a model turn still open
    let mut open = false;
    for (i, m) in rest.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("user");
        if role == "tool" {
            continue;
        }
        let model = role == "assistant";
        let before = rest[..i].iter().rev().find(|p| p["role"] != "tool").and_then(|p| p["role"].as_str());
        // a model message straight after another carries on the same turn
        if !(model && before == Some("assistant")) {
            s += &format!("<|turn>{}\n", if model { "model" } else { role });
        }
        let calls = m["tool_calls"].as_array().map(|v| v.as_slice()).unwrap_or(&[]);
        for c in calls {
            let f = &c["function"];
            let inner = arg(&args_of(f), false);
            let inner = inner.strip_prefix('{').and_then(|x| x.strip_suffix('}')).unwrap_or("");
            s += &format!("{CALL}call:{}{{{inner}}}{CALL_END}", f["name"].as_str().unwrap_or(""));
        }
        let mut answered = false;
        if !calls.is_empty() {
            for t in rest[i + 1..].iter().take_while(|t| t["role"] == "tool") {
                let name = calls
                    .iter()
                    .find(|c| !c["id"].is_null() && c["id"] == t["tool_call_id"])
                    .and_then(|c| c["function"]["name"].as_str())
                    .or(t["name"].as_str())
                    .unwrap_or("unknown");
                s += &format!(
                    "{RESPONSE}response:{name}{{value:{}}}<tool_response|>",
                    arg(&Value::String(text_of(t)), false)
                );
                answered = true;
            }
        }
        let text = if model { no_thinking(&text_of(m)) } else { text_of(m).trim().to_string() };
        s += &text;
        open = if !calls.is_empty() && !answered {
            // the calls have no results yet: the turn waits for them
            s += RESPONSE;
            true
        } else if answered && text.is_empty() {
            true
        } else {
            s += "<turn|>\n";
            false
        };
    }
    if !open {
        s += OPEN;
        s += NO_THOUGHT;
    }
    s
}

/// Reads `call:NAME{key:value,...}` as the model writes it: the tool's name and its arguments as JSON.
pub fn parse_call(body: &str) -> Option<(String, Value)> {
    let b = body.trim().strip_prefix("call:")?;
    let brace = b.find('{')?;
    let mut at = brace;
    let args = value(b, &mut at, 0)?;
    Some((b[..brace].trim().to_string(), args))
}

/// The deepest a call's arguments may nest: deeper is not a real call, and each level is a frame on the stack.
const MAX_DEPTH: usize = 64;

/// One value at `at`, which is left just past it, `depth` levels down.
fn value(s: &str, at: &mut usize, depth: usize) -> Option<Value> {
    if depth > MAX_DEPTH {
        return None;
    }
    let skip = |at: &mut usize| {
        while s[*at..].starts_with([' ', '\n', '\t', '\r']) {
            *at += 1
        }
    };
    skip(at);
    let r = &s[*at..];
    if let Some(r) = r.strip_prefix(Q) {
        let end = r.find(Q)?;
        *at += Q.len() + end + Q.len();
        return Some(Value::String(r[..end].to_string()));
    }
    if r.starts_with('{') {
        *at += 1;
        let mut m = serde_json::Map::new();
        loop {
            skip(at);
            if s[*at..].starts_with('}') {
                *at += 1;
                return Some(Value::Object(m));
            }
            let r = &s[*at..];
            let key = if let Some(r) = r.strip_prefix(Q) {
                let end = r.find(Q)?;
                *at += Q.len() + end + Q.len();
                r[..end].to_string()
            } else {
                let end = r.find(':')?;
                *at += end;
                r[..end].trim().to_string()
            };
            skip(at);
            if !s[*at..].starts_with(':') {
                return None;
            }
            *at += 1;
            m.insert(key, value(s, at, depth + 1)?);
            skip(at);
            if s[*at..].starts_with(',') {
                *at += 1;
            }
        }
    }
    if r.starts_with('[') {
        *at += 1;
        let mut a = Vec::new();
        loop {
            skip(at);
            if s[*at..].starts_with(']') {
                *at += 1;
                return Some(Value::Array(a));
            }
            a.push(value(s, at, depth + 1)?);
            skip(at);
            if s[*at..].starts_with(',') {
                *at += 1;
            }
        }
    }
    // a number, true, false or null; anything else is kept as the text it is
    let end = r.find([',', '}', ']']).unwrap_or(r.len());
    if end == 0 {
        return None;
    }
    *at += end;
    let word = r[..end].trim();
    Some(serde_json::from_str(word).unwrap_or(Value::String(word.to_string())))
}

/// Stands between the reply and the client when the request has tools: text goes through, a tool call is kept back
/// and collected, and the reply is over when the model starts waiting for a tool's result.
#[derive(Default)]
pub struct Calls {
    held: String,
    pub bodies: Vec<String>,
    /// the model has asked for its tools' results
    pub waiting: bool,
}

impl Calls {
    /// Takes the next piece of the reply. False once the client has gone.
    pub fn push(&mut self, piece: &str, emit: &mut dyn FnMut(bool, &str) -> bool) -> bool {
        self.held += piece;
        loop {
            let call = self.held.find(CALL);
            let wait = self.held.find(RESPONSE);
            if let Some(w) = wait.filter(|&w| call.is_none_or(|c| w < c)) {
                let before = self.held[..w].to_string();
                self.held.clear();
                self.waiting = true;
                return before.is_empty() || emit(false, &before);
            }
            let Some(c) = call else { break };
            let Some(e) = self.held[c..].find(CALL_END) else {
                // the call is still being written: send what came before it and keep the rest
                let before: String = self.held.drain(..c).collect();
                return before.is_empty() || emit(false, &before);
            };
            let before = self.held[..c].to_string();
            self.bodies.push(self.held[c + CALL.len()..c + e].to_string());
            self.held.drain(..c + e + CALL_END.len());
            if !before.is_empty() && !emit(false, &before) {
                return false;
            }
        }
        // keep back an ending that could be the start of a marker
        let keep = (1..=self.held.len().min(RESPONSE.len() - 1))
            .rev()
            .find(|&n| {
                self.held.is_char_boundary(self.held.len() - n)
                    && (CALL.starts_with(&self.held[self.held.len() - n..])
                        || RESPONSE.starts_with(&self.held[self.held.len() - n..]))
            })
            .unwrap_or(0);
        let out: String = self.held.drain(..self.held.len() - keep).collect();
        out.is_empty() || emit(false, &out)
    }

    /// The end of the reply: whatever was kept back and is not part of a call goes out.
    pub fn flush(&mut self, emit: &mut dyn FnMut(bool, &str) -> bool) {
        if !self.held.is_empty() && !self.held.starts_with(CALL) {
            emit(false, &self.held);
        }
        self.held.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"]}}})
    }

    const DECL: &str = "declaration:get_weather{description:<|\"|>Get the weather<|\"|>,parameters:{properties:{\
        city:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>},unit:{enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}},\
        required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}";

    #[test]
    fn declares_a_tool_as_the_template_does() {
        assert_eq!(declaration(&weather()), DECL);
        let nested = json!({"name": "f", "description": "d", "parameters": {"type": "object", "properties": {
            "tags": {"type": "array", "items": {"type": "string"}}, "opt": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
            "maybe": {"type": "number", "nullable": true}}}});
        assert_eq!(
            declaration(&nested),
            "declaration:f{description:<|\"|>d<|\"|>,parameters:{properties:{\
            maybe:{nullable:true,type:<|\"|>NUMBER<|\"|>},\
            opt:{properties:{n:{type:<|\"|>INTEGER<|\"|>}},required:[<|\"|>n<|\"|>],type:<|\"|>OBJECT<|\"|>},\
            tags:{items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>}},type:<|\"|>OBJECT<|\"|>}}"
        );
        assert_eq!(
            declaration(&json!({"name": "now", "description": "The time"})),
            "declaration:now{description:<|\"|>The time<|\"|>}"
        );
    }

    #[test]
    fn writes_the_turns_with_tools() {
        let mut msgs = vec![
            json!({"role": "system", "content": "Be brief. "}),
            json!({"role": "user", "content": "Weather in Paris?"}),
        ];
        let tools = [weather()];
        let head =
            format!("<|turn>system\nBe brief.<|tool>{DECL}<tool|><turn|>\n<|turn>user\nWeather in Paris?<turn|>\n");
        assert_eq!(prompt_with_tools(&msgs, &tools), format!("{head}<|turn>model\n{NO_THOUGHT}"));

        // the model called the tool: the turn waits for the result
        msgs.push(json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"days\":2}"}}]}));
        let call = "<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|>";
        assert_eq!(prompt_with_tools(&msgs, &tools), format!("{head}{call}<|tool_response>"));

        // the result is in: the model carries on in the same turn
        msgs.push(json!({"role": "tool", "tool_call_id": "c1", "content": "18C, sunny"}));
        let result = "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}<tool_response|>";
        assert_eq!(prompt_with_tools(&msgs, &tools), format!("{head}{call}{result}"));

        // its answer closes the turn, and the chat goes on
        msgs.push(json!({"role": "assistant", "content": "It is 18C."}));
        msgs.push(json!({"role": "user", "content": "Thanks"}));
        assert_eq!(
            prompt_with_tools(&msgs, &tools),
            format!("{head}{call}{result}It is 18C.<turn|>\n<|turn>user\nThanks<turn|>\n<|turn>model\n{NO_THOUGHT}")
        );
    }

    #[test]
    fn no_tools_and_no_system_message() {
        let msgs = [json!({"role": "user", "content": "Hi"})];
        assert_eq!(
            prompt_with_tools(&msgs, &[weather()]),
            format!("<|turn>system\n<|tool>{DECL}<tool|><turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n{NO_THOUGHT}")
        );
        // the plain path is the one a request without tools takes, and it is not this one
        assert_eq!(prompt_of(&msgs), format!("<|turn>user\nHi<turn|>\n<|turn>model\n{NO_THOUGHT}"));
    }

    #[test]
    fn reads_a_call() {
        let (name, args) = parse_call("call:get_weather{city:<|\"|>Paris, France {x}<|\"|>,days:2,opts:{deep:true,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]},x:null,r:1.5}").unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(
            args,
            json!({"city": "Paris, France {x}", "days": 2, "opts": {"deep": true, "tags": ["a", "b"]}, "x": null, "r": 1.5})
        );
        assert_eq!(parse_call("call:now{}").unwrap(), ("now".to_string(), json!({})));
        assert!(parse_call("not a call").is_none());
        assert!(parse_call("call:broken{city:<|\"|>unclosed").is_none());
        // nesting past MAX_DEPTH is refused, not followed down the stack
        let deep = format!("call:f{{a:{}1{}}}", "[".repeat(5000), "]".repeat(5000));
        assert!(parse_call(&deep).is_none());
    }

    /// Feeds the pieces through the reader and gives back what the client was sent.
    fn through(pieces: &[&str]) -> (String, Calls) {
        let (mut sent, mut r) = (String::new(), Calls::default());
        for p in pieces {
            assert!(r.push(p, &mut |_, t| {
                sent += t;
                true
            }));
            if r.waiting {
                break;
            }
        }
        r.flush(&mut |_, t| {
            sent += t;
            true
        });
        (sent, r)
    }

    #[test]
    fn keeps_calls_out_of_the_text() {
        let (sent, r) =
            through(&["Sure. ", "<|tool_call>", "call:now{}", "<tool_call|>", "<|tool_response>", "never read"]);
        assert_eq!(sent, "Sure. ");
        assert_eq!(r.bodies, ["call:now{}"]);
        assert!(r.waiting);

        // markers split across pieces, and two calls
        let (sent, r) = through(&[
            "abc<|to",
            "ol_call>call:f{a:1}<tool_",
            "call|><|tool_call>call:g{}<tool_call|>",
            "<|tool_resp",
            "onse>",
        ]);
        assert_eq!(sent, "abc");
        assert_eq!(r.bodies, ["call:f{a:1}", "call:g{}"]);
        assert!(r.waiting);

        // text that only looks like the start of a marker still goes out
        let (sent, r) = through(&["a < b and <|t", "ag> stays"]);
        assert_eq!(sent, "a < b and <|tag> stays");
        assert!(r.bodies.is_empty() && !r.waiting);
    }
}
