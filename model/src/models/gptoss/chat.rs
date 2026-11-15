//! chat.rs - gpt-oss's chat format: OpenAI's harmony format as gpt-oss's chat template writes it (the system line,
//! the reasoning effort, the tools as a TypeScript-like namespace in the developer message), and the reader that
//! splits the reply by channel: reasoning, text and calls.

use crate::api::Chat;
use crate::chat::format::{text_of, tojson};
use crate::serve::{defused, defused_tools};
use serde_json::Value;

/// The request's prompt: the plain harmony format when it has nothing to do with tools, the tools' format when it does.
pub fn prompt(c: &Chat, tok: &crate::tok::Tokenizer) -> String {
    // what the user, the tools and the tool servers wrote, made safe to put in the prompt (the harmony markers are all
    // control tokens)
    let d: &dyn Fn(&str) -> std::borrow::Cow<str> = &|t| tok.defuse(t, &[]);
    let (msgs, tools) = (defused(&c.msgs, d), defused_tools(&c.tools, d));
    let plain = tools.is_empty() && !msgs.iter().any(|m| m["role"] == "tool" || filled(&m["tool_calls"]));
    if plain { prompt_of(&msgs, &effort_of(c)) } else { prompt_with_tools(&msgs, &tools, &effort_of(c)) }
}

/// The reasoning effort of a request: low, medium or high as it names them (the harmony format has only those three,
/// so "none" and "minimal" are low), else medium, or low when it turns thinking off.
fn effort_of(c: &Chat) -> String {
    match c.effort.as_deref() {
        Some("medium") => "medium",
        Some("high") => "high",
        Some(_) => "low",
        None if c.think == Some(false) => "low",
        None => "medium",
    }
    .to_string()
}

/// The messages in the harmony format, with the assistant's turn opened.
pub fn prompt_of(msgs: &[Value], effort: &str) -> String {
    // the system line is gpt-oss's own default, as its Harmony format gives it
    let mut s = format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
        Knowledge cutoff: 2024-06\n\nReasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel \
        must be included for every message.<|end|>"
    );
    for m in msgs {
        let role = match m["role"].as_str().unwrap_or("user") {
            "system" | "developer" => "developer",
            "assistant" => "assistant",
            _ => "user",
        };
        let content = text_of(m);
        if role == "assistant" {
            s += &format!("<|start|>assistant<|channel|>final<|message|>{content}<|end|>");
        } else if role == "developer" {
            s += &format!("<|start|>developer<|message|># Instructions\n\n{content}<|end|>");
        } else {
            s += &format!("<|start|>user<|message|>{content}<|end|>");
        }
    }
    s + "<|start|>assistant"
}

// ---- tools and channels, in gpt-oss's Harmony format. Its chat template is the reference for every string here.

fn filled(v: &Value) -> bool {
    match v {
        Value::Object(m) => !m.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    }
}

/// A parameter's schema as the TypeScript-like type the template writes for it.
fn ts_type(spec: &Value) -> String {
    let null = if spec["nullable"].as_bool() == Some(true) { " | null" } else { "" };
    let ty = &spec["type"];
    if ty == "array" {
        let items = &spec["items"];
        let of = match items["type"].as_str() {
            _ if !filled(items) => "any[]".to_string(),
            Some("string") => "string[]".to_string(),
            Some("number" | "integer") => "number[]".to_string(),
            Some("boolean") => "boolean[]".to_string(),
            _ => {
                let inner = ts_type(items);
                if inner == "object | object" || inner.chars().count() > 50 {
                    "any[]".to_string()
                } else {
                    inner + "[]"
                }
            }
        };
        return of + null;
    }
    if let Some(a) = ty.as_array().filter(|a| !a.is_empty()) {
        return a.iter().filter_map(|t| t.as_str()).collect::<Vec<_>>().join(" | ");
    }
    if let Some(vs) = spec["oneOf"].as_array().filter(|a| !a.is_empty()) {
        if vs.len() > 1 && vs.iter().any(|v| v["type"] == "object") {
            return "any".to_string();
        }
        return vs.iter().map(ts_type).collect::<Vec<_>>().join(" | ");
    }
    match ty.as_str() {
        Some("string") => match spec["enum"].as_array().filter(|e| !e.is_empty()) {
            Some(e) => format!(
                "\"{}\"",
                e.iter()
                    .map(|x| x.as_str().map_or_else(|| x.to_string(), String::from))
                    .collect::<Vec<_>>()
                    .join("\" | \"")
            ),
            None => format!("string{null}"),
        },
        Some("number" | "integer") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some("object") => match spec["properties"].as_object().filter(|p| !p.is_empty()) {
            Some(props) => {
                let need: Vec<&str> = spec["required"]
                    .as_array()
                    .map(|r| r.iter().filter_map(|x| x.as_str()).collect())
                    .unwrap_or_default();
                let fields: Vec<String> = props
                    .iter()
                    .map(|(k, p)| format!("{k}{}: {}", if need.contains(&k.as_str()) { "" } else { "?" }, ts_type(p)))
                    .collect();
                format!("{{\n{}}}", fields.join(", "))
            }
            None => "object".to_string(),
        },
        _ => "any".to_string(),
    }
}

/// The tools as the developer message lists them: one namespace, a type for each function.
fn namespace(tools: &[Value]) -> String {
    let mut s = String::from("## functions\n\nnamespace functions {\n\n");
    for t in tools {
        let f = if t["function"].is_object() { &t["function"] } else { t };
        s += &format!("// {}\ntype {} = ", f["description"].as_str().unwrap_or(""), f["name"].as_str().unwrap_or(""));
        let p = &f["parameters"];
        match p["properties"].as_object().filter(|m| !m.is_empty()) {
            Some(props) => {
                let need: Vec<&str> =
                    p["required"].as_array().map(|r| r.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
                s += "(_: {\n";
                for (k, spec) in props {
                    if let Some(d) = spec["description"].as_str().filter(|d| !d.is_empty()) {
                        s += &format!("// {d}\n");
                    }
                    s += &format!("{k}{}: {}", if need.contains(&k.as_str()) { "" } else { "?" }, ts_type(spec));
                    let d = &spec["default"];
                    if !d.is_null() {
                        let shown = d.as_str().map_or_else(|| d.to_string(), String::from);
                        if filled(&spec["enum"]) {
                            s += &format!(", // default: {shown}");
                        } else if filled(&spec["oneOf"]) {
                            s += &format!("// default: {shown}");
                        } else {
                            s += &format!(", // default: {}", tojson(d));
                        }
                    }
                    s += ",\n";
                }
                s += "}) => any;\n\n";
            }
            None => s += "() => any;\n\n",
        }
    }
    s + "} // namespace functions"
}

/// A request's arguments as JSON, whether it sent them as text (OpenAI) or as an object.
fn args_of(f: &Value) -> Value {
    match &f["arguments"] {
        Value::String(a) => serde_json::from_str(a).unwrap_or(Value::String(a.clone())),
        v => v.clone(),
    }
}

/// The messages and the tools in the harmony format, with the assistant's turn opened. The tools go in the developer
/// message, a call the assistant made is a commentary message addressed to the function, and a tool's result is a
/// message from that function back to the assistant.
fn prompt_with_tools(msgs: &[Value], tools: &[Value], effort: &str) -> String {
    let mut s = format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
        Knowledge cutoff: 2024-06\n\nReasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel \
        must be included for every message."
    );
    if !tools.is_empty() {
        s += "\nCalls to these tools must go to the commentary channel: 'functions'.";
    }
    s += "<|end|>";
    let lead = msgs.first().is_some_and(|m| matches!(m["role"].as_str(), Some("system" | "developer")));
    let rest = if lead { &msgs[1..] } else { msgs };
    if lead || !tools.is_empty() {
        s += "<|start|>developer<|message|>";
        if lead {
            s += &format!("# Instructions\n\n{}\n\n", text_of(&msgs[0]));
        }
        if !tools.is_empty() {
            s += &format!("# Tools\n\n{}", namespace(tools));
        }
        s += "<|end|>";
    }
    // the calls of the last assistant message that made any: (id, name), for the results that follow it
    let mut made: Vec<(Value, String)> = Vec::new();
    for (i, m) in rest.iter().enumerate() {
        match m["role"].as_str().unwrap_or("user") {
            "assistant" => match m["tool_calls"].as_array().filter(|c| !c.is_empty()) {
                Some(calls) => {
                    // the thinking that led to a call is kept only until the assistant has given its answer
                    let answered = rest[i + 1..].iter().any(|n| n["role"] == "assistant" && !filled(&n["tool_calls"]));
                    let text = text_of(m);
                    if !text.is_empty() && !answered {
                        s += &format!("<|start|>assistant<|channel|>analysis<|message|>{text}<|end|>");
                    }
                    made.clear();
                    for c in calls {
                        let f = if c["function"].is_object() { &c["function"] } else { c };
                        let name = f["name"].as_str().unwrap_or("");
                        s += &format!(
                            "<|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{}<|call|>",
                            tojson(&args_of(f))
                        );
                        made.push((c["id"].clone(), name.to_string()));
                    }
                }
                None => s += &format!("<|start|>assistant<|channel|>final<|message|>{}<|end|>", text_of(m)),
            },
            "tool" => {
                let name = made
                    .iter()
                    .find(|(id, _)| !id.is_null() && *id == m["tool_call_id"])
                    .or(made.last())
                    .map_or("unknown", |(_, n)| n.as_str());
                s += &format!(
                    "<|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{}<|end|>",
                    Value::String(text_of(m))
                );
            }
            "system" | "developer" => {
                s += &format!("<|start|>developer<|message|># Instructions\n\n{}<|end|>", text_of(m))
            }
            _ => s += &format!("<|start|>user<|message|>{}<|end|>", text_of(m)),
        }
    }
    s + "<|start|>assistant"
}

const MESSAGE: &str = "<|message|>";
const END: &str = "<|end|>";

/// What the message being written is.
#[derive(Clone, PartialEq)]
enum Part {
    /// its header (channel, and who it is for) is still coming
    Header,
    /// the analysis channel: reasoning
    Thinking,
    /// the final channel, or commentary meant for the user: the reply's text
    Text,
    /// commentary addressed to a function: a call's arguments
    Call(String),
}

/// Reads the reply as the model writes it, message by message: reasoning and text go to the client as what they
/// are, and a call to a function is collected.
pub struct Channels {
    held: String,
    part: Part,
    /// the calls made: the function's name and its arguments as the model wrote them
    pub calls: Vec<(String, String)>,
    args: String,
}

impl Channels {
    pub fn new() -> Self {
        Channels { held: String::new(), part: Part::Header, calls: Vec::new(), args: String::new() }
    }

    /// Whether what is being written now is reasoning.
    pub fn thinking(&self) -> bool {
        self.part == Part::Thinking
    }

    fn out(&mut self, text: &str, emit: &mut dyn FnMut(bool, &str) -> bool) -> bool {
        if matches!(self.part, Part::Call(_)) {
            self.args += text;
            return true;
        }
        text.is_empty() || emit(self.part == Part::Thinking, text)
    }

    /// Takes the next piece of the reply. False once the client has gone.
    pub fn push(&mut self, piece: &str, emit: &mut dyn FnMut(bool, &str) -> bool) -> bool {
        self.held += piece;
        loop {
            if self.part == Part::Header {
                let Some(m) = self.held.find(MESSAGE) else { return true };
                let head = self.held[..m].to_string();
                self.held.drain(..m + MESSAGE.len());
                self.part = match head.find("to=functions.") {
                    Some(f) => {
                        let name = &head[f + 13..];
                        Part::Call(name[..name.find([' ', '<', '\n']).unwrap_or(name.len())].to_string())
                    }
                    None if head.contains("analysis") => Part::Thinking,
                    None => Part::Text,
                };
                continue;
            }
            if let Some(e) = self.held.find(END) {
                let body = self.held[..e].to_string();
                self.held.drain(..e + END.len());
                if !self.out(&body, emit) {
                    return false;
                }
                self.close();
                continue;
            }
            // keep back an ending that could be the start of the end marker
            let keep = (1..END.len().min(self.held.len() + 1))
                .rev()
                .find(|&n| {
                    self.held.is_char_boundary(self.held.len() - n)
                        && END.starts_with(&self.held[self.held.len() - n..])
                })
                .unwrap_or(0);
            let body: String = self.held.drain(..self.held.len() - keep).collect();
            return self.out(&body, emit);
        }
    }

    /// The message being written is over.
    fn close(&mut self) {
        if let Part::Call(name) = std::mem::replace(&mut self.part, Part::Header) {
            self.calls.push((name, std::mem::take(&mut self.args)));
        }
    }

    /// The end of the reply: what was kept back goes out, and a call being written is complete.
    pub fn finish(&mut self, emit: &mut dyn FnMut(bool, &str) -> bool) {
        let rest = std::mem::take(&mut self.held);
        if self.part == Part::Header {
            // no header ever came: plain text
            self.part = Part::Text;
        }
        self.out(&rest, emit);
        self.close();
    }
}

impl Default for Channels {
    fn default() -> Self {
        Self::new()
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

    const NS: &str = "## functions\n\nnamespace functions {\n\n// Get the weather\ntype get_weather = (_: {\n// City name\ncity: string,\n\
        unit?: \"c\" | \"f\",\n}) => any;\n\n} // namespace functions";
    const SYSTEM: &str = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\n\
        Reasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.";

    #[test]
    fn lists_the_tools_as_the_template_does() {
        assert_eq!(namespace(&[weather()]), NS);
        let more = json!({"name": "f", "description": "d", "parameters": {"type": "object", "properties": {
            "tags": {"type": "array", "items": {"type": "string"}}, "n": {"type": "integer", "default": 3},
            "opt": {"type": "object", "properties": {"deep": {"type": "boolean"}}}, "any": {}}, "required": ["tags"]}});
        assert_eq!(
            namespace(&[more, json!({"name": "now", "description": "The time"})]),
            "## functions\n\nnamespace functions {\n\n// d\ntype f = (_: {\ntags: string[],\nn?: number, // default: 3,\nopt?: {\ndeep?: boolean},\nany?: any,\n}) => any;\n\n\
            // The time\ntype now = () => any;\n\n} // namespace functions"
        );
    }

    #[test]
    fn writes_the_messages_with_tools() {
        let mut msgs = vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "Weather in Paris?"}),
        ];
        let tools = [weather()];
        let head = format!(
            "{SYSTEM}\nCalls to these tools must go to the commentary channel: 'functions'.<|end|>\
            <|start|>developer<|message|># Instructions\n\nBe brief.\n\n# Tools\n\n{NS}<|end|><|start|>user<|message|>Weather in Paris?<|end|>"
        );
        assert_eq!(prompt_with_tools(&msgs, &tools, "medium"), format!("{head}<|start|>assistant"));

        msgs.push(json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"days\":2}"}}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": "c1", "content": "18C, sunny"}));
        let call = "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Paris\", \"days\": 2}<|call|>\
            <|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>\"18C, sunny\"<|end|>";
        assert_eq!(prompt_with_tools(&msgs, &tools, "medium"), format!("{head}{call}<|start|>assistant"));

        msgs.push(json!({"role": "assistant", "content": "It is 18C."}));
        msgs.push(json!({"role": "user", "content": "Thanks"}));
        assert_eq!(
            prompt_with_tools(&msgs, &tools, "medium"),
            format!(
                "{head}{call}<|start|>assistant<|channel|>final<|message|>It is 18C.<|end|><|start|>user<|message|>Thanks<|end|><|start|>assistant"
            )
        );
    }

    #[test]
    fn a_request_without_tools_gets_the_plain_format() {
        let msgs = [json!({"role": "user", "content": "Hi"})];
        assert_eq!(
            prompt_of(&msgs, "medium"),
            format!("{SYSTEM}<|end|><|start|>user<|message|>Hi<|end|><|start|>assistant")
        );
    }

    /// Feeds the pieces through the reader: the reasoning and the text the client was sent, and the reader.
    fn through(pieces: &[&str]) -> (String, String, Channels) {
        let (mut thought, mut text, mut r) = (String::new(), String::new(), Channels::new());
        for p in pieces {
            assert!(r.push(p, &mut |th, t| {
                if th {
                    thought += t
                } else {
                    text += t
                };
                true
            }));
        }
        r.finish(&mut |th, t| {
            if th {
                thought += t
            } else {
                text += t
            };
            true
        });
        (thought, text, r)
    }

    #[test]
    fn splits_the_reply_by_channel() {
        let (thought, text, r) = through(&[
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Think",
            "ing",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "Hello",
            " there",
        ]);
        assert_eq!((thought.as_str(), text.as_str()), ("Thinking", "Hello there"));
        assert!(r.calls.is_empty());

        // markers that arrive in bits, and text that only looks like the start of one
        let (thought, text, _) = through(&["<|chan", "nel|>final<|mess", "age|>a <| b <|en", "x <|e", "nd|>"]);
        assert_eq!((thought.as_str(), text.as_str()), ("", "a <| b <|enx "));

        // a reply with no header at all is its text
        let (_, text, _) = through(&["plain words"]);
        assert_eq!(text, "plain words");
    }

    #[test]
    fn reads_a_call() {
        let (thought, text, r) = through(&[
            "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather ",
            "<|constrain|>json<|message|>{\"city\":",
            "\"Paris\"}",
        ]);
        assert_eq!((thought.as_str(), text.as_str()), ("Need the weather.", ""));
        assert_eq!(r.calls, [("get_weather".to_string(), "{\"city\":\"Paris\"}".to_string())]);

        let (_, _, r) = through(&["<|start|>assistant to=functions.now<|channel|>commentary json<|message|>{}"]);
        assert_eq!(r.calls, [("now".to_string(), "{}".to_string())]);
    }
}
