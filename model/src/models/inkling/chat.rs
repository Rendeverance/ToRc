//! chat.rs - Inkling 2's chat format, as its chat template writes it, and the reader that splits the reply into
//! reasoning, text and calls.
//!
//! Every message is a role token, a content token and an end: `<|message_user|><|content_text|>...<|end_message|>`.
//! The tools are declared first, as JSON in a system message. A system message `Thinking effort level: N` goes in
//! before the first message that is not a system one (N from 0 to 0.99: the request's reasoning effort, 0.9 when it
//! names none, 0 when it turns thinking off). The model's turn is not opened for it: it writes its own
//! `<|message_model|>` messages (thinking, text, a call addressed to a tool by name) and ends them with
//! `<|content_model_end_sampling|>`.

use crate::api::Chat;
use crate::chat::format::{calls_of, text_of};
use crate::serve::{defused, defused_tools};
use serde_json::Value;

const END: &str = "<|end_message|>";
const THINK: &str = "<|content_thinking|>";
const TEXT: &str = "<|content_text|>";
const CALL: &str = "<|content_invoke_tool_json|>";

/// JSON as the template's tojson(sort_keys=true, separators=(",", ":")) writes it.
fn json_sorted(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut kv: Vec<(&String, &Value)> = m.iter().collect();
            kv.sort_by(|a, b| a.0.cmp(b.0));
            let parts: Vec<String> =
                kv.iter().map(|(k, x)| format!("{}:{}", Value::String((*k).clone()), json_sorted(x))).collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(json_sorted).collect::<Vec<_>>().join(",")),
        _ => v.to_string(),
    }
}

/// The request's thinking effort as the template writes the number.
fn effort_of(c: &Chat) -> &'static str {
    match c.effort.as_deref().map(str::trim) {
        Some("none") => "0",
        Some("minimal") => "0.1",
        Some("low") => "0.2",
        Some("medium") => "0.7",
        Some("high") => "0.9",
        Some("xhigh" | "max") => "0.99",
        _ if c.think == Some(false) => "0",
        _ => "0.9",
    }
}

/// The request's prompt.
pub fn prompt(c: &Chat, tok: &crate::tok::Tokenizer) -> String {
    // what the user, the tools and the tool servers wrote, made safe to put in the prompt
    let d: &dyn Fn(&str) -> std::borrow::Cow<str> = &|t| tok.defuse(t, &[]);
    let (msgs, tools) = (defused(&c.msgs, d), defused_tools(&c.tools, d));
    let mut s = String::new();
    if !tools.is_empty() {
        let specs: Vec<Value> = tools
            .iter()
            .map(|t| {
                let f = if t["function"].is_object() { &t["function"] } else { t };
                serde_json::json!({
                    "description": f["description"].as_str().unwrap_or(""),
                    "name": f["name"],
                    "parameters": if f["parameters"].is_object() { f["parameters"].clone() } else { serde_json::json!({}) },
                    "type": t["type"].as_str().unwrap_or("function"),
                })
            })
            .collect();
        s += &format!("<|message_system|>tool_declare<|content_xml|>{}{END}", json_sorted(&Value::Array(specs)));
    }
    let effort = format!("<|message_system|>{TEXT}Thinking effort level: {}{END}", effort_of(c));
    let mut said = false;
    for m in &msgs {
        let role = m["role"].as_str().unwrap_or("user");
        if !said && role != "system" && role != "developer" {
            s += &effort;
            said = true;
        }
        let content = text_of(m);
        match role {
            "system" | "developer" => s += &format!("<|message_system|>{TEXT}{content}{END}"),
            "assistant" => {
                let calls = calls_of(m);
                let thought = m["reasoning_content"].as_str().unwrap_or("");
                if !thought.is_empty() {
                    s += &format!("<|message_model|>{THINK}{thought}{END}");
                }
                if !content.is_empty() || (calls.is_empty() && thought.is_empty()) {
                    s += &format!("<|message_model|>{TEXT}{content}{END}");
                }
                for (name, args) in &calls {
                    s += &format!(
                        "<|message_model|>{name}{CALL}{{\"name\":{},\"args\":{}}}{END}",
                        Value::String(name.clone()),
                        json_sorted(&Value::Object(args.clone()))
                    );
                }
                s += "<|content_model_end_sampling|>";
            }
            "tool" => {
                // a tool's answer carries the tool's name: its own, or that of the call it answers
                let name = m["name"].as_str().filter(|n| !n.is_empty()).map(str::to_string).or_else(|| {
                    let id = m["tool_call_id"].as_str()?;
                    msgs.iter().flat_map(|p| p["tool_calls"].as_array().into_iter().flatten()).find_map(|tc| {
                        (tc["id"].as_str() == Some(id)).then(|| tc["function"]["name"].as_str().map(str::to_string))?
                    })
                });
                s += &format!("<|message_tool|>{}{TEXT}{content}{END}", name.unwrap_or_default());
            }
            _ => s += &format!("<|message_user|>{TEXT}{content}{END}"),
        }
    }
    if !said {
        s += &effort;
    }
    s
}

#[derive(PartialEq)]
enum Part {
    /// between messages: the role token and the content token are still to come
    Header,
    Thinking,
    Text,
    Call,
}

/// Reads the reply as the model writes it, message by message: reasoning and text go to the client as what they
/// are, and a call is collected.
pub struct Reader {
    held: String,
    part: Part,
    /// the calls made: the tool's name and its arguments as JSON text
    pub calls: Vec<(String, String)>,
    body: String,
}

impl Reader {
    pub fn new() -> Self {
        Reader { held: String::new(), part: Part::Header, calls: Vec::new(), body: String::new() }
    }

    /// Whether what is being written now is reasoning.
    pub fn thinking(&self) -> bool {
        self.part == Part::Thinking
    }

    fn out(&mut self, text: &str, emit: &mut dyn FnMut(bool, &str) -> bool) -> bool {
        if self.part == Part::Call {
            self.body += text;
            return true;
        }
        text.is_empty() || emit(self.part == Part::Thinking, text)
    }

    /// Takes the next piece of the reply. False once the client has gone.
    pub fn push(&mut self, piece: &str, emit: &mut dyn FnMut(bool, &str) -> bool) -> bool {
        self.held += piece;
        loop {
            if self.part == Part::Header {
                // the first content token decides what the message is
                let found = [(THINK, Part::Thinking), (TEXT, Part::Text), (CALL, Part::Call)]
                    .into_iter()
                    .filter_map(|(mark, part)| self.held.find(mark).map(|at| (at, mark.len(), part)))
                    .min_by_key(|f| f.0);
                let Some((at, len, part)) = found else { return true };
                self.held.drain(..at + len);
                self.part = part;
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

    /// The message being written is over: a call's JSON gives its name and arguments.
    fn close(&mut self) {
        if std::mem::replace(&mut self.part, Part::Header) == Part::Call {
            let body = std::mem::take(&mut self.body);
            if let Ok(v) = serde_json::from_str::<Value>(body.trim())
                && let Some(name) = v["name"].as_str()
            {
                let args = if v["args"].is_object() { v["args"].to_string() } else { "{}".to_string() };
                self.calls.push((name.to_string(), args));
            }
        }
    }

    /// The end of the reply: what was kept back goes out, and a call being written is complete.
    pub fn finish(&mut self, emit: &mut dyn FnMut(bool, &str) -> bool) {
        let rest = std::mem::take(&mut self.held);
        if self.part == Part::Header {
            // no content token ever came: plain text
            self.part = Part::Text;
        }
        self.out(&rest, emit);
        self.close();
    }
}

impl Default for Reader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(pieces: &[&str]) -> (String, String, Vec<(String, String)>) {
        let (mut think, mut text) = (String::new(), String::new());
        let mut r = Reader::new();
        let mut emit = |t: bool, s: &str| {
            if t {
                think.push_str(s)
            } else {
                text.push_str(s)
            }
            true
        };
        for p in pieces {
            r.push(p, &mut emit);
        }
        r.finish(&mut emit);
        (think, text, r.calls)
    }

    #[test]
    fn splits_thinking_text_and_calls() {
        let (think, text, calls) = read(&[
            "<|message_model|><|content_thinking|>Let me",
            " see.<|end_",
            "message|><|message_model|><|content_text|>Paris.<|end_message|>",
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}<|end_message|>",
        ]);
        assert_eq!(think, "Let me see.");
        assert_eq!(text, "Paris.");
        assert_eq!(calls, vec![("get_weather".to_string(), "{\"city\":\"Paris\"}".to_string())]);
    }

    #[test]
    fn sorted_json_is_compact() {
        let v: Value = serde_json::from_str("{\"b\": 1, \"a\": {\"d\": [1, 2], \"c\": \"x\"}}").unwrap();
        assert_eq!(json_sorted(&v), "{\"a\":{\"c\":\"x\",\"d\":[1,2]},\"b\":1}");
    }
}
