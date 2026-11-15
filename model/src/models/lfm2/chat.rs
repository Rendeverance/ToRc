//! chat.rs - LFM2's chat format, from the chat template the model ships with: ChatML turns; the tools go in the system
//! turn as "List of tools: [..]" after the caller's own system text; a call is a Python-style list,
//! <|tool_call_start|>[name(key='value')]<|tool_call_end|>; and each tool answer is a "tool" turn of its own.

use crate::chat::format::{calls_of, split_system, text_of, tojson};
use crate::serve::Call;
use serde_json::{Map, Value};

/// The tags a call is written between: tokens in LFM2's vocabulary.
pub const CALL_TAGS: (&str, &str) = ("<|tool_call_start|>", "<|tool_call_end|>");

pub fn prompt(msgs: &[Value], tools: &[Value]) -> String {
    let (sys, rest) = split_system(msgs);
    let mut sys = sys.unwrap_or_default();
    if !tools.is_empty() {
        if !sys.is_empty() {
            sys += "\n";
        }
        sys += &format!("List of tools: [{}]", tools.iter().map(tojson).collect::<Vec<_>>().join(", "));
    }
    let mut s = if sys.is_empty() { String::new() } else { format!("<|im_start|>system\n{sys}<|im_end|>\n") };
    for m in rest {
        let role = m["role"].as_str().unwrap_or("user");
        let mut body = text_of(m);
        let calls = calls_of(m);
        if role == "assistant" && !calls.is_empty() {
            let list: Vec<String> = calls
                .into_iter()
                .map(|(name, args)| {
                    // as the template writes them: text in single quotes, as it is; an object as JSON; the rest as
                    // Python
                    let a: Vec<String> = args
                        .into_iter()
                        .map(|(k, v)| match v {
                            Value::String(x) => format!("{k}='{x}'"),
                            Value::Object(_) => format!("{k}={}", tojson(&v)),
                            v => format!("{k}={}", py_str(&v)),
                        })
                        .collect();
                    format!("{name}({})", a.join(", "))
                })
                .collect();
            body += &format!("<|tool_call_start|>[{}]<|tool_call_end|>", list.join(", "));
        }
        s += &format!("<|im_start|>{role}\n{body}<|im_end|>\n");
    }
    s + "<|im_start|>assistant\n"
}

/// A value as Python's str() writes it.
fn py_str(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Null => "None".to_string(),
        Value::String(s) => format!("'{s}'"),
        Value::Array(a) => format!("[{}]", a.iter().map(py_str).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => {
            format!("{{{}}}", o.iter().map(|(k, x)| format!("'{k}': {}", py_str(x))).collect::<Vec<_>>().join(", "))
        }
        Value::Number(n) => n.to_string(),
    }
}

/// The calls in a reply: a list of Python-style calls, [get_weather(city="Paris"), now()]. Values are Python or JSON
/// literals: strings in either quote, numbers, True/False/None (or true/false/null), lists and dicts.
pub fn read(bodies: &[String]) -> Vec<Call> {
    let mut out = Vec::new();
    for b in bodies {
        let mut p = Py { s: b.trim().as_bytes(), i: 0 };
        let list = p.eat(b'[');
        loop {
            p.ws();
            let Some(name) = p.ident() else { break };
            if !p.eat(b'(') {
                break;
            }
            let mut args = Map::new();
            loop {
                p.ws();
                if p.eat(b')') {
                    break;
                }
                let Some(key) = p.ident() else { break };
                p.ws();
                if !p.eat(b'=') {
                    break;
                }
                let Some(v) = p.value() else { break };
                args.insert(key, v);
                p.ws();
                p.eat(b',');
            }
            out.push(Call { name, args });
            p.ws();
            if !(list && p.eat(b',')) {
                break;
            }
        }
    }
    out
}

/// A small reader for Python literals, enough for LFM2's calls.
struct Py<'a> {
    s: &'a [u8],
    i: usize,
}

impl Py<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        let hit = self.s.get(self.i) == Some(&c);
        self.i += hit as usize;
        hit
    }

    fn ident(&mut self) -> Option<String> {
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || b"_.-".contains(&self.s[self.i])) {
            self.i += 1;
        }
        (self.i > start).then(|| String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    fn value(&mut self) -> Option<Value> {
        self.ws();
        match *self.s.get(self.i)? {
            q @ (b'"' | b'\'') => {
                self.i += 1;
                let mut out = Vec::new();
                while let Some(&c) = self.s.get(self.i) {
                    self.i += 1;
                    match c {
                        b'\\' => {
                            let e = *self.s.get(self.i)?;
                            self.i += 1;
                            out.extend_from_slice(match e {
                                b'n' => b"\n",
                                b't' => b"\t",
                                b'r' => b"\r",
                                _ => std::slice::from_ref(&e),
                            });
                        }
                        c if c == q => return Some(Value::String(String::from_utf8_lossy(&out).into_owned())),
                        c => out.push(c),
                    }
                }
                None
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                while !self.eat(b']') {
                    a.push(self.value()?);
                    self.eat(b',');
                }
                Some(Value::Array(a))
            }
            b'{' => {
                self.i += 1;
                let mut o = Map::new();
                while !self.eat(b'}') {
                    let k = match self.value()? {
                        Value::String(k) => k,
                        v => v.to_string(),
                    };
                    if !self.eat(b':') {
                        return None;
                    }
                    o.insert(k, self.value()?);
                    self.eat(b',');
                }
                Some(Value::Object(o))
            }
            _ => {
                let start = self.i;
                while self.i < self.s.len() && !b",)]} \t\r\n".contains(&self.s[self.i]) {
                    self.i += 1;
                }
                // nothing read (a stray bracket where a value should be): not a value, so a list or a dict can't
                // go round on it for ever
                if self.i == start {
                    return None;
                }
                let w = std::str::from_utf8(&self.s[start..self.i]).ok()?;
                Some(match w {
                    "True" | "true" => Value::Bool(true),
                    "False" | "false" => Value::Bool(false),
                    "None" | "null" => Value::Null,
                    _ => serde_json::from_str(w).unwrap_or_else(|_| Value::String(w.to_string())),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::format::fixtures::{chat, named, plain, tools, want};
    use serde_json::json;

    // The prompt text, pinned byte for byte: the same as the model's own template gives for this chat. A plain chat is
    // ChatML, as the Qwen models write it.
    #[test]
    fn the_prompt_text() {
        assert_eq!(prompt(&plain(), &[]), crate::models::qwen::chat::prompt(&plain(), &[]), "plain");
        assert_eq!(
            prompt(&chat(), &tools()),
            want(
                r##""<|im_start|>system\nBe brief.\nList of tools: [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}]<|im_end|>\n<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|><|im_end|>\n<|im_start|>tool\n17C, light rain<|im_end|>\n<|im_start|>assistant\nIt is 17C with light rain.<|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n""##
            ),
            "tools"
        );
    }

    #[test]
    fn calls_are_read() {
        let bodies = vec![
            "[get_weather(city=\"Paris\")]".to_string(),
            "[calc(n=42, note='it\\'s 1', opts={\"a\": [1, 2.5]}, on=True, off=None), now()]".to_string(),
            "[broken(".to_string(),
        ];
        assert_eq!(
            named(&read(&bodies)),
            vec![
                ("get_weather".to_string(), json!({"city": "Paris"})),
                (
                    "calc".to_string(),
                    json!({"n": 42, "note": "it's 1", "opts": {"a": [1, 2.5]}, "on": true, "off": null})
                ),
                ("now".to_string(), json!({})),
                ("broken".to_string(), json!({})),
            ]
        );
    }
}
