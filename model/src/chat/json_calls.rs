//! json_calls.rs - the tool format of Qwen2.5's chat template (Apache-2.0, THIRD-PARTY-NOTICES.md), for models whose
//! own templates have no tool format (ERNIE 4.5, Moonlight): the tools go in the system text as JSON, and a call is a
//! JSON object {"name": .., "arguments": {..}} between <tool_call> and </tool_call>. Those models have no tokens for
//! the tags, so their calls are found in the text as it is written (TextCalls).

use super::format::{calls_of, text_of, tojson};
use crate::serve::Call;
use serde_json::{Map, Value};

/// The system text with the tools after it, as Qwen2.5's chat template writes them.
pub fn with_tools(sys: Option<String>, tools: &[Value]) -> Option<String> {
    if tools.is_empty() {
        return sys;
    }
    let mut s = sys.map(|s| s + "\n\n").unwrap_or_default();
    s += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
          You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    for t in tools {
        s += "\n";
        s += &tojson(t);
    }
    s += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within \
          <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": \
          <args-json-object>}\n</tool_call>";
    Some(s)
}

/// An assistant message's calls, written as with_tools asks for them.
pub fn written(m: &Value) -> String {
    let mut s = String::new();
    for (k, (name, args)) in calls_of(m).into_iter().enumerate() {
        if k > 0 || !text_of(m).is_empty() {
            s += "\n";
        }
        s += &format!(
            "<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
            Value::String(name),
            tojson(&Value::Object(args))
        );
    }
    s
}

/// The calls in the bodies TextCalls found (the arguments may come as a JSON string, and some models write
/// "parameters"). A body that is not a call is left out.
pub fn read(bodies: &[String]) -> Vec<Call> {
    bodies
        .iter()
        .filter_map(|b| {
            let v: Value = serde_json::from_str(b.trim()).ok()?;
            let name = v["name"].as_str()?.to_string();
            let args = match v.get("arguments").or(v.get("parameters")) {
                Some(Value::Object(o)) => o.clone(),
                Some(Value::String(s)) => serde_json::from_str::<Map<String, Value>>(s).unwrap_or_default(),
                _ => Map::new(),
            };
            Some(Call { name, args })
        })
        .collect()
}

/// Finds calls written as plain text. The reply's text goes through `feed` as it is made: what can't be part of a call
/// comes back to be sent on, a call's body goes into `calls`, and text that might be the start of "<tool_call>" waits
/// until it is clear either way.
#[derive(Default)]
pub struct TextCalls {
    held: String,
    in_call: bool,
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

impl TextCalls {
    pub fn feed(&mut self, s: &str, calls: &mut Vec<String>) -> String {
        self.held.push_str(s);
        let mut out = String::new();
        loop {
            if self.in_call {
                let Some(j) = self.held.find(CLOSE) else { return out };
                calls.push(self.held[..j].to_string());
                self.held.drain(..j + CLOSE.len());
                self.in_call = false;
            } else if let Some(i) = self.held.find(OPEN) {
                out.push_str(&self.held[..i]);
                self.held.drain(..i + OPEN.len());
                self.in_call = true;
            } else {
                // keep back the longest end of the text that "<tool_call>" could still start with
                let keep = (1..OPEN.len()).rev().find(|&k| self.held.ends_with(&OPEN[..k])).unwrap_or(0);
                let cut = self.held.len() - keep;
                out.push_str(&self.held[..cut]);
                self.held.drain(..cut);
                return out;
            }
        }
    }

    /// At the reply's end: a call cut off before its close is kept (the reader decides), and held text is sent.
    pub fn finish(&mut self, calls: &mut Vec<String>) -> String {
        let rest = std::mem::take(&mut self.held);
        if std::mem::take(&mut self.in_call) {
            calls.push(rest);
            String::new()
        } else {
            rest
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::format::fixtures::named;
    use serde_json::json;

    #[test]
    fn calls_are_read() {
        let bodies = vec![
            "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n".to_string(),
            "{\"name\": \"calc\", \"arguments\": \"{\\\"n\\\": 42}\"}".to_string(),
            "{\"name\": \"now\"}".to_string(),
            "get_weather(Paris)".to_string(),
        ];
        assert_eq!(
            named(&read(&bodies)),
            vec![
                ("get_weather".to_string(), json!({"city": "Paris"})),
                ("calc".to_string(), json!({"n": 42})),
                ("now".to_string(), json!({})),
            ]
        );
    }

    #[test]
    fn text_calls_split_anywhere() {
        let reply = "Let me check. <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n\
                     </tool_call> and a < sign <tool_ca";
        // every way of cutting the reply into pieces gives the same text and the same call
        for step in 1..=reply.len() {
            let (mut tc, mut calls, mut text) = (TextCalls::default(), Vec::new(), String::new());
            let mut i = 0;
            while i < reply.len() {
                let j = (i + step).min(reply.len());
                text += &tc.feed(&reply[i..j], &mut calls);
                i = j;
            }
            text += &tc.finish(&mut calls);
            assert_eq!(text, "Let me check.  and a < sign <tool_ca", "step {step}");
            assert_eq!(named(&read(&calls)), vec![("get_weather".to_string(), json!({"city": "Paris"}))]);
        }
        // a call cut off by the length limit is kept for the reader
        let (mut tc, mut calls) = (TextCalls::default(), Vec::new());
        assert_eq!(tc.feed("<tool_call>{\"name\": \"now\"}", &mut calls), "");
        assert_eq!(tc.finish(&mut calls), "");
        assert_eq!(calls, vec!["{\"name\": \"now\"}".to_string()]);
    }
}
