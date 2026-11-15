//! chat.rs - GLM-4.7-Flash's chat format, from the chat template the model ships with: "[gMASK]<sop>", then
//! "<|system|>", "<|user|>" and "<|assistant|>" turns with no newlines, and its own tool format.

use crate::chat::format::{calls_of, props, text_of, tojson, value};
use crate::serve::Call;
use serde_json::{Map, Value};

/// The tags a call is written between: tokens in GLM's vocabulary.
pub const CALL_TAGS: (&str, &str) = ("<tool_call>", "</tool_call>");

/// The prompt, as llama-server renders the template (--jinja). An assistant turn opens with its reasoning as
/// "<think>...</think>" when its content carries one, else with "</think>"; the prompt ends with "<|assistant|>" (the
/// opener, "<think>" or "</think>", comes from the caller). With tools, the template's own tool section comes first,
/// an assistant's calls are written as <tool_call>NAME<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>,
/// and the tools' answers as <tool_response> blocks after one <|observation|>.
pub fn prompt(msgs: &[Value], tools: &[Value]) -> String {
    let mut s = String::from("[gMASK]<sop>");
    if !tools.is_empty() {
        s += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
              You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
        for t in tools {
            s += &tojson(t);
            s += "\n";
        }
        s += "</tools>\n\nFor each function call, output the function name and arguments within the following XML \
              format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>\
              <arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }
    let mut after_tool = false;
    for m in msgs {
        let body = text_of(m);
        let role = m["role"].as_str().unwrap_or("user");
        match role {
            "user" => s += &format!("<|user|>{body}"),
            "system" => s += &format!("<|system|>{body}"),
            "assistant" => {
                s += "<|assistant|>";
                match body.split_once("</think>") {
                    Some((r, c)) => {
                        let r = r.rsplit("<think>").next().unwrap_or("").trim();
                        s += &format!("<think>{r}</think>{}", c.trim());
                    }
                    None => s += &format!("</think>{}", body.trim()),
                }
                for (name, args) in calls_of(m) {
                    s += &format!("<tool_call>{name}");
                    for (k, v) in args {
                        let v = match v {
                            Value::String(x) => x,
                            v => tojson(&v),
                        };
                        s += &format!("<arg_key>{k}</arg_key><arg_value>{v}</arg_value>");
                    }
                    s += "</tool_call>";
                }
            }
            "tool" => {
                if !after_tool {
                    s += "<|observation|>";
                }
                s += &format!("<tool_response>{body}</tool_response>");
            }
            _ => {}
        }
        after_tool = role == "tool";
    }
    s + "<|assistant|>"
}

/// The calls in a reply: the name, then <arg_key>/<arg_value> pairs. The template writes a string value as it is and
/// anything else as JSON, so a value is read back by its schema. A body that is not a call is left out.
pub fn read(bodies: &[String], tools: &[Value]) -> Vec<Call> {
    bodies
        .iter()
        .filter_map(|b| {
            let (name, mut rest) = match b.find("<arg_key>") {
                Some(i) => (b[..i].trim(), &b[i..]),
                None => (b.trim(), ""),
            };
            if name.is_empty() || name.contains(char::is_whitespace) {
                return None;
            }
            let props = props(tools, name);
            let mut args = Map::new();
            while let Some(k0) = rest.find("<arg_key>").map(|i| i + 9) {
                let Some(k1) = rest[k0..].find("</arg_key>").map(|i| k0 + i) else { break };
                let Some(v0) = rest[k1..].find("<arg_value>").map(|i| k1 + i + 11) else { break };
                let v1 = rest[v0..].find("</arg_value>").map_or(rest.len(), |i| v0 + i);
                let key = rest[k0..k1].trim();
                args.insert(key.to_string(), value(props, key, &rest[v0..v1]));
                rest = &rest[(v1 + 12).min(rest.len())..];
            }
            Some(Call { name: name.to_string(), args })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::format::fixtures::{chat, named, plain, tools, typed_tools, want};
    use serde_json::json;

    // The prompt text, pinned byte for byte: the same as the model's own template gives for this chat.
    #[test]
    fn the_prompt_text() {
        assert_eq!(
            prompt(&plain(), &[]),
            want(
                r##""[gMASK]<sop><|system|>Be brief.<|user|>What is the weather in Paris?<|assistant|></think>It is 17C with light rain.<|user|>Thanks<|assistant|>""##
            ),
            "plain"
        );
        assert_eq!(
            prompt(&chat(), &tools()),
            want(
                r##""[gMASK]<sop><|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call><|system|>Be brief.<|user|>What is the weather in Paris?<|assistant|></think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><|observation|><tool_response>17C, light rain</tool_response><|assistant|></think>It is 17C with light rain.<|user|>Thanks<|assistant|>""##
            ),
            "tools"
        );
    }

    #[test]
    fn calls_are_read() {
        let bodies = vec![
            "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>".to_string(),
            "calc\n<arg_key>n</arg_key>\n<arg_value>42</arg_value>\n<arg_key>note</arg_key><arg_value>123</arg_value>\
             <arg_key>opts</arg_key><arg_value>{\"a\": [1, 2]}</arg_value>"
                .to_string(),
            "now".to_string(),
            "not a call at all".to_string(),
        ];
        assert_eq!(
            named(&read(&bodies, &typed_tools())),
            vec![
                ("get_weather".to_string(), json!({"city": "Paris"})),
                ("calc".to_string(), json!({"n": 42, "note": "123", "opts": {"a": [1, 2]}})),
                ("now".to_string(), json!({})),
            ]
        );
    }
}
