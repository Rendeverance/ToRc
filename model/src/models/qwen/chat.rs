//! chat.rs - the Qwen models' chat format: ChatML, with tools written the way the Qwen3.5 / 3.6 chat template writes
//! them (its XML dialect), and the reader of the calls a reply makes in it.

use crate::chat::format::{props, text_of, tojson, tool_answers, value};
use crate::serve::Call;
use serde_json::{Map, Value};

/// The tags a call is written between: tokens in the Qwen vocabularies.
pub const CALL_TAGS: (&str, &str) = ("<tool_call>", "</tool_call>");

/// The call format that follows the tools: the Qwen3.5 / 3.6 chat template's own text (THIRD-PARTY-NOTICES.md).
const TOOL_CALL_RULES: &str = "\n\n\
If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n\
</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified \
format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural \
language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question \
like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The ChatML text of the messages, ending with the assistant's turn opened. With tools, it writes them the way the
/// Qwen3.5 / 3.6 template does: the tools and the call format go in the system message, followed by the caller's own
/// system text; the assistant's calls become <tool_call> blocks; and consecutive tool answers become one user turn of
/// <tool_response> blocks.
pub fn prompt(msgs: &[Value], tools: &[Value]) -> String {
    let mut s = String::new();
    let mut i = 0;
    if !tools.is_empty() {
        let mut sys = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            sys += "\n";
            sys += &tojson(t);
        }
        sys += "\n</tools>";
        sys += TOOL_CALL_RULES;
        if msgs.first().is_some_and(|m| m["role"].as_str() == Some("system")) {
            let u = text_of(&msgs[0]);
            if !u.trim().is_empty() {
                sys += "\n\n";
                sys += u.trim();
            }
            i = 1;
        }
        s += &format!("<|im_start|>system\n{sys}<|im_end|>\n");
    }
    // Qwen3.5 / 3.6 chat template: an assistant turn that follows the final user message is rendered with an
    // (empty) think block, and every turn before that message is rendered with no reasoning at all
    let lastq = msgs.iter().rposition(|m| m["role"].as_str() == Some("user")).unwrap_or(0);
    while i < msgs.len() {
        let m = &msgs[i];
        let role = m["role"].as_str().unwrap_or("user");
        if role == "tool" {
            s += &format!("<|im_start|>user\n{}<|im_end|>\n", tool_answers(msgs, &mut i));
            continue;
        }
        let mut body = text_of(m);
        if role == "assistant" {
            let tn = body.trim().len();
            for (k, c) in m["tool_calls"].as_array().map(|v| v.as_slice()).unwrap_or(&[]).iter().enumerate() {
                body += match (k, tn) {
                    (0, 0) => "<tool_call>\n<function=",
                    (0, _) => "\n\n<tool_call>\n<function=",
                    _ => "\n<tool_call>\n<function=",
                };
                body += c["function"]["name"].as_str().unwrap_or("");
                body += ">\n";
                // the arguments: a JSON string, as OpenAI sends them, or an object
                let args = match &c["function"]["arguments"] {
                    Value::String(a) => serde_json::from_str::<Value>(a).unwrap_or(Value::Null),
                    v => v.clone(),
                };
                if let Value::Object(o) = args {
                    for (k, v) in o {
                        body += &format!("<parameter={k}>\n");
                        body += &match v {
                            Value::String(x) => x,
                            v => tojson(&v),
                        };
                        body += "\n</parameter>\n";
                    }
                }
                body += "</function>\n</tool_call>";
            }
        }
        let think = if role == "assistant" && i > lastq { "<think>\n\n</think>\n\n" } else { "" };
        s += &format!("<|im_start|>{role}\n{think}{body}<|im_end|>\n");
        i += 1;
    }
    s + "<|im_start|>assistant\n"
}

/// The calls in the bodies of a reply's <tool_call> blocks: <function=NAME> followed by <parameter=KEY>\nVALUE\n
/// </parameter> blocks. A body that is not a call is left out.
pub fn read(bodies: &[String], tools: &[Value]) -> Vec<Call> {
    let find = |s: &str, w: &str| s.find(w);
    bodies
        .iter()
        .filter_map(|b| {
            let f = find(b, "<function=")? + 10;
            let gt = f + find(&b[f..], ">")?;
            let name = &b[f..gt];
            let props = props(tools, name);
            let mut args = Map::new();
            let mut p = gt + 1;
            while let Some(q) = find(&b[p..], "<parameter=") {
                let q = p + q + 11;
                let Some(kg) = find(&b[q..], ">").map(|x| q + x) else { break };
                let v0 = kg + 1;
                let Some(ve) = find(&b[v0..], "</parameter>").map(|x| v0 + x) else { break };
                p = ve + 12;
                let v = b[v0..ve].strip_prefix('\n').unwrap_or(&b[v0..ve]);
                let v = v.strip_suffix('\n').unwrap_or(v);
                let key = &b[q..kg];
                args.insert(key.to_string(), value(props, key, v));
            }
            Some(Call { name: name.to_string(), args })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::format::fixtures::{chat, plain, tools, typed_tools, want};
    use crate::serve::tool_calls;

    // The prompt text, pinned byte for byte. A slip in a template doesn't crash: the model just answers worse.
    #[test]
    fn the_prompt_text() {
        assert_eq!(
            prompt(&plain(), &[]),
            want(
                r##""<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\nIt is 17C with light rain.<|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n""##
            ),
            "plain"
        );
        assert_eq!(
            prompt(&chat(), &tools()),
            want(
                r##""<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n17C, light rain\n</tool_response><|im_end|>\n<|im_start|>assistant\nIt is 17C with light rain.<|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n""##
            ),
            "tools"
        );
    }

    #[test]
    fn calls_as_openai_writes_them() {
        let bodies = vec![
            "\n<function=Read>\n<parameter=path>\na/b.txt\n</parameter>\n</function>\n".to_string(),
            "\n<function=calc>\n<parameter=n>\n42\n</parameter>\n<parameter=opts>\n{\"a\": 1, \"b\": [true, null]}\n</parameter>\n<parameter=tags>\n[\"x\", \"y\"]\n</parameter>\n<parameter=note>\n123\n</parameter>\n<parameter=any>\ntrue\n</parameter>\n<parameter=mode>\nfast\n</parameter>\n<parameter=bad>\nnot json\n</parameter>\n</function>\n".to_string(),
            "<function=unknown_tool><parameter=k>{\"k\":1}</parameter></function>".to_string(),
            "\n<function=Write>\n<parameter=text>\nline one\nline two\n\n  indented \"quoted\" \\ back\n</parameter>\n<parameter=path>\nC:\\x\\y.txt\n</parameter>\n</function>\n".to_string(),
            "no function here".to_string(),
            "\n<function=now>\n</function>\n".to_string(),
        ];
        let want: Value = serde_json::from_str(r##"[{"index":0,"id":"call_0","type":"function","function":{"name":"Read","arguments":"{\"path\":\"a/b.txt\"}"}},{"index":1,"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"n\":42,\"opts\":{\"a\":1,\"b\":[true,null]},\"tags\":[\"x\",\"y\"],\"note\":\"123\",\"any\":\"true\",\"mode\":\"fast\",\"bad\":\"not json\"}"}},{"index":2,"id":"call_2","type":"function","function":{"name":"unknown_tool","arguments":"{\"k\":\"{\\\"k\\\":1}\"}"}},{"index":3,"id":"call_3","type":"function","function":{"name":"Write","arguments":"{\"text\":\"line one\\nline two\\n\\n  indented \\\"quoted\\\" \\\\ back\",\"path\":\"C:\\\\x\\\\y.txt\"}"}},{"index":4,"id":"call_4","type":"function","function":{"name":"now","arguments":"{}"}}]"##).unwrap();
        assert_eq!(tool_calls(&read(&bodies, &typed_tools())), want);
        assert_eq!(tool_calls(&read(&bodies, &typed_tools())).to_string(), want.to_string());
    }
}
