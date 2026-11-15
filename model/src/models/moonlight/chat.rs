//! chat.rs - Moonlight's chat format (Kimi's), from the chat template the model ships with: each turn is
//! "<|im_system|>", "<|im_user|>" or "<|im_assistant|>", the role name, "<|im_middle|>", the text and "<|im_end|>"; a
//! conversation without a system message starts with "You are a helpful assistant"; and
//! "<|im_assistant|>assistant<|im_middle|>" opens the reply. The template has no tool format, so the tools and the
//! calls are written in Qwen2.5's JSON format (chat/json_calls.rs), and the tools' answers come back as a user turn.

use crate::chat::format::{split_system, text_of, tool_answers};
use crate::chat::json_calls;
use serde_json::Value;

pub fn prompt(msgs: &[Value], tools: &[Value]) -> String {
    let turn = |role: &str, body: &str| {
        let tag = match role {
            "system" => "<|im_system|>",
            "assistant" => "<|im_assistant|>",
            _ => "<|im_user|>",
        };
        format!("{tag}{role}<|im_middle|>{body}<|im_end|>")
    };
    let (sys, rest) = split_system(msgs);
    let sys = sys.or_else(|| (!msgs.is_empty()).then(|| "You are a helpful assistant".to_string()));
    let mut s = json_calls::with_tools(sys, tools).map_or(String::new(), |sys| turn("system", &sys));
    let mut i = 0;
    while i < rest.len() {
        let m = &rest[i];
        match m["role"].as_str().unwrap_or("user") {
            "assistant" => s += &turn("assistant", &format!("{}{}", text_of(m), json_calls::written(m))),
            "tool" => {
                s += &turn("user", &tool_answers(rest, &mut i));
                continue;
            }
            role @ ("user" | "system") => s += &turn(role, &text_of(m)),
            _ => {}
        }
        i += 1;
    }
    s + "<|im_assistant|>assistant<|im_middle|>"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::format::fixtures::{chat, plain, tools, want};

    // The prompt text, pinned byte for byte: the plain chat as the model's own template gives it.
    #[test]
    fn the_prompt_text() {
        assert_eq!(
            prompt(&plain(), &[]),
            want(
                r##""<|im_system|>system<|im_middle|>Be brief.<|im_end|><|im_user|>user<|im_middle|>What is the weather in Paris?<|im_end|><|im_assistant|>assistant<|im_middle|>It is 17C with light rain.<|im_end|><|im_user|>user<|im_middle|>Thanks<|im_end|><|im_assistant|>assistant<|im_middle|>""##
            ),
            "plain"
        );
        assert_eq!(
            prompt(&chat(), &tools()),
            want(
                r##""<|im_system|>system<|im_middle|>Be brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|><|im_user|>user<|im_middle|>What is the weather in Paris?<|im_end|><|im_assistant|>assistant<|im_middle|><tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|im_end|><|im_user|>user<|im_middle|><tool_response>\n17C, light rain\n</tool_response><|im_end|><|im_assistant|>assistant<|im_middle|>It is 17C with light rain.<|im_end|><|im_user|>user<|im_middle|>Thanks<|im_end|><|im_assistant|>assistant<|im_middle|>""##
            ),
            "tools"
        );
    }
}
